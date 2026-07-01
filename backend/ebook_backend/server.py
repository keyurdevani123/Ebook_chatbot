import os
from contextlib import asynccontextmanager
from logging import getLogger
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import JSONResponse
from ebook_backend.routers.files import get_current_user_id

from ebook_backend.agents import RetrieverAgent, SynthesizerAgent
from ebook_backend.database.messages import Messages
from ebook_backend.database.files import UserFiles
from ebook_backend.database.users import Users
from ebook_backend.database.sessions import ChatSessions
from ebook_backend.llm import LLMClient
from pydantic import BaseModel
from ebook_backend.models.api import (
    CrossBookChatRequest,
    CrossBookChatResponse,
    DeleteMessagesNotFoundResponse,
    DeleteMessagesSuccessResponse,
    GetMessagesResponse,
    PostMessage,
    PostMessageResponse,
    SearchEbookResponse,
    SearchQuery,
)
from ebook_backend.store.pinecone.client import PineconeDBClient
from fastapi.middleware.cors import CORSMiddleware
from ebook_backend.routers.auth import router as auth_router
from ebook_backend.routers.files import router as files_router

logger = getLogger(__name__)

import asyncio
import httpx

# ---------------------------------------------------------------------------
# Module-level stubs — populated during lifespan startup, NOT at import time.
# This ensures tests can mock these constructors before startup fires.
# ---------------------------------------------------------------------------
llm_client: Optional[LLMClient] = None
vectordb_client: Optional[PineconeDBClient] = None
messages_db: Optional[Messages] = None
files_db: Optional[UserFiles] = None
sessions_db: Optional[ChatSessions] = None
users_db: Optional[Users] = None
retriever: Optional[RetrieverAgent] = None
synthesizer: Optional[SynthesizerAgent] = None

async def keep_awake_task(url: str):
    """Background task to ping the server every 14 minutes to prevent sleeping on free tiers."""
    while True:
        try:
            await asyncio.sleep(14 * 60)
            async with httpx.AsyncClient() as client:
                await client.get(f"{url}/api/ping", timeout=10.0)
                logger.info("Server | keep_awake_task | Pinged self successfully")
        except Exception as e:
            logger.warning(f"Server | keep_awake_task | Failed to ping self: {e}")

# ---------------------------------------------------------------------------
# Application lifespan — initializes all heavy clients on startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize clients on startup so tests can mock before they're called."""
    global llm_client, vectordb_client, messages_db, files_db, sessions_db, users_db, retriever, synthesizer

    logger.info("Server | Startup | Initializing clients...")
    llm_client = LLMClient()
    vectordb_client = PineconeDBClient()
    vectordb_client.initialize_collection()
    messages_db = Messages()
    messages_db.initialize_indexes()
    files_db = UserFiles()
    files_db.initialize_indexes()
    sessions_db = ChatSessions()
    sessions_db.initialize_indexes()
    users_db = Users()
    users_db.initialize_indexes()
    retriever = RetrieverAgent(vectordb_client, llm_client)
    synthesizer = SynthesizerAgent(retriever, llm_client)
    logger.info("Server | Startup | Ready.")

    render_url = os.getenv("RENDER_EXTERNAL_URL")
    keep_awake = None
    if render_url:
        logger.info(f"Server | Startup | Starting keep_awake task for {render_url}")
        keep_awake = asyncio.create_task(keep_awake_task(render_url))

    yield  # Server runs here

    if keep_awake:
        keep_awake.cancel()

    logger.info("Server | Shutdown.")


app = FastAPI(
    title="E-Book RAG API",
    description="RAG-powered multi-agent knowledge system for e-books.",
    version="2.0.0",
    docs_url="/",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(files_router)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health", tags=["System"])
async def health():
    """Simple liveness probe."""
    return {"status": "ok"}

@app.get("/api/ping", tags=["System"])
async def ping():
    """Endpoint for keep-awake cron jobs."""
    return {"status": "pong"}


# ---------------------------------------------------------------------------
# E-Book — Message history
# ---------------------------------------------------------------------------

@app.get(
    "/api/users/{user_id}/books/{book_id}/messages",
    tags=["E-Book"],
    response_model=GetMessagesResponse,
)
async def get_ebook_messages(
    user_id: str,
    book_id: str,
    skip: int = 0,
    limit: int = 10,
):
    """Get paginated chat history for a user + book pair."""
    try:
        msgs = messages_db.get_messages(
            user_id=user_id, book_id=book_id, skip=skip, limit=limit
        )
        return JSONResponse({"status": True, "data": msgs}, 200)
    except Exception as exc:
        logger.exception("get_ebook_messages | %s", exc)
        return JSONResponse({"status": False, "error": str(exc)}, 500)


@app.post(
    "/api/users/{user_id}/books/{book_id}/messages",
    tags=["E-Book"],
    response_model=PostMessageResponse,
)
async def post_ebook_message(
    user_id: str,
    book_id: str,
    message: PostMessage,
):
    """Chat with a single book using RAG + Groq.

    Set **debug=true** to include citation details in the response metadata.
    """
    try:
        messages_db.store_message(
            message.text,
            user_id=user_id,
            book_id=book_id,
            type="Human",
            lang=message.lang,
            model=message.model,
        )

        history = messages_db.get_last_six_messages(
            user_id=user_id, book_id=book_id, skip=1
        )
        language = message.lang or "same language as input"

        result = await synthesizer.synthesize(
            query=message.text,
            book_ids=[book_id],
            history=history,
            model=message.model,
            language=language,
        )

        message_id = messages_db.store_message(
            result.answer,
            user_id=user_id,
            book_id=book_id,
            type="AI",
            lang=message.lang,
            model=message.model,
        )

        metadata = {}
        if message.debug:
            metadata = {
                "citations": [c.model_dump() for c in result.citations],
                "confidence_scores": result.confidence_scores,
            }

        return JSONResponse(
            {
                "status": True,
                "data": {
                    "_id": str(message_id),
                    "text": result.answer,
                    "metadata": metadata,
                },
            },
            200,
        )
    except Exception as exc:
        logger.exception("post_ebook_message | %s", exc)
        return JSONResponse({"status": False, "error": str(exc)}, 500)


@app.delete(
    "/api/users/{user_id}/books/{book_id}/messages",
    tags=["E-Book"],
    responses={
        200: {"model": DeleteMessagesSuccessResponse},
        404: {"model": DeleteMessagesNotFoundResponse},
    },
)
async def delete_ebook_messages(user_id: str, book_id: str):
    """Clear the entire chat history for a user + book pair."""
    try:
        deleted_count = messages_db.delete_messages(user_id=user_id, book_id=book_id)
        if deleted_count > 0:
            return JSONResponse(
                {"status": True, "data": {"deleted_count": deleted_count}}, 200
            )
        return JSONResponse(
            {"status": False, "message": "No messages found to delete"}, 404
        )
    except Exception as exc:
        logger.exception("delete_ebook_messages | %s", exc)
        return JSONResponse({"status": False, "error": str(exc)}, 500)


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------

class CreateSessionRequest(BaseModel):
    title: str = "New Chat"

@app.get("/api/users/{user_id}/sessions", tags=["Sessions"])
async def get_user_sessions(user_id: str):
    """Get all chat sessions for a user."""
    try:
        sessions = sessions_db.get_user_sessions(user_id=user_id)
        return JSONResponse({"status": True, "data": sessions}, 200)
    except Exception as exc:
        logger.exception("get_user_sessions | %s", exc)
        return JSONResponse({"status": False, "error": str(exc)}, 500)

@app.post("/api/users/{user_id}/sessions", tags=["Sessions"])
async def create_user_session(user_id: str, request: CreateSessionRequest):
    """Create a new chat session."""
    try:
        session_id = sessions_db.create_session(user_id=user_id, title=request.title)
        return JSONResponse({"status": True, "data": {"session_id": session_id, "title": request.title}}, 200)
    except Exception as exc:
        logger.exception("create_user_session | %s", exc)
        return JSONResponse({"status": False, "error": str(exc)}, 500)

class UpdateSessionRequest(BaseModel):
    title: str

@app.put("/api/users/{user_id}/sessions/{session_id}")
def rename_session(user_id: str, session_id: str, payload: dict, token: str = Depends(get_current_user_id)):
    title = payload.get("title")
    if not title:
        raise HTTPException(status_code=400, detail="Title is required")
    
    from bson.objectid import ObjectId
    success = sessions_db.update_one(
        {"_id": ObjectId(session_id), "user_id": user_id},
        {"$set": {"title": title}}
    )
    return {"status": success}

@app.put("/api/users/{user_id}/sessions/{session_id}/auto-title")
def auto_title_session(user_id: str, session_id: str, payload: dict, token: str = Depends(get_current_user_id)):
    first_message = payload.get("message")
    if not first_message:
        raise HTTPException(status_code=400, detail="Message is required")
        
    prompt = f"Generate a very short, concise title (max 5 words) for a chat session that starts with this message. Do not use quotes or prefixes. Message: '{first_message}'"
    messages = [{"role": "user", "content": prompt}]
    title = llm_client.chat(messages, model="llama-3.1-8b-instant", temperature=0.7).strip('"\'. ')
    
    from bson.objectid import ObjectId
    success = sessions_db.update_one(
        {"_id": ObjectId(session_id), "user_id": user_id},
        {"$set": {"title": title}}
    )
    return {"status": success, "title": title}

@app.delete("/api/users/{user_id}/sessions/{session_id}", tags=["Sessions"])
async def delete_user_session(user_id: str, session_id: str):
    """Delete a chat session and all its messages."""
    try:
        deleted_count = sessions_db.delete_session(user_id=user_id, session_id=session_id)
        messages_db.delete_messages(user_id=user_id, book_id=session_id)
        if deleted_count > 0:
            return JSONResponse({"status": True, "data": {"deleted_count": deleted_count}}, 200)
        return JSONResponse({"status": False, "message": "No session found to delete"}, 404)
    except Exception as exc:
        logger.exception("delete_user_session | %s", exc)
        return JSONResponse({"status": False, "error": str(exc)}, 500)


# ---------------------------------------------------------------------------
# Session Message history
# ---------------------------------------------------------------------------

@app.get(
    "/api/users/{user_id}/sessions/{session_id}/messages",
    tags=["Global"],
    response_model=GetMessagesResponse,
)
async def get_session_messages(
    user_id: str,
    session_id: str,
    skip: int = 0,
    limit: int = 10,
):
    """Get paginated chat history for a session."""
    try:
        msgs = messages_db.get_messages(
            user_id=user_id, book_id=session_id, skip=skip, limit=limit
        )
        return JSONResponse({"status": True, "data": msgs}, 200)
    except Exception as exc:
        logger.exception("get_session_messages | %s", exc)
        return JSONResponse({"status": False, "error": str(exc)}, 500)


@app.post(
    "/api/users/{user_id}/sessions/{session_id}/messages",
    tags=["Global"],
    response_model=PostMessageResponse,
)
async def post_session_message(
    user_id: str,
    session_id: str,
    message: PostMessage,
):
    """Chat globally across all of the user's uploaded PDFs within a session."""
    try:
        messages_db.store_message(
            message.text,
            user_id=user_id,
            book_id=session_id,
            type="Human",
            lang=message.lang,
            model=message.model,
        )

        history = messages_db.get_last_six_messages(
            user_id=user_id, book_id=session_id, skip=1
        )
        language = message.lang or "same language as input"

        # Retrieve all of the user's uploaded PDFs
        user_files = files_db.get_user_files(user_id=user_id)
        all_book_ids = [str(f["_id"]) for f in user_files]

        result = await synthesizer.synthesize(
            query=message.text,
            book_ids=all_book_ids,
            history=history,
            model=message.model,
            language=language,
        )

        message_id = messages_db.store_message(
            result.answer,
            user_id=user_id,
            book_id=session_id,
            type="AI",
            lang=message.lang,
            model=message.model,
        )

        metadata = {}
        if message.debug:
            metadata = {
                "citations": [c.model_dump() for c in result.citations],
                "confidence_scores": result.confidence_scores,
            }

        return JSONResponse(
            {
                "status": True,
                "data": {
                    "_id": str(message_id),
                    "text": result.answer,
                    "metadata": metadata,
                },
            },
            200,
        )
    except Exception as exc:
        logger.exception("post_session_message | %s", exc)
        return JSONResponse({"status": False, "error": str(exc)}, 500)


@app.delete(
    "/api/users/{user_id}/sessions/{session_id}/messages",
    tags=["Global"],
    responses={
        200: {"model": DeleteMessagesSuccessResponse},
        404: {"model": DeleteMessagesNotFoundResponse},
    },
)
async def delete_session_messages(user_id: str, session_id: str):
    """Clear the entire chat history for a session."""
    try:
        deleted_count = messages_db.delete_messages(user_id=user_id, book_id=session_id)
        if deleted_count > 0:
            return JSONResponse(
                {"status": True, "data": {"deleted_count": deleted_count}}, 200
            )
        return JSONResponse(
            {"status": False, "message": "No messages found to delete"}, 404
        )
    except Exception as exc:
        logger.exception("delete_session_messages | %s", exc)
        return JSONResponse({"status": False, "error": str(exc)}, 500)

# ---------------------------------------------------------------------------
# E-Book — Internal vector search
# ---------------------------------------------------------------------------

@app.post(
    "/api/users/{user_id}/books/{book_id}/search",
    name="Search Ebook Content",
    tags=["E-Book"],
    response_model=SearchEbookResponse,
)
async def search_similar_ebook_content(
    user_id: str,
    book_id: str,
    query: SearchQuery,
):
    """[Internal] Semantic similarity search — returns raw chunk texts without LLM."""
    try:
        chunks = retriever.retrieve(
            query=query.text,
            book_ids=[book_id],
            limit=query.limit,
        )
        results = [c.text for c in chunks]
        return JSONResponse(
            {"status": True, "data": {"count": len(results), "results": results}},
            200,
        )
    except Exception as exc:
        logger.exception("search_similar_ebook_content | %s", exc)
        return JSONResponse({"status": False, "error": str(exc)}, 500)


# ---------------------------------------------------------------------------
# Cross-book synthesis
# ---------------------------------------------------------------------------

@app.post(
    "/api/books/chat",
    tags=["Cross-Book"],
    response_model=CrossBookChatResponse,
)
async def cross_book_chat(request: CrossBookChatRequest):
    """Chat across multiple books — parallel retrieval with weighted confidence scores."""
    try:
        result = await synthesizer.synthesize(
            query=request.text,
            book_ids=request.book_ids,
            model=request.model,
            language=request.lang or "same language as input",
        )

        response_data: dict = {
            "text": result.answer,
            "confidence_scores": result.confidence_scores,
        }
        if request.debug:
            response_data["citations"] = [c.model_dump() for c in result.citations]

        return JSONResponse({"status": True, "data": response_data}, 200)
    except Exception as exc:
        logger.exception("cross_book_chat | %s", exc)
        return JSONResponse({"status": False, "error": str(exc)}, 500)
