import os
import shutil
from fastapi import APIRouter, File, UploadFile, Depends, HTTPException, Header
import pdfplumber
import jwt
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_core.documents import Document
import uuid

from ..database.files import UserFiles
# To avoid circular imports, we'll import server clients dynamically inside the endpoints

router = APIRouter(prefix="/api/pdfs", tags=["PDFs"])

JWT_SECRET = os.getenv("JWT_SECRET", "super-secret-key-change-in-prod")

def get_current_user_id(authorization: str = Header(...)):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid token")
    token = authorization.split(" ")[1]
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        return payload["user_id"]
    except Exception:
        raise HTTPException(status_code=401, detail="Token expired or invalid")

@router.post("")
async def upload_pdf(file: UploadFile = File(...), user_id: str = Depends(get_current_user_id)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    upload_dir = os.path.join("uploads", user_id)
    os.makedirs(upload_dir, exist_ok=True)
    filepath = os.path.join(upload_dir, file.filename)
    
    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Extract text using pdfplumber
    text = ""
    try:
        with pdfplumber.open(filepath) as pdf:
            for page in pdf.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
    except Exception as e:
        os.remove(filepath)
        raise HTTPException(status_code=400, detail=f"Failed to read PDF: {str(e)}")
        
    size = os.path.getsize(filepath)
    
    from ..server import files_db, llm_client, vectordb_client
    # Save file metadata
    file_doc = files_db.store_file(user_id, file.filename, filepath, size)
    file_id = str(file_doc)
    
    # Create Vector Embeddings with SemanticChunker
    embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    text_splitter = SemanticChunker(embeddings)
    raw_docs = text_splitter.create_documents([text])
    
    docs = []
    db_chunks = []
    for i, doc in enumerate(raw_docs):
        chunk_id = str(uuid.uuid4())
        doc.metadata = {
            "bookid": file_id,
            "user_id": user_id,
            "chunk_id": chunk_id,
            "chunk_index": i
        }
        docs.append(doc)
        db_chunks.append({
            "chunk_id": chunk_id,
            "user_id": user_id,
            "book_id": file_id,
            "text": doc.page_content,
            "chunk_index": i
        })
        
    if docs:
        vector_docs = llm_client.embed_documents(docs)
        # Store in Pinecone
        vectordb_client.insert_documents(vector_docs)
        
        # Store raw text chunks in MongoDB for Hybrid Search
        files_db.store_chunks(db_chunks)
    
    return {"status": True, "file_id": file_id, "filename": file.filename}

@router.get("")
def get_pdfs(user_id: str = Depends(get_current_user_id)):
    from ..server import files_db
    files = files_db.get_user_files(user_id)
    return {"status": True, "data": files}

@router.delete("/{file_id}")
def delete_pdf(file_id: str, user_id: str = Depends(get_current_user_id)):
    from ..server import files_db, vectordb_client
    file_doc = files_db.get_file(user_id, file_id)
    if not file_doc:
        raise HTTPException(status_code=404, detail="File not found")
        
    # Delete from filesystem
    if os.path.exists(file_doc["filepath"]):
        os.remove(file_doc["filepath"])
        
    # Delete from MongoDB
    files_db.delete_file(user_id, file_id)
    
    # Delete from Pinecone
    try:
        if vectordb_client.index:
            vectordb_client.index.delete(filter={"bookid": {"$eq": file_id}})
    except Exception as e:
        print("Warning: failed to delete vectors from Pinecone", e)
        
    return {"status": True, "message": "Deleted successfully"}
