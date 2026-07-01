from enum import Enum
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from .document import CitedChunk


# ---------------------------------------------------------------------------
# Shared
# ---------------------------------------------------------------------------

class MessageType(str, Enum):
    ai = "AI"
    human = "Human"


class Message(BaseModel):
    _id: str
    text: str
    user_id: str
    book_id: str
    type: MessageType
    createdAt: int


class PostMessage(BaseModel):
    model: Literal[
        "llama-3.1-8b-instant",
        "llama3-70b-8192",
        "mixtral-8x7b-32768",
    ] = Field("llama-3.1-8b-instant")
    text: str
    lang: Optional[str] = Field(None)
    debug: bool = Field(False)


class SearchQuery(BaseModel):
    text: str
    limit: int = Field(4, ge=1, le=20)


# ---------------------------------------------------------------------------
# E-Book single-book responses
# ---------------------------------------------------------------------------

class GetMessagesResponse(BaseModel):
    status: bool
    data: List[Message]


class MessageMetadata(BaseModel):
    citations: Optional[List[CitedChunk]] = None
    confidence_scores: Optional[Dict[str, float]] = None


class MessageData(BaseModel):
    _id: str
    text: str
    metadata: Optional[MessageMetadata] = None


class PostMessageResponse(BaseModel):
    status: bool
    data: MessageData


class DeleteMessagesData(BaseModel):
    deleted_count: int


class DeleteMessagesSuccessResponse(BaseModel):
    status: bool = True
    data: DeleteMessagesData


class DeleteMessagesNotFoundResponse(BaseModel):
    status: bool = False
    message: str = "No messages found to delete"


class SearchEbookData(BaseModel):
    count: int
    results: List[str]


class SearchEbookResponse(BaseModel):
    status: bool
    data: SearchEbookData


# ---------------------------------------------------------------------------
# Cross-book synthesis
# ---------------------------------------------------------------------------

class CrossBookChatRequest(BaseModel):
    """Request body for cross-book chat using the SynthesizerAgent."""
    book_ids: List[str] = Field(..., min_length=1, description="List of book IDs to query across")
    text: str = Field(..., description="User question")
    lang: Optional[str] = Field(None, description="Response language (default: auto-detect)")
    model: Literal[
        "llama-3.1-8b-instant",
        "llama3-70b-8192",
        "mixtral-8x7b-32768",
    ] = Field("llama-3.1-8b-instant")
    debug: bool = Field(False)


class CrossBookChatData(BaseModel):
    text: str
    citations: List[CitedChunk]
    confidence_scores: Dict[str, float]


class CrossBookChatResponse(BaseModel):
    status: bool
    data: CrossBookChatData
