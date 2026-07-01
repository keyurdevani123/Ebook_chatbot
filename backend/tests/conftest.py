"""conftest.py — Shared pytest fixtures for all test modules."""

import os
import tempfile
from typing import List
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from ebook_backend.models.document import CitedChunk, ScoredDocument, VectorDocument


# ---------------------------------------------------------------------------
# Environment stub — prevents real API calls by injecting fake keys
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def mock_env(monkeypatch):
    """Inject dummy env vars so imports don't fail without a real .env.
    These run before every test, including before lifespan fires.
    """
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
    monkeypatch.setenv("PINECONE_API_KEY", "test-pinecone-key")
    monkeypatch.setenv("PINECONE_REGION", "us-east-1")
    monkeypatch.setenv("MONGODB_HOST", "mongodb://localhost:27017")


# ---------------------------------------------------------------------------
# Sample data helpers
# ---------------------------------------------------------------------------

def make_cited_chunk(
    text: str = "Sample text content",
    book_id: str = "book-001",
    score: float = 0.85,
    page: int = 5,
) -> CitedChunk:
    return CitedChunk(
        text=text,
        book_id=book_id,
        source=f"{book_id}.pdf",
        page=page,
        chapter="Chapter 1",
        chunk_index=0,
        score=score,
    )


def make_scored_document(score: float = 0.85, book_id: str = "book-001") -> ScoredDocument:
    return ScoredDocument(
        id="vec-123",
        score=score,
        metadata={
            "text": "Sample chunk text",
            "book_id": book_id,
            "source": f"{book_id}.pdf",
            "page": 5,
            "chapter": "Introduction",
            "chunk_index": 0,
        },
        vector=[],
    )


# ---------------------------------------------------------------------------
# Sample text fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_pdf_path():
    """Create a temporary plain-text file for chunker tests (no PDF binary needed)."""
    content = "\n".join([
        f"Page {i}: " + "The quick brown fox jumps over the lazy dog. " * 30
        for i in range(1, 6)
    ])
    with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False, encoding="utf-8") as f:
        f.write(content)
        return f.name


# ---------------------------------------------------------------------------
# Mock LLMClient
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_llm_client():
    client = MagicMock()
    client.embed_query.return_value = [0.1] * 384
    client.embed_documents.return_value = [
        VectorDocument(vector=[0.1] * 384, metadata={"text": "doc", "book_id": "b1"})
    ]
    client.chat.return_value = "This is a synthesized answer."
    return client


# ---------------------------------------------------------------------------
# Mock PineconeDBClient
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_pinecone_client():
    client = MagicMock()
    client.query.return_value = [
        make_scored_document(score=0.90, book_id="book-001"),
        make_scored_document(score=0.75, book_id="book-001"),
    ]
    return client


# ---------------------------------------------------------------------------
# FastAPI TestClient — uses context manager to trigger lifespan AFTER mocks
# ---------------------------------------------------------------------------

@pytest.fixture
def api_client(mock_llm_client, mock_pinecone_client):
    """TestClient with all external services mocked.

    Using ``with TestClient(app) as client:`` triggers FastAPI's lifespan
    startup handler *inside* the patch context, so LLMClient() and
    PineconeDBClient() calls during startup receive the mocks instead of
    trying to make real API calls.
    """
    mock_messages_db = MagicMock()
    mock_messages_db.get_messages.return_value = []
    mock_messages_db.get_last_six_messages.return_value = []
    mock_messages_db.store_message.return_value = "msg-id-123"
    mock_messages_db.delete_messages.return_value = 2

    with (
        patch("ebook_backend.server.LLMClient", return_value=mock_llm_client),
        patch("ebook_backend.server.PineconeDBClient", return_value=mock_pinecone_client),
        patch("ebook_backend.server.Messages", return_value=mock_messages_db),
    ):
        from ebook_backend.server import app

        # The `with TestClient(app) as client:` block triggers the lifespan
        # startup AFTER patches are in place — no real API calls.
        with TestClient(app) as client:
            yield client
