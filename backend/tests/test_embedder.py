"""test_embedder.py — Unit tests for EmbedderAgent."""

from langchain_core.documents import Document

from ebook_backend.agents.embedder import EmbedderAgent
from ebook_backend.models.document import VectorDocument


def make_docs(n: int = 3):
    return [
        Document(page_content=f"Document {i} content about networking.", metadata={"book_id": "b1"})
        for i in range(n)
    ]


class TestEmbedderAgent:

    def test_returns_correct_count(self, mock_llm_client):
        """EmbedderAgent should return one VectorDocument per input Document."""
        mock_llm_client.embed_documents.return_value = [
            VectorDocument(vector=[0.1] * 384, metadata={"text": f"doc {i}", "book_id": "b1"})
            for i in range(3)
        ]
        embedder = EmbedderAgent(mock_llm_client)
        docs = make_docs(3)
        result = embedder.embed(docs)
        assert len(result) == 3

    def test_returns_vector_documents(self, mock_llm_client):
        """Output items should be VectorDocument instances."""
        mock_llm_client.embed_documents.return_value = [
            VectorDocument(vector=[0.2] * 384, metadata={"text": "doc", "book_id": "b1"})
        ]
        embedder = EmbedderAgent(mock_llm_client)
        result = embedder.embed(make_docs(1))
        assert all(isinstance(v, VectorDocument) for v in result)

    def test_vector_has_correct_dimension(self, mock_llm_client):
        """Vectors must be 384-dimensional (fastembed bge-small-en-v1.5)."""
        mock_llm_client.embed_documents.return_value = [
            VectorDocument(vector=[0.1] * 384, metadata={"text": "doc", "book_id": "b1"})
        ]
        embedder = EmbedderAgent(mock_llm_client)
        result = embedder.embed(make_docs(1))
        assert len(result[0].vector) == 384

    def test_empty_input_returns_empty_list(self, mock_llm_client):
        """Embedding zero documents should return an empty list without calling LLM."""
        embedder = EmbedderAgent(mock_llm_client)
        result = embedder.embed([])
        assert result == []
        mock_llm_client.embed_documents.assert_not_called()

    def test_metadata_preserved(self, mock_llm_client):
        """VectorDocument metadata should include source document metadata."""
        mock_llm_client.embed_documents.return_value = [
            VectorDocument(
                vector=[0.1] * 384,
                metadata={"text": "doc 0 content", "book_id": "b1", "page": 3},
            )
        ]
        embedder = EmbedderAgent(mock_llm_client)
        result = embedder.embed(make_docs(1))
        assert result[0].metadata.get("book_id") == "b1"
