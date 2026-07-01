"""test_synthesizer.py — Unit tests for SynthesizerAgent."""

import pytest

from ebook_backend.agents.retriever import RetrieverAgent
from ebook_backend.agents.synthesizer import SynthesizerAgent
from ebook_backend.models.document import CitedChunk, SynthesisResult


def make_mock_chunks(book_id: str, scores: list[float]) -> list[CitedChunk]:
    return [
        CitedChunk(
            text=f"Chunk {i} from {book_id}: networking and security content.",
            book_id=book_id,
            source=f"{book_id}.pdf",
            page=i + 1,
            score=score,
        )
        for i, score in enumerate(scores)
    ]


@pytest.fixture
def mock_retriever(mock_llm_client, mock_pinecone_client):
    retriever = RetrieverAgent(mock_pinecone_client, mock_llm_client)
    return retriever


class TestSynthesizerAgent:

    @pytest.mark.asyncio
    async def test_synthesize_returns_synthesis_result(self, mock_retriever, mock_llm_client):
        """synthesize() must return a SynthesisResult."""
        synthesizer = SynthesizerAgent(mock_retriever, mock_llm_client)
        result = await synthesizer.synthesize(
            query="What is subnetting?",
            book_ids=["book-001"],
        )
        assert isinstance(result, SynthesisResult)

    @pytest.mark.asyncio
    async def test_synthesize_answer_not_empty(self, mock_retriever, mock_llm_client):
        """The generated answer must be a non-empty string."""
        synthesizer = SynthesizerAgent(mock_retriever, mock_llm_client)
        result = await synthesizer.synthesize(
            query="Explain TCP/IP layers.",
            book_ids=["book-001"],
        )
        assert isinstance(result.answer, str)
        assert len(result.answer.strip()) > 0

    @pytest.mark.asyncio
    async def test_confidence_scores_are_normalized(self, mock_retriever, mock_llm_client, mock_pinecone_client):
        """Per-book confidence scores must each be between 0 and 1."""
        mock_pinecone_client.query.return_value = [
            type("D", (), {
                "score": 0.8, "metadata": {
                    "text": "t", "book_id": "b1", "source": "b1.pdf", "page": 1,
                    "chunk_index": 0,
                }
            })(),
        ]
        synthesizer = SynthesizerAgent(mock_retriever, mock_llm_client)
        result = await synthesizer.synthesize(
            query="Firewall concepts",
            book_ids=["b1"],
        )
        for score in result.confidence_scores.values():
            assert 0.0 <= score <= 1.0, f"Confidence score out of range: {score}"

    @pytest.mark.asyncio
    async def test_parallel_retrieval_fires_per_book(
        self, mock_retriever, mock_llm_client, mock_pinecone_client
    ):
        """synthesize() with 3 books should trigger 3 Pinecone queries."""
        synthesizer = SynthesizerAgent(mock_retriever, mock_llm_client)
        await synthesizer.synthesize(
            query="Cross-book question",
            book_ids=["b1", "b2", "b3"],
        )
        assert mock_pinecone_client.query.call_count == 3

    @pytest.mark.asyncio
    async def test_empty_retrieval_still_returns_answer(self, mock_retriever, mock_llm_client, mock_pinecone_client):
        """When no chunks are retrieved, synthesizer should still produce an answer."""
        mock_pinecone_client.query.return_value = []
        synthesizer = SynthesizerAgent(mock_retriever, mock_llm_client)
        result = await synthesizer.synthesize(
            query="What is something obscure?",
            book_ids=["b1"],
        )
        assert isinstance(result.answer, str)
        assert result.citations == []
        assert result.confidence_scores == {}

    @pytest.mark.asyncio
    async def test_duplicate_chunks_across_books_are_deduplicated(
        self, mock_retriever, mock_llm_client, mock_pinecone_client
    ):
        """Identical chunk text from two books should appear only once in citations."""
        same_text = "Identical content that appears in both books."
        mock_pinecone_client.query.return_value = [
            type("D", (), {
                "score": 0.85, "metadata": {
                    "text": same_text, "book_id": "bX", "source": "bX.pdf",
                    "page": 1, "chunk_index": 0,
                }
            })(),
        ]
        synthesizer = SynthesizerAgent(mock_retriever, mock_llm_client)
        result = await synthesizer.synthesize(
            query="Some query",
            book_ids=["b1", "b2"],
        )
        texts = [c.text for c in result.citations]
        assert texts.count(same_text) <= 1, "Duplicate chunk appeared multiple times in citations"

    @pytest.mark.asyncio
    async def test_confidence_scores_keyed_by_book_id(self, mock_retriever, mock_llm_client):
        """confidence_scores keys must match the requested book_ids."""
        synthesizer = SynthesizerAgent(mock_retriever, mock_llm_client)
        book_ids = ["bookA", "bookB"]
        result = await synthesizer.synthesize(
            query="Security protocols",
            book_ids=book_ids,
        )
        # All requested books should appear in confidence_scores
        for bid in book_ids:
            assert bid in result.confidence_scores
