from typing import Dict, List, Optional

from pydantic import BaseModel


class VectorDocument(BaseModel):
    """A document with its embedding vector and metadata, ready for Pinecone upsert."""
    vector: List[float]
    metadata: dict


class ScoredDocument(BaseModel):
    """A document returned from a Pinecone similarity query with a relevance score."""
    id: str
    score: float
    metadata: dict
    vector: List[float]


class CitedChunk(BaseModel):
    """A retrieved text chunk with full citation metadata and relevance score."""
    text: str
    book_id: str
    source: str                    # Original filename (e.g. "networking_guide.pdf")
    page: Optional[int] = None     # Page number (PDF) or None (EPUB)
    chapter: Optional[str] = None  # Chapter title when extractable
    chunk_index: Optional[int] = None
    chunk_id: Optional[str] = None
    score: float = 0.0             # Pinecone cosine similarity score
    weight: float = 0.0            # Normalized confidence weight (0–1)


class SynthesisResult(BaseModel):
    """Final output from the SynthesizerAgent: answer + citations + per-book confidence."""
    answer: str
    citations: List[CitedChunk]
    confidence_scores: Dict[str, float]  # { book_id: weighted_confidence }
