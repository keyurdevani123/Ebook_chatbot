import os
from typing import List
from uuid import uuid4

from pinecone import Pinecone, ServerlessSpec

from ebook_backend.models.document import ScoredDocument, VectorDocument

# fastembed BAAI/bge-small-en-v1.5 produces 384-dimensional vectors
EMBEDDING_DIMENSIONS = 384


class PineconeDBClient:
    """Client for Pinecone vector database operations.

    Uses serverless Pinecone with 384-dim embeddings (fastembed bge-small-en-v1.5).
    Default index: ``ebooks-production`` — scoped to books only.
    """

    def __init__(
        self,
        api_key: str | None = None,
        collection_name: str = "ebooks-production",
    ) -> None:
        self.api_key = api_key or os.getenv("PINECONE_API_KEY")
        self.region = os.getenv("PINECONE_REGION", "us-east-1")
        self.index_name = collection_name
        self.client = Pinecone(api_key=self.api_key) if self.api_key else None
        self.index = None

    def initialize_collection(self, dimensions: int = EMBEDDING_DIMENSIONS) -> None:
        """Create the Pinecone index if it doesn't exist, then connect to it."""
        if not self.client:
            return
            
        existing = [idx["name"] for idx in self.client.list_indexes().get("indexes", [])]

        if self.index_name not in existing:
            self.client.create_index(
                name=self.index_name,
                dimension=dimensions,
                spec=ServerlessSpec(cloud="aws", region=self.region),
            )

        self.index = self.client.Index(name=self.index_name)

    def insert_documents(self, docs: List[VectorDocument]) -> None:
        """Upsert a batch of VectorDocuments into the Pinecone index."""
        if not docs or not self.index:
            return

        self.index.upsert(
            vectors=[
                (str(uuid4()), docs[i].vector, docs[i].metadata)
                for i in range(len(docs))
            ],
            batch_size=100,  # Smaller batches — friendlier on free-tier rate limits
        )

    def query(self, vector: List[float], filter: dict, limit: int) -> List[ScoredDocument]:
        """Run a similarity query and return scored results with metadata."""
        if not self.index:
            return []
            
        results = self.index.query(
            vector=[vector],
            top_k=limit,
            include_values=False,
            include_metadata=True,
            filter=filter,
        )

        return [
            ScoredDocument(
                id=doc["id"],
                score=doc["score"],
                metadata=doc["metadata"],
                vector=doc.get("values", []),
            )
            for doc in results.get("matches", [])
        ]
