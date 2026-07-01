import os
from typing import List
from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    ScoredPoint,
    VectorParams,
)

from ebook_backend.models.document import VectorDocument


class QdrantDBClient:

    def __init__(self, host: str | None = None, collection_name: str = "books-production") -> None:
        self.host = host or os.getenv("QDRANT_HOST") or "http://localhost:6333"

        # self.client = QdrantClient(path="vectors", force_disable_check_same_thread=True)
        self.client = QdrantClient(url=self.host)

        self.collection_name = collection_name

        self.collection_path = f"/collections/{self.collection_name}"
        self.collection_store_path = f"{self.collection_path}/points"
        self.collection_search_path = f"{self.collection_store_path}/search"

    def initialize_collection(self, dimensions: int = 1536) -> None:
        if self.client.collection_exists(collection_name=self.collection_name):
            return

        self.client.recreate_collection(
            collection_name=self.collection_name, vectors_config=VectorParams(size=dimensions, distance=Distance.COSINE)
        )

    def save(self, docs: List[VectorDocument]) -> None:
        self.client.upsert(
            collection_name=self.collection_name,
            points=[PointStruct(id=uuid4().hex, vector=doc.vector, payload=doc.metadata) for doc in docs],
        )

    def query(self, vector: List[float], book_id: str, limit: int) -> List[ScoredPoint]:
        return self.client.search(
            collection_name=self.collection_name,
            query_vector=vector,
            query_filter=Filter(must=[FieldCondition(key="book_id", match=MatchValue(value=book_id))]),
            # score_threshold=0.75,
            limit=limit,
        )
