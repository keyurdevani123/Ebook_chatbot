"""RetrieverAgent — semantic search with citation-level source attribution.

Embeds the user query, queries Pinecone, and returns CitedChunks that carry
full provenance: book_id, source filename, page number, chapter, and score.
"""

import asyncio
from logging import getLogger
from typing import List, Optional

from ..llm import LLMClient
from ..models.document import CitedChunk
from ..store.pinecone import PineconeDBClient

logger = getLogger(__name__)


class RetrieverAgent:
    """Performs similarity search against the Pinecone index.

    Args:
        vectordb_client: Initialized PineconeDBClient instance.
        llm_client: LLMClient used for query embedding.
    """

    def __init__(self, vectordb_client: PineconeDBClient, llm_client: LLMClient) -> None:
        self.vectordb = vectordb_client
        self.llm = llm_client

    # ------------------------------------------------------------------
    # Synchronous retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        book_ids: Optional[List[str]] = None,
        limit: int = 6,
    ) -> List[CitedChunk]:
        """Embed ``query`` and return the top-``limit`` cited chunks.

        Args:
            query: Natural language question from the user.
            book_ids: Optional list of book IDs to scope the search.
                      Pass None or [] to search across all indexed books.
            limit: Maximum number of results to return.

        Returns:
            List of CitedChunk objects sorted by relevance score (descending).
        """
        vector = self.llm.embed_query(query)

        # Build Pinecone metadata filter
        pinecone_filter: dict = {}
        if book_ids:
            pinecone_filter = {"bookid": {"$in": book_ids}}

        scored_docs = self.vectordb.query(
            vector=vector,
            filter=pinecone_filter,
            limit=limit,
        )

        cited_chunks: List[CitedChunk] = []
        for doc in scored_docs:
            meta = doc.metadata
            cited_chunks.append(
                CitedChunk(
                    text=meta.get("text", ""),
                    book_id=meta.get("bookid", ""),
                    source=meta.get("source", ""),
                    page=meta.get("page"),
                    chapter=meta.get("chapter"),
                    chunk_index=meta.get("chunk_index"),
                    chunk_id=meta.get("chunk_id"),
                    score=doc.score,
                )
            )

        # 2. Exact Keyword Search via MongoDB
        keyword_chunks: List[CitedChunk] = []
        try:
            from ..server import files_db
            if files_db and book_ids:
                kw_results = files_db.keyword_search(book_ids=book_ids, query=query, limit=limit)
                for doc in kw_results:
                    keyword_chunks.append(
                        CitedChunk(
                            text=doc.get("text", ""),
                            book_id=str(doc.get("book_id", "")),
                            source=doc.get("source", ""),
                            chunk_index=doc.get("chunk_index"),
                            chunk_id=doc.get("chunk_id"),
                            score=doc.get("score", 0.0),
                        )
                    )
        except Exception as e:
            logger.warning("Keyword search failed: %s", e)

        # 3. Reciprocal Rank Fusion (RRF)
        fused_scores = {}
        k_fusion = 60
        
        for rank, doc in enumerate(cited_chunks, 1):
            cid = doc.chunk_id or str(hash(doc.text))
            if cid not in fused_scores:
                fused_scores[cid] = {"score": 0.0, "doc": doc}
            fused_scores[cid]["score"] += 1.0 / (k_fusion + rank)
            
        for rank, doc in enumerate(keyword_chunks, 1):
            cid = doc.chunk_id or str(hash(doc.text))
            if cid not in fused_scores:
                fused_scores[cid] = {"score": 0.0, "doc": doc}
            fused_scores[cid]["score"] += 1.0 / (k_fusion + rank)
            
        sorted_fused = sorted(fused_scores.values(), key=lambda x: x["score"], reverse=True)
        final_chunks = [item["doc"] for item in sorted_fused][:limit]

        logger.debug(
            "RetrieverAgent | Query: %r | Books: %s | Vector: %d | Keyword: %d | Fused: %d chunks",
            query[:60],
            book_ids,
            len(cited_chunks),
            len(keyword_chunks),
            len(final_chunks)
        )

        return final_chunks

    # ------------------------------------------------------------------
    # Async wrapper for parallel synthesis
    # ------------------------------------------------------------------

    async def retrieve_async(
        self,
        query: str,
        book_ids: Optional[List[str]] = None,
        limit: int = 6,
    ) -> List[CitedChunk]:
        """Async-safe wrapper — runs the sync Pinecone call in a thread pool.

        This allows SynthesizerAgent to use asyncio.gather() for parallel
        retrieval across multiple books without blocking the event loop.
        """
        return await asyncio.to_thread(self.retrieve, query, book_ids, limit)
