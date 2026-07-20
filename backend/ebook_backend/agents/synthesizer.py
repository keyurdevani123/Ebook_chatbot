"""SynthesizerAgent — cross-book parallel retrieval with weighted confidence scoring.

Given a user query and a list of book IDs, this agent:
  1. Fires parallel Pinecone retrievals across every book (asyncio.gather).
  2. Merges and deduplicates results by content hash.
  3. Computes a weighted confidence score for each chunk and book.
  4. Builds a context string with inline citation labels.
  5. Calls Groq LLM and returns a SynthesisResult with answer + citations + scores.
"""

import asyncio
import hashlib
from logging import getLogger
from typing import Dict, List, Optional

from ..llm import LLMClient
from ..llm.prompts import luna_system_prompt
from ..models.document import CitedChunk, SynthesisResult
from .retriever import RetrieverAgent

logger = getLogger(__name__)

# Maximum chunks injected into context — keeps prompt size manageable on free server
MAX_CONTEXT_CHUNKS = 15
# Chunks fetched per book during parallel retrieval
CHUNKS_PER_BOOK = 8


class SynthesizerAgent:
    """Orchestrates multi-book RAG with parallel retrieval and confidence scoring.

    Args:
        retriever: Initialized RetrieverAgent instance.
        llm_client: Shared LLMClient for chat generation.
    """

    def __init__(self, retriever: RetrieverAgent, llm_client: LLMClient) -> None:
        self.retriever = retriever
        self.llm = llm_client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def synthesize(
        self,
        query: str,
        book_ids: List[str],
        history: Optional[List[dict]] = None,
        model: str = "llama-3.1-8b-instant",
        language: str = "same language as input",
    ) -> SynthesisResult:
        """Generate a cited answer by querying multiple books in parallel.

        Args:
            query: The user's question.
            book_ids: Book IDs to retrieve evidence from.
            history: Recent conversation turns (oldest first).
            model: Groq model name.
            language: Response language instruction for Luna.

        Returns:
            SynthesisResult with answer, citation list, and per-book confidence.
        """
        history = history or []

        # ----------------------------------------------------------------
        # Step 1: Parallel retrieval across all books
        # ----------------------------------------------------------------
        
        # Pre-compute query vector once to avoid redundant API calls & rate limits
        try:
            import asyncio
            query_vector = await asyncio.to_thread(self.llm.embed_query, query)
        except Exception as e:
            logger.error("SynthesizerAgent | Failed to embed query: %s", e)
            query_vector = None
            
        tasks = [
            self.retriever.retrieve_async(query, [bid], limit=CHUNKS_PER_BOOK, vector=query_vector)
            for bid in book_ids
        ]
        per_book_results: List[List[CitedChunk]] = await asyncio.gather(*tasks)

        # ----------------------------------------------------------------
        # Step 2: Merge + deduplicate by content SHA-256
        # ----------------------------------------------------------------
        all_chunks: List[CitedChunk] = []
        seen_hashes: set[str] = set()

        for book_chunks in per_book_results:
            for chunk in book_chunks:
                content_hash = hashlib.sha256(chunk.text.encode("utf-8")).hexdigest()
                if content_hash not in seen_hashes:
                    seen_hashes.add(content_hash)
                    all_chunks.append(chunk)

        # Edge case: nothing retrieved
        if not all_chunks:
            logger.warning("SynthesizerAgent | No chunks retrieved for query: %r", query[:60])
            system_prompt = luna_system_prompt.format(
                context="No document context found for this query.",
                language=language,
                content_type="Book",
            )
            messages = [{"role": "system", "content": system_prompt}]
            messages.extend(history[::-1])
            messages.append({"role": "user", "content": query})
            
            answer = await asyncio.to_thread(
                self.llm.chat,
                messages,
                model=model,
                temperature=0.1
            )
            return SynthesisResult(answer=answer, citations=[], confidence_scores={})

        # ----------------------------------------------------------------
        # Step 3: Weighted confidence scoring
        # Normalize each chunk's Pinecone score → relative weight (sums to 1)
        # ----------------------------------------------------------------
        total_score = sum(c.score for c in all_chunks) or 1.0
        for chunk in all_chunks:
            chunk.weight = round(chunk.score / total_score, 6)

        # Sort by weight descending and cap at MAX_CONTEXT_CHUNKS
        all_chunks.sort(key=lambda c: c.weight, reverse=True)
        top_chunks = all_chunks[:MAX_CONTEXT_CHUNKS]

        # ----------------------------------------------------------------
        # Step 4: Per-book confidence (sum of weights for that book's chunks)
        # ----------------------------------------------------------------
        confidence_scores: Dict[str, float] = {}
        for bid in book_ids:
            book_weight = sum(c.weight for c in top_chunks if c.book_id == bid)
            confidence_scores[bid] = round(book_weight, 4)

        # ----------------------------------------------------------------
        # Step 5: Build context string with inline citation labels
        # ----------------------------------------------------------------
        context_parts: List[str] = []
        for chunk in top_chunks:
            label_parts = [f"Book: {chunk.book_id}"]
            if chunk.page is not None:
                label_parts.append(f"Page: {chunk.page}")
            if chunk.chapter:
                label_parts.append(f"Chapter: {chunk.chapter}")
            label_parts.append(f"Confidence: {chunk.weight:.2%}")
            citation_label = "[" + " | ".join(label_parts) + "]"
            context_parts.append(f"{citation_label}\n{chunk.text}")

        context_str = "\n\n---\n\n".join(context_parts)

        # ----------------------------------------------------------------
        # Step 6: Build message list and call Groq LLM
        # ----------------------------------------------------------------
        system_prompt = luna_system_prompt.format(
            context=context_str,
            language=language,
            content_type="Book",
        )

        messages = [{"role": "system", "content": system_prompt}]
        # History is stored newest-first in MongoDB; reverse for chronological order
        messages.extend(history[::-1])
        messages.append({"role": "user", "content": query})

        # Run the synchronous Groq API call in a threadpool to prevent freezing the event loop
        answer = await asyncio.to_thread(self.llm.chat, messages, model=model, temperature=0.1)

        logger.info(
            "SynthesizerAgent | Books: %s | Chunks used: %d | Confidence: %s",
            book_ids,
            len(top_chunks),
            confidence_scores,
        )

        return SynthesisResult(
            answer=answer,
            citations=top_chunks,
            confidence_scores=confidence_scores,
        )
