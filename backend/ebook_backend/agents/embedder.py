"""EmbedderAgent — converts LangChain Documents into VectorDocuments.

Wraps LLMClient.embed_documents() as a dedicated pipeline stage with
logging and timing so embedding throughput can be monitored.
"""

import time
from logging import getLogger
from typing import List

from langchain_core.documents import Document

from ..llm import LLMClient
from ..models.document import VectorDocument

logger = getLogger(__name__)


class EmbedderAgent:
    """Embeds a list of Document chunks using the Hugging Face Inference API.

    Embedding is done via a remote API — no heavy local models are loaded into RAM.
    This saves significant memory on constrained servers.

    Args:
        llm_client: Shared LLMClient instance (contains the embedding model).
    """

    def __init__(self, llm_client: LLMClient) -> None:
        self.llm = llm_client

    def embed(self, docs: List[Document]) -> List[VectorDocument]:
        """Embed documents and return VectorDocuments ready for Pinecone upsert.

        Args:
            docs: LangChain Documents with page_content and metadata.

        Returns:
            List of VectorDocuments with 384-dim vectors and full metadata.
        """
        if not docs:
            return []

        logger.info("EmbedderAgent | Embedding %d chunks...", len(docs))
        t0 = time.perf_counter()

        vectors = self.llm.embed_documents(docs)

        elapsed = time.perf_counter() - t0
        logger.info(
            "EmbedderAgent | Embedded %d chunks in %.2fs (%.1f chunks/s)",
            len(vectors),
            elapsed,
            len(vectors) / elapsed if elapsed > 0 else float("inf"),
        )

        return vectors
