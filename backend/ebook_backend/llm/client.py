import os
from typing import List

from groq import Groq
from langchain_core.documents import Document

from ..models.document import VectorDocument


class LLMClient:
    """Unified client for LLM chat (Groq) and embeddings (fastembed).

    Groq provides extremely fast free-tier inference via LPU hardware.
    fastembed provides lightweight local ONNX embeddings (no API calls, no cost).
    Model: BAAI/bge-small-en-v1.5 → 384-dimensional vectors.
    """

    EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
    DEFAULT_CHAT_MODEL = "llama-3.1-8b-instant"

    def __init__(self) -> None:
        self.groq = Groq(api_key=os.environ["GROQ_API_KEY"])
        self._embedding_model = None  # Lazy-loaded on first use

    @property
    def embedding_model(self):
        """Lazy-load fastembed model to avoid slowing down server startup."""
        if self._embedding_model is None:
            from fastembed import TextEmbedding  # noqa: import-here to defer load
            self._embedding_model = TextEmbedding(self.EMBEDDING_MODEL)
        return self._embedding_model

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    def embed_documents(self, docs: List[Document]) -> List[VectorDocument]:
        """Embed a list of LangChain Documents, returning VectorDocuments."""
        texts = [doc.page_content for doc in docs]
        vectors = list(self.embedding_model.embed(texts))
        return [
            VectorDocument(
                vector=v.tolist(),
                metadata={**doc.metadata, "text": doc.page_content},
            )
            for v, doc in zip(vectors, docs)
        ]

    def embed_query(self, query: str) -> List[float]:
        """Embed a single query string for similarity search."""
        vectors = list(self.embedding_model.embed([query]))
        return vectors[0].tolist()

    # ------------------------------------------------------------------
    # Chat
    # ------------------------------------------------------------------

    def chat(self, messages: list, model: str = DEFAULT_CHAT_MODEL, temperature: float = 0.6) -> str:
        """Send a list of messages to Groq and return the text response.

        Also accepts a (system_prompt, user_message) tuple for convenience.
        """
        if isinstance(messages, tuple) and len(messages) == 2:
            messages = [
                {"role": "system", "content": messages[0]},
                {"role": "user", "content": messages[1]},
            ]

        # Validate model — fall back to fast default if unsupported model passed
        allowed_models = {
            "llama-3.1-8b-instant",
            "llama3-70b-8192",
            "mixtral-8x7b-32768",
            "llama-3.3-70b-versatile",
        }
        if model not in allowed_models:
            model = self.DEFAULT_CHAT_MODEL

        import time
        max_retries = 3
        base_delay = 2
        for attempt in range(max_retries):
            try:
                completion = self.groq.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=1024,
                )
                return completion.choices[0].message.content
            except Exception as e:
                if "429" in str(e) or "Too Many Requests" in str(e):
                    if attempt < max_retries - 1:
                        sleep_time = base_delay * (2 ** attempt)
                        print(f"Rate limit hit. Retrying in {sleep_time} seconds...")
                        time.sleep(sleep_time)
                    else:
                        raise e
                else:
                    raise e
