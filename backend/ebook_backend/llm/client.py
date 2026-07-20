import os
from typing import List

from groq import Groq
from langchain_core.documents import Document

from ..models.document import VectorDocument


class LLMClient:
    """Unified client for LLM chat (Groq) and embeddings (HuggingFace API).

    Groq provides extremely fast free-tier inference via LPU hardware.
    HuggingFace API provides free cloud-based embeddings (0 RAM usage).
    Model: BAAI/bge-small-en-v1.5 → 384-dimensional vectors.
    """

    EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
    DEFAULT_CHAT_MODEL = "llama-3.1-8b-instant"

    def __init__(self) -> None:
        self.groq = Groq(api_key=os.environ["GROQ_API_KEY"])
        self._embedding_model = None  # Lazy-loaded on first use
        self._model_cooldowns = {}  # Tracks {model_name: timestamp_when_available}

    @property
    def embedding_model(self):
        """Lazy-load HuggingFace API to avoid slowing down server startup and save RAM."""
        if self._embedding_model is None:
            from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
            hf_token = os.environ.get("HF_TOKEN")
            if not hf_token:
                raise ValueError("HF_TOKEN environment variable is missing in .env! Please get a free token from huggingface.co")
            
            self._embedding_model = HuggingFaceInferenceAPIEmbeddings(
                api_key=hf_token,
                model_name=self.EMBEDDING_MODEL
            )
        return self._embedding_model

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    def embed_documents(self, docs: List[Document]) -> List[VectorDocument]:
        """Embed a list of LangChain Documents, returning VectorDocuments."""
        texts = [doc.page_content for doc in docs]
        vectors = self.embedding_model.embed_documents(texts)
        return [
            VectorDocument(
                vector=v,
                metadata={**doc.metadata, "text": doc.page_content},
            )
            for v, doc in zip(vectors, docs)
        ]

    def embed_query(self, query: str) -> List[float]:
        """Embed a single query string for similarity search."""
        return self.embedding_model.embed_query(query)

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
            "qwen/qwen3-32b",
            "meta-llama/llama-4-scout-17b-16e-instruct"
        }
        if model not in allowed_models:
            model = self.DEFAULT_CHAT_MODEL

        # Define a list of high-availability Groq models as fallbacks.
        # It tries the requested model first, then cascades down this list.
        fallback_models = [
            model,
            "llama-3.1-8b-instant",
            "qwen/qwen3-32b",
            "llama-3.3-70b-versatile",
            "meta-llama/llama-4-scout-17b-16e-instruct",
            "mixtral-8x7b-32768"
        ]
        
        import time
        
        # Remove duplicates while preserving order, AND skip models in cooldown
        unique_fallbacks = []
        for m in fallback_models:
            if m not in unique_fallbacks:
                # If model is on cooldown (hit 429 recently), skip it
                cooldown_until = self._model_cooldowns.get(m, 0)
                if time.time() >= cooldown_until:
                    unique_fallbacks.append(m)

        # If ALL models are currently on cooldown, just try them all anyway
        # (Groq rate limits reset quickly, so it's worth a shot)
        if not unique_fallbacks:
            for m in fallback_models:
                if m not in unique_fallbacks:
                    unique_fallbacks.append(m)

        last_exception = None
        for current_model in unique_fallbacks:
            try:
                completion = self.groq.chat.completions.create(
                    model=current_model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=1024,
                )
                if current_model != model:
                    print(f"Successfully switched to fallback model: {current_model}")
                return completion.choices[0].message.content
            except Exception as e:
                error_str = str(e).lower()
                if "429" in error_str or "too many requests" in error_str or "rate limit" in error_str:
                    # Determine if this is a daily limit or a minute limit
                    if "per day" in error_str:
                        cooldown_seconds = 86400  # 24 hours
                        print(f"Daily rate limit hit for {current_model}. Putting in 24-hour cooldown and switching...")
                    else:
                        cooldown_seconds = 60     # 1 minute
                        print(f"Minute rate limit hit for {current_model}. Putting in 60s cooldown and switching...")
                    
                    # Remember that this model is exhausted for the appropriate duration
                    self._model_cooldowns[current_model] = time.time() + cooldown_seconds
                    last_exception = e
                    continue
                else:
                    # If it's a different error (e.g. bad request, token auth), fail immediately.
                    raise e

        # If we exhausted the entire list and still got 429s
        raise last_exception or Exception("All fallback models failed due to rate limits.")
