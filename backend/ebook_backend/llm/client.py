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
        """No longer used. We use raw HTTP for embeddings."""
        return None

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    def embed_documents(self, docs: List[Document]) -> List[VectorDocument]:
        """Embed a list of LangChain Documents using raw HTTP to Hugging Face."""
        import httpx
        
        texts = [doc.page_content for doc in docs]
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            raise ValueError("HF_TOKEN environment variable is missing in .env!")
            
        api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{self.EMBEDDING_MODEL}"
        headers = {"Authorization": f"Bearer {hf_token}"}
        
        # Batch requests to Hugging Face API to prevent timeouts on large PDFs
        BATCH_SIZE = 32
        vectors = []
        
        # Use a synchronous httpx client (or requests) since this is a synchronous function
        with httpx.Client(timeout=120.0) as client:
            for i in range(0, len(texts), BATCH_SIZE):
                batch_texts = texts[i:i + BATCH_SIZE]
                response = client.post(
                    api_url, 
                    headers=headers, 
                    json={"inputs": batch_texts, "options": {"wait_for_model": True}}
                )
                response.raise_for_status()
                batch_vectors = response.json()
                vectors.extend(batch_vectors)

        return [
            VectorDocument(
                vector=v,
                metadata={**doc.metadata, "text": doc.page_content},
            )
            for v, doc in zip(vectors, docs)
        ]

    def embed_query(self, query: str) -> List[float]:
        """Embed a single query string for similarity search using raw HTTP."""
        import httpx
        
        hf_token = os.environ.get("HF_TOKEN")
        api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{self.EMBEDDING_MODEL}"
        headers = {"Authorization": f"Bearer {hf_token}"}
        
        with httpx.Client(timeout=60.0) as client:
            response = client.post(
                api_url, 
                headers=headers, 
                json={"inputs": [query], "options": {"wait_for_model": True}}
            )
            response.raise_for_status()
            vectors = response.json()
            return vectors[0]

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
