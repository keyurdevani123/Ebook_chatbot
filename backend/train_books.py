"""Training pipeline — indexes e-books into Pinecone using the agent architecture.

Usage:
    python train_books.py

Place PDF/EPUB files in: data/docs/<YEAR>/
State is tracked at two levels:
  - Book level  : training/ebooks/states/<YEAR>.json  (which books are done)
  - Chunk level : data/chunk_hashes.jsonl             (deduplication across runs)
"""

import datetime
import os

from dotenv import load_dotenv

from ebook_backend.agents.chunker import ChunkerAgent
from ebook_backend.agents.embedder import EmbedderAgent
from ebook_backend.llm import LLMClient
from ebook_backend.store import PineconeDBClient
from training.ebooks.state_handler import read_state, write_state

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DOC_DIR_NAME = "2024"
STATE_FILE = f"{DOC_DIR_NAME}.json"
DOC_DIR = f"{os.path.dirname(__file__)}/data/docs/{DOC_DIR_NAME}"

# Supported file extensions
SUPPORTED_EXTENSIONS = (".epub", ".pdf")
BOOKS = [
    f for f in os.listdir(DOC_DIR) if f.lower().endswith(SUPPORTED_EXTENSIONS)
] if os.path.isdir(DOC_DIR) else []

# ---------------------------------------------------------------------------
# State tracking (book-level — which books have been fully indexed)
# ---------------------------------------------------------------------------

STATE_JSON = read_state(STATE_FILE)
if "SUCCESS" not in STATE_JSON:
    STATE_JSON["SUCCESS"] = []
if "ERROR" not in STATE_JSON:
    STATE_JSON["ERROR"] = []

print(f"{datetime.datetime.now().isoformat()} | Training start | Dir: {DOC_DIR_NAME} ({DOC_DIR})")
print(f"{datetime.datetime.now().isoformat()} | Files to train: {BOOKS}")

# ---------------------------------------------------------------------------
# Initialize agent pipeline
# ---------------------------------------------------------------------------

llm_client = LLMClient()
chunker = ChunkerAgent(state_dir="data")      # chunk-level SHA-256 dedup
embedder = EmbedderAgent(llm_client)           # fastembed local embeddings
vectordb_client = PineconeDBClient()
vectordb_client.initialize_collection()


# ---------------------------------------------------------------------------
# Training functions
# ---------------------------------------------------------------------------

def train_single(book_id: str) -> None:
    """Index a single book file into Pinecone."""
    if book_id in STATE_JSON["SUCCESS"]:
        print(f"{datetime.datetime.now().isoformat()} | SKIP | {book_id} already trained.")
        return

    path = os.path.join(DOC_DIR, book_id)
    clean_book_id = os.path.splitext(book_id)[0]  # Remove extension for Pinecone filter

    print(f"{datetime.datetime.now().isoformat()} | START | {book_id}")

    # Step 1 — Chunk (extract, split, deduplicate)
    docs = chunker.chunk(
        path=path,
        metadata={
            "book_id": clean_book_id,
            "source": book_id,
        },
    )
    print(f"{datetime.datetime.now().isoformat()} | CHUNKS | {book_id} → {len(docs)} new chunks")

    if not docs:
        print(f"{datetime.datetime.now().isoformat()} | SKIP | {book_id} — all chunks already indexed.")
        STATE_JSON["SUCCESS"].append(book_id)
        write_state(STATE_FILE, STATE_JSON)
        return

    # Debug — show last chunk
    print(f"{datetime.datetime.now().isoformat()} | LAST CHUNK | {docs[-1]}")

    # Step 2 — Embed (local fastembed, no API cost)
    vectors = embedder.embed(docs)

    # Step 3 — Upsert into Pinecone
    vectordb_client.save(vectors)

    print(f"{datetime.datetime.now().isoformat()} | DONE | {book_id}")

    STATE_JSON["SUCCESS"].append(book_id)
    write_state(STATE_FILE, STATE_JSON)
    print("=" * 60)


def train_all(books: list) -> None:
    """Iterate over all discovered book files and index each one."""
    for book_id in books:
        try:
            train_single(book_id)
        except Exception as err:
            print(f"{datetime.datetime.now().isoformat()} | ERROR | {book_id} | {err}")
            STATE_JSON["ERROR"].append({"book_id": book_id, "error": str(err)})
            write_state(STATE_FILE, STATE_JSON)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__" or __name__ == "train_books":
    train_all(BOOKS)
    print(f"{datetime.datetime.now().isoformat()} | Training complete.")
