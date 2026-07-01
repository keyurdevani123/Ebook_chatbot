"""ChunkerAgent — pre-processes PDF and EPUB files into text chunks.

Responsibilities:
  1. Load the source file (PDF with page metadata, or EPUB).
  2. Split text into overlapping chunks via RecursiveCharacterTextSplitter.
  3. Deduplicate chunks via SHA-256 hash stored in a JSONL state file.
     Re-indexing the same file will silently skip already-seen chunks.
"""

import hashlib
import json
import os
import time
from typing import List

import ebooklib
import pdfplumber
from bs4 import BeautifulSoup
from ebooklib import epub
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class ChunkerAgent:
    """Loads, splits, and deduplicates document chunks.

    Args:
        state_dir: Directory where the JSONL dedup state file is stored.
        chunk_size: Target token/character size per chunk.
        chunk_overlap: Overlap between adjacent chunks.
    """

    CHUNK_SIZE = 1500
    CHUNK_OVERLAP = 150
    STATE_FILENAME = "chunk_hashes.jsonl"

    def __init__(self, state_dir: str = "data") -> None:
        self.state_dir = state_dir
        self.state_path = os.path.join(state_dir, self.STATE_FILENAME)
        os.makedirs(state_dir, exist_ok=True)
        self._seen_hashes: set[str] = self._load_state()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk(self, path: str, metadata: dict = {}) -> List[Document]:
        """Load a PDF or EPUB file, split it, and return deduplicated chunks.

        Each returned Document carries citation metadata:
          - page (int | None)   — page number from PDF; None for EPUB
          - chapter (str | None)— chapter name when detectable
          - chunk_index (int)   — position of chunk within the book
          - source (str)        — original filename
          - book_id (str)       — passed in via metadata

        Args:
            path: Absolute or relative path to a .pdf or .epub file.
            metadata: Extra metadata merged into every chunk (e.g. book_id).

        Returns:
            List of deduplicated LangChain Documents ready for embedding.
        """
        ext = os.path.splitext(path)[1].lower()

        if ext == ".pdf":
            raw_docs = self._load_pdf(path)
        elif ext == ".epub":
            raw_docs = self._load_epub(path)
        else:
            raise ValueError(f"Unsupported file format: '{ext}'. Use .pdf or .epub.")

        chunks = self._split(raw_docs)
        return self._deduplicate(chunks, metadata)

    def reset_state(self) -> None:
        """Clear the deduplication state file (use before full re-index)."""
        self._seen_hashes.clear()
        if os.path.exists(self.state_path):
            os.remove(self.state_path)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_state(self) -> set[str]:
        """Read previously seen chunk hashes from the JSONL state file."""
        seen: set[str] = set()
        if not os.path.exists(self.state_path):
            return seen

        with open(self.state_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entry = json.loads(line)
                        seen.add(entry["hash"])
                    except (json.JSONDecodeError, KeyError):
                        continue  # Skip malformed lines
        return seen

    def _persist_hash(self, chunk_hash: str, book_id: str) -> None:
        """Append a new chunk hash to the JSONL state file (atomic append)."""
        entry = {
            "hash": chunk_hash,
            "book_id": book_id,
            "ts": int(time.time()),
        }
        with open(self.state_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    @staticmethod
    def _compute_hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _load_pdf(self, path: str) -> List[Document]:
        """Extract text from each PDF page, keeping page numbers as metadata."""
        docs: List[Document] = []
        with pdfplumber.open(path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                if text.strip():
                    docs.append(
                        Document(
                            page_content=text,
                            metadata={"page": page_num},
                        )
                    )
        return docs

    def _load_epub(self, path: str) -> List[Document]:
        """Load EPUB using ebooklib + BeautifulSoup4.

        Extracts text from each EPUB document item. Chapter title is pulled
        from the first heading tag (h1/h2/h3) found in the HTML content.
        This approach avoids the heavy `unstructured` dependency.
        """
        book = epub.read_epub(path)
        docs: List[Document] = []

        for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
            soup = BeautifulSoup(item.content, "lxml")
            text = soup.get_text(separator="\n").strip()

            if not text:
                continue

            # Extract chapter title from the first heading found
            heading = soup.find(["h1", "h2", "h3"])
            chapter = heading.get_text().strip() if heading else None

            docs.append(
                Document(
                    page_content=text,
                    metadata={"chapter": chapter} if chapter else {},
                )
            )

        return docs

    def _split(self, raw_docs: List[Document]) -> List[Document]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.CHUNK_SIZE,
            chunk_overlap=self.CHUNK_OVERLAP,
        )
        return splitter.split_documents(raw_docs)

    def _deduplicate(self, chunks: List[Document], extra_metadata: dict) -> List[Document]:
        """Remove chunks already seen in previous runs, tag with metadata."""
        book_id = extra_metadata.get("book_id", "")
        result: List[Document] = []

        for idx, chunk in enumerate(chunks):
            content_hash = self._compute_hash(chunk.page_content)

            if content_hash in self._seen_hashes:
                continue  # Already indexed in a previous run — skip

            # Merge provided metadata and add chunk position
            chunk.metadata = {
                **chunk.metadata,
                **extra_metadata,
                "chunk_index": idx,
            }

            result.append(chunk)
            self._seen_hashes.add(content_hash)
            self._persist_hash(content_hash, book_id)

        return result
