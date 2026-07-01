"""test_chunker.py — Unit tests for ChunkerAgent.

Tests cover:
  - Chunk count is within a reasonable range
  - SHA-256 deduplication works across two runs
  - Chunk metadata carries book_id and chunk_index
  - No empty chunks are produced
  - State file is created and readable
"""

import os
import tempfile

import pytest

from ebook_backend.agents.chunker import ChunkerAgent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_text_file(content: str, suffix: str = ".txt") -> str:
    """Write content to a temp file and return its path."""
    f = tempfile.NamedTemporaryFile(
        suffix=suffix, mode="w", encoding="utf-8", delete=False
    )
    f.write(content)
    f.close()
    return f.name


LONG_TEXT = (
    "Networking fundamentals cover OSI model, TCP/IP, subnets, routing, and switching. "
    * 200  # ~8000 chars — should produce several chunks
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestChunkerAgentBasics:

    def setup_method(self):
        """Use a fresh temp dir for each test so state doesn't leak."""
        self.state_dir = tempfile.mkdtemp()
        self.chunker = ChunkerAgent(state_dir=self.state_dir)

    def test_produces_chunks_from_text_file(self, sample_pdf_path):
        """ChunkerAgent should produce at least one chunk for non-trivial content."""
        # Re-use the .txt fixture as if it were a plain split target
        # (We test EPUB/PDF separately; here we call _split directly)
        from langchain_core.documents import Document
        docs = [Document(page_content=LONG_TEXT, metadata={})]
        chunks = self.chunker._split(docs)
        assert len(chunks) >= 2, "Expected multiple chunks for long text"

    def test_chunk_count_reasonable(self):
        """Number of chunks should scale with content length."""
        from langchain_core.documents import Document
        docs = [Document(page_content=LONG_TEXT, metadata={})]
        chunks = self.chunker._deduplicate(self.chunker._split(docs), {"book_id": "b1"})
        assert 2 <= len(chunks) <= 30, f"Unexpected chunk count: {len(chunks)}"

    def test_no_empty_chunks(self):
        """No chunk should have empty or whitespace-only content."""
        from langchain_core.documents import Document
        docs = [Document(page_content=LONG_TEXT, metadata={})]
        chunks = self.chunker._deduplicate(self.chunker._split(docs), {"book_id": "b1"})
        for chunk in chunks:
            assert chunk.page_content.strip(), "Found empty chunk"

    def test_chunk_metadata_has_book_id(self):
        """Every chunk must carry the book_id in its metadata."""
        from langchain_core.documents import Document
        docs = [Document(page_content=LONG_TEXT, metadata={})]
        chunks = self.chunker._deduplicate(self.chunker._split(docs), {"book_id": "test-book"})
        for chunk in chunks:
            assert chunk.metadata.get("book_id") == "test-book"

    def test_chunk_metadata_has_chunk_index(self):
        """Every chunk must carry chunk_index metadata."""
        from langchain_core.documents import Document
        docs = [Document(page_content=LONG_TEXT, metadata={})]
        chunks = self.chunker._deduplicate(self.chunker._split(docs), {"book_id": "b1"})
        for chunk in chunks:
            assert "chunk_index" in chunk.metadata


class TestChunkerDeduplication:

    def setup_method(self):
        self.state_dir = tempfile.mkdtemp()
        self.chunker = ChunkerAgent(state_dir=self.state_dir)

    def test_deduplication_skips_seen_chunks(self):
        """Re-running on the same content should produce zero new chunks."""
        from langchain_core.documents import Document
        docs = [Document(page_content=LONG_TEXT, metadata={})]
        split = self.chunker._split(docs)

        # First run — all chunks are new
        first_run = self.chunker._deduplicate(split, {"book_id": "b1"})
        assert len(first_run) > 0, "First run should yield chunks"

        # Second run with a new chunker reading the same state file
        chunker2 = ChunkerAgent(state_dir=self.state_dir)
        second_run = chunker2._deduplicate(split, {"book_id": "b1"})
        assert len(second_run) == 0, "Second run on same content should produce no new chunks"

    def test_state_file_is_created(self):
        """JSONL state file should be created after first chunk run."""
        from langchain_core.documents import Document
        docs = [Document(page_content=LONG_TEXT, metadata={})]
        split = self.chunker._split(docs)
        self.chunker._deduplicate(split, {"book_id": "b1"})

        assert os.path.exists(self.chunker.state_path), "State file not created"
        assert os.path.getsize(self.chunker.state_path) > 0, "State file is empty"

    def test_state_file_is_valid_jsonl(self):
        """State file entries must be valid JSON with 'hash' and 'book_id' keys."""
        import json
        from langchain_core.documents import Document
        docs = [Document(page_content=LONG_TEXT, metadata={})]
        split = self.chunker._split(docs)
        self.chunker._deduplicate(split, {"book_id": "b2"})

        with open(self.chunker.state_path, "r") as f:
            for line in f:
                entry = json.loads(line)
                assert "hash" in entry
                assert "book_id" in entry

    def test_reset_state_clears_hashes(self):
        """reset_state() should allow re-indexing the same content."""
        from langchain_core.documents import Document
        docs = [Document(page_content=LONG_TEXT, metadata={})]
        split = self.chunker._split(docs)
        self.chunker._deduplicate(split, {"book_id": "b1"})

        self.chunker.reset_state()
        third_run = self.chunker._deduplicate(split, {"book_id": "b1"})
        assert len(third_run) > 0, "After reset, chunks should be accepted again"
