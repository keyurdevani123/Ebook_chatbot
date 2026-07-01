import os
from typing import List

import ebooklib
import pdfplumber
from bs4 import BeautifulSoup
from ebooklib import epub
from langchain_core.documents import Document
from langchain_community.document_loaders.text import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE = 1500
CHUNK_OVERLAP = 150


def _split_documents(raw_docs: List[Document], metadata: dict) -> List[Document]:
    """Split a list of Documents and merge with extra metadata."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = text_splitter.split_documents(raw_docs)
    return [
        Document(page_content=chunk.page_content, metadata={**chunk.metadata, **metadata})
        for chunk in chunks
    ]


def load_and_split_text(path: str, metadata: dict = {}) -> List[Document]:
    """Load a plain text file and split into chunks."""
    loader = TextLoader(path, encoding="utf-8")
    return _split_documents(loader.load(), metadata)


def load_and_split_epub(path: str, metadata: dict = {}) -> List[Document]:
    """Load an EPUB file using ebooklib + BeautifulSoup4 and split into chunks.

    Extracts per-chapter text. Chapter title is pulled from the first h1/h2/h3
    heading in each EPUB document item.
    """
    book = epub.read_epub(path)
    raw_docs: List[Document] = []

    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        soup = BeautifulSoup(item.content, "lxml")
        text = soup.get_text(separator="\n").strip()

        if not text:
            continue

        heading = soup.find(["h1", "h2", "h3"])
        chapter = heading.get_text().strip() if heading else None

        raw_docs.append(
            Document(
                page_content=text,
                metadata={"chapter": chapter} if chapter else {},
            )
        )

    return _split_documents(raw_docs, metadata)


def load_and_split_pdf(path: str, metadata: dict = {}) -> List[Document]:
    """Load a PDF file page-by-page (with page numbers) and split into chunks.

    Uses pdfplumber for accurate text extraction with page-level metadata.
    Each resulting chunk carries a ``page`` field in its metadata.
    """
    raw_docs: List[Document] = []

    with pdfplumber.open(path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                raw_docs.append(
                    Document(
                        page_content=text,
                        metadata={"page": page_num},
                    )
                )

    return _split_documents(raw_docs, metadata)


def split_text(text: str, metadata: dict = {}) -> List[Document]:
    """Split a raw text string into chunks (used by training pipeline)."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    return text_splitter.split_documents(
        [Document(page_content=text, metadata=metadata)]
    )
