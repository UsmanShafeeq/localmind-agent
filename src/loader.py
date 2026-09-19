"""Document loading and text splitting pipeline."""
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)

from src.config import CHUNK_SIZE, CHUNK_OVERLAP, RAW_DOCS_PATH


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".markdown"}


def _loader_for(path: Path):
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return PyPDFLoader(str(path))
    if suffix in {".md", ".markdown"}:
        return UnstructuredMarkdownLoader(str(path))
    return TextLoader(str(path), encoding="utf-8")


def load_documents(raw_dir: str = RAW_DOCS_PATH) -> List[Document]:
    """Load every supported file under raw_dir into LangChain Documents."""
    root = Path(raw_dir)
    if not root.exists():
        raise FileNotFoundError(f"Raw docs directory not found: {root}")

    docs: List[Document] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            try:
                docs.extend(_loader_for(path).load())
            except Exception as exc:  # noqa: BLE001
                print(f"[loader] Failed to load {path}: {exc}")
    return docs


def split_documents(docs: List[Document]) -> List[Document]:
    """Split documents into overlapping chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(docs)


def build_chunks(raw_dir: str = RAW_DOCS_PATH) -> List[Document]:
    """Convenience: load + split in one call."""
    return split_documents(load_documents(raw_dir))