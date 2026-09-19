"""ChromaDB setup and retriever functions."""
from pathlib import Path
from typing import List, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.config import DB_PATH, COLLECTION_NAME, TOP_K
from src.embeddings import get_embeddings


def get_vectorstore() -> Chroma:
    """Open (or create) the persistent Chroma collection."""
    Path(DB_PATH).mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=DB_PATH,
    )


def index_documents(docs: List[Document], reset: bool = False) -> int:
    """Embed and store documents. Returns number of chunks indexed."""
    if not docs:
        return 0

    if reset and Path(DB_PATH).exists():
        store = get_vectorstore()
        try:
            store.delete_collection()
        except Exception:  # noqa: BLE001
            pass

    store = get_vectorstore()
    store.add_documents(docs)
    return len(docs)


def get_retriever(k: int = TOP_K):
    """Return a similarity-search retriever."""
    return get_vectorstore().as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )


def search(query: str, k: int = TOP_K) -> List[Document]:
    """Direct similarity search helper."""
    return get_vectorstore().similarity_search(query, k=k)


def collection_size() -> int:
    """Return the number of items currently stored."""
    try:
        return get_vectorstore()._collection.count()  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        return 0