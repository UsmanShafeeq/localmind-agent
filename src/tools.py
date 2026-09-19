"""Custom tools exposed to the agent."""
from typing import List

from langchain_core.tools import tool

from src.config import TOP_K
from src.vectorstore import search


def _format_docs(docs: List) -> str:
    if not docs:
        return "No matching documents were found in the vector store."
    blocks = []
    for i, doc in enumerate(docs, start=1):
        src = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page")
        loc = f"{src}" + (f" (page {page + 1})" if isinstance(page, int) else "")
        blocks.append(f"[Chunk {i} | {loc}]\n{doc.page_content.strip()}")
    return "\n\n---\n\n".join(blocks)


@tool
def vector_store_search(query: str) -> str:
    """Search the local vector store for document chunks relevant to `query`.

    Use this FIRST for any question about the contents of the local documents.
    Returns the top matching chunks with source citations.
    """
    docs = search(query, k=TOP_K)
    return _format_docs(docs)


ALL_TOOLS = [vector_store_search]