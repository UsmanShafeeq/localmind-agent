"""Central configuration loaded from environment variables."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_NAME = os.getenv("MODEL_NAME", "llama3.2")
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
DB_PATH = str((BASE_DIR / os.getenv("DB_PATH", "data/vector_db")).resolve())
RAW_DOCS_PATH = str((BASE_DIR / os.getenv("RAW_DOCS_PATH", "data/raw_docs")).resolve())
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "localmind_docs")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))
TOP_K = int(os.getenv("TOP_K", "4"))

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

PROMPTS_DIR = BASE_DIR / "prompts"
SYSTEM_PROMPT_PATH = PROMPTS_DIR / "system_prompt.txt"
RAG_PROMPT_PATH = PROMPTS_DIR / "rag_prompt.txt"


def load_prompt(path: Path) -> str:
    """Read a prompt template from disk."""
    return path.read_text(encoding="utf-8")