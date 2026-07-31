"""
Centralized configuration for the Agentic Multi-Tool AI Assistant backend.
All secrets are loaded from environment variables (.env) — never hardcoded.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root (works whether run from /backend or repo root)
ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")


class Settings:
    # --- API keys ---
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")

    # --- Model config ---
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    # --- Pollinations (free, keyless image API) ---
    POLLINATIONS_BASE_URL: str = os.getenv(
        "POLLINATIONS_BASE_URL", "https://image.pollinations.ai/prompt"
    )

    # --- Storage paths ---
    DATA_DIR: Path = ROOT_DIR / "data"
    CHROMA_PERSIST_DIR: str = os.getenv(
        "CHROMA_PERSIST_DIR", str(DATA_DIR / "chroma")
    )
    SQLITE_PATH: str = os.getenv(
        "SQLITE_PATH", str(DATA_DIR / "sqlite" / "chat_history.db")
    )

    # --- Memory / RAG tuning ---
    MEMORY_TOP_K: int = int(os.getenv("MEMORY_TOP_K", "4"))
    RAG_TOP_K: int = int(os.getenv("RAG_TOP_K", "4"))
    RAG_CHUNK_SIZE: int = int(os.getenv("RAG_CHUNK_SIZE", "1000"))
    RAG_CHUNK_OVERLAP: int = int(os.getenv("RAG_CHUNK_OVERLAP", "150"))

    # --- Server ---
    HOST: str = os.getenv("BACKEND_HOST", "0.0.0.0")
    PORT: int = int(os.getenv("BACKEND_PORT", "8000"))
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    def validate(self) -> list:
        """Return a list of missing/soft-required settings (used for startup warnings)."""
        warnings = []
        if not self.GROQ_API_KEY:
            warnings.append("GROQ_API_KEY is not set — LLM calls will fail.")
        if not self.TAVILY_API_KEY:
            warnings.append("TAVILY_API_KEY is not set — research agent will fail.")
        return warnings


settings = Settings()

# Ensure local storage directories exist
Path(settings.CHROMA_PERSIST_DIR).mkdir(parents=True, exist_ok=True)
Path(settings.SQLITE_PATH).parent.mkdir(parents=True, exist_ok=True)
