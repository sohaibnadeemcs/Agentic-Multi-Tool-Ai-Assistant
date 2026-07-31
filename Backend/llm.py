"""
Thin wrapper around the Groq API shared by every agent node.
Centralizing this makes it trivial to swap models/providers later.
"""
from typing import Iterator, List, Dict
from groq import Groq

from backend.config import settings
from backend.logger import get_logger

logger = get_logger(__name__)

_client = Groq(api_key=settings.GROQ_API_KEY) if settings.GROQ_API_KEY else None


def _require_client() -> Groq:
    if _client is None:
        raise RuntimeError(
            "GROQ_API_KEY is not configured. Add it to your .env file."
        )
    return _client


def chat_completion(messages: List[Dict[str, str]], temperature: float = 0.3, model: str = None) -> str:
    """Non-streaming completion. Returns the full text."""
    client = _require_client()
    model = model or settings.GROQ_MODEL
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
        )
        return response.choices[0].message.content
    except Exception:
        logger.exception("Groq chat_completion failed")
        raise


def chat_completion_stream(messages: List[Dict[str, str]], temperature: float = 0.3, model: str = None) -> Iterator[str]:
    """Streaming completion. Yields text chunks as they arrive."""
    client = _require_client()
    model = model or settings.GROQ_MODEL
    try:
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
    except Exception:
        logger.exception("Groq chat_completion_stream failed")
        raise
