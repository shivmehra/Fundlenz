from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import settings


@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    return SentenceTransformer(settings.embed_model)


def embed(texts: list[str]) -> np.ndarray:
    vecs = _model().encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    return vecs.astype("float32")


def chunk_text(text: str, max_tokens: int, overlap: int) -> list[str]:
    """Whitespace-based chunking. Approximates tokens as words."""
    words = text.split()
    if not words:
        return []
    chunks: list[str] = []
    start = 0
    step = max(1, max_tokens - overlap)
    while start < len(words):
        chunk = words[start : start + max_tokens]
        if not chunk:
            break
        chunks.append(" ".join(chunk))
        start += step
    return chunks
