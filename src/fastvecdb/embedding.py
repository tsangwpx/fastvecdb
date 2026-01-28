"""Embedding model placeholder."""

import hashlib

from fastvecdb.config import EMBEDDING_DIMENSION


async def get_embedding(content_type: str | None, data: bytes | str) -> list[float]:
    """
    Get embedding for content.

    Placeholder function - returns a dummy vector for now.
    In production, this would call an actual embedding model.

    Args:
        content_type: MIME type of the content (e.g., 'text/plain')
        data: The content to embed (bytes or string)

    Returns:
        A vector of dimension EMBEDDING_DIMENSION
    """
    # TODO: Replace with actual embedding model (e.g., OpenAI, Hugging Face, etc.)
    # For now, return a dummy vector

    # Convert data to string for hashing
    if isinstance(data, bytes):
        text = data.decode("utf-8", errors="ignore")
    else:
        text = data

    # Create deterministic dummy embedding based on hash of text
    hash_obj = hashlib.sha256(text.encode())
    hash_bytes = hash_obj.digest()

    # Convert hash bytes to floats
    embedding: list[float] = []
    for i in range(EMBEDDING_DIMENSION):
        # Use bytes cyclically to generate dimension values
        byte_val = hash_bytes[i % len(hash_bytes)]
        # Normalize to [-1, 1]
        value = (byte_val / 127.5) - 1.0
        embedding.append(float(value))

    return embedding
