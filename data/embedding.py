# smarthub/data/embedding.py
import httpx
from typing import List

OLLAMA_URL = "http://localhost:11434/api/embeddings"


async def embed_texts(texts: List[str], model: str = "nomic-embed-text") -> List[List[float]]:
    """
    Call Ollama embedding API.
    Returns a list of embedding vectors (list of floats).
    """
    vectors: List[List[float]] = []
    async with httpx.AsyncClient(timeout=60.0) as client:
        for text in texts:
            resp = await client.post(
                OLLAMA_URL,
                json={"model": model, "prompt": text},
            )
            resp.raise_for_status()
            data = resp.json()
            vec = data.get("embedding")
            if not vec:
                raise RuntimeError(f"No embedding in response: {data}")
            vectors.append(vec)
    return vectors
