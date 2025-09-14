# smarthub/data/embedding.py
from __future__ import annotations

from typing import List, Iterable, Optional
import os
import httpx

# Ollama embeddings endpoint expects a single string under "prompt"
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
EMBED_ENDPOINT = os.environ.get("OLLAMA_EMBED_ENDPOINT", "/api/embeddings")
EMBED_URL = f"{OLLAMA_URL.rstrip('/')}{EMBED_ENDPOINT}"
DEFAULT_TIMEOUT = float(os.environ.get("EMBED_TIMEOUT_S", "60.0"))

async def _embed_one(text: str, model: str, client: httpx.AsyncClient) -> List[float]:
    payload = {"model": model, "prompt": text or " "}
    r = await client.post(EMBED_URL, json=payload, headers={"Content-Type": "application/json"})
    r.raise_for_status()
    data = r.json()
    emb = data.get("embedding")
    if not isinstance(emb, list) or not emb:
        raise RuntimeError(f"Embedding missing/empty; response keys={list(data.keys())}")
    return [float(x) for x in emb]

async def embed_texts(texts: Iterable[str], model: str) -> List[List[float]]:
    items = list(texts)
    if not items:
        return []
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as cli:
        out: List[List[float]] = []
        for t in items:  # simple & predictable
            out.append(await _embed_one(t, model, cli))
        return out
