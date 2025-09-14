from abc import ABC, abstractmethod
from typing import List, Dict
import httpx, os

class LLMClient(ABC):
    @abstractmethod
    async def chat(self, system: str, messages: List[Dict[str, str]], model: str) -> str: ...

class OllamaClient(LLMClient):
    def __init__(self, base_url: str | None = None):
        self.base = base_url or os.getenv("OLLAMA_URL", "http://localhost:11434")

    async def chat(self, system: str, messages: List[Dict[str, str]], model: str) -> str:
        # Simple prompt format; you can switch to /api/chat later
        prompt = (system + "\n\n" if system else "") + "\n".join([f"{m['role'].upper()}: {m['content']}" for m in messages])
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(f"{self.base}/api/generate", json={
                "model": model,
                "prompt": prompt,
                "stream": False
            })
        r.raise_for_status()
        return r.json()["response"]
