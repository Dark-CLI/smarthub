# core/llm_client.py

import time, httpx, os

class OllamaClient:
    def __init__(self, base_url=None):
        self.base = base_url or os.getenv("OLLAMA_URL", "http://localhost:11434")

    async def chat(self, system: str, messages, model: str, num_ctx=4096):
        """
        messages: Can be
          - a string (just user content)
          - a list[dict] (old format: [{'role':'user', 'content':'msg'}])
        """
        # Accept both formats for backward compatibility
        if isinstance(messages, list):
            user = "\n".join([m["content"] for m in messages])
        else:
            user = str(messages)
        # Build prompt
        prompt = (system.strip() + "\n" + user.strip()).strip()
        t0 = time.perf_counter()
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(f"{self.base}/api/generate", json={
                "model": model,
                "system": system.strip(),
                "prompt": user.strip(),
                "num_ctx": num_ctx,
                "stream": False
            })
        resp.raise_for_status()
        result = resp.json()
        t1 = time.perf_counter()
        print(f"LLM call: {model:20} {(t1-t0)*1000:.1f} ms")
        return result["response"]
