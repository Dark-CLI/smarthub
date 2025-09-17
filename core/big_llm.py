# core/big_llm.py
import json
from typing import Any, Dict, List
from core.llm_client import OllamaClient

SYSTEM = """
You are the single-turn decision & reply layer for a smart-home assistant.
Return EXACTLY ONE LINE JSON (no prose). Choose one:

EXECUTE:
{"mode":"EXECUTE","device":"<entity_id>","action":"<domain.service>","args":{},"reply":"<short confirmation>"}

REPLY:
{"mode":"REPLY","text":"<short answer>"}

Inputs:
- user_message: raw user text
- context: current room/context
- recent: compact JSON (messages/memory/executions)
- keywords: short keywords from a small parser (free text)
- devices: candidate devices from search (each item has entity_id|key, name, domain, area, services[])

Output ONE line of JSON only.
"""

async def run_big_llm(
    user_message: str,
    context: Dict[str, Any],
    recent_json: str,
    keywords_text: str,
    devices: List[Dict[str, Any]],
    model: str = "llama3.1:latest",
) -> str:
    """
    Just pass inputs to the big LLM and return its RAW one-line output.
    No post-processing.
    """
    user_blob = (
        f"user_message={json.dumps(user_message)}\n"
        f"context={json.dumps(context, ensure_ascii=False, separators=(',',':'))}\n"
        f"recent={recent_json}\n"
        f"keywords={json.dumps(keywords_text)}\n"
        f"devices={json.dumps(devices, ensure_ascii=False, separators=(',',':'))}"
    )
    out = await OllamaClient().chat(
        SYSTEM,
        [{"role":"user","content":user_blob}],
        model=model
    )
    return out.strip() if isinstance(out, str) else str(out)
