# core/big_llm.py
import json
from typing import Any, Dict, List
from core.llm_client import OllamaClient

SYSTEM = """
You are the single-turn decision & reply layer for a smart-home assistant.
Return EXACTLY ONE LINE JSON (no prose). One of:

{"mode":"EXECUTE","device":"<entity_id>","action":"<domain.service>","args":{},"reply":"<short confirmation>"}
{"mode":"REPLY","text":"<short answer>"}

Inputs:
- user_message, context, recent, keywords
- devices: [{entity_id, name, domain, area}]
- actions: [{action, domain, service, fields}]
Output ONE line of JSON only.
"""

async def run_big_llm(
    user_message: str,
    context: Dict[str, Any],
    # recent_json: str,
    # keywords_text: str,
    devices: List[Dict[str, Any]],
    actions: List[Dict[str, Any]],
    model: str = "qwen2.5:7b-instruct",
) -> str:
    user_blob = (
        f"user_message={json.dumps(user_message)}\n"
        f"context={json.dumps(context, ensure_ascii=False, separators=(',',':'))}\n"
        # f"recent={recent_json}\n"
        f"devices={json.dumps(devices, ensure_ascii=False, separators=(',',':'))}\n"
        f"actions={json.dumps(actions, ensure_ascii=False, separators=(',',':'))}"
    )
    out = await OllamaClient().chat(
        SYSTEM,
        [{"role":"user","content":user_blob}],
        model=model
    )
    return out.strip() if isinstance(out, str) else str(out)
