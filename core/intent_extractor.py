# core/intent_extractor.py
import os
import json
import re
from typing import Dict, Any
from core.llm_client import OllamaClient

SMALL_MODEL = os.getenv("SMALL_MODEL", "qwen2.5:3b-instruct")
SYSTEM = (
    "You guess the user's smart-home action using short keywords.\n"
    "ALWAYS return ONE LINE JSON with exactly these keys:\n"
    '{"intent":"<what action>","target":"<what/where>","service":"<how/function>"}\n'
    "Rules:\n"
    "- Keep each field to a short phrase (≤6 words).\n"
    "- If uncertain, still guess (no empty fields).\n"
    "- Use nouns for target (device/area/room/object).\n"
    "- Use verbs/nouns for service (turn_on, dim, set_temp, open_curtain, fan_speed, etc.).\n"
    "- No prose, no extra keys, one line only.\n\n"
)
# Strip bidi/control chars that can confuse the model (e.g., \u200E seen in logs)
_CTRL = re.compile(r'[\u200E\u200F\u202A-\u202E\u2066-\u2069]')

def _clean(s: str) -> str:
    return _CTRL.sub('', s)

async def extract_intents(message: str, context: Dict[str, Any]) -> str:
    """
    Call the small LLM and return its RAW one-line output as text.
    - No JSON parsing.
    - No post-processing or fallback.
    - We only clean control/bidi chars from inputs to avoid {} outputs.
    """
    ctx_txt = _clean(json.dumps(context, ensure_ascii=False, separators=(",", ":")))
    msg_txt = _clean(message)
    user = f"Context:{ctx_txt}\nUser:{msg_txt}"
    out = await OllamaClient().chat(SYSTEM, [{"role": "user", "content": user}], model=SMALL_MODEL)
    return out.strip() if isinstance(out, str) else str(out)
