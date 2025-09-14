import os, json
from typing import Dict, Any
from core.llm_client import OllamaClient
from utils.jsonio import parse_one_line_json, clamp_value

BIG_MODEL = os.getenv("BIG_MODEL", "llama3.1:8b-instruct")

_SYSTEM = (
    "Return ONE JSON line only, no extra text.\n"
    "Modes:\n"
    '  EXECUTE_AND_REPLY: {"mode":"EXECUTE_AND_REPLY","device_id":"...","action_id":"...","args":{...},"reply_text":"..."}\n'
    '  FETCH_MORE: {"mode":"FETCH_MORE","fetch":"devices_for_area","params":{"area":"..."}}  # example fetch\n'
    '  REPLY: {"mode":"REPLY","text":"..."}\n'
    "Use ONLY provided candidates; respect schema_hint (clamp ranges if needed)."
)

async def decide_and_reply(bundle: Dict[str, Any], summary: str) -> Dict[str, Any]:
    user = "Summary:" + summary + "\nInput:\n" + json.dumps(bundle, ensure_ascii=False)
    out = await OllamaClient().chat(_SYSTEM, [{"role":"user","content":user}], model=BIG_MODEL)
    data = parse_one_line_json(out)
    if not data:
        return {"mode":"REPLY","text":"I couldn’t parse that. Please rephrase."}
    # Optional clamp if model tweaked args:
    if data.get("mode") == "EXECUTE_AND_REPLY":
        # try clamp using first candidate's hint if present
        if bundle.get("candidates"):
            hint = bundle["candidates"][0].get("schema_hint", {})
            data["args"] = clamp_value(data.get("args", {}), hint)
    return data
