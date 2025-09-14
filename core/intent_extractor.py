import os
from typing import Any, Dict
from core.llm_client import OllamaClient
from utils.jsonio import parse_one_line_json

SMALL_MODEL = os.getenv("SMALL_MODEL", "qwen2.5:3b-instruct")

async def extract_intent_or_reply(message: str, context: Dict[str, Any], summary: str) -> Dict[str, Any]:
    """
    Returns:
      {"mode":"reply","text": "..."} OR
      {"mode":"intent","intent": {"intent":"...","targets":[...],"args":{...},"confidence":0.0-1.0}}
    """
    system = (
        "You extract smart-home intents. If no action is needed, reply in plain text.\n"
        'Else output ONE LINE JSON: {"intent":"...","targets":["..."],"args":{...},"confidence":0.0-1.0}\n'
        "Intents: turn_on | turn_off | set_brightness | set_temp | open_curtain | close_curtain\n"
        "Return only one mode."
    )
    user = f'Context: {context}\nSummary:{summary}\nUser:"{message}"'
    out = await OllamaClient().chat(system, [{"role":"user","content":user}], model=SMALL_MODEL)

    data = parse_one_line_json(out)
    if data is None:
        return {"mode":"reply", "text": out.strip()}
    return {"mode":"intent", "intent": data}
