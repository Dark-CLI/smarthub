# core/history.py
import json

def compact_recent() -> str:
    history = {
        "messages": [
            {"role":"user","text":"hi"},
            {"role":"assistant","text":"hello, what can I do?"},
            {"role":"user","text":"lights were too bright this morning"},
            {"role":"assistant","text":"we can dim them next time"},
            {"role":"user","text":"bedroom felt cold last night"},
            {"role":"assistant","text":"we can set temp to 21C at night"},
        ],
        "memory": {  # fake memory
            "lighting_pref":"softer mornings",
            "temp_pref":"21–22C in bedroom at night",
            "mood":"often tired in evenings",
        },
        "executions": [  # actions previously executed by the LLM
            {"service":"light.turn_on","device":"light.living_room_main","args":{},"result":"ok"},
            {"service":"climate.set_temperature","device":"climate.bedroom","args":{"temperature":21},"result":"ok"},
        ],
    }
    return json.dumps(history, ensure_ascii=False, separators=(",",":"))
