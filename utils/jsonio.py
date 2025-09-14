import json, re

def parse_one_line_json(text: str):
    m = re.search(r"\{.*\}", text.strip(), flags=re.DOTALL)
    if not m: return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None

def clamp_value(args: dict, hint: dict) -> dict:
    rng = hint.get("value_range")
    if rng and "value" in args:
        try:
            lo, hi = map(int, rng.split("-"))
            args["value"] = max(lo, min(hi, int(args["value"])))
        except Exception:
            pass
    return args
