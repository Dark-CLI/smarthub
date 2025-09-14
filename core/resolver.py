from typing import Any, Dict, List
from data.repo import Repo
from utils.jsonio import clamp_value

def _normalize_args(intent_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    # Example: map "low/med/high" to numbers for brightness
    mapping = {"low":25, "medium":50, "med":50, "high":75, "max":100}
    if "value" in args and isinstance(args["value"], str):
        v = args["value"].lower().strip()
        if v in mapping: args["value"] = mapping[v]
        else:
            try: args["value"] = int(float(v))
            except: pass
    return args

async def build_bundle_from_intent(repo: Repo, intent: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    # 1) pick devices via vector search + aliases + room bias
    targets: List[str] = intent.get("targets", [])
    devices = repo.search_devices(targets, context)[:3]

    candidates = []
    for dev in devices:
        action = repo.map_intent_to_action(dev, intent["intent"])
        if not action:
            continue
        args = _normalize_args(intent["intent"], dict(intent.get("args", {})))
        hint = repo.load_schema_hint(action.schema_hash)
        args = clamp_value(args, hint)  # one clamp
        candidates.append({
            "device":{"id":dev.id,"name":dev.name,"type":dev.type},
            "action":{"id":action.id,"name":action.name},
            "schema_hint": hint,
            "args_proposed": args
        })

    bundle = {
        "user": intent,
        "context": context,
        "candidates": candidates,
        "examples": repo.sample_examples(intent.get("intent",""), context)  # ≤3 small
    }
    return bundle
