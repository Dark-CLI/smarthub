# ha/syncer.py
import json
from typing import Dict, Any, List

from data.embedding import embed_texts
from data.vectors_devices import add_or_update as add_devices
from data.vectors_actions import add_or_update as add_actions
from ha.client import HAClient

def _compact_device_json(state: Dict[str, Any]) -> str:
    """
    Minimal but useful JSON for embedding. NO dynamic state values.
    Keep entity_id to tie back later.
    """
    entity_id = state["entity_id"]
    domain = entity_id.split(".", 1)[0] if "." in entity_id else None
    name = (state.get("attributes") or {}).get("friendly_name")
    obj = {
        "entity_id": entity_id,
        "name": name,
        "domain": domain,
        # area: wire later if you fetch area registry
    }
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))

def _compact_action_json(domain: str, service: str, schema: Dict[str, Any]) -> str:
    """
    Minimal JSON for action semantics. Include fields (arg names) only.
    """
    fields = list((schema or {}).get("fields", {}).keys())
    obj = {"action": f"{domain}.{service}", "domain": domain, "service": service, "fields": fields}
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))

async def sync_all(embed_model: str = "nomic-embed-text") -> Dict[str, int]:
    ha = HAClient()

    # DEVICES
    states: List[Dict[str, Any]] = await ha.states()
    device_texts: List[str] = [_compact_device_json(st) for st in states]
    device_vecs = await embed_texts(device_texts, model=embed_model)
    dev_rows = []
    for st, snap, vec in zip(states, device_texts, device_vecs):
        key = f"entity:{st['entity_id']}"
        dev_rows.append({"key": key, "vector": vec, "snapshot": snap})
    add_devices(dev_rows)

    # ACTIONS
    svc_map: Dict[str, Dict[str, Any]] = await ha.services_map()  # {"light":{"turn_on":{schema},...},...}
    action_pairs: List[tuple[str, str, Dict[str, Any]]] = []
    for domain, svcs in (svc_map or {}).items():
        for service, schema in svcs.items():
            action_pairs.append((domain, service, schema))
    action_texts = [_compact_action_json(d, s, schema) for (d, s, schema) in action_pairs]
    action_vecs = await embed_texts(action_texts, model=embed_model)
    act_rows = []
    for (domain, service, _schema), snap, vec in zip(action_pairs, action_texts, action_vecs):
        key = f"service:{domain}.{service}"
        act_rows.append({"key": key, "vector": vec, "snapshot": snap})
    add_actions(act_rows)

    return {"devices_indexed": len(dev_rows), "actions_indexed": len(act_rows)}
