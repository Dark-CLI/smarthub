# data/search_interface.py
from typing import Any, Dict, List, Tuple
from data.embedding import embed_texts
from data.vectors import query as vector_query
from ha.client import HAClient

async def search_devices(text: str, top_k: int = 12, embed_model: str = "nomic-embed-text") -> List[Dict[str, Any]]:
    """
    Text → embed → vector search → resolve with HA.
    Returns a list of dicts ready for Big LLM:
      { "key": str, "entity_id": Optional[str], "name": str,
        "domain": Optional[str], "area": Optional[str], "services": List[str] }
    """
    # 1) embed
    qvec = (await embed_texts([text], model=embed_model))[0]

    # 2) vector search ([(key, score), ...])
    hits: List[Tuple[str, float]] = vector_query(qvec, top_k=top_k) or []

    # 3) resolve via live HA
    ha = HAClient()
    states = await ha.states()
    state_map = {s["entity_id"]: s for s in states}
    svc_map = await ha.services_map()  # {"light": {"turn_on": {...}, ...}, ...}

    devices: List[Dict[str, Any]] = []
    for key, _score in hits:
        try:
            kind, ident = key.split(":", 1)
        except ValueError:
            devices.append({"key": key, "entity_id": None, "name": key, "domain": None, "area": None, "services": []})
            continue

        if kind == "entity":
            entity_id = ident
            domain = entity_id.split(".", 1)[0] if "." in entity_id else None
            st = state_map.get(entity_id) or {}
            attrs = (st.get("attributes") or {})
            name = attrs.get("friendly_name") or entity_id
            # HA states don’t usually include area; leave None unless you have a resolver
            area = attrs.get("area") or None
            services = list((svc_map.get(domain) or {}).keys()) if domain else []
            devices.append({
                "key": key,
                "entity_id": entity_id,
                "name": name,
                "domain": domain,
                "area": area,
                "services": services,
            })
        elif kind == "domain":
            domain = ident
            services = list((svc_map.get(domain) or {}).keys())
            devices.append({
                "key": key,
                "entity_id": None,
                "name": domain,
                "domain": domain,
                "area": None,
                "services": services,
            })
        else:
            devices.append({"key": key, "entity_id": None, "name": key, "domain": None, "area": None, "services": []})

    return devices
