# data/search_actions.py
from typing import Any, Dict, List, Tuple
from data.embedding import embed_texts
from data.vectors_actions import query as query_actions
from ha.client import HAClient

async def search_actions(text: str, top_k: int = 6, embed_model: str = "nomic-embed-text") -> List[Dict[str, Any]]:
    """
    Text -> embed -> search actions_index -> return fresh resolved actions.
    We also fetch HA service map to later validate args or expose fields.
    """
    qvec = (await embed_texts([text], model=embed_model))[0]
    hits: List[Tuple[str, float]] = query_actions(qvec, top_k=top_k) or []

    ha = HAClient()
    svc_map = await ha.services_map()  # {"light":{"turn_on":{...},...},...}

    out: List[Dict[str, Any]] = []
    for key, _ in hits:
        kind, ident = key.split(":", 1)
        if kind != "service":
            continue
        domain, service = ident.split(".", 1)
        fields = list((svc_map.get(domain, {}).get(service, {}).get("fields", {}) or {}).keys())
        out.append({
            "key": key,
            "action": ident,      # "light.turn_on"
            "domain": domain,
            "service": service,
            "fields": fields,     # to help big LLM see possible args (still no coaching)
        })
    return out
