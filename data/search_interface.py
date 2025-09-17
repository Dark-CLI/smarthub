# data/search.py
from __future__ import annotations
import time
from typing import Any, Dict, List, Tuple, Optional

from data.embedding import embed_texts
from ha.client import HAClient

# Import device/actions query funcs
from data.vectors_devices import query as _query_devices  # (qvec, top_k) -> List[Tuple[key, score]]
try:
    from data.vectors_actions import query as _query_actions  # (qvec, top_k) -> List[Tuple[key, score]]
except Exception:
    _query_actions = None  # actions index not present yet → return []

# tiny cache for HA services to avoid hammering
_SVC_CACHE: Dict[str, Any] = {}
_SVC_CACHE_AT = 0.0
_SVC_TTL = 30.0

async def _services_map(ha: HAClient) -> Dict[str, Dict[str, Any]]:
    global _SVC_CACHE_AT
    now = time.time()
    if _SVC_CACHE and (now - _SVC_CACHE_AT) < _SVC_TTL:
        return _SVC_CACHE
    m = await ha.services_map()
    _SVC_CACHE.clear()
    _SVC_CACHE.update(m or {})
    _SVC_CACHE_AT = now
    return _SVC_CACHE

async def _resolve_devices(ha: HAClient, hits: List[Tuple[str, float]]) -> List[Dict[str, Any]]:
    """[(key, score)] -> [{key, entity_id?, name, domain, area?, services[]}] (no scores in output)"""
    states = await ha.states()
    state_map = {s["entity_id"]: s for s in states}
    svc_map = await _services_map(ha)

    out: List[Dict[str, Any]] = []
    for key, _ in hits:
        try:
            kind, ident = key.split(":", 1)
        except ValueError:
            out.append({"key": key, "entity_id": None, "name": key, "domain": None, "area": None, "services": []})
            continue

        if kind == "entity":
            entity_id = ident
            domain = entity_id.split(".", 1)[0] if "." in entity_id else None
            st = state_map.get(entity_id) or {}
            attrs = (st.get("attributes") or {})
            friendly = attrs.get("friendly_name") or entity_id
            area = None  # TODO: wire HA area registry if you want real areas
            services = list((svc_map.get(domain) or {}).keys()) if domain else []
            out.append({
                "key": key,
                "entity_id": entity_id,
                "name": friendly,
                "domain": domain,
                "area": area,
                "services": services,
            })
        elif kind == "domain":
            domain = ident
            services = list((svc_map.get(domain) or {}).keys())
            out.append({
                "key": key,
                "entity_id": None,
                "name": domain,
                "domain": domain,
                "area": None,
                "services": services,
            })
        else:
            out.append({"key": key, "entity_id": None, "name": key, "domain": None, "area": None, "services": []})
    return out

async def _resolve_actions(ha: HAClient, hits: List[Tuple[str, float]]) -> List[Dict[str, Any]]:
    """[(key, score)] -> [{key, action, domain, service, args_schema?}]"""
    if not hits:
        return []
    svc_map = await _services_map(ha)

    out: List[Dict[str, Any]] = []
    for key, _ in hits:
        # Expected shapes:
        # - "service:light.turn_on"
        # - or your actions table may already store fields; if so, keep it minimal
        action = None
        domain = None
        service = None
        try:
            kind, ident = key.split(":", 1)
            if kind == "service" and "." in ident:
                domain, service = ident.split(".", 1)
                action = ident
        except ValueError:
            pass

        if not action and "." in key:
            # fallback: parse raw key if it contains a dot
            action = key.split(":", 1)[-1]

        if not action:
            out.append({"key": key, "action": None, "domain": None, "service": None})
            continue

        if not domain or not service:
            if "." in action:
                domain, service = action.split(".", 1)
        schema = (svc_map.get(domain) or {}).get(service) if (domain and service) else None
        out.append({
            "key": key,
            "action": action,
            "domain": domain,
            "service": service,
            "args_schema": schema or None,  # pass through if you want; big LLM can ignore
        })
    return out

class Searcher:
    """
    One-shot searcher:
      - embed text ONCE
      - query device index AND action index
      - resolve to ready JSON lists for Big LLM
    """

    def __init__(self, top_k_devices: int = 6, top_k_actions: int = 6, embed_model: str = "nomic-embed-text"):
        self.top_k_devices = top_k_devices
        self.top_k_actions = top_k_actions
        self.embed_model = embed_model

    async def search(self, text: str) -> Dict[str, Any]:
        # 1) embed once
        qvec = (await embed_texts([text], model=self.embed_model))[0]

        # 2) vector queries
        dev_hits: List[Tuple[str, float]] = _query_devices(qvec, top_k=self.top_k_devices) or []
        act_hits: List[Tuple[str, float]] = []
        if _query_actions is not None:
            act_hits = _query_actions(qvec, top_k=self.top_k_actions) or []

        # 3) resolve
        ha = HAClient()
        devices = await _resolve_devices(ha, dev_hits)
        actions = await _resolve_actions(ha, act_hits)

        return {"devices": devices, "actions": actions}
