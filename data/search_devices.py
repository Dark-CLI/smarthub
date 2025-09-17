# data/search_devices.py
from typing import Any, Dict, List, Tuple
import json
from data.embedding import embed_texts
from data.vectors_devices import query as query_devices
from ha.client import HAClient
from utils.filters import filter_entity_map


async def search_devices(text: str, top_k: int = 6, embed_model: str = "nomic-embed-text") -> List[Dict[str, Any]]:
    """
    Text -> embed -> search devices_index -> return fresh resolved devices.
    We fetch *fresh* state to get up-to-date friendly_name and preserve entity_id.
    """
    qvec = (await embed_texts([text], model=embed_model))[0]
    hits: List[Tuple[str, float]] = query_devices(qvec, top_k=top_k) or []

    ha = HAClient()
    entity_ids = [ident for (kind, ident) in [k.split(":", 1) for k, _ in hits] if kind == "entity"]
    states = await ha.states_batch(entity_ids)
    state_map = filter_entity_map({s["entity_id"]: s for s in states})
    return list(state_map.items())
