# smarthub/ha/syncer.py
from __future__ import annotations
from typing import Dict, Any, List, Tuple

from ha.client import HAClient
from data.embedding import embed_texts
from data.vectors import add_or_update_batch_with_meta, get_last_hash
from data.builders import build_static_descriptor


async def sync_embeddings(embed_model: str = "nomic-embed-text",
                          batch: int = 32) -> Dict[str, int]:
    """
    Incremental sync (no repo):
      - fetch /api/states and /api/services via HAClient
      - build compact descriptors (drop empties, decode features)
      - hash stable content; embed ONLY when changed
      - upsert vectors + metadata to LanceDB
    """
    ha = HAClient()
    states = await ha.states()
    services_map = await ha.services_map()  # domain -> services dict

    to_embed: List[Tuple[str, str, Dict[str, Any]]] = []  # (entity_id, descriptor_text, meta)

    for st in states:
        eid = st.get("entity_id")
        if not eid or "." not in eid:
            continue
        dom = eid.split(".", 1)[0]
        desc_text, content, content_hash = build_static_descriptor(st, services_map.get(dom, {}))
        if not desc_text:
            continue

        prev_hash = get_last_hash(("entity", eid))
        if prev_hash != content_hash:
            meta = {
                "domain": content.get("domain"),
                "area": content.get("area"),
                "content_hash": content_hash,
                "last_embedded_hash": content_hash,
            }
            to_embed.append((eid, desc_text, meta))

    embedded = 0
    # simple batching; embedding itself is sequential per text for now
    for i in range(0, len(to_embed), batch):
        chunk = to_embed[i : i + batch]
        texts = [t for (_, t, __) in chunk]
        vecs = await embed_texts(texts, model=embed_model)
        pairs = []
        for j, (eid, _text, meta) in enumerate(chunk):
            pairs.append((("entity", eid), vecs[j], meta))
        add_or_update_batch_with_meta(pairs)
        embedded += len(pairs)

    return {
        "entities_scanned": len(states),
        "entities_embedded": embedded,
    }
