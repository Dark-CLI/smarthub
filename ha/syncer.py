# smarthub/ha/syncer.py
from __future__ import annotations

from typing import Dict, Any, List
from data.repo import Repo
from data.builders import embed_text_from_static
from data.embedding import embed_texts
from data.vectors import add_or_update, add_or_update_batch


async def sync_embeddings(embed_model: str = "nomic-embed-text",
                          embed_version: str = "v1",
                          batch: int = 32) -> Dict[str, int]:
    """
    Hydrate Repo from HA (once), build embed texts, embed via Ollama, upsert into LanceDB.
    Idempotent & safe to call repeatedly (we just overwrite vectors by key).
    """
    repo = Repo()
    await repo.ensure_hydrated()

    # ---- Entities: build texts
    ent_keys: List[str] = []
    ent_texts: List[str] = []
    for eid, static_json in repo.iter_entities():
        ent_keys.append(eid)
        ent_texts.append(embed_text_from_static(static_json))

    # ---- Domains: build texts
    dom_keys: List[str] = []
    dom_texts: List[str] = []
    for dom in repo.iter_domains():
        dom_keys.append(dom)
        dom_texts.append(repo.domain_embed_text(dom))

    # ---- Embed (batched) + upsert vectors
    entities_embedded = 0
    domains_embedded = 0

    # entities
    for i in range(0, len(ent_texts), batch):
        chunk = ent_texts[i : i + batch]
        vecs = await embed_texts(chunk, model=embed_model)
        pairs = [(("entity", ent_keys[i + j]), vecs[j]) for j in range(len(vecs))]
        add_or_update_batch(pairs)
        for j in range(len(vecs)):
            repo.mark_entity_embedded(ent_keys[i + j], embed_model, embed_version)
        entities_embedded += len(vecs)

    # domains
    for i in range(0, len(dom_texts), batch):
        chunk = dom_texts[i : i + batch]
        vecs = await embed_texts(chunk, model=embed_model)
        pairs = [(("domain", dom_keys[i + j]), vecs[j]) for j in range(len(vecs))]
        add_or_update_batch(pairs)
        for j in range(len(vecs)):
            repo.mark_domain_embedded(dom_keys[i + j], embed_model, embed_version)
        domains_embedded += len(vecs)

    return {
        "entities_embedded": entities_embedded,
        "domains_embedded": domains_embedded,
    }


# Backwards-compatible helper (if you already call sync_once elsewhere)
async def sync_once(embed_model: str = "nomic-embed-text", embed_version: str = "v1") -> Dict[str, int]:
    """
    Kept for compatibility with your scripts. Just delegates to sync_embeddings.
    """
    return await sync_embeddings(embed_model=embed_model, embed_version=embed_version)
