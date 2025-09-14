from ha.client import HAClient
from data.repo import Repo
from data.builders import static_snapshot_from_state, embed_text_from_static, dict_hash
from data.embedding import embed_texts
from data.vectors import add_or_update


async def sync_once(embed_model: str, embed_version: str) -> dict:
    ha = HAClient()
    repo = Repo()

    states = await ha.states()          # list of entity states
    services_list = await ha.services() # list of {domain, services}
    # Build a domain->services dict for quick lookup
    svc_by_domain = {b["domain"]: b.get("services", {}) for b in services_list if isinstance(b, dict)}

    # 1) upsert entities (static-only)
    changed_entities = []
    for st in states:
        dom = st["entity_id"].split(".", 1)[0]
        static_json = static_snapshot_from_state(st, {dom: svc_by_domain.get(dom, {})})
        content_hash = dict_hash(static_json)
        if repo.upsert_entity(static_json, content_hash):  # returns True if new/changed
            changed_entities.append(st["entity_id"])

    # 2) upsert domain services (one row per domain)
    changed_domains = []
    for dom, svcs in svc_by_domain.items():
        srv_json = {"domain": dom, "services": svcs}
        h = dict_hash(srv_json)
        if repo.upsert_domain_services(dom, srv_json, h):
            changed_domains.append(dom)

    # 3) embed what’s needed (new/changed or model bumped)
    need_ent = repo.entities_needing_embedding(embed_model, embed_version)
    texts_ent = [embed_text_from_static(repo.get_entity_static(eid)) for eid in need_ent]
    if texts_ent:
        vecs = await embed_texts(texts_ent, model=embed_model)
        for eid, txt, vec in zip(need_ent, texts_ent, vecs):
            vid = add_or_update(key=("entity", eid), vector=vec)
            repo.save_entity_embedding(eid, vid, txt, embed_model, embed_version)

    need_dom = repo.services_needing_embedding(embed_model, embed_version)
    texts_dom = [f"domain: {d}\n" + repo.domain_embed_text(d) for d in need_dom]
    if texts_dom:
        vecs = await embed_texts(texts_dom, model=embed_model)
        for dom, txt, vec in zip(need_dom, texts_dom, vecs):
            vid = add_or_update(key=("domain", dom), vector=vec)
            repo.save_domain_embedding(dom, vid, txt, embed_model, embed_version)

    return {
        "entities_changed": len(changed_entities),
        "domains_changed": len(changed_domains),
        "entities_embedded": len(need_ent),
        "domains_embedded": len(need_dom),
    }
