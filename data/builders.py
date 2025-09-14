import json, hashlib

def dict_hash(d: dict) -> str:
    """Stable hash for a dict, used to detect changes in static data."""
    return hashlib.sha1(json.dumps(d, sort_keys=True, separators=(",",":")).encode()).hexdigest()

def static_snapshot_from_state(state: dict, domain_services: dict) -> dict:
    """Build static-only snapshot for embedding (no live values)."""
    eid = state["entity_id"]; domain = eid.split(".", 1)[0]
    attrs = state.get("attributes", {}) or {}
    return {
        "entity_id": eid,
        "domain": domain,
        "friendly_name": attrs.get("friendly_name", eid),
        "attribute_keys": sorted(list(attrs.keys())),
        "service_fields": {
            name: sorted(list((defn.get("fields") or {}).keys()))
            for name, defn in (domain_services.get(domain, {}) or {}).items()
        }
    }

def embed_text_from_static(static_json: dict) -> str:
    """Convert snapshot dict into a flat text for embedding."""
    parts = [
        f"name: {static_json.get('friendly_name')}",
        f"entity_id: {static_json.get('entity_id')}",
        f"domain: {static_json.get('domain')}",
        "attributes: " + ", ".join(static_json.get("attribute_keys", [])),
    ]
    if static_json.get("service_fields"):
        svc = [f"{k}({', '.join(v)})" for k, v in static_json["service_fields"].items()]
        parts.append("services: " + "; ".join(sorted(svc)))
    return "\n".join(parts)
