# smarthub/data/repo.py
from __future__ import annotations

from typing import Any, Dict, List, Optional
from data.models import Session, Message, Device, Action, Schema
from core.summarizer import update_summary as _update_summary_fn

# -----------------------------
# In-memory "DB" (dev/demo)
# -----------------------------

_SESS: Dict[str, Session] = {}                  # chat_id -> Session
_MSGS: Dict[str, List[Message]] = {}            # chat_id -> [Message]
_DEVICES: List[Device] = []                     # seeded demo devices
_ACTIONS: List[Action] = []                     # seeded demo actions
_SCHEMAS: Dict[str, Schema] = {}                # schema_hash -> Schema

# Indexes for HA sync + embeddings (static-only)
_DEVICE_INDEX: Dict[str, Dict[str, Any]] = {}   # entity_id -> row dict
_DOMAIN_INDEX: Dict[str, Dict[str, Any]] = {}   # domain -> row dict


def _now() -> int:
    import time
    return int(time.time())


# -----------------------------
# Repository
# -----------------------------
class Repo:
    # ===== Sessions / Messages =====
    def load_or_init_session(self, chat_id: str, tenant_id: Optional[str] = None) -> Session:
        s = _SESS.get(chat_id)
        if not s:
            s = Session(chat_id=chat_id, tenant_id=tenant_id, summary_text="", updated_at=_now())
            _SESS[chat_id] = s
            _MSGS[chat_id] = []
        return s

    def add_message(self, chat_id: str, role: str, content: str) -> None:
        m = Message(chat_id=chat_id, role=role, content=content, created_at=_now())
        _MSGS.setdefault(chat_id, []).append(m)

    def load_summary(self, chat_id: str) -> str:
        s = _SESS.get(chat_id)
        return s.summary_text if s else ""

    def update_summary(self, chat_id: str, user: str, assistant: str) -> None:
        s = _SESS.get(chat_id)
        if not s:
            return
        s.summary_text = _update_summary_fn(s.summary_text, user, assistant)
        s.updated_at = _now()

    def list_messages(self, chat_id: str) -> List[Message]:
        return list(_MSGS.get(chat_id, []))

    # ===== Devices / Actions / Schemas (demo) =====
    def search_devices(self, targets: List[str], context: Dict[str, Any]) -> List[Device]:
        """Very naive: substring match on device name + optional room bias."""
        room = (context or {}).get("room", "").lower()
        outs: List[Device] = []
        for d in _DEVICES:
            name = d.name.lower()
            area = (d.area or "").lower()
            if any(t.lower() in name for t in targets) or (room and room == area):
                outs.append(d)
        return outs or _DEVICES[:1]

    def map_intent_to_action(self, dev: Device, intent: str) -> Optional[Action]:
        """Pick the first action on device whose name overlaps the intent string (naive)."""
        needle = intent.replace("_", " ").lower()
        for a in _ACTIONS:
            if a.device_id == dev.id and needle in a.name.lower():
                return a
        return None

    def load_schema_hint(self, schema_hash: str) -> Dict[str, Any]:
        sch = _SCHEMAS.get(schema_hash)
        if not sch:
            return {}
        # Minimal hint extraction (generic)
        props = (sch.schema_json or {}).get("properties", {})
        rng = props.get("value", {}).get("range", "0-100")
        return {"value_range": rng, **(sch.hint_json or {})}

    def sample_examples(self, intent: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        # Tiny canned example for bootstrapping
        return [{
            "user": "make it dimmer",
            "resolved": {"device_id": "dev_42", "action_id": "act_brightness"},
            "args": {"value": 25},
            "result": "ok"
        }]

    def devices_for_area(self, area: str) -> List[Dict[str, Any]]:
        outs: List[Dict[str, Any]] = []
        for d in _DEVICES:
            if d.area.lower() == (area or "").lower():
                outs.append({
                    "device": {"id": d.id, "name": d.name, "type": d.type},
                    "action": {"id": "act_on", "name": "turn_on"},
                    "schema_hint": {"toggle": True},
                    "args_proposed": {}
                })
        return outs

    # ===== HA Sync: static snapshots & embeddings (in-memory) =====
    def upsert_entity(self, static_json: Dict[str, Any], content_hash: str) -> bool:
        """
        Store/refresh a static device snapshot (no live values).
        Returns True if it's new/changed (triggers re-embedding).
        """
        eid = static_json["entity_id"]
        row = _DEVICE_INDEX.get(eid)
        if row and row.get("content_hash") == content_hash:
            return False
        _DEVICE_INDEX[eid] = {
            **(row or {}),
            "static_json": static_json,
            "content_hash": content_hash,
            "updated_at": _now(),
        }
        return True

    def upsert_domain_services(self, domain: str, content_json: Dict[str, Any], content_hash: str) -> bool:
        """
        Store/refresh a domain services block (as returned by /api/services for that domain).
        Returns True if new/changed.
        """
        row = _DOMAIN_INDEX.get(domain)
        if row and row.get("content_hash") == content_hash:
            return False
        _DOMAIN_INDEX[domain] = {
            **(row or {}),
            "content_json": content_json,
            "content_hash": content_hash,
            "updated_at": _now(),
        }
        return True

    def entities_needing_embedding(self, model: str, version: str) -> List[str]:
        """Which entity snapshots require embedding (missing or model/version mismatch)."""
        need: List[str] = []
        for eid, row in _DEVICE_INDEX.items():
            if row.get("embed_model") != model or row.get("embed_version") != version or not row.get("embed_text"):
                need.append(eid)
        return need

    def services_needing_embedding(self, model: str, version: str) -> List[str]:
        need: List[str] = []
        for dom, row in _DOMAIN_INDEX.items():
            if row.get("embed_model") != model or row.get("embed_version") != version or not row.get("embed_text"):
                need.append(dom)
        return need

    def get_entity_static(self, entity_id: str) -> Dict[str, Any]:
        row = _DEVICE_INDEX.get(entity_id)
        if not row:
            raise KeyError(entity_id)
        return row["static_json"]

    def domain_embed_text(self, domain: str) -> str:
        """Build a simple text summary from stored domain services JSON."""
        row = _DOMAIN_INDEX.get(domain)
        if not row:
            return f"domain: {domain}"
        data = row["content_json"]
        services = data.get("services", {}) or {}
        parts = [f"domain: {domain}", "services:"]
        for name, spec in services.items():
            fields = ", ".join(sorted((spec.get("fields") or {}).keys()))
            parts.append(f"- {name}({fields})")
        return "\n".join(parts)

    def save_entity_embedding(self, entity_id: str, vector_id: int, embed_text: str, model: str, version: str) -> None:
        row = _DEVICE_INDEX.get(entity_id) or {}
        row.update({
            "vector_id": vector_id,
            "embed_text": embed_text,
            "embed_model": model,
            "embed_version": version,
            "embedded_at": _now(),
        })
        _DEVICE_INDEX[entity_id] = row

    def save_domain_embedding(self, domain: str, vector_id: int, embed_text: str, model: str, version: str) -> None:
        row = _DOMAIN_INDEX.get(domain) or {}
        row.update({
            "vector_id": vector_id,
            "embed_text": embed_text,
            "embed_model": model,
            "embed_version": version,
            "embedded_at": _now(),
        })
        _DOMAIN_INDEX[domain] = row

    # ===== Debug / Introspection =====
    def debug_counts(self) -> Dict[str, int]:
        return {
            "sessions": len(_SESS),
            "messages": sum(len(v) for v in _MSGS.values()),
            "devices_seeded": len(_DEVICES),
            "actions_seeded": len(_ACTIONS),
            "schemas_seeded": len(_SCHEMAS),
            "device_index": len(_DEVICE_INDEX),
            "domain_index": len(_DOMAIN_INDEX),
        }


# -----------------------------
# Demo seed (optional)
# -----------------------------
if not _DEVICES:
    _DEVICES.extend([
        Device(
            id="dev_42",
            tenant_id="t_demo",
            name="Living Room Lamp",
            area="living_room",
            domain="light",
            type="light",
            aliases=["lamp"],
            capabilities=["brightness"],
            hash="h1",
        ),
        Device(
            id="dev_77",
            tenant_id="t_demo",
            name="Ceiling Light",
            area="living_room",
            domain="light",
            type="light",
            aliases=["ceiling"],
            capabilities=["on_off"],
            hash="h2",
        ),
    ])
    _ACTIONS.extend([
        Action(id="act_brightness", device_id="dev_42", name="set_brightness", schema_hash="sch_bright"),
        Action(id="act_on",         device_id="dev_77", name="turn_on",       schema_hash="sch_on"),
    ])
    _SCHEMAS["sch_bright"] = Schema(
        schema_hash="sch_bright",
        schema_json={"properties": {"value": {"range": "0-100"}}},
        hint_json={"value_range": "0-100"},
    )
    _SCHEMAS["sch_on"] = Schema(
        schema_hash="sch_on",
        schema_json={"properties": {}},
        hint_json={"toggle": True},
    )
