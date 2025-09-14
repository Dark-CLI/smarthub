# smarthub/data/repo.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Iterable, Tuple
import time

# Optional: your chat/session/demo models (keep if you use them elsewhere)
from data.models import Session, Message, Device, Action, Schema  # noqa: F401
from core.summarizer import update_summary as _update_summary_fn  # noqa: F401

# -----------------------------
# In-memory store (dev/demo)
# -----------------------------
_SESS: Dict[str, Session] = {}
_MSGS: Dict[str, List[Message]] = {}

# Static snapshots (filled from HA)
_DEVICE_INDEX: Dict[str, Dict[str, Any]] = {}   # entity_id -> static_json + hashes + embed meta
_DOMAIN_INDEX: Dict[str, Dict[str, Any]] = {}   # domain    -> services block JSON + hashes + embed meta


def _now() -> int:
    return int(time.time())


class Repo:
    """
    In-memory repo that can HYDRATE ITSELF from Home Assistant on first use.
    Persists nothing; meant for fast runtime lookups. Embeddings live in LanceDB.
    """
    def __init__(self) -> None:
        self._hydrated = False

    # ---------- lazy hydration ----------
    async def ensure_hydrated(self) -> None:
        if self._hydrated:
            return
        # Local imports to avoid import cycles at module import time
        from ha.client import HAClient
        from data.builders import static_snapshot_from_state, dict_hash

        ha = HAClient()
        states = await ha.states()
        services_list = await ha.services()

        svc_by_domain: Dict[str, Dict[str, Any]] = {}
        for blk in services_list:
            if not isinstance(blk, dict):
                continue
            dom = blk.get("domain")
            if not dom:
                continue
            svc_by_domain[dom] = blk.get("services", {}) or {}

        # Entities → static snapshot (no live values)
        for st in states:
            eid = st.get("entity_id", "")
            if not eid:
                continue
            dom = eid.split(".", 1)[0]
            static_json = static_snapshot_from_state(st, {dom: svc_by_domain.get(dom, {})})
            self._upsert_entity_static(static_json, dict_hash(static_json))

        # Domains → full services block
        for dom, svcs in svc_by_domain.items():
            payload = {"domain": dom, "services": svcs}
            self._upsert_domain_block(dom, payload, dict_hash(payload))

        self._hydrated = True

    # ---------- internal upserts (no embedding here) ----------
    def _upsert_entity_static(self, static_json: Dict[str, Any], content_hash: str) -> None:
        eid = static_json["entity_id"]
        row = _DEVICE_INDEX.get(eid) or {}
        row.update({
            "static_json": static_json,
            "content_hash": content_hash,
            "updated_at": _now(),
        })
        _DEVICE_INDEX[eid] = row

    def _upsert_domain_block(self, domain: str, content_json: Dict[str, Any], content_hash: str) -> None:
        row = _DOMAIN_INDEX.get(domain) or {}
        row.update({
            "content_json": content_json,
            "content_hash": content_hash,
            "updated_at": _now(),
        })
        _DOMAIN_INDEX[domain] = row

    # ---------- public helpers used by syncer / runtime ----------
    def get_entity_static(self, entity_id: str) -> Dict[str, Any]:
        row = _DEVICE_INDEX.get(entity_id)
        if not row:
            raise KeyError(entity_id)
        return row["static_json"]

    def try_get_entity_static(self, entity_id: str) -> Optional[Dict[str, Any]]:
        row = _DEVICE_INDEX.get(entity_id)
        return None if not row else row["static_json"]

    def domain_embed_text(self, domain: str) -> str:
        """Render a compact text of a domain's services/fields for embedding."""
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

    def iter_entities(self) -> Iterable[Tuple[str, Dict[str, Any]]]:
        for eid, row in _DEVICE_INDEX.items():
            yield eid, row["static_json"]

    def iter_domains(self) -> Iterable[str]:
        return iter(_DOMAIN_INDEX.keys())

    # ---------- optional: store embed meta in-memory (not required) ----------
    def mark_entity_embedded(self, entity_id: str, model: str, version: str) -> None:
        row = _DEVICE_INDEX.get(entity_id) or {}
        row.update({
            "embed_model": model,
            "embed_version": version,
            "embedded_at": _now(),
        })
        _DEVICE_INDEX[entity_id] = row

    def mark_domain_embedded(self, domain: str, model: str, version: str) -> None:
        row = _DOMAIN_INDEX.get(domain) or {}
        row.update({
            "embed_model": model,
            "embed_version": version,
            "embedded_at": _now(),
        })
        _DOMAIN_INDEX[domain] = row

    # ---------- (optional) chat/session helpers you already had ----------
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

    # ---------- debug ----------
    def debug_counts(self) -> Dict[str, int]:
        return {
            "device_index": len(_DEVICE_INDEX),
            "domain_index": len(_DOMAIN_INDEX),
            "sessions": len(_SESS),
            "messages": sum(len(v) for v in _MSGS.values()),
        }
