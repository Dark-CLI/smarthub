# smarthub/data/builders.py
from __future__ import annotations
from typing import Any, Dict, List, Tuple
import hashlib
import json

# ----------------- small utils -----------------

def _norm(s: Any) -> str:
    if s is None:
        return ""
    if isinstance(s, (int, float, bool)):
        return str(s)
    return str(s).strip()

def _nonempty(parts: List[str]) -> List[str]:
    return [p for p in parts if p and p.strip()]

def _hash_dict(d: Dict[str, Any]) -> str:
    s = json.dumps(d, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

# ----------------- supported_features decoding (compact, useful) -----------------

_FAN_BITS = {
    1:  "can_set_speed",
    2:  "can_oscillate",
    4:  "can_set_direction",
    8:  "has_preset_modes",
}

_MEDIA_BITS = {
    1 << 0:  "can_pause",
    1 << 1:  "can_seek",
    1 << 3:  "can_next_track",
    1 << 4:  "can_previous_track",
    1 << 5:  "can_turn_on",
    1 << 6:  "can_turn_off",
    1 << 7:  "can_volume_set",
    1 << 8:  "can_volume_mute",
    1 << 9:  "can_volume_step",
    1 << 10: "can_select_source",
    1 << 11: "can_select_sound_mode",
}

_LIGHT_FEATURE_BITS = {
    4:  "has_effects",
    8:  "can_flash",
    32: "has_transition",
}

def _decode_bits(mask: int, table: Dict[int, str]) -> List[str]:
    out = []
    for bit, name in table.items():
        if mask & bit:
            out.append(name)
    return out

def _decode_supported_features(domain: str, attrs: Dict[str, Any]) -> List[str]:
    feats: List[str] = []
    mask = attrs.get("supported_features") if isinstance(attrs.get("supported_features"), int) else None
    domain = (domain or "").lower()

    if domain == "fan":
        if mask is not None:
            feats.extend(_decode_bits(mask, _FAN_BITS))

    elif domain == "media_player":
        if mask is not None:
            feats.extend(_decode_bits(mask, _MEDIA_BITS))

    elif domain == "light":
        modes = attrs.get("supported_color_modes")
        if isinstance(modes, list):
            modes_lower = [str(m).lower() for m in modes]
            if "brightness" in modes_lower:
                feats.append("has_brightness")
            if "color_temp" in modes_lower:
                feats.append("has_color_temperature")
            if any(m in modes_lower for m in ("hs", "xy", "rgb", "rgbw", "rgbww")):
                feats.append("has_color")
            if "white" in modes_lower:
                feats.append("has_white")
        if mask is not None:
            feats.extend(_decode_bits(mask, _LIGHT_FEATURE_BITS))

    # dedupe
    seen, dedup = set(), []
    for f in feats:
        if f not in seen:
            dedup.append(f); seen.add(f)
    return dedup

# ----------------- services summarization -----------------

def _service_names_for_domain(domain_services_block: Dict[str, Any]) -> List[str]:
    if not isinstance(domain_services_block, dict):
        return []
    names = sorted(domain_services_block.keys())
    priority = [
        "turn_on", "turn_off", "toggle", "set_percentage", "oscillate",
        "set_temperature", "open", "close", "pause", "play",
        "volume_set", "volume_up", "volume_down",
    ]
    prio_set = set(priority)
    pref = [n for n in priority if n in names]
    rest = [n for n in names if n not in prio_set]
    # keep concise; retrieval works better with short descriptors
    return (pref + rest)[:12]

# ----------------- main builder -----------------

def build_static_descriptor(
    state: Dict[str, Any],
    domain_services_block: Dict[str, Any],
) -> Tuple[str, Dict[str, Any], str]:
    """
    Produce:
      - descriptor_text (for embedding): short, human-friendly, no empty fields
      - content (stable dict): used for hashing + metadata
      - content_hash (sha1 of content)
    """
    eid = _norm(state.get("entity_id"))
    attrs = state.get("attributes") or {}
    friendly = _norm(attrs.get("friendly_name")) or eid
    domain = eid.split(".", 1)[0] if "." in eid else ""
    area = _norm(attrs.get("area") or attrs.get("area_id") or attrs.get("room") or "")

    feats = _decode_supported_features(domain, attrs)
    svc_names = _service_names_for_domain(domain_services_block)

    content: Dict[str, Any] = {
        "entity_id": eid,
        "friendly_name": friendly,
        "domain": domain,
    }
    if area:
        content["area"] = area
    if feats:
        content["capabilities"] = feats
    if svc_names:
        content["services"] = svc_names

    content_hash = _hash_dict(content)

    # compact descriptor text
    parts: List[str] = []
    title_bits = [friendly]
    if domain:
        title_bits.append(f"({domain} domain)")
    if area:
        title_bits.append(f"in {area.replace('_', ' ')}")
    title = " ".join(title_bits).strip()
    parts.append(title + ".")
    if feats:
        parts.append("Capabilities: " + ", ".join(feats) + ".")
    if svc_names:
        parts.append("Services: " + ", ".join(svc_names) + ".")
    descriptor_text = " ".join(_nonempty(parts))

    return descriptor_text, content, content_hash
