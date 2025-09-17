# smarthub/ha/client.py
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional
import httpx

# --- NEW: load .env early ---
try:
    from dotenv import load_dotenv, find_dotenv  # pip install python-dotenv
    _DOTENV_PATH = find_dotenv(usecwd=True) or ".env"
    if _DOTENV_PATH and os.path.exists(_DOTENV_PATH):
        load_dotenv(_DOTENV_PATH)
except Exception:
    # If python-dotenv isn't installed, we just skip; rely on real env
    pass

def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    return os.environ.get(name, default)

def _build_ha_base_url() -> str:
    url = _env("HA_URL")
    if url:
        return url.rstrip("/")
    scheme = _env("HA_SCHEME", "http")
    host = _env("HA_HOST", "localhost")
    port = _env("HA_PORT", "8123")
    return f"{scheme}://{host}:{port}"

class HAClient:
    """
    Minimal Home Assistant client using REST API.
    Auth: Long-lived token in HA_TOKEN env (loaded from .env if present).
    """
    def __init__(self, timeout: float = 15.0):
        self.base = _build_ha_base_url()
        self.token = _env("HA_TOKEN")
        if not self.token:
            raise RuntimeError("HA_TOKEN is not set.")
        self._timeout = timeout
        self._headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }
        self._services_cache: Optional[List[Dict[str, Any]]] = None

    async def _get(self, path: str) -> Any:
        async with httpx.AsyncClient(timeout=self._timeout) as cli:
            r = await cli.get(f"{self.base}{path}", headers=self._headers)
            r.raise_for_status()
            return r.json()

    async def _post(self, path: str, json: Dict[str, Any]) -> Any:
        async with httpx.AsyncClient(timeout=self._timeout) as cli:
            r = await cli.post(f"{self.base}{path}", headers=self._headers, json=json)
            r.raise_for_status()
            ctype = r.headers.get("content-type", "")
            return r.json() if ctype.startswith("application/json") else None

    async def states(self) -> List[Dict[str, Any]]:
        return await self._get("/api/states")

    async def state(self, entity_id: str) -> Dict[str, Any]:
        return await self._get(f"/api/states/{entity_id}")

    async def states_batch(self, entity_ids: list[str]) -> list[dict]:
        # Current implementation: dumb loop; optimized later
        out = []
        for eid in entity_ids:
            try:
                state = await self.state(eid)
                if state: out.append(state)
            except Exception:
                pass
        return out

    async def services(self) -> List[Dict[str, Any]]:
        if self._services_cache is None:
            self._services_cache = await self._get("/api/services")
        return self._services_cache

    async def services_map(self) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for item in await self.services():
            dom = item.get("domain")
            svcs = item.get("services") or {}
            if isinstance(dom, str) and isinstance(svcs, dict):
                out[dom] = svcs
        return out

    async def domain_services(self, domain: str) -> Dict[str, Any]:
        return (await self.services_map()).get(domain, {})

    async def call_service(
        self,
        domain: str,
        service: str,
        service_data: Optional[Dict[str, Any]] = None,
        entity_id: Optional[str] = None,
        area_id: Optional[str] = None,
        device_id: Optional[str] = None,
    ) -> Any:
        payload: Dict[str, Any] = dict(service_data or {})
        if entity_id:
            payload["entity_id"] = entity_id
        if area_id:
            payload["area_id"] = area_id
        if device_id:
            payload["device_id"] = device_id
        return await self._post(f"/api/services/{domain}/{service}", payload)

    async def execute(self, device_id: str, action_id: str, args: Dict[str, Any]) -> Dict[str, Any]:
        if "." in action_id:
            domain, service = action_id.split(".", 1)
        else:
            domain, service = "light", "turn_on"
        entity_id = args.pop("entity_id", None)
        area_id = args.pop("area_id", None)
        result = await self.call_service(domain, service, args, entity_id=entity_id, device_id=device_id, area_id=area_id)
        return {"status": "ok", "service": f"{domain}.{service}", "result": result}
