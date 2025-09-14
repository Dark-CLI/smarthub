import httpx
from typing import Any, Dict, List, Optional
from utils.env import build_ha_base_url, env

class HAClient:
    """
    Minimal Home Assistant client using the REST API.
    Auth: Long-lived access token (HA_TOKEN).
    Base URL sourced from HA_URL or HA_SCHEME/HA_HOST/HA_PORT.
    """
    def __init__(self, timeout: float = 15.0):
        self.base = build_ha_base_url()
        self.token = env("HA_TOKEN")
        if not self.token:
            raise RuntimeError("HA_TOKEN is not set (put it in .env or your environment).")
        self._timeout = timeout
        self._headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

    # ---------- Core REST helpers ----------

    async def _get(self, path: str) -> Any:
        async with httpx.AsyncClient(timeout=self._timeout) as cli:
            r = await cli.get(f"{self.base}{path}", headers=self._headers)
            r.raise_for_status()
            return r.json()

    async def _post(self, path: str, json: Dict[str, Any]) -> Any:
        async with httpx.AsyncClient(timeout=self._timeout) as cli:
            r = await cli.post(f"{self.base}{path}", headers=self._headers, json=json)
            r.raise_for_status()
            # HA returns 200 with JSON for services
            return r.json() if r.headers.get("content-type","").startswith("application/json") else None

    # ---------- Common endpoints ----------

    async def states(self) -> List[Dict[str, Any]]:
        """GET /api/states – All entities with current state/attributes."""
        return await self._get("/api/states")

    async def state(self, entity_id: str) -> Dict[str, Any]:
        return await self._get(f"/api/states/{entity_id}")

    async def services(self) -> Dict[str, Any]:
        """GET /api/services – Available domains/services (and fields)."""
        return await self._get("/api/services")

    async def call_service(
        self,
        domain: str,
        service: str,
        service_data: Optional[Dict[str, Any]] = None,
        entity_id: Optional[str] = None,
        area_id: Optional[str] = None,
        device_id: Optional[str] = None,
    ) -> Any:
        """
        POST /api/services/{domain}/{service}
        Targeting via entity_id/area_id/device_id as per HA.
        """
        payload: Dict[str, Any] = service_data.copy() if service_data else {}
        # Targets
        if entity_id:
            payload["entity_id"] = entity_id
        if area_id:
            payload["area_id"] = area_id
        if device_id:
            payload["device_id"] = device_id
        return await self._post(f"/api/services/{domain}/{service}", payload)

    # ---------- Convenience wrappers ----------

    async def execute(self, device_id: str, action_id: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Your project’s execute adapter:
        - Map action_id → (domain, service)
        - Prefer entity_id targeting; device_id/area_id as fallback
        """
        # Example mapping—replace with your real repo mapping:
        # action_id like "light.turn_on" or "act_brightness"
        if "." in action_id:
            domain, service = action_id.split(".", 1)
        else:
            # naive fallback: infer from args or your DB
            domain, service = "light", "turn_on"

        entity_id = args.pop("entity_id", None)
        area_id   = args.pop("area_id", None)
        result = await self.call_service(domain, service, args, entity_id=entity_id, device_id=device_id, area_id=area_id)
        return {"status": "ok", "service": f"{domain}.{service}", "result": result}

    async def find_entities(self, name_substring: str) -> List[Dict[str, Any]]:
        """Naive search overstates by friendly name/entity_id."""
        name_substring = name_substring.lower()
        out = []
        for s in await self.states():
            name = (s.get("attributes", {}).get("friendly_name") or "").lower()
            eid  = s.get("entity_id","").lower()
            if name_substring in name or name_substring in eid:
                out.append(s)
        return out

    async def available_actions(self, entity_id: str) -> List[Dict[str, Any]]:
        """
        Given an entity_id, return the list of applicable actions/services
        with their allowed fields, filtered by supported_features/attributes.
        """
        entity = await self.state(entity_id)
        domain = entity_id.split(".")[0]
        features = entity.get("attributes", {}).get("supported_features", 0)
        attrs = entity.get("attributes", {})

        services = await self.services()
        domain_block = next((b for b in services if b.get("domain") == domain), None)
        if not domain_block:
            return []

        available = []
        for svc_name, svc_def in domain_block.get("services", {}).items():
            # Check target.entity.supported_features
            targets = svc_def.get("target", {}).get("entity", [])
            skip = False
            for t in targets:
                required = t.get("supported_features")
                if required:
                    # if list, all must match
                    if isinstance(required, list):
                        if not all((features & bit) == bit for bit in required):
                            skip = True
                            break
                    else:
                        if (features & required) != required:
                            skip = True
                            break
            if skip:
                continue

            # Filter fields
            fields_out = {}
            for fname, fdef in svc_def.get("fields", {}).items():
                f_ok = True
                flt = fdef.get("filter")
                if flt:
                    if "supported_features" in flt:
                        req = flt["supported_features"]
                        if isinstance(req, list):
                            if not all((features & bit) == bit for bit in req):
                                f_ok = False
                        else:
                            if (features & req) != req:
                                f_ok = False
                    if "attribute" in flt:
                        for key, val in flt["attribute"].items():
                            if attrs.get(key) != val:
                                f_ok = False
                if f_ok:
                    fields_out[fname] = fdef

            available.append({
                "service": f"{domain}.{svc_name}",
                "description": svc_def.get("description"),
                "fields": fields_out
            })
        return available