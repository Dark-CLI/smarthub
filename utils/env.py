import os
from dotenv import load_dotenv

load_dotenv(override=False)  # loads .env if present

def env(key: str, default: str | None = None) -> str | None:
    return os.getenv(key, default)

def build_ha_base_url() -> str:
    # Prefer HA_URL. Else compose from scheme/host/port.
    url = env("HA_URL")
    if url:
        return url.rstrip("/")
    scheme = env("HA_SCHEME", "http")
    host = env("HA_HOST", "localhost")
    port = env("HA_PORT", "8123")
    return f"{scheme}://{host}:{port}".rstrip("/")
