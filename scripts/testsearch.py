# smarthub/scripts/testsearch.py
import asyncio
import json
import time

from core.intent_extractor import extract_intents
from data.search_devices import search_devices
from ha.client import HAClient

def _pp(obj):
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))

async def main():
    # Hardcoded example input (change as needed)
    message = "it's getting hot in her"
    context = {"room": "kitchen"}
    top_k = 4

    print(f"Message: {message}")
    print(f"Context: {_pp(context)}")

    # 1) small LLM → intent keywords
    t0 = time.perf_counter()
    intent = await extract_intents(message, context)
    t1 = time.perf_counter()
    print(f"Small LLM text: {intent}  ({(t1 - t0)*1000:.1f} ms)")

    search_text = intent or message

    # 2) search devices
    t2 = time.perf_counter()
    devices = await search_devices(search_text, top_k=top_k)
    t3 = time.perf_counter()

    # 3) for each unique domain in device results, get services using our HA API
    ha = HAClient()

    domains = sorted({
        key.split(".", 1)[0] for key, state in devices if isinstance(key, str) and "." in key
    })

    actions = []
    for domain in domains:
        svcs = await ha.domain_services(domain)
        for svc, schema in (svcs or {}).items():
            actions.append({
                "action": f"{domain}.{svc}",
                "domain": domain,
                "service": svc,
                "fields": list((schema.get("fields") or {}).keys()),
                "description": schema.get("description") or ""
            })

    t4 = time.perf_counter()
    print(json.dumps(devices, indent=2, ensure_ascii=False))
    print(json.dumps(actions, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())