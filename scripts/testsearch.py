# smarthub/scripts/testsearch.py
import asyncio
import argparse
import time

from data.embedding import embed_texts
from data.vectors import query
from ha.client import HAClient

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("q", nargs="*", help="query text, e.g.: turn on living room lamp")
    ap.add_argument("--k", type=int, default=5, help="top-k")
    args = ap.parse_args()

    qtext = " ".join(args.q) or "turn on living room lamp"
    print(f"Query: {qtext}")

    # time embedding
    t0 = time.perf_counter()
    qvec = (await embed_texts([qtext], model="nomic-embed-text"))[0]
    t1 = time.perf_counter()

    # time vector search
    t2 = time.perf_counter()
    hits = query(qvec, top_k=args.k)
    t3 = time.perf_counter()

    # resolve friendly names using HA live state
    ha = HAClient()
    states = await ha.states()
    state_map = {s["entity_id"]: s for s in states}

    print("Top matches:")
    for key, score in hits:
        kind, ident = key.split(":", 1)
        if kind == "entity":
            st = state_map.get(ident)
            name = (st or {}).get("attributes", {}).get("friendly_name") or ident
            print(f"  {score:.3f}  {ident}  →  {name}")
        else:
            # domain-level rows would look like "domain:light"
            print(f"  {score:.3f}  {ident}  (domain)")

    print("\n--- Timings ---")
    print(f"Embed query:   {(t1 - t0)*1000:.1f} ms")
    print(f"Vector search: {(t3 - t2)*1000:.1f} ms")
    print(f"Total:         {(t3 - t0)*1000:.1f} ms")

if __name__ == "__main__":
    asyncio.run(main())
