# scripts/test_search.py
# scripts/test_search.py
import asyncio, sys, argparse, time
from data.embedding import embed_texts
from data.vectors import query
from data.repo import Repo
from ha.syncer import sync_embeddings  # keep available

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("q", nargs="*", help="query text")
    ap.add_argument("--sync", action="store_true", help="(one-time) rebuild embeddings")
    args = ap.parse_args()

    qtext = " ".join(args.q) or "I am feeling cold"

    t0 = time.perf_counter()

    repo = Repo()
    await repo.ensure_hydrated()
    t1 = time.perf_counter()

    if args.sync:
        print("Syncing embeddings once...")
        await sync_embeddings(embed_model="nomic-embed-text", embed_version="v1")
    t2 = time.perf_counter()

    qvec = (await embed_texts([qtext], model="nomic-embed-text"))[0]
    t3 = time.perf_counter()

    hits = query(qvec, top_k=5)
    t4 = time.perf_counter()

    print(f"\nQuery: {qtext}")
    print("Top matches:")
    for key, score in hits:
        kind, ident = key.split(":", 1)
        if kind == "entity":
            static = repo.try_get_entity_static(ident) or (await (repo.ensure_hydrated()) or repo.get_entity_static(ident))
            name = static.get("friendly_name") or ident
            print(f"  {score:.3f}  {ident}  →  {name}")
        else:
            print(f"  {score:.3f}  {ident}  (domain)")

    print("\n--- Timings ---")
    print(f"Hydrate repo:     {(t1 - t0)*1000:.1f} ms")
    print(f"Sync embeddings:  {(t2 - t1)*1000:.1f} ms")
    print(f"Embed query:      {(t3 - t2)*1000:.1f} ms")
    print(f"Vector search:    {(t4 - t3)*1000:.1f} ms")
    print(f"Total:            {(t4 - t0)*1000:.1f} ms")

if __name__ == "__main__":
    asyncio.run(main())

