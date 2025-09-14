# smarthub/scripts/testsync.py
import asyncio
import time
from ha.syncer import sync_embeddings

async def main():
    t0 = time.perf_counter()
    r1 = await sync_embeddings(embed_model="nomic-embed-text", batch=32)
    t1 = time.perf_counter()
    print(f"[first]  {r1}  |  {(t1 - t0)*1000:.1f} ms")

    # run immediately again; should embed 0 if hashes persisted
    t2 = time.perf_counter()
    r2 = await sync_embeddings(embed_model="nomic-embed-text", batch=32)
    t3 = time.perf_counter()
    print(f"[second] {r2}  |  {(t3 - t2)*1000:.1f} ms")

if __name__ == "__main__":
    asyncio.run(main())
