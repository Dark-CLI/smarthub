# scripts/test_search.py
import asyncio
from data.embedding import embed_texts
from data.vectors import query
from data.repo import Repo

async def main():
    qvec = (await embed_texts(["turn on living room lamp"], model="nomic-embed-text"))[0]
    hits = query(qvec, top_k=5)
    print("HITS:", hits)
    # map keys back to entities/domains using Repo
    repo = Repo()
    for key, score in hits:
        kind, ident = key.split(":", 1)
        if kind == "entity":
            static = repo.get_entity_static(ident)
            print(f"{score:.3f}  {ident}  →  {static.get('friendly_name')}")
        else:
            print(f"{score:.3f}  {key}")

if __name__ == "__main__":
    asyncio.run(main())
