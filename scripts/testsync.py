import asyncio
from ha.syncer import sync_once

async def main():
    res = await sync_once(embed_model="nomic-embed-text", embed_version="v1")
    print(res)

if __name__ == "__main__":
    asyncio.run(main())
