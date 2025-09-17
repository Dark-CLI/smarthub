#!/usr/bin/env python3
import asyncio, time, argparse
from ha.syncer import sync_all
from data.vectors_devices import reset as reset_devices
from data.vectors_actions import reset as reset_actions

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true", help="drop both indices before syncing")
    ap.add_argument("--model", default="nomic-embed-text")
    args = ap.parse_args()

    if args.reset:
        reset_devices()
        reset_actions()

    print("Starting full sync (devices + actions)...")
    t0 = time.perf_counter()
    result = await sync_all(embed_model=args.model)
    dt = (time.perf_counter() - t0) * 1000
    print(f"Sync complete in {dt:.1f} ms")
    print(f"Devices indexed: {result['devices_indexed']}")
    print(f"Actions indexed: {result['actions_indexed']}")

if __name__ == "__main__":
    asyncio.run(main())
