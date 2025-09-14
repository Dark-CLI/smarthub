#!/usr/bin/env python3
import asyncio, time, json
import pandas as pd
from core.intent_extractor import extract_intents

MESSAGES = [
    # in-room (infer from context)
    "turn on the light",
    "make it warmer",
    "close the curtains",
    "it's too bright here",
    "turn off the fan",
    # cross-room (explicit other room)
    "dim the living room lights",
    "turn on the kitchen light",
    "set bedroom to 21C",
    "open the curtains in the study",
    "turn off the bathroom fan",
    # global / well-known (no room needed)
    "start the vacuum",
    "turn on the water heater",
    "run the water pump for 5 minutes",
    # combo / out-of-domain probes
    "vacuum the living room",
    "play music in the office",
]

CONTEXTS = [
    {"room":"kitchen"},
    {"room":"bedroom"},
    {"room":"living_room"},
    {"room":"office"},
    {"room":"bathroom"},
    {"room":"kitchen"},
    {"room":"bedroom"},
    {"room":"office"},
    {"room":"hallway"},
    {"room":"living_room"},
    {"room":"kitchen"},
    {"room":"bedroom"},
    {"room":"office"},
    {"room":"kitchen"},
    {"room":"bedroom"},
]

LTR = "\u200E"  # force left-to-right display for mixed scripts

async def main():
    rows = []
    for msg, ctx in zip(MESSAGES, CONTEXTS):
        t0 = time.perf_counter()
        try:
            raw = await extract_intents(msg, ctx)
            elapsed = (time.perf_counter() - t0) * 1000
            rows.append({
                "message": LTR + msg,
                "context": LTR + json.dumps(ctx, separators=(",",":"), ensure_ascii=False),
                "time_ms": round(elapsed, 1),
                "model_text": LTR + raw,  # whatever the model returned (JSON-ish or not)
            })
        except Exception as e:
            elapsed = (time.perf_counter() - t0) * 1000
            rows.append({
                "message": LTR + msg,
                "context": LTR + json.dumps(ctx, separators=(",",":"), ensure_ascii=False),
                "time_ms": round(elapsed, 1),
                "model_text": f"error:{type(e).__name__}",
            })

    df = pd.DataFrame(rows, columns=["message","context","time_ms","model_text"])
    with pd.option_context("display.max_colwidth", 120):
        print(df.to_string(index=False))

if __name__ == "__main__":
    asyncio.run(main())
