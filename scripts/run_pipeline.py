#!/usr/bin/env python3
import asyncio, time, json
import pandas as pd
from core.interface import Interface

CASES = [
    {"msg":"turn on the light","ctx":{"room":"kitchen"}},
    # {"msg":"it's too bright here","ctx":{"room":"office"}},
    # {"msg":"I'm cold","ctx":{"room":"bedroom"}},
    # {"msg":"dim the living room lights","ctx":{"room":"kitchen"}},
    # {"msg":"turn on the kitchen light","ctx":{"room":"bedroom"}},
    # {"msg":"close the curtains","ctx":{"room":"living_room"}},
    # {"msg":"start the vacuum","ctx":{"room":"hallway"}},
    # {"msg":"play music in the office","ctx":{"room":"bedroom"}},
    # {"msg":"open the bedroom window","ctx":{"room":"bedroom"}},
]

LTR = "\u200E"

async def main():
    iface = Interface(top_k=12)
    rows = []
    for c in CASES:
        t0 = time.perf_counter()
        res = await iface.handle_message(c["msg"], c["ctx"])
        dt = (time.perf_counter() - t0) * 1000
        rows.append({
            "message": LTR + res["message"],
            "context": LTR + json.dumps(res["context"], ensure_ascii=False),
            "keywords": LTR + (res["keywords"] or ""),
            "devices": len(res["devices"]),
            "decision": LTR + res["decision"],
            "time_ms": round(dt, 1),
        })
    df = pd.DataFrame(rows, columns=["message","context","keywords","devices","decision","time_ms"])
    with pd.option_context("display.max_colwidth", 140):
        print(df.to_string(index=False))

if __name__ == "__main__":
    asyncio.run(main())
