from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import Any, Dict
from data.repo import Repo
from core.intent_extractor import extract_intent_or_reply
from core.resolver import build_bundle_from_intent
from core.decide_and_reply import decide_and_reply
from ha.client import HAClient

router = APIRouter()

class TurnIn(BaseModel):
    chat_id: str
    user_last_message: str
    context: Dict[str, Any] = Field(default_factory=dict)
    tenant_id: str | None = None

@router.post("/turn")
async def chat_turn(body: TurnIn):
    repo = Repo()
    ha = HAClient()

    # 1) persist user msg
    repo.add_message(body.chat_id, "user", body.user_last_message)

    # 2) small LLM → intent or plain reply
    intent_or_reply = await extract_intent_or_reply(
        message=body.user_last_message,
        context=body.context,
        summary=repo.load_summary(body.chat_id)
    )
    if intent_or_reply["mode"] == "reply":
        reply = intent_or_reply["text"]
        repo.add_message(body.chat_id, "assistant", reply)
        repo.update_summary(body.chat_id, body.user_last_message, reply)
        return {"reply": reply}

    # 3) resolver → bundle
    bundle = await build_bundle_from_intent(repo, intent_or_reply["intent"], body.context)

    # 4) large LLM: Decide & Reply (bounded loop if it asks to fetch more)
    for _ in range(2):
        decision = await decide_and_reply(bundle=bundle, summary=repo.load_summary(body.chat_id))
        mode = decision.get("mode")
        if mode == "REPLY":
            reply = decision["text"]
            repo.add_message(body.chat_id, "assistant", reply)
            repo.update_summary(body.chat_id, body.user_last_message, reply)
            return {"reply": reply}
        if mode == "FETCH_MORE":
            # Minimal demo: we only handle a sample fetch kind
            if decision.get("fetch") == "devices_for_area":
                area = decision.get("params", {}).get("area")
                extra = repo.devices_for_area(area)
                bundle["candidates"].extend(extra)
                continue
            return {"reply": "Need more info to proceed."}
        if mode == "EXECUTE_AND_REPLY":
            # Validate again (defense in depth)
            # In demo we trust args; production: repo.full_validate(...)
            _ = decision["args"]
            tool_result = await ha.execute(decision["device_id"], decision["action_id"], decision["args"])
            reply = decision["reply_text"]
            repo.add_message(body.chat_id, "assistant", reply)
            repo.update_summary(body.chat_id, body.user_last_message, reply)
            return {"reply": reply}

    # fallback
    reply = "Sorry, I need more details."
    repo.add_message(body.chat_id, "assistant", reply)
    return {"reply": reply}
