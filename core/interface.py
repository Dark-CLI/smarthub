# core/interface.py
from typing import Any, Dict, List
from core.history import compact_recent
from core.intent_extractor import extract_intents
from core.big_llm import run_big_llm
from data.search_interface import search_devices

class Interface:
    """
    Coordinator for one user turn:
    user_message -> small LLM (keywords) -> search_devices(text) -> big LLM
    """

    def __init__(self, top_k: int = 12, big_model: str = "llama3.1:latest"):
        self.top_k = top_k
        self.big_model = big_model

    async def handle_message(self, user_message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        # 1) static history/memory/executions
        recent_json = compact_recent()

        # 2) small LLM (raw line)
        keywords_text = await extract_intents(user_message, context)

        # 3) search interface (text → ready devices JSON)
        search_text = keywords_text or user_message
        devices = await search_devices(search_text, top_k=self.top_k)

        # 4) big LLM
        decision = await run_big_llm(
            user_message=user_message,
            context=context,
            recent_json=recent_json,
            keywords_text=keywords_text,
            devices=devices,
            model=self.big_model,
        )

        return {
            "message": user_message,
            "context": context,
            "keywords": keywords_text,
            "devices": devices,
            "decision": decision,
        }
