# core/interface.py
from typing import Any, Dict
from core.history import compact_recent
from core.intent_extractor import extract_intents
from core.big_llm import run_big_llm
from data.search_devices import search_devices
from data.search_actions import search_actions
from ha.client import HAClient


class Interface:
    """
    Orchestrates one user turn:
    user_message -> small keywords -> search devices/actions -> big LLM
    """
    def __init__(self, top_k: int = 6, big_model: str = "llama3.1:latest"):
        self.top_k = top_k
        self.big_model = big_model

    async def handle_message(self, user_message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        recent = compact_recent()
        keywords = await extract_intents(user_message, context)
        qtext = keywords or user_message

        devices = await search_devices(qtext, top_k=self.top_k)
        actions = await search_actions(qtext, top_k=self.top_k)
        ha = HAClient()
        domains = sorted({
            key.split(".", 1)[0] for key, state in devices if isinstance(key, str) and "." in key
        })
        for domain in domains:
            svcs = await ha.domain_services(domain)
            for svc, schema in (svcs or {}).items():
                actions.append({
                    "action": f"{domain}.{svc}",
                    "domain": domain,
                    "service": svc,
                    "fields": list((schema.get("fields") or {}).keys()),
                    "description": schema.get("description") or ""
                })

        decision = await run_big_llm(
            user_message=user_message,
            context=context,
            # recent_json=recent,
            # keywords_text=keywords,
            devices=devices,
            actions=actions,
            model=self.big_model,
        )
        return {
            "message": user_message,
            "context": context,
            "keywords": keywords,
            "devices": devices,
            "actions": actions,
            "decision": decision,
        }
