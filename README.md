# SmartHub (LLM → Home Assistant)

Pipeline (no arbiter):
1) HA → server: `{chat_id, user_last_message, context}`
2) Persist history + load summary
3) Small LLM → intent (JSON) OR plain reply
4) Resolver (code): map targets→device_ids, intent→action_id, normalize+validate args, build Candidate Bundle (+ examples)
5) Large LLM → **Decide & Reply**:
   - EXECUTE_AND_REPLY  (has device_id, action_id, args, reply_text)
   - FETCH_MORE (server fetch; then call Decide & Reply again)
   - REPLY (plain text)
6) Execute via HA if needed; return reply
7) Update summary/aliases

Run:
- `pip install -r requirements.txt`
- `uvicorn app.main:app --reload`
