import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_min_turn_reply_path():
    r = client.post("/chat/turn", json={
        "chat_id":"c1",
        "user_last_message":"thanks",
        "context":{"room":"living_room"}
    })
    assert r.status_code == 200

def test_min_turn_action_path():
    r = client.post("/chat/turn", json={
        "chat_id":"c2",
        "user_last_message":"dim the living room lamp to 30%",
        "context":{"room":"living_room"}
    })
    assert r.status_code == 200
