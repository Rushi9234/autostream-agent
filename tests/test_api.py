"""
Tests for api.py's access control.

Regression test for a real gap a security pass found: /approve had no
authentication at all -- anyone who knew or guessed a thread_id could
approve/reject someone else's lead. Doesn't need a live LLM call since these
requests never reach a pending-approval state in the first place (they're
rejected by the auth check before that).
"""

import os

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("ADMIN_TOKEN", "test-token-123")

    import agent
    # Keep the test checkpoint DB out of the repo dir -- not chdir'ing,
    # since api.py's StaticFiles("static") is resolved relative to cwd.
    agent.CHECKPOINT_DB_PATH = str(tmp_path / "checkpoints.sqlite")

    import api
    return TestClient(api.app)


def test_approve_rejects_missing_token(client):
    response = client.post("/approve", json={"thread_id": "any-thread", "approved": True})
    assert response.status_code == 401


def test_approve_rejects_wrong_token(client):
    response = client.post(
        "/approve",
        json={"thread_id": "any-thread", "approved": True},
        headers={"X-Admin-Token": "wrong-token"},
    )
    assert response.status_code == 401


def test_approve_with_correct_token_gets_past_auth(client):
    """
    Correct token should clear the auth check -- the 409 here confirms we got
    *past* auth and hit the actual "not currently pending" business logic
    instead (this thread never started a lead flow, so nothing's pending).
    """
    response = client.post(
        "/approve",
        json={"thread_id": "any-thread", "approved": True},
        headers={"X-Admin-Token": "test-token-123"},
    )
    assert response.status_code == 409


def test_health_endpoint_needs_no_auth(client):
    response = client.get("/health")
    assert response.status_code == 200


def test_chat_rejects_whitespace_only_message(client):
    response = client.post("/chat", json={"thread_id": "test-1", "message": "   "})
    assert response.status_code == 400
    assert "empty or whitespace-only" in response.json()["detail"]


def test_chat_history_endpoint(client, monkeypatch):
    import agent
    monkeypatch.setattr(agent, "invoke_llm", lambda prompt: "Hello!")

    client.post("/chat", json={"thread_id": "hist-thread-1", "message": "Hi"})

    res = client.get("/chat/history?thread_id=hist-thread-1")
    assert res.status_code == 200
    data = res.json()
    assert data["thread_id"] == "hist-thread-1"
    assert len(data["messages"]) >= 2
    assert data["messages"][0]["role"] == "user"
    assert data["messages"][0]["content"] == "Hi"


def test_repeated_approval_attempt_returns_409(client):
    res1 = client.post(
        "/approve",
        json={"thread_id": "thread-not-pending", "approved": True},
        headers={"X-Admin-Token": "test-token-123"},
    )
    assert res1.status_code == 409

    res2 = client.post(
        "/approve",
        json={"thread_id": "thread-not-pending", "approved": True},
        headers={"X-Admin-Token": "test-token-123"},
    )
    assert res2.status_code == 409


def test_chat_returns_sources_metadata(client, monkeypatch):
    import agent

    def fake_llm(prompt):
        if "Classify" in prompt:
            return "inquiry"
        return "The Pro plan is $79/mo."

    monkeypatch.setattr(agent, "invoke_llm", fake_llm)
    monkeypatch.setattr(
        agent,
        "retrieve_with_sources",
        lambda q, k=3: ("Context text", [{"source": "autostream_kb.md", "section": "Pricing Plans → Pro Plan"}]),
    )

    res = client.post("/chat", json={"thread_id": "src-thread", "message": "How much is Pro?"})
    assert res.status_code == 200
    data = res.json()
    assert "sources" in data
    assert len(data["sources"]) == 1
    assert data["sources"][0]["section"] == "Pricing Plans → Pro Plan"


def test_api_leads_endpoint(client):
    res = client.get("/api/leads")
    assert res.status_code == 200
    data = res.json()
    assert "leads" in data
    assert "count" in data
    assert isinstance(data["leads"], list)


def test_api_threads_endpoint(client):
    res = client.get("/api/threads")
    assert res.status_code == 200
    data = res.json()
    assert "threads" in data
    assert isinstance(data["threads"], list)



