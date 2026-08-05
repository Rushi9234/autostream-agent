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
