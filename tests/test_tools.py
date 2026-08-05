"""Tests for the SQLite-backed lead store in tools.py."""

import tools


def test_mock_lead_capture_persists_and_lists(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "DB_PATH", tmp_path / "leads.db")

    saved = tools.mock_lead_capture(name="Alex", email="alex@example.com", platform="YouTube")
    assert saved["name"] == "Alex"
    assert saved["id"] == 1

    leads = tools.list_leads()
    assert len(leads) == 1
    assert leads[0]["email"] == "alex@example.com"


def test_multiple_leads_are_ordered_most_recent_first(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "DB_PATH", tmp_path / "leads.db")

    tools.mock_lead_capture(name="Alex", email="alex@example.com", platform="YouTube")
    tools.mock_lead_capture(name="Sam", email="sam@example.com", platform="TikTok")

    leads = tools.list_leads()
    assert [lead["name"] for lead in leads] == ["Sam", "Alex"]
