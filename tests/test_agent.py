"""
Unit tests for the parts of agent.py that don't need a live LLM call.

The routing guard and JSON extraction are the two pieces of logic most likely
to silently break, and neither needs an API key to test — they're pure
functions once you mock out invoke_llm.
"""

import agent


def test_parse_extraction_json_plain():
    raw = '{"name": "Alex", "email": null, "platform": null}'
    assert agent.parse_extraction_json(raw) == {
        "name": "Alex",
        "email": None,
        "platform": None,
    }


def test_parse_extraction_json_strips_markdown_fence():
    raw = '```json\n{"name": "Alex", "email": null, "platform": null}\n```'
    assert agent.parse_extraction_json(raw) == {
        "name": "Alex",
        "email": None,
        "platform": None,
    }


def test_parse_extraction_json_invalid_returns_empty_dict():
    assert agent.parse_extraction_json("not json at all") == {}


def test_invoke_llm_retries_then_succeeds(monkeypatch):
    calls = {"count": 0}

    class FakeLLM:
        def invoke(self, prompt):
            calls["count"] += 1
            if calls["count"] < 2:
                raise RuntimeError("transient failure")
            return type("Resp", (), {"content": "ok"})()

    monkeypatch.setattr(agent, "get_llm", lambda: FakeLLM())
    result = agent.invoke_llm("some prompt", retries=2, backoff_seconds=0)

    assert result == "ok"
    assert calls["count"] == 2


def test_invoke_llm_returns_none_after_exhausting_retries(monkeypatch):
    class FakeLLM:
        def invoke(self, prompt):
            raise RuntimeError("always fails")

    monkeypatch.setattr(agent, "get_llm", lambda: FakeLLM())
    result = agent.invoke_llm("some prompt", retries=1, backoff_seconds=0)

    assert result is None


def test_route_turn_stays_in_lead_flow_once_started(monkeypatch):
    """
    Regression test for the routing guard: once we've captured a partial
    lead (e.g. just a name), the next turn must stay on the lead path even
    though a bare reply like "Alex" would otherwise look like casual chat.
    """

    def fail_if_called(*args, **kwargs):
        raise AssertionError("classifier should not be called mid lead-flow")

    monkeypatch.setattr(agent, "invoke_llm", fail_if_called)

    state = {
        "messages": [],
        "intent": "",
        "lead_info": {"name": "Alex"},
        "lead_captured": False,
    }
    result = agent.route_turn(state)
    assert result == {"intent": "high_intent"}


def test_route_turn_falls_back_to_casual_when_llm_unavailable(monkeypatch):
    from langchain_core.messages import HumanMessage

    monkeypatch.setattr(agent, "invoke_llm", lambda prompt: None)

    state = {
        "messages": [HumanMessage(content="hello")],
        "intent": "",
        "lead_info": {},
        "lead_captured": False,
    }
    result = agent.route_turn(state)
    assert result == {"intent": "casual"}


def test_pick_after_lead_routes_to_escalation_once_complete():
    complete_state = {
        "lead_info": {"name": "Alex", "email": "a@x.com", "platform": "YouTube"},
        "lead_captured": False,
    }
    assert agent.pick_after_lead(complete_state) == "escalate_node"


def test_pick_after_lead_ends_turn_when_fields_still_missing():
    partial_state = {"lead_info": {"name": "Alex"}, "lead_captured": False}
    assert agent.pick_after_lead(partial_state) == agent.END


def test_pick_after_lead_ends_turn_once_already_captured():
    # Guards against re-escalating a lead that already went through review.
    captured_state = {
        "lead_info": {"name": "Alex", "email": "a@x.com", "platform": "YouTube"},
        "lead_captured": True,
    }
    assert agent.pick_after_lead(captured_state) == agent.END


def test_escalate_node_approved_captures_lead_and_confirms(monkeypatch, tmp_path):
    import tools

    monkeypatch.setattr(tools, "DB_PATH", tmp_path / "leads.db")
    monkeypatch.setattr(agent, "interrupt", lambda payload: {"approved": True})

    state = {"lead_info": {"name": "Alex", "email": "alex@example.com", "platform": "YouTube"}}
    result = agent.escalate_node(state)

    assert result["lead_captured"] is True
    assert "Alex" in result["messages"][0].content
    assert tools.list_leads()[0]["email"] == "alex@example.com"


def test_escalate_node_rejected_clears_lead_info(monkeypatch, tmp_path):
    import tools

    monkeypatch.setattr(tools, "DB_PATH", tmp_path / "leads.db")
    monkeypatch.setattr(agent, "interrupt", lambda payload: {"approved": False})

    state = {"lead_info": {"name": "Alex", "email": "alex@example.com", "platform": "YouTube"}}
    result = agent.escalate_node(state)

    assert result["lead_captured"] is False
    assert result["lead_info"] == {}
    assert tools.list_leads() == []


def test_rag_node_falls_back_gracefully_when_retrieve_fails(monkeypatch):
    """
    Regression test: retrieve() calls Gemini's embeddings API directly to
    build/query the FAISS index, and unlike invoke_llm() it originally had
    no error handling — a transient embeddings-API failure crashed the
    whole turn instead of degrading like every other node does.
    """
    from langchain_core.messages import HumanMessage

    def boom(*args, **kwargs):
        raise RuntimeError("embeddings API unreachable")

    monkeypatch.setattr(agent, "retrieve", boom)

    state = {"messages": [HumanMessage(content="What's your refund policy?")]}
    result = agent.rag_node(state)

    assert result["messages"][0].content == agent.LLM_UNAVAILABLE_REPLY
