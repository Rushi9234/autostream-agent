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
    lead (e.g. just a name), the next turn stays on lead path for casual/short text.
    """
    from langchain_core.messages import HumanMessage

    monkeypatch.setattr(agent, "invoke_llm", lambda prompt: "high_intent")

    state = {
        "messages": [HumanMessage(content="Alex")],
        "intent": "",
        "lead_info": {"name": "Alex"},
        "lead_captured": False,
    }
    result = agent.route_turn(state)
    assert result["intent"] == "high_intent"
    assert result["lead_info"] == {"name": "Alex"}


def test_mixed_intent_preserves_slots_and_routes_to_inquiry(monkeypatch):
    from langchain_core.messages import HumanMessage

    def fake_llm(prompt):
        if "Extract lead information" in prompt:
            return '{"name": null, "email": "rushikesh@example.com", "platform": null}'
        return "inquiry"

    monkeypatch.setattr(agent, "invoke_llm", fake_llm)

    state = {
        "messages": [HumanMessage(content="My email is rushikesh@example.com, but what is your refund policy?")],
        "intent": "",
        "lead_info": {"name": "Rushikesh"},
        "lead_captured": False,
    }
    result = agent.route_turn(state)

    assert result["intent"] == "inquiry"
    assert result["lead_info"]["name"] == "Rushikesh"
    assert result["lead_info"]["email"] == "rushikesh@example.com"


def test_dynamic_continuation_in_rag_node(monkeypatch):
    from langchain_core.messages import HumanMessage

    monkeypatch.setattr(
        agent,
        "retrieve_with_sources",
        lambda q, k=3: ("AutoStream refund policy: 7 days refund.", [{"source": "autostream_kb.md", "section": "Company Policies → Refund Policy"}]),
    )
    monkeypatch.setattr(agent, "invoke_llm", lambda prompt: "AutoStream offers refunds within 7 days.")

    state = {
        "messages": [HumanMessage(content="What is your refund policy?")],
        "lead_info": {"name": "Rushikesh"},
        "lead_captured": False,
    }
    result = agent.rag_node(state)
    content = result["messages"][0].content

    assert "7 days" in content
    assert "what's the best email to reach you at" in content.lower()


def test_invalid_email_regex_validation():
    assert agent.is_valid_email("user@example.com") is True
    assert agent.is_valid_email("user.name+tag@domain.co.uk") is True
    assert agent.is_valid_email("not-an-email") is False
    assert agent.is_valid_email("user@domain") is False
    assert agent.is_valid_email("   ") is False


def test_off_topic_query_returns_decline_and_preserves_lead_info(monkeypatch):
    from langchain_core.messages import HumanMessage

    monkeypatch.setattr(agent, "retrieve_with_sources", lambda q, k=3: ("", []))

    state = {
        "messages": [HumanMessage(content="What is the weather in Tokyo?")],
        "lead_info": {"name": "Rushikesh"},
        "lead_captured": False,
    }
    result = agent.rag_node(state)

    assert result["messages"][0].content == agent.OFF_TOPIC_REPLY
    assert result["lead_info"] == {"name": "Rushikesh"}
    assert result["sources"] == []
    assert "(By the way" not in result["messages"][0].content



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
    assert result["intent"] == "casual"


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
    from langchain_core.messages import HumanMessage

    def boom(*args, **kwargs):
        raise RuntimeError("embeddings API unreachable")

    monkeypatch.setattr(agent, "retrieve", boom)
    monkeypatch.setattr(agent, "retrieve_with_sources", boom)

    state = {"messages": [HumanMessage(content="What's your refund policy?")]}
    result = agent.rag_node(state)

    assert result["messages"][0].content == agent.LLM_UNAVAILABLE_REPLY


def test_extract_slots_bare_name():
    lead_info, invalid_email = agent.extract_slots_from_message("kedar rushi", {}, lead_in_progress=True)
    assert lead_info.get("name") == "Kedar Rushi"
    assert not invalid_email

    lead_info, _ = agent.extract_slots_from_message("Alex", {}, lead_in_progress=True)
    assert lead_info.get("name") == "Alex"


def test_extract_slots_detects_malformed_email():
    lead_info, invalid_email = agent.extract_slots_from_message("rushikedar40gmail", {}, lead_in_progress=True)
    assert invalid_email is True
    assert lead_info.get("email") is None

    lead_info, invalid_email = agent.extract_slots_from_message("rushi@40", {}, lead_in_progress=True)
    assert invalid_email is True
    assert lead_info.get("email") is None


def test_check_personal_state_queries():
    state_info = {"name": "Kedar Rushi", "email": "rushikedar40@gmail.com", "platform": "YouTube"}
    reply_email = agent.check_personal_state_query("what is my mail", state_info, lead_captured=True)
    assert "rushikedar40@gmail.com" in reply_email

    reply_name = agent.check_personal_state_query("what is my name", state_info, lead_captured=True)
    assert "Kedar Rushi" in reply_name

    reply_details = agent.check_personal_state_query("what details do you have", state_info, lead_captured=True)
    assert "Kedar Rushi" in reply_details
    assert "rushikedar40@gmail.com" in reply_details
    assert "YouTube" in reply_details

