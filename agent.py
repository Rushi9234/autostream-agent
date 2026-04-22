"""
LangGraph agent for AutoStream.

Graph flow:
    user message -> route_turn -> (casual_node OR rag_node OR lead_node) -> END

The router decides which node handles the turn based on classified intent.
State is persisted across turns using MemorySaver (keyed by thread_id in main.py).
"""

import json
import os
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from prompts import (
    CASUAL_REPLY_PROMPT,
    INTENT_CLASSIFIER_PROMPT,
    LEAD_EXTRACTION_PROMPT,
    RAG_ANSWER_PROMPT,
)
from rag import retrieve
from tools import mock_lead_capture


# Single LLM instance shared by all nodes. Lazily initialized so we don't hit
# the API on import.
_llm = None


def get_llm():
    global _llm
    if _llm is None:
        _llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.2)
    return _llm


# State schema for the graph. `messages` uses add_messages so LangGraph
# appends to the list instead of replacing it on each turn.
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    intent: str
    lead_info: dict
    lead_captured: bool


def format_history(messages, limit=6):
    """Format last N messages as a plain transcript for the prompt."""
    recent = messages[-limit:]
    lines = []
    for m in recent:
        role = "User" if isinstance(m, HumanMessage) else "Assistant"
        lines.append(f"{role}: {m.content}")
    return "\n".join(lines) if lines else "(no prior messages)"


def latest_user_message(messages):
    """Return the text of the most recent user message."""
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            return m.content
    return ""


# -----------------------------
# Nodes
# -----------------------------

def route_turn(state: AgentState):
    """
    Decide which handler should run this turn.

    Normally we classify the user message into casual / inquiry / high_intent.
    But if we've already started collecting lead info (name/email/platform)
    and haven't finished, we stay in the lead flow. Otherwise a short reply
    like "Alex" gets classified as casual and the signup breaks.
    """
    user_message = latest_user_message(state["messages"])
    history = format_history(state["messages"][:-1])

    lead_info = state.get("lead_info", {}) or {}
    already_captured = state.get("lead_captured", False)

    # Check if we're in the middle of collecting lead details
    in_lead_flow = any(lead_info.get(f) for f in ("name", "email", "platform"))
    if in_lead_flow and not already_captured:
        return {"intent": "high_intent"}

    prompt = INTENT_CLASSIFIER_PROMPT.format(
        history=history,
        user_message=user_message,
    )
    raw = get_llm().invoke(prompt).content.strip().lower()

    if "high_intent" in raw:
        intent = "high_intent"
    elif "inquiry" in raw:
        intent = "inquiry"
    else:
        intent = "casual"

    return {"intent": intent}


def casual_node(state: AgentState):
    """Handle greetings and small talk."""
    user_message = latest_user_message(state["messages"])
    history = format_history(state["messages"][:-1])

    prompt = CASUAL_REPLY_PROMPT.format(history=history, user_message=user_message)
    reply = get_llm().invoke(prompt).content.strip()
    return {"messages": [AIMessage(content=reply)]}


def rag_node(state: AgentState):
    """Answer questions using retrieved knowledge base context."""
    user_message = latest_user_message(state["messages"])
    history = format_history(state["messages"][:-1])

    context = retrieve(user_message, k=3)
    prompt = RAG_ANSWER_PROMPT.format(
        context=context,
        history=history,
        user_message=user_message,
    )
    reply = get_llm().invoke(prompt).content.strip()
    return {"messages": [AIMessage(content=reply)]}


def parse_extraction_json(raw):
    """
    Parse the JSON returned by the extraction prompt.
    Gemini sometimes wraps JSON in ```json ... ``` fences, so we strip those.
    """
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        # Remove opening fence (```json or just ```)
        cleaned = cleaned.split("\n", 1)[-1] if "\n" in cleaned else cleaned
        # Remove closing fence
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {}


def lead_node(state: AgentState):
    """
    Collect lead info and fire the tool once all 3 fields are present.

    On each turn we try to extract any new values from the user's latest
    message, update state, and either:
      - call mock_lead_capture() if we now have name + email + platform
      - ask the user for whichever fields are still missing
    """
    user_message = latest_user_message(state["messages"])
    lead_info = dict(state.get("lead_info", {}) or {})

    # Extract any new info from the user's latest message
    extraction_prompt = LEAD_EXTRACTION_PROMPT.format(
        name=lead_info.get("name"),
        email=lead_info.get("email"),
        platform=lead_info.get("platform"),
        user_message=user_message,
    )
    raw = get_llm().invoke(extraction_prompt).content
    extracted = parse_extraction_json(raw)

    for field in ("name", "email", "platform"):
        value = extracted.get(field)
        if value and value != "null" and not lead_info.get(field):
            lead_info[field] = str(value).strip()

    # Figure out what's still missing
    missing = [f for f in ("name", "email", "platform") if not lead_info.get(f)]

    # All 3 fields collected -> fire the tool
    if not missing:
        mock_lead_capture(
            name=lead_info["name"],
            email=lead_info["email"],
            platform=lead_info["platform"],
        )
        reply = (
            f"Perfect, you're all set, {lead_info['name']}! I've captured your "
            f"details and our team will reach out to {lead_info['email']} shortly "
            f"to get you started on the Pro plan for your {lead_info['platform']} "
            f"channel. Welcome to AutoStream!"
        )
        return {
            "messages": [AIMessage(content=reply)],
            "lead_info": lead_info,
            "lead_captured": True,
        }

    # Still missing fields -> ask for them
    had_any_before = any((state.get("lead_info") or {}).get(f) for f in ("name", "email", "platform"))
    intro = ""
    if not had_any_before:
        intro = "Awesome, I'd love to help you get started on the Pro plan! I just need a few quick details. "

    field_questions = {
        "name": "Could I get your name",
        "email": "what's the best email to reach you at",
        "platform": "and which platform do you create content on (YouTube, Instagram, TikTok, etc.)",
    }
    asks = [field_questions[f] for f in missing]
    if len(asks) == 1:
        question = asks[0][0].upper() + asks[0][1:] + "?"
    else:
        question = ", ".join(asks[:-1]) + ", " + asks[-1] + "?"
        question = question[0].upper() + question[1:]

    reply = (intro + question).strip()
    return {
        "messages": [AIMessage(content=reply)],
        "lead_info": lead_info,
    }


# -----------------------------
# Routing
# -----------------------------

def pick_handler(state: AgentState):
    """Conditional edge: route to the right node based on intent."""
    intent = state.get("intent", "casual")
    if intent == "high_intent":
        return "lead_node"
    if intent == "inquiry":
        return "rag_node"
    return "casual_node"


def build_graph():
    """Build and compile the LangGraph with an in-memory checkpointer."""
    graph = StateGraph(AgentState)

    graph.add_node("route_turn", route_turn)
    graph.add_node("casual_node", casual_node)
    graph.add_node("rag_node", rag_node)
    graph.add_node("lead_node", lead_node)

    graph.add_edge(START, "route_turn")
    graph.add_conditional_edges("route_turn", pick_handler)
    graph.add_edge("casual_node", END)
    graph.add_edge("rag_node", END)
    graph.add_edge("lead_node", END)

    # MemorySaver keeps state across turns for each thread_id.
    return graph.compile(checkpointer=MemorySaver())