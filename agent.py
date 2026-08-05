"""
LangGraph agent for AutoStream — the core piece I built to learn how LangGraph
models multi-turn, stateful conversations as an explicit graph instead of a
single prompt-in/text-out loop.

Graph flow:
    user message -> route_turn -> (casual_node OR rag_node OR lead_node) -> END

The router decides which node handles the turn based on classified intent.
Conversation state is persisted across turns using a SQLite-backed checkpointer
(keyed by thread_id in main.py / api.py), so a restart doesn't wipe every
in-progress conversation the way the original in-memory checkpointer did.
"""

import json
import logging
import sqlite3
import time
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command, interrupt

from prompts import (
    CASUAL_REPLY_PROMPT,
    INTENT_CLASSIFIER_PROMPT,
    LEAD_EXTRACTION_PROMPT,
    RAG_ANSWER_PROMPT,
)
from rag import retrieve
from tools import mock_lead_capture

logger = logging.getLogger(__name__)

CHECKPOINT_DB_PATH = "checkpoints.sqlite"
LOG_FILE_PATH = "agent.log"


def configure_logging():
    """
    Send per-node observability logs (see log_node below) to agent.log, and
    keep the console limited to warnings/errors so a normal chat session
    isn't drowned in log lines. Called once by build_graph(), which every
    front-end (CLI/API/Gradio) goes through.
    """
    root = logging.getLogger()
    if any(isinstance(h, logging.FileHandler) for h in root.handlers):
        return  # already configured (e.g. tests import agent more than once)

    root.setLevel(logging.INFO)

    file_handler = logging.FileHandler(LOG_FILE_PATH, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    root.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    root.addHandler(console_handler)


def log_node(node_name):
    """
    Decorator that logs one line per node execution — which node ran, the
    resulting intent (if any), and how long it took. This is the cheapest
    version of "observability": no LangSmith account or extra service
    needed, just a plain agent.log a reviewer can tail to see exactly how a
    conversation was routed turn by turn.
    """

    def decorator(fn):
        def wrapper(state):
            start = time.monotonic()
            result = fn(state)
            elapsed_ms = (time.monotonic() - start) * 1000
            intent = result.get("intent", state.get("intent", ""))
            logger.info("node=%s intent=%s latency_ms=%.1f", node_name, intent, elapsed_ms)
            return result

        return wrapper

    return decorator

# Fallback reply used whenever every retry against the LLM fails. Keeping this
# as a constant instead of inlining it means every node fails the same, honest
# way instead of each one improvising its own error text.
LLM_UNAVAILABLE_REPLY = (
    "Sorry, I'm having trouble reaching the AI service right now. "
    "Please try again in a moment."
)

# Single LLM instance shared by all nodes. Lazily initialized so importing this
# module (e.g. for tests) doesn't hit the API or require an API key.
_llm = None


def get_llm():
    global _llm
    if _llm is None:
        _llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.2)
    return _llm


def invoke_llm(prompt, retries=2, backoff_seconds=1.5):
    """
    Call the LLM with retries and a graceful failure mode.

    This is the fix for the biggest gap in the first version of this file: every
    node called get_llm().invoke(...) directly, so a single dropped request or
    rate-limit response crashed the whole turn. Real APIs fail transiently all
    the time, so a short retry with backoff is worth the few extra lines.

    Returns the response text, or None if every attempt failed — callers decide
    what fallback text to show the user.
    """
    last_error = None
    for attempt in range(retries + 1):
        try:
            response = get_llm().invoke(prompt)
            return response.content.strip()
        except Exception as exc:  # LLM SDKs raise a variety of transport/API errors
            last_error = exc
            logger.warning("LLM call failed (attempt %d/%d): %s", attempt + 1, retries + 1, exc)
            if attempt < retries:
                time.sleep(backoff_seconds * (attempt + 1))

    logger.error("LLM call failed after %d attempts: %s", retries + 1, last_error)
    return None


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

@log_node("route_turn")
def route_turn(state: AgentState):
    """
    Decide which handler should run this turn.

    Normally we classify the user message into casual / inquiry / high_intent.
    But if we've already started collecting lead info (name/email/platform)
    and haven't finished, we stay in the lead flow. Otherwise a short reply
    like "Alex" gets classified as casual and the signup flow breaks. This
    guard is the one piece of routing logic that isn't obvious from a basic
    LangGraph tutorial — it only showed up once I actually tested a multi-turn
    signup by hand.
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
    raw = invoke_llm(prompt)

    if raw is None:
        # Can't classify right now — default to casual, which is the safest
        # (least action-taking) bucket rather than accidentally starting a
        # lead flow or promising an answer we can't ground.
        return {"intent": "casual"}

    raw = raw.lower()
    if "high_intent" in raw:
        intent = "high_intent"
    elif "inquiry" in raw:
        intent = "inquiry"
    else:
        intent = "casual"

    return {"intent": intent}


@log_node("casual_node")
def casual_node(state: AgentState):
    """Handle greetings and small talk."""
    user_message = latest_user_message(state["messages"])
    history = format_history(state["messages"][:-1])

    prompt = CASUAL_REPLY_PROMPT.format(history=history, user_message=user_message)
    reply = invoke_llm(prompt) or LLM_UNAVAILABLE_REPLY
    return {"messages": [AIMessage(content=reply)]}


@log_node("rag_node")
def rag_node(state: AgentState):
    """Answer questions using retrieved knowledge base context."""
    user_message = latest_user_message(state["messages"])
    history = format_history(state["messages"][:-1])

    try:
        context = retrieve(user_message, k=3)
    except Exception as exc:
        # retrieve() calls Gemini's embeddings API directly (to build/query
        # the FAISS index) and, unlike invoke_llm(), had no error handling
        # at all -- a transient embeddings-API failure crashed the whole
        # turn instead of degrading gracefully like every other node does.
        logger.error("RAG retrieval failed: %s", exc)
        return {"messages": [AIMessage(content=LLM_UNAVAILABLE_REPLY)]}

    prompt = RAG_ANSWER_PROMPT.format(
        context=context,
        history=history,
        user_message=user_message,
    )
    reply = invoke_llm(prompt) or LLM_UNAVAILABLE_REPLY
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


@log_node("lead_node")
def lead_node(state: AgentState):
    """
    Collect lead info from the conversation. Once all 3 fields are present,
    this hands off to escalate_node for human sign-off instead of firing the
    capture tool itself — see escalate_node for why.

    On each turn we try to extract any new values from the user's latest
    message, update state, and either:
      - confirm the details are complete and let the graph route to
        escalate_node
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
    raw = invoke_llm(extraction_prompt)
    extracted = parse_extraction_json(raw) if raw is not None else {}

    for field in ("name", "email", "platform"):
        value = extracted.get(field)
        if value and value != "null" and not lead_info.get(field):
            lead_info[field] = str(value).strip()

    # Figure out what's still missing
    missing = [f for f in ("name", "email", "platform") if not lead_info.get(f)]

    # All 3 fields collected -> acknowledge and let pick_after_lead() route to
    # escalate_node. We don't call the capture tool here anymore.
    if not missing:
        reply = (
            f"Got it, {lead_info['name']}! I'm passing your details along for a "
            f"quick review before we finalize your Pro plan signup for "
            f"{lead_info['platform']} — you'll hear from us shortly."
        )
        return {
            "messages": [AIMessage(content=reply)],
            "lead_info": lead_info,
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


@log_node("escalate_node")
def escalate_node(state: AgentState):
    """
    Pause the graph and wait for a human to approve or reject the lead
    before mock_lead_capture() ever fires.

    This is the human-in-the-loop piece: a bare "if all 3 fields are
    present, save it" flow means the agent alone decides who counts as a
    customer. interrupt() suspends execution here — graph.invoke() returns
    to the caller immediately with no reply yet — and the run only resumes
    once something calls graph.invoke(Command(resume=...), config=...) with
    an approval decision (see api.py's /approve endpoint or main.py's CLI
    prompt). The state (including lead_info) is safely checkpointed to
    checkpoints.sqlite while paused, so this survives a process restart too.
    """
    lead_info = state["lead_info"]
    decision = interrupt({"type": "lead_approval", "lead_info": lead_info})

    approved = bool(decision.get("approved")) if isinstance(decision, dict) else bool(decision)

    if approved:
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
        return {"messages": [AIMessage(content=reply)], "lead_captured": True}

    reply = (
        "Thanks for your interest! Our team wants to follow up personally before "
        "finalizing this — someone will reach out to go over the details with you."
    )
    # Clear lead_info on rejection too, not just leave lead_captured=False —
    # otherwise route_turn's in_lead_flow guard (it checks lead_info, not
    # lead_captured) would keep routing every future message straight back
    # into lead_node/escalate_node forever.
    return {"messages": [AIMessage(content=reply)], "lead_captured": False, "lead_info": {}}


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


def pick_after_lead(state: AgentState):
    """
    Conditional edge out of lead_node: once name/email/platform are all
    present and the lead hasn't been through review yet, hand off to
    escalate_node for human approval instead of ending the turn.
    """
    lead_info = state.get("lead_info") or {}
    is_complete = all(lead_info.get(f) for f in ("name", "email", "platform"))
    if is_complete and not state.get("lead_captured"):
        return "escalate_node"
    return END


def build_graph():
    """
    Build and compile the LangGraph with a SQLite-backed checkpointer.

    Originally used MemorySaver (pure in-process RAM), which meant every
    conversation vanished on restart. SqliteSaver persists checkpoints to
    checkpoints.sqlite so a thread_id's history survives a process restart —
    the same interface swap would apply to a Postgres-backed checkpointer for
    a real multi-instance deployment.
    """
    configure_logging()

    graph = StateGraph(AgentState)

    graph.add_node("route_turn", route_turn)
    graph.add_node("casual_node", casual_node)
    graph.add_node("rag_node", rag_node)
    graph.add_node("lead_node", lead_node)
    graph.add_node("escalate_node", escalate_node)

    graph.add_edge(START, "route_turn")
    graph.add_conditional_edges("route_turn", pick_handler)
    graph.add_edge("casual_node", END)
    graph.add_edge("rag_node", END)
    graph.add_conditional_edges("lead_node", pick_after_lead)
    graph.add_edge("escalate_node", END)

    # SqliteSaver.from_conn_string() is a context manager meant for short-lived
    # scripts; a long-running server/CLI needs the connection to stay open for
    # the process lifetime, so we open it directly instead.
    conn = sqlite3.connect(CHECKPOINT_DB_PATH, check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    return graph.compile(checkpointer=checkpointer)


def get_pending_lead_approval(graph, config):
    """
    Return the lead_approval interrupt payload if this thread is currently
    paused at escalate_node, or None if it isn't waiting on anything.

    Front-ends (main.py, api.py) poll this after every invoke() to know
    whether to show a normal reply or surface a pending-approval prompt.
    """
    snapshot = graph.get_state(config)
    for task in snapshot.tasks:
        for pending in task.interrupts:
            if isinstance(pending.value, dict) and pending.value.get("type") == "lead_approval":
                return pending.value
    return None


def resume_with_lead_decision(graph, config, approved):
    """Resume a thread paused at escalate_node with an approve/reject decision."""
    return graph.invoke(Command(resume={"approved": approved}), config=config)
