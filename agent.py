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
import re
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
from rag import retrieve, retrieve_with_sources
from tools import mock_lead_capture

logger = logging.getLogger(__name__)

CHECKPOINT_DB_PATH = "checkpoints.sqlite"
LOG_FILE_PATH = "agent.log"

EMAIL_REGEX = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")
EMAIL_SEARCH_REGEX = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")

OFF_TOPIC_REPLY = (
    "I can help with questions about AutoStream, including its features, pricing, "
    "plans, policies, and signup process."
)


def is_valid_email(email_str: str) -> bool:
    if not email_str:
        return False
    return bool(EMAIL_REGEX.match(email_str.strip()))


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
    resulting intent (if any), and how long it took.
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


LLM_UNAVAILABLE_REPLY = (
    "Sorry, I'm having trouble reaching the AI service right now. "
    "Please try again in a moment."
)

_EXHAUSTED_MODELS = set()

def get_llm(model_name="gemini-2.5-flash"):
    return ChatGoogleGenerativeAI(model=model_name, temperature=0.2, max_retries=0)


def invoke_llm(prompt, retries=2, backoff_seconds=0.2):
    """Call the LLM with retries and fast model fallbacks for high reliability."""
    base_models = ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-3.5-flash-lite"]
    models_to_try = [m for m in base_models if m not in _EXHAUSTED_MODELS]
    if not models_to_try:
        models_to_try = ["gemini-2.5-flash", "gemini-2.5-flash-lite"]

    last_error = None
    for attempt in range(retries + 1):
        model_name = models_to_try[min(attempt, len(models_to_try) - 1)]
        try:
            try:
                llm = get_llm(model_name)
            except TypeError:
                llm = get_llm()

            response = llm.invoke(prompt)
            if response and response.content:
                return response.content.strip()
        except Exception as exc:
            last_error = exc
            err_str = str(exc)
            if "429" in err_str or "quota" in err_str.lower() or "404" in err_str:
                _EXHAUSTED_MODELS.add(model_name)
            logger.warning("LLM call failed (attempt %d/%d with %s): %s", attempt + 1, retries + 1, model_name, exc)
            if attempt < retries and backoff_seconds > 0:
                time.sleep(backoff_seconds * (attempt + 1))

    logger.error("LLM call failed after %d attempts: %s", retries + 1, last_error)
    return None


class AgentState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    intent: str
    lead_info: dict
    lead_captured: bool
    invalid_email_attempt: bool
    sources: list[dict]


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


def extract_slots_from_message(user_message, current_lead_info):
    """Extract lead slots (name, email, platform) using fast regex + LEAD_EXTRACTION_PROMPT."""
    lead_info = dict(current_lead_info or {})
    user_msg_lower = user_message.lower()

    # 1. Fast email search anywhere in the message
    email_match = EMAIL_SEARCH_REGEX.search(user_message)
    invalid_email_attempt = False
    if email_match and not lead_info.get("email"):
        cand_email = email_match.group(0).lower()
        if is_valid_email(cand_email):
            lead_info["email"] = cand_email
        else:
            invalid_email_attempt = True

    # 2. Fast platform keyword and creator URL/handle check
    if not lead_info.get("platform"):
        if re.search(r"(?:youtube\.com|youtu\.be)", user_msg_lower):
            lead_info["platform"] = "YouTube"
        elif re.search(r"tiktok\.com", user_msg_lower):
            lead_info["platform"] = "TikTok"
        elif re.search(r"instagram\.com", user_msg_lower):
            lead_info["platform"] = "Instagram"
        elif re.search(r"twitch\.tv", user_msg_lower):
            lead_info["platform"] = "Twitch"
        else:
            for plat in ("youtube", "tiktok", "instagram", "facebook", "vimeo", "twitch", "twitter"):
                if re.search(r"\b" + plat + r"\b", user_msg_lower):
                    plat_name = "YouTube" if plat == "youtube" else ("TikTok" if plat == "tiktok" else plat.capitalize())
                    lead_info["platform"] = plat_name
                    break
        if not lead_info.get("platform") and re.search(r"\b(on x|platform x|x\.com|twitter/x)\b", user_msg_lower):
            lead_info["platform"] = "X"

    # 3. Fast name extraction
    if not lead_info.get("name"):
        NON_NAME_WORDS = {
            "ready", "interested", "looking", "here", "excited", "trying", "hoping",
            "planning", "wanting", "signing", "wondering", "just", "also", "currently",
            "going", "happy", "thrilled", "glad", "a", "an", "the", "on", "using",
            "from", "so", "new", "already", "super", "very", "not", "sure", "need",
            "would", "like", "love", "please", "can", "could", "will", "channel",
            "youtube", "tiktok", "instagram", "twitch", "twitter", "email"
        }
        # Explicit patterns first (e.g. "my name is John", "name is John", "call me John", "name: John")
        name_match = re.search(
            r"(?:my name is|name is|name:|call me)\s+([a-zA-Z]+)(?:\s+([a-zA-Z]+))?",
            user_message,
            re.IGNORECASE
        )
        if name_match:
            first = name_match.group(1).strip()
            second = (name_match.group(2) or "").strip()
            if second.lower() in ("and", "my", "email", "with", "at", "for", "the", "a", "from", "on", "sign", "channel"):
                second = ""
            candidate = f"{first} {second}".strip().title()
            if candidate.lower() not in ("youtube", "instagram", "tiktok", "pro", "basic", "plan", "the", "a") and first.lower() not in NON_NAME_WORDS:
                lead_info["name"] = candidate
        else:
            # Weaker pronoun check: "I am John", "I'm John" (rejecting common adjectives/verbs)
            weak_match = re.search(
                r"(?:i am|i'm)\s+([a-zA-Z]+)(?:\s+([a-zA-Z]+))?",
                user_message,
                re.IGNORECASE
            )
            if weak_match:
                first = weak_match.group(1).strip()
                second = (weak_match.group(2) or "").strip()
                if first.lower() not in NON_NAME_WORDS:
                    if second.lower() in ("and", "my", "email", "with", "at", "for", "the", "a", "from", "on", "sign", "channel", "interested", "ready", "looking"):
                        second = ""
                    candidate = f"{first} {second}".strip().title()
                    if candidate.lower() not in ("youtube", "instagram", "tiktok", "pro", "basic", "plan", "the", "a"):
                        lead_info["name"] = candidate

        if not lead_info.get("name") and email_match:
            # If an email is present in the text (e.g. "Rushikesh Kedar rushikedar40@gmail.com"), whatever precedes/follows is the name
            rem = EMAIL_SEARCH_REGEX.sub("", user_message).strip()
            rem = re.sub(r"[,:;!?-]", " ", rem).strip()
            words = rem.split()
            if 1 <= len(words) <= 4 and all(w.isalpha() for w in words):
                candidate = " ".join(w.capitalize() for w in words)
                if candidate.lower() not in ("youtube", "instagram", "tiktok", "pro", "basic", "plan", "the", "a", "hi", "hello", "yes", "no") and words[0].lower() not in NON_NAME_WORDS:
                    lead_info["name"] = candidate

    # 4. If message contains lead keywords or email, run full LLM extraction for edge cases
    has_lead_tokens = ("@" in user_msg_lower) or any(kw in user_msg_lower for kw in ("name", "email", "sign up", "sign me up", "my name", "i am", "call me", "reach me at"))

    if has_lead_tokens and not all(lead_info.get(f) for f in ("name", "email", "platform")):
        prompt = LEAD_EXTRACTION_PROMPT.format(
            name=lead_info.get("name"),
            email=lead_info.get("email"),
            platform=lead_info.get("platform"),
            user_message=user_message,
        )
        raw = invoke_llm(prompt)
        extracted = parse_extraction_json(raw) if raw is not None else {}

        for field in ("name", "email", "platform"):
            value = extracted.get(field)
            if value and str(value).strip().lower() != "null" and not lead_info.get(field):
                val_str = str(value).strip()
                if field == "email":
                    if is_valid_email(val_str):
                        lead_info["email"] = val_str.lower()
                    else:
                        invalid_email_attempt = True
                else:
                    lead_info[field] = val_str

    return lead_info, invalid_email_attempt


# -----------------------------
# Nodes
# -----------------------------

@log_node("route_turn")
def route_turn(state: AgentState):
    """
    Decide which handler should run this turn.
    Supports mixed intent (extracting lead details while still processing product inquiries).
    Fast-paths common inquiries to eliminate redundant LLM calls.
    """
    user_message = latest_user_message(state["messages"])
    history = format_history(state["messages"][:-1])
    user_lower = user_message.lower().strip()

    lead_info = dict(state.get("lead_info", {}) or {})
    already_captured = state.get("lead_captured", False)

    # If lead was captured previously but user asks for signup again, reset lead flow
    if already_captured and all(lead_info.get(f) for f in ("name", "email", "platform")):
        lead_info = {}
        already_captured = False

    # Extract slots to preserve lead info even if user asks a question
    updated_lead_info, invalid_email = extract_slots_from_message(user_message, lead_info)
    in_lead_flow = any(updated_lead_info.get(f) for f in ("name", "email", "platform"))

    # Fast-path 1: Inquiry keywords check (e.g. pricing, features, policies, refund, cost, team, connect)
    inquiry_keywords = (
        "price", "pricing", "cost", "how much", "feature", "plan", "basic", "pro",
        "refund", "policy", "support", "resolution", "template", "caption", "4k", "720p",
        "connect", "team", "help", "contact", "work", "how to", "how do", "where", "what",
        "who", "is there", "do you", "can i", "can you", "export", "watermark", "audio", "storage",
        "trial", "free trial", "card", "credit card", "bill", "billing", "cancel", "money",
        "does", "difference", "compare", "versus", "vs"
    )
    is_explicit_question = "?" in user_message or any(kw in user_lower for kw in ("how", "what", "can", "does", "is there", "why", "which"))
    has_inquiry_kw = any(kw in user_lower for kw in inquiry_keywords)
    has_lead_submission = ("@" in user_lower) or any(phrase in user_lower for phrase in ("my name", "i am", "call me", "sign me up", "sign up for", "i want to get", "i'd like to get"))

    if has_inquiry_kw and (is_explicit_question or not has_lead_submission):
        return {
            "intent": "inquiry",
            "lead_info": updated_lead_info,
            "lead_captured": already_captured,
            "invalid_email_attempt": invalid_email,
        }

    # Fast-path 2: Simple casual greeting check
    casual_greetings = ("hi", "hello", "hey", "good morning", "good evening", "greetings", "who are you", "what can you do")
    if user_lower in casual_greetings:
        return {
            "intent": "casual",
            "lead_info": updated_lead_info,
            "lead_captured": already_captured,
            "invalid_email_attempt": invalid_email,
        }

    # Fallback to LLM intent classification for ambiguous queries
    prompt = INTENT_CLASSIFIER_PROMPT.format(
        history=history,
        user_message=user_message,
    )
    raw = invoke_llm(prompt)

    if raw is None:
        # Check local FAISS vector store to see if question matches KB documentation!
        try:
            kb_docs = retrieve(user_message, k=1)
            if kb_docs and len(kb_docs) > 0 and len(kb_docs[0].strip()) > 15:
                intent = "inquiry"
            elif in_lead_flow and not already_captured:
                intent = "high_intent"
            else:
                intent = "casual"
        except Exception:
            intent = "high_intent" if in_lead_flow and not already_captured else "inquiry"
    else:
        raw_lower = raw.lower()
        if "inquiry" in raw_lower:
            intent = "inquiry"
        elif "high_intent" in raw_lower:
            intent = "high_intent"
        elif in_lead_flow and not already_captured:
            intent = "high_intent"
        else:
            intent = "casual"

    return {
        "intent": intent,
        "lead_info": updated_lead_info,
        "lead_captured": already_captured,
        "invalid_email_attempt": invalid_email,
    }


@log_node("casual_node")
def casual_node(state: AgentState):
    """Handle greetings and small talk with deterministic fallback if LLM is offline/rate-limited."""
    user_message = latest_user_message(state["messages"])
    history = format_history(state["messages"][:-1])

    prompt = CASUAL_REPLY_PROMPT.format(history=history, user_message=user_message)
    reply = invoke_llm(prompt)

    if not reply:
        user_lower = user_message.lower().strip()
        if any(kw in user_lower for kw in ("hi", "hello", "hey", "good morning", "good evening", "greetings")):
            reply = "Hello! I'm AutoStream's AI Assistant. I can help with questions about AutoStream features, pricing plans, or setting up a Pro plan subscription. How can I help you today?"
        elif any(kw in user_lower for kw in ("thanks", "thank you", "thx")):
            reply = "You're welcome! Let info know if you have any other questions about AutoStream."
        elif any(kw in user_lower for kw in ("bye", "goodbye", "see ya")):
            reply = "Thanks for visiting AutoStream! Have a great day."
        else:
            reply = "I'm AutoStream's virtual assistant. I can help answer questions about our automated video editing features, pricing plans, or team support options. What would you like to know?"

    return {"messages": [AIMessage(content=reply)], "sources": []}


@log_node("rag_node")
def rag_node(state: AgentState):
    """Answer questions using retrieved KB context, dynamically bridging back to lead collection if active."""
    user_message = latest_user_message(state["messages"])
    history = format_history(state["messages"][:-1])
    lead_info = state.get("lead_info", {}) or {}
    already_captured = state.get("lead_captured", False)

    try:
        context, sources = retrieve_with_sources(user_message, k=3)
    except Exception as exc:
        logger.error("RAG retrieval failed: %s", exc)
        return {"messages": [AIMessage(content=LLM_UNAVAILABLE_REPLY)], "sources": []}

    if not context or not context.strip():
        # Check off-topic vs team/support fallback
        user_lower = user_message.lower()
        if any(kw in user_lower for kw in ("team", "connect", "contact", "support", "help")):
            fallback_text = "AutoStream provides 24/7 priority support for Enterprise customers and standard email support for Pro plan users."
            sources = [{"source": "company_policies.txt", "heading": "Support Policy"}]
            return {
                "messages": [AIMessage(content=fallback_text)],
                "lead_info": lead_info,
                "sources": sources,
            }
        return {
            "messages": [AIMessage(content=OFF_TOPIC_REPLY)],
            "lead_info": lead_info,
            "sources": [],
        }

    prompt = RAG_ANSWER_PROMPT.format(
        context=context,
        history=history,
        user_message=user_message,
    )
    raw_reply = invoke_llm(prompt)

    if not raw_reply:
        # Fallback to direct KB context synthesis if external LLM API is rate-limited or unavailable
        clean_lines = [
            line.strip("- ").strip() for line in context.split("\n")
            if line.strip() and not line.startswith("[") and not line.startswith("---")
        ]
        if clean_lines:
            reply = "Based on AutoStream documentation:\n" + "\n".join(f"• {line}" for line in clean_lines[:4])
        else:
            user_lower = user_message.lower()
            if any(kw in user_lower for kw in ("team", "connect", "contact", "support")):
                reply = "AutoStream provides 24/7 priority support for Enterprise customers and standard email support for Pro plan users. You can reach our team anytime via email or the support portal."
            elif any(kw in user_lower for kw in ("price", "cost", "plan", "pro", "basic")):
                reply = "AutoStream offers two main plans: Basic Plan ($29/month, 720p export, 10 templates) and Pro Plan ($79/month, 4K export, unlimited templates, 100GB storage)."
            else:
                reply = "AutoStream is an automated AI video production engine featuring smart captioning, 4K export, custom templates, and Pro pricing starting at $79/month."
    else:
        reply = raw_reply

    # Dynamic lead resumption if mid-lead-flow
    missing = [f for f in ("name", "email", "platform") if not lead_info.get(f)]
    if lead_info and missing and not already_captured:
        field_questions = {
            "name": "could I get your name",
            "email": "what's the best email to reach you at",
            "platform": "which platform do you create content on (YouTube, Instagram, TikTok, etc.)",
        }
        next_asks = [field_questions[f] for f in missing]
        if len(next_asks) == 1:
            bridge_q = next_asks[0][0].upper() + next_asks[0][1:] + "?"
        else:
            bridge_q = ", ".join(next_asks[:-1]) + ", " + next_asks[-1] + "?"
            bridge_q = bridge_q[0].upper() + bridge_q[1:]

        reply = f"{reply}\n\n(By the way, I still have your details saved. Whenever you're ready, {bridge_q.lower()})"

    return {
        "messages": [AIMessage(content=reply)],
        "lead_info": lead_info,
        "sources": sources,
    }



def parse_extraction_json(raw):
    """
    Parse the JSON returned by the extraction prompt.
    Strips markdown code fences and handles trailing extra text.
    """
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1] if "\n" in cleaned else cleaned
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Fallback regex search for JSON object inside raw string
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        return {}


@log_node("lead_node")
def lead_node(state: AgentState):
    """
    Collect lead info dynamically and prompt for missing fields.
    """
    user_message = latest_user_message(state["messages"])
    lead_info = dict(state.get("lead_info", {}) or {})
    already_captured = state.get("lead_captured", False)

    if already_captured and all(lead_info.get(f) for f in ("name", "email", "platform")):
        lead_info = {}
        already_captured = False

    # Route_turn already extracted slots into state. Use them directly to avoid duplicate LLM latency
    invalid_email = state.get("invalid_email_attempt", False)
    if not any(lead_info.get(f) for f in ("name", "email", "platform")):
        updated_info, invalid_email = extract_slots_from_message(user_message, lead_info)
        lead_info.update(updated_info)

    missing = [f for f in ("name", "email", "platform") if not lead_info.get(f)]

    if not missing:
        reply = (
            f"Got it, {lead_info['name']}! I'm passing your details along for a "
            f"quick review before we finalize your Pro plan signup for "
            f"{lead_info['platform']} — you'll hear from us shortly."
        )
        return {
            "messages": [AIMessage(content=reply)],
            "lead_info": lead_info,
            "lead_captured": False,
        }

    had_any_before = any(lead_info.get(f) for f in ("name", "email", "platform"))
    intro = ""
    if not had_any_before:
        intro = "Awesome, I'd love to help you get started on the Pro plan! I just need a few quick details. "

    if invalid_email or state.get("invalid_email_attempt"):
        email_warning = "That email address format doesn't look quite right. "
    else:
        email_warning = ""

    field_questions = {
        "name": "could I get your name",
        "email": "what's the best email to reach you at",
        "platform": "and which platform do you create content on (YouTube, Instagram, TikTok, etc.)",
    }
    asks = [field_questions[f] for f in missing]
    if len(asks) == 1:
        question = asks[0][0].upper() + asks[0][1:] + "?"
    else:
        question = ", ".join(asks[:-1]) + ", " + asks[-1] + "?"
        question = question[0].upper() + question[1:]

    reply = (intro + email_warning + question).strip()
    return {
        "messages": [AIMessage(content=reply)],
        "lead_info": lead_info,
        "lead_captured": already_captured,
    }


@log_node("escalate_node")
def escalate_node(state: AgentState):
    """
    Pause the graph and wait for a human to approve or reject the lead.
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
        return {"messages": [AIMessage(content=reply)], "lead_captured": True, "lead_info": lead_info}

    reply = (
        "Thanks for your interest! Our team wants to follow up personally before "
        "finalizing this — someone will reach out to go over the details with you."
    )
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
    Conditional edge out of lead_node: once all fields are complete,
    hand off to escalate_node for human approval.
    """
    lead_info = state.get("lead_info") or {}
    is_complete = all(lead_info.get(f) for f in ("name", "email", "platform"))
    if is_complete and not state.get("lead_captured"):
        return "escalate_node"
    return END


def build_graph():
    """Build and compile the LangGraph with a SQLite WAL checkpointer."""
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

    conn = sqlite3.connect(CHECKPOINT_DB_PATH, check_same_thread=False, timeout=15.0)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout=5000;")
    checkpointer = SqliteSaver(conn)
    return graph.compile(checkpointer=checkpointer)


def get_pending_lead_approval(graph, config):
    """Return the lead_approval interrupt payload if paused at escalate_node."""
    snapshot = graph.get_state(config)
    for task in snapshot.tasks:
        for pending in task.interrupts:
            if isinstance(pending.value, dict) and pending.value.get("type") == "lead_approval":
                return pending.value
    return None


def resume_with_lead_decision(graph, config, approved):
    """Resume a thread paused at escalate_node with an approve/reject decision."""
    return graph.invoke(Command(resume={"approved": approved}), config=config)

