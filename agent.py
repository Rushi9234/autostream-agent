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

def get_llm(model_name="gemini-2.5-flash-lite"):
    return ChatGoogleGenerativeAI(model=model_name, temperature=0.2, max_retries=0)


def invoke_llm(prompt, retries=1, backoff_seconds=0.1):
    """Call the LLM with retries and fast model fallbacks for high reliability."""
    base_models = ["gemini-2.5-flash-lite", "gemini-2.5-flash"]
    models_to_try = [m for m in base_models if m not in _EXHAUSTED_MODELS]
    if not models_to_try:
        models_to_try = ["gemini-2.5-flash-lite", "gemini-2.5-flash"]

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
    lead_in_progress: bool
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


EXCLUDED_NAME_WORDS = {
    "hi", "hello", "hey", "good", "morning", "evening", "greetings", "howdy",
    "thanks", "thank", "thx", "bye", "goodbye", "cya", "see", "ya", "later",
    "yes", "yeah", "yep", "sure", "ok", "okay", "alright", "no", "nope", "nah",
    "pro", "basic", "plan", "plans", "pricing", "price", "cost", "features", "feature",
    "policy", "policies", "refund", "refunds", "support", "help", "contact", "team",
    "connect", "question", "questions", "tell", "show", "give", "info", "information",
    "youtube", "tiktok", "instagram", "twitch", "twitter", "facebook", "vimeo", "x",
    "email", "mail", "gmail", "platform", "channel", "handle", "username", "account",
    "sign", "signup", "register", "join", "start", "get", "started", "onboarding",
    "roi", "calculator", "raw", "video", "videos", "editing", "editor", "hours",
    "saved", "value", "creator", "tier", "select", "recommend", "recommended",
    "what", "who", "where", "when", "why", "how", "which", "can", "could", "would",
    "should", "will", "do", "does", "did", "is", "are", "am", "was", "were", "be",
    "been", "being", "have", "has", "had", "my", "your", "his", "her", "their", "our",
    "the", "a", "an", "this", "that", "these", "those", "here", "there", "ready",
    "interested", "looking", "excited", "trying", "hoping", "planning", "wanting",
    "wondering", "just", "also", "currently", "going", "happy", "thrilled", "glad",
    "super", "very", "not", "need", "like", "love", "please", "with", "at", "for",
    "from", "on", "in", "to", "by", "about", "into", "over", "after"
}


def check_personal_state_query(user_message: str, lead_info: dict, lead_captured: bool = False) -> str | None:
    """Answer personal state queries (e.g. what is my email/name/platform/details) directly from conversation state."""
    msg = user_message.lower().strip()
    msg_clean = re.sub(r"[?!.,'\"]", " ", msg).strip()
    info = lead_info or {}

    # If the user is providing or updating details, it's NOT a state lookup query!
    if EMAIL_SEARCH_REGEX.search(user_message) or any(w in msg.split() for w in ("update", "change", "replace", "switch")):
        return None

    is_email_query = bool(re.search(r"\b(?:(?:what(?:s| is| was)?|tell me|show me|do you (?:have|know)) (?:my|the) (?:e-?mail|mail)|what mail|what is mail)\b", msg_clean)) or msg in ("my email", "my mail", "email?", "mail?")
    is_name_query = bool(re.search(r"\b(?:(?:what(?:s| is| was)?|tell me|show me|do you (?:have|know)) (?:my|the) name|who am i)\b", msg_clean)) or msg in ("my name", "name?")
    is_platform_query = bool(re.search(r"\b(?:(?:what(?:s| is| was)?|tell me|show me|do you (?:have|know)) (?:my|the) platform|which platform did i (?:say|choose|pick|give))\b", msg_clean)) or msg in ("my platform", "platform?")
    is_details_query = bool(re.search(r"\b(?:what (?:details|info|data|information) do you (?:have|know)|what(?:s| is) (?:saved|recorded) for me|what do you know about me|show my (?:details|info|data))\b", msg_clean)) or msg in ("my details", "my info")

    if is_email_query:
        email = info.get("email")
        if email:
            return f"Your email on file is {email}."
        return "I don't have your email saved yet. What's the best email to reach you at?"

    if is_name_query:
        name = info.get("name")
        if name:
            return f"You told me your name is {name}."
        return "I don't have your name yet. Could I get your name?"

    if is_platform_query:
        platform = info.get("platform")
        if platform:
            return f"You mentioned you create content on {platform}."
        return "I don't have your platform saved yet. Which platform do you create content on (YouTube, Instagram, TikTok, etc.)?"

    if is_details_query:
        name_str = info.get("name", "Not provided yet")
        email_str = info.get("email", "Not provided yet")
        plat_str = info.get("platform", "Not provided yet")
        summary = (
            f"Here are the details I have saved for you so far:\n"
            f"- Name: {name_str}\n"
            f"- Email: {email_str}\n"
            f"- Platform: {plat_str}"
        )
        missing = [f for f in ("name", "email", "platform") if not info.get(f)]
        if missing and not lead_captured:
            field_questions = {
                "name": "could I get your name",
                "email": "what's the best email to reach you at",
                "platform": "which platform do you create content on",
            }
            asks = [field_questions[f] for f in missing]
            if len(asks) == 1:
                q = asks[0][0].upper() + asks[0][1:] + "?"
            else:
                q = ", ".join(asks[:-1]) + ", and " + asks[-1] + "?"
                q = q[0].upper() + q[1:]
            summary += f"\n\nWhenever you're ready, {q}"
        return summary

    return None


def extract_slots_from_message(user_message, current_lead_info, lead_in_progress=False):
    """Extract lead slots (name, email, platform) using fast regex + bare name check + LEAD_EXTRACTION_PROMPT."""
    lead_info = dict(current_lead_info or {})
    user_msg_lower = user_message.lower().strip()

    # 1. Fast email search anywhere in the message
    email_match = EMAIL_SEARCH_REGEX.search(user_message)
    invalid_email_attempt = False
    if email_match and not lead_info.get("email"):
        cand_email = email_match.group(0).lower()
        if is_valid_email(cand_email):
            lead_info["email"] = cand_email
        else:
            invalid_email_attempt = True
    elif not lead_info.get("email"):
        # Detect malformed email attempts (missing '@' or domain, e.g. rushikedar40gmail, rushi@40)
        has_at = "@" in user_message
        domain_tokens = ("gmail", "yahoo", "outlook", "hotmail", "icloud", "proton", "mail.co", ".com", ".net", ".org", ".io")
        if has_at or any(d in user_msg_lower for d in domain_tokens):
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
        # Explicit patterns first (e.g. "my name is John", "name is John", "call me John", "name: John")
        name_match = re.search(
            r"(?:my name is|name is|name:|call me)\s+([a-zA-Z]+)(?:\s+([a-zA-Z]+))?",
            user_message,
            re.IGNORECASE
        )
        if name_match:
            first = name_match.group(1).strip()
            second = (name_match.group(2) or "").strip()
            if second.lower() in ("and", "my", "email", "with", "at", "for", "the", "a", "from", "on", "sign", "channel", "youtube", "tiktok", "instagram", "twitch", "twitter"):
                second = ""
            candidate = f"{first} {second}".strip().title()
            if candidate.lower() not in ("youtube", "instagram", "tiktok", "pro", "basic", "plan", "the", "a") and first.lower() not in EXCLUDED_NAME_WORDS:
                lead_info["name"] = candidate
        else:
            # Weaker pronoun check: "I am John", "I'm John"
            weak_match = re.search(
                r"(?:i am|i'm)\s+([a-zA-Z]+)(?:\s+([a-zA-Z]+))?",
                user_message,
                re.IGNORECASE
            )
            if weak_match:
                first = weak_match.group(1).strip()
                second = (weak_match.group(2) or "").strip()
                if first.lower() not in EXCLUDED_NAME_WORDS:
                    if second.lower() in ("and", "my", "email", "with", "at", "for", "the", "a", "from", "on", "sign", "channel", "interested", "ready", "looking", "youtube", "tiktok", "instagram", "twitch"):
                        second = ""
                    candidate = f"{first} {second}".strip().title()
                    if candidate.lower() not in ("youtube", "instagram", "tiktok", "pro", "basic", "plan", "the", "a"):
                        lead_info["name"] = candidate

        if not lead_info.get("name") and email_match:
            # If an email is present in the text (e.g. "Rushikesh Kedar rushikedar40@gmail.com for youtube")
            rem = EMAIL_SEARCH_REGEX.sub("", user_message).strip()
            rem = re.sub(r"[,:;!?-]", " ", rem).strip()
            rem = re.sub(r"\b(youtube|tiktok|instagram|twitch|twitter|facebook|vimeo|x)\b", " ", rem, flags=re.IGNORECASE)
            rem = re.sub(r"\b(for|on|at|via|with|channel|platform|my|the|a|an|sign|signup|handle|here|creator|pro|plan)\b", " ", rem, flags=re.IGNORECASE)
            rem = re.sub(r"\s+", " ", rem).strip()
            words = rem.split()
            if 1 <= len(words) <= 4 and all(w.isalpha() for w in words):
                candidate = " ".join(w.capitalize() for w in words)
                if candidate.lower() not in ("youtube", "instagram", "tiktok", "pro", "basic", "plan", "the", "a", "hi", "hello", "yes", "no") and words[0].lower() not in EXCLUDED_NAME_WORDS:
                    lead_info["name"] = candidate

        # Bare name check: If message is 1-3 clean alphabetic words without inquiry, excluded, or punctuation terms
        if not lead_info.get("name") and not email_match:
            clean = re.sub(r"[,:;!?'\"]", " ", user_message).strip()
            words = clean.split()
            if 1 <= len(words) <= 3:
                if all(w.isalpha() and len(w) >= 2 for w in words):
                    if all(w.lower() not in EXCLUDED_NAME_WORDS for w in words):
                        candidate = " ".join(w.capitalize() for w in words)
                        lead_info["name"] = candidate

    # 4. If message contains lead keywords or in lead flow, run full LLM extraction for conversational phrasing
    has_lead_tokens = ("@" in user_msg_lower) or any(kw in user_msg_lower for kw in ("name", "email", "sign up", "sign me up", "my name", "i am", "call me", "reach me at"))
    should_run_llm_extract = (has_lead_tokens or lead_in_progress) and not all(lead_info.get(f) for f in ("name", "email", "platform"))
    is_pure_question = "?" in user_message or user_msg_lower.startswith(("how", "what", "where", "why", "when", "can", "does"))

    if should_run_llm_extract and not is_pure_question:
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
                    if is_valid_email(val_str) and EMAIL_SEARCH_REGEX.search(user_message):
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
    lead_in_progress = state.get("lead_in_progress", False)

    lead_intent_phrases = (
        "sign up", "sign me up", "sign up for", "i want to get", "i'd like to get",
        "get started with", "try the pro", "onboarding", "subscribe", "buy pro", "try pro"
    )
    if any(phrase in user_lower for phrase in lead_intent_phrases):
        lead_in_progress = True

    # Extract slots to preserve lead info even if user asks a question
    updated_lead_info, invalid_email = extract_slots_from_message(user_message, lead_info, lead_in_progress)
    if any(updated_lead_info.get(f) for f in ("name", "email", "platform")):
        lead_in_progress = True

    # Personal state queries check (e.g. what is my email / name / platform / details)
    state_reply = check_personal_state_query(user_message, updated_lead_info, already_captured)
    if state_reply:
        return {
            "intent": "inquiry",
            "lead_info": updated_lead_info,
            "lead_captured": already_captured,
            "lead_in_progress": lead_in_progress,
            "invalid_email_attempt": invalid_email,
        }

    # Fast-path 1: Inquiry keywords check (e.g. pricing, features, policies, refund, cost, team, connect)
    inquiry_keywords = (
        "price", "pricing", "cost", "how much", "feature", "plan", "basic", "pro",
        "refund", "policy", "policies", "support", "resolution", "template", "caption", "4k", "720p",
        "connect", "team", "help", "contact", "work", "how to", "how do", "where", "what",
        "who", "is there", "do you", "can i", "can you", "export", "watermark", "audio", "storage",
        "trial", "free trial", "card", "credit card", "bill", "billing", "cancel", "money",
        "does", "difference", "compare", "versus", "vs"
    )
    is_explicit_question = "?" in user_message or any(kw in user_lower for kw in ("how", "what", "can", "does", "is there", "why", "which"))
    has_inquiry_kw = any(kw in user_lower for kw in inquiry_keywords)
    has_lead_submission = ("@" in user_lower) or any(phrase in user_lower for phrase in lead_intent_phrases)

    if has_inquiry_kw and (is_explicit_question or not has_lead_submission):
        return {
            "intent": "inquiry",
            "lead_info": updated_lead_info,
            "lead_captured": already_captured,
            "lead_in_progress": lead_in_progress,
            "invalid_email_attempt": invalid_email,
        }

    # If user made an invalid email attempt, stay in lead flow!
    if invalid_email:
        return {
            "intent": "high_intent",
            "lead_info": updated_lead_info,
            "lead_captured": already_captured,
            "lead_in_progress": True,
            "invalid_email_attempt": True,
            "sources": [],
        }

    missing_slots = [f for f in ("name", "email", "platform") if not updated_lead_info.get(f)]

    if (lead_in_progress or any(updated_lead_info.get(f) for f in ("name", "email", "platform"))):
        if not already_captured or has_lead_submission or EMAIL_SEARCH_REGEX.search(user_message) or any(w in user_lower for w in ("update", "change", "new")):
            if not updated_lead_info.get("email") and not already_captured:
                if any(tok in user_lower for tok in ("gmail", "yahoo", "outlook", "mail", ".com", ".net", ".io")) or "@" in user_message:
                    invalid_email = True
            return {
                "intent": "high_intent",
                "lead_info": updated_lead_info,
                "lead_captured": already_captured,
                "lead_in_progress": not already_captured,
                "invalid_email_attempt": invalid_email,
                "sources": [],
            }

    # Fast-path 2: Simple casual greeting check
    casual_greetings = ("hi", "hello", "hey", "good morning", "good evening", "greetings", "who are you", "what can you do")
    if user_lower in casual_greetings:
        return {
            "intent": "casual",
            "lead_info": updated_lead_info,
            "lead_captured": already_captured,
            "lead_in_progress": lead_in_progress,
            "invalid_email_attempt": invalid_email,
        }

    # Fallback to LLM intent classification for ambiguous queries
    prompt = INTENT_CLASSIFIER_PROMPT.format(
        history=history,
        user_message=user_message,
    )
    raw = invoke_llm(prompt)

    if raw is None:
        try:
            kb_docs = retrieve(user_message, k=1)
            if kb_docs and len(kb_docs) > 0 and len(kb_docs[0].strip()) > 15:
                intent = "inquiry"
            elif (lead_in_progress or any(updated_lead_info.get(f) for f in ("name", "email", "platform"))) and not already_captured:
                intent = "high_intent"
            else:
                intent = "casual"
        except Exception:
            intent = "high_intent" if (lead_in_progress and not already_captured) else "inquiry"
    else:
        raw_lower = raw.lower()
        if "inquiry" in raw_lower:
            intent = "inquiry"
        elif "high_intent" in raw_lower:
            intent = "high_intent"
        elif (lead_in_progress or any(updated_lead_info.get(f) for f in ("name", "email", "platform"))) and not already_captured:
            intent = "high_intent"
        else:
            intent = "casual"

    return {
        "intent": intent,
        "lead_info": updated_lead_info,
        "lead_captured": already_captured,
        "lead_in_progress": lead_in_progress,
        "invalid_email_attempt": invalid_email,
    }


@log_node("casual_node")
def casual_node(state: AgentState):
    """Handle greetings and small talk with deterministic fallback if LLM is offline/rate-limited."""
    user_message = latest_user_message(state["messages"])
    history = format_history(state["messages"][:-1])
    lead_info = state.get("lead_info", {}) or {}
    already_captured = state.get("lead_captured", False)
    lead_in_progress = state.get("lead_in_progress", False)

    state_reply = check_personal_state_query(user_message, lead_info, already_captured)
    if state_reply:
        return {
            "messages": [AIMessage(content=state_reply)],
            "lead_info": lead_info,
            "lead_captured": already_captured,
            "lead_in_progress": lead_in_progress,
            "sources": [],
        }

    prompt = CASUAL_REPLY_PROMPT.format(history=history, user_message=user_message)
    reply = invoke_llm(prompt)

    if not reply:
        user_lower = user_message.lower().strip()
        if any(kw in user_lower for kw in ("hi", "hello", "hey", "good morning", "good evening", "greetings")):
            reply = "Hello! Welcome to AutoStream Studio. I can answer questions about our video editing features, compare Basic vs. Pro plans, or help you get started. How can I assist you today?"
        elif any(kw in user_lower for kw in ("thanks", "thank you", "thx", "appreciate it")):
            reply = "You're very welcome! Feel free to ask if you want to explore more features or test out our Pro plan."
        elif any(kw in user_lower for kw in ("bye", "goodbye", "see ya", "cya")):
            reply = "Thanks for visiting AutoStream! Wishing you great success with your video creation."
        else:
            reply = "I'm AutoStream's virtual assistant. I can help answer questions about our automated video editing features, pricing plans, or team support options. What would you like to know?"

    # If lead is in progress and not captured, append gentle reminder
    missing = [f for f in ("name", "email", "platform") if not lead_info.get(f)]
    if lead_in_progress and missing and not already_captured:
        field_questions = {
            "name": "could I get your name",
            "email": "what's the best email to reach you at",
            "platform": "which platform do you create content on (YouTube, Instagram, TikTok, etc.)",
        }
        next_asks = [field_questions[f] for f in missing]
        if len(next_asks) == 1:
            bridge_q = next_asks[0][0].upper() + next_asks[0][1:] + "?"
        else:
            bridge_q = ", ".join(next_asks[:-1]) + ", and " + next_asks[-1] + "?"
            bridge_q = bridge_q[0].upper() + bridge_q[1:]
        reply = f"{reply}\n\n(Whenever you're ready, {bridge_q.lower()})"

    return {
        "messages": [AIMessage(content=reply)],
        "lead_info": lead_info,
        "lead_captured": already_captured,
        "lead_in_progress": lead_in_progress,
        "sources": [],
    }


def get_cached_kb_response(user_message: str):
    """
    Tier-1 Fast response cache for top standard AutoStream knowledge queries.
    Returns (reply_text, sources_list) or (None, None).
    Zero latency, zero external API quota, high quality consultative copy.
    """
    msg = user_message.lower().strip()

    # 1. Compare Basic vs Pro
    if any(k in msg for k in ("compare", "basic vs pro", "pro vs basic", "compare basic and pro")) or (
        "difference" in msg and ("basic" in msg or "pro" in msg)
    ):
        reply = (
            "Here is a side-by-side comparison of AutoStream's plans:\n\n"
            "**Basic Plan ($29/month)**:\n"
            "• Up to 10 video exports per month\n"
            "• Automated cuts and trimming\n"
            "• 720p maximum export resolution\n"
            "• Background music library & standard templates\n"
            "• Standard business-hours email support\n\n"
            "**Pro Plan ($79/month)**:\n"
            "• Unlimited video exports\n"
            "• Ultra-HD 4K export resolution\n"
            "• AI-generated captions in multiple languages\n"
            "• Priority rendering queue & advanced editing templates\n"
            "• 24/7 dedicated support across email and chat\n\n"
            "For active creators publishing weekly, the Pro Plan typically pays for itself within the first few videos."
        )
        sources = [
            {"source": "features.txt", "heading": "Basic Plan Features"},
            {"source": "features.txt", "heading": "Pro Plan Features"},
            {"source": "pricing_plans.txt", "heading": "Pricing Plans"},
        ]
        return reply, sources

    # 2. Pro plan cost & features
    if any(re.search(p, msg) for p in (
        r"\bhow much (?:is|does) (?:the )?pro\b",
        r"\bcost of (?:the )?pro\b",
        r"\bpro plan (?:price|cost)\b",
        r"\bprice of (?:the )?pro\b",
    )):
        reply = (
            "The **Pro Plan is $79 per month**. It is built for active video creators, "
            "providing unlimited video exports, 4K resolution, multilingual AI captions, "
            "priority render queue access, and 24/7 support across email and chat."
        )
        sources = [
            {"source": "pricing_plans.txt", "heading": "Pro Plan"},
            {"source": "features.txt", "heading": "Pro Plan Features"},
        ]
        return reply, sources

    # 3. Basic plan cost & features
    if any(re.search(p, msg) for p in (
        r"\bhow much (?:is|does) (?:the )?basic\b",
        r"\bcost of (?:the )?basic\b",
        r"\bbasic plan (?:price|cost)\b",
        r"\bprice of (?:the )?basic\b",
    )):
        reply = (
            "The **Basic Plan is $29 per month**. It includes up to 10 video exports per month, "
            "720p maximum resolution, automated cuts & trimming, background music library, "
            "standard templates, and business-hours email support."
        )
        sources = [
            {"source": "pricing_plans.txt", "heading": "Basic Plan"},
            {"source": "features.txt", "heading": "Basic Plan Features"},
        ]
        return reply, sources

    # 4. Refund policy
    if any(k in msg for k in ("refund policy", "refunds", "money back", "can i get a refund", "return policy")) or (
        "refund" in msg and any(w in msg for w in ("what", "how", "policy", "days", "guarantee"))
    ):
        reply = (
            "AutoStream offers refunds within **7 days of your original purchase date**. "
            "After 7 days have passed from the purchase date, no refunds are provided."
        )
        sources = [
            {"source": "company_policies.txt", "heading": "Refund Policy"},
        ]
        return reply, sources

    # 5. Support policy
    if any(k in msg for k in ("support policy", "support hours", "customer support", "24/7 support", "how does support work")):
        reply = (
            "AutoStream provides two levels of customer support:\n"
            "• **Basic Plan:** Standard business-hours email support.\n"
            "• **Pro Plan:** 24/7 priority support across both email and chat."
        )
        sources = [
            {"source": "company_policies.txt", "heading": "Support Policy"},
        ]
        return reply, sources

    # 6. Resolution / 4K
    if any(k in msg for k in ("4k", "resolution", "export quality", "can i export 4k", "720p")):
        reply = (
            "AutoStream supports up to **4K resolution** on the **Pro Plan** ($79/mo) with unlimited exports. "
            "The **Basic Plan** ($29/mo) exports at up to **720p resolution** with 10 videos per month."
        )
        sources = [
            {"source": "features.txt", "heading": "Pro Plan Features"},
            {"source": "features.txt", "heading": "Basic Plan Features"},
        ]
        return reply, sources

    # 7. AI Captions / Subtitles
    if any(k in msg for k in ("caption", "subtitle", "ai caption", "transcription", "subtitles")):
        reply = (
            "AutoStream includes **AI-generated captions in multiple languages** as part of the **Pro Plan**. "
            "It automatically transcribes speech, synchronizes subtitle timings, and renders customizable animated captions directly on your video."
        )
        sources = [
            {"source": "features.txt", "heading": "Pro Plan Features"},
        ]
        return reply, sources

    # 8. ROI Calculator Explanation
    if any(k in msg for k in ("roi calculator", "how does the calculator work", "how is roi calculated", "hours saved")):
        reply = (
            "Our Video Editing ROI Calculator estimates your monthly time and cost savings. "
            "By entering your weekly video count and average raw video length, it calculates the hours "
            "AutoStream saves you by automating cuts, pauses, and captions, and recommends either the Basic or Pro tier."
        )
        sources = [
            {"source": "features.txt", "heading": "Pro Plan Features"},
        ]
        return reply, sources

    return None, None


@log_node("rag_node")
def rag_node(state: AgentState):
    """Answer questions using retrieved KB context, dynamically bridging back to lead collection if active."""
    user_message = latest_user_message(state["messages"])
    history = format_history(state["messages"][:-1])
    lead_info = state.get("lead_info", {}) or {}
    already_captured = state.get("lead_captured", False)
    lead_in_progress = state.get("lead_in_progress", False)

    # 1. Check for personal state queries first (e.g. 'what is my mail')
    state_reply = check_personal_state_query(user_message, lead_info, already_captured)
    if state_reply:
        return {
            "messages": [AIMessage(content=state_reply)],
            "lead_info": lead_info,
            "lead_captured": already_captured,
            "lead_in_progress": lead_in_progress,
            "sources": [],
        }

    # 2. Retrieve KB context using FAISS vector store
    try:
        context, sources = retrieve_with_sources(user_message, k=3)
    except Exception as exc:
        logger.error("RAG retrieval failed: %s", exc)
        return {
            "messages": [AIMessage(content=LLM_UNAVAILABLE_REPLY)],
            "lead_info": lead_info,
            "lead_captured": already_captured,
            "lead_in_progress": lead_in_progress,
            "sources": [],
        }

    # 3. Fast Tier-1 Response Cache (Instant, 0 tokens used, 100% reliable)
    cached_reply, cached_sources = get_cached_kb_response(user_message)
    if cached_reply:
        reply = cached_reply
        if not sources and cached_sources:
            sources = cached_sources
    else:
        # 4. If no exact cache match, verify relevance or invoke LLM (Tier 2: Gemini 2.5 Flash Lite)
        if not context or not context.strip():
            user_lower = user_message.lower()
            if any(kw in user_lower for kw in ("team", "connect", "contact", "support", "help")):
                fallback_text = "AutoStream provides 24/7 priority support for Pro plan users and standard email support for Basic plan users."
                sources = [{"source": "company_policies.txt", "heading": "Support Policy"}]
                return {
                    "messages": [AIMessage(content=fallback_text)],
                    "lead_info": lead_info,
                    "lead_captured": already_captured,
                    "lead_in_progress": lead_in_progress,
                    "sources": sources,
                }
            return {
                "messages": [AIMessage(content=OFF_TOPIC_REPLY)],
                "lead_info": lead_info,
                "lead_captured": already_captured,
                "lead_in_progress": lead_in_progress,
                "sources": [],
            }

        prompt = RAG_ANSWER_PROMPT.format(
            context=context,
            history=history,
            user_message=user_message,
        )
        raw_reply = invoke_llm(prompt)

        # 5. Tier 3: Polished conversational fallback if LLM is unreachable
        if not raw_reply:
            clean_lines = [
                line.strip("- ").strip() for line in context.split("\n")
                if line.strip() and not line.startswith("[") and not line.startswith("---")
            ]
            if clean_lines:
                points = "\n".join(f"• {line}" for line in clean_lines[:4])
                reply = f"Based on AutoStream documentation:\n{points}\n\nLet me know if you would like more details or want to get started!"
            else:
                user_lower = user_message.lower()
                if any(kw in user_lower for kw in ("price", "cost", "plan", "pro", "basic")):
                    reply = "AutoStream offers two main plans: the Basic Plan at $29/month (up to 10 videos, 720p) and the Pro Plan at $79/month (unlimited videos, 4K, AI captions, 24/7 support)."
                else:
                    reply = "AutoStream is an automated AI video production platform featuring smart trimming, AI captions, 4K exports, and priority render queue."
        else:
            reply = raw_reply

    # Dynamic lead resumption if mid-lead-flow
    missing = [f for f in ("name", "email", "platform") if not lead_info.get(f)]
    if (lead_info or lead_in_progress) and missing and not already_captured:
        field_questions = {
            "name": "could I get your name",
            "email": "what's the best email to reach you at",
            "platform": "which platform do you create content on (YouTube, Instagram, TikTok, etc.)",
        }
        next_asks = [field_questions[f] for f in missing]
        if len(next_asks) == 1:
            bridge_q = next_asks[0][0].upper() + next_asks[0][1:] + "?"
        else:
            bridge_q = ", ".join(next_asks[:-1]) + ", and " + next_asks[-1] + "?"
            bridge_q = bridge_q[0].upper() + bridge_q[1:]

        reply = f"{reply}\n\n(By the way, I still have your details saved. Whenever you're ready, {bridge_q.lower()})"

    return {
        "messages": [AIMessage(content=reply)],
        "lead_info": lead_info,
        "lead_captured": already_captured,
        "lead_in_progress": lead_in_progress,
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
    lead_in_progress = True

    invalid_email = state.get("invalid_email_attempt", False)
    updated_info, inv = extract_slots_from_message(user_message, lead_info, lead_in_progress)
    lead_info.update(updated_info)
    if inv:
        invalid_email = True

    # If user has already completed onboarding in this thread:
    if already_captured and all(lead_info.get(f) for f in ("name", "email", "platform")):
        email_match = EMAIL_SEARCH_REGEX.search(user_message)
        has_new_details = False
        if email_match:
            new_email = email_match.group(0).lower()
            if is_valid_email(new_email) and new_email != lead_info.get("email"):
                lead_info["email"] = new_email
                has_new_details = True
        for plat in ("youtube", "tiktok", "instagram", "facebook", "vimeo", "twitch", "twitter"):
            if re.search(r"\b" + plat + r"\b", user_message.lower()):
                pname = "YouTube" if plat == "youtube" else ("TikTok" if plat == "tiktok" else plat.capitalize())
                if pname != lead_info.get("platform"):
                    lead_info["platform"] = pname
                    has_new_details = True

        if not has_new_details:
            reply = (
                f"You're already all set on the Pro plan, {lead_info['name']}! "
                f"We have your {lead_info['platform']} channel registered under {lead_info['email']}. "
                f"If you'd like to update your details or add another channel, just let me know!"
            )
            return {
                "messages": [AIMessage(content=reply)],
                "lead_info": lead_info,
                "lead_captured": True,
                "lead_in_progress": False,
                "sources": [],
            }
        else:
            mock_lead_capture(
                name=lead_info["name"],
                email=lead_info["email"],
                platform=lead_info["platform"],
            )
            reply = (
                f"Got it, {lead_info['name']}! I've updated your Pro plan details "
                f"for your {lead_info['platform']} channel to {lead_info['email']}. Welcome to AutoStream!"
            )
            return {
                "messages": [AIMessage(content=reply)],
                "lead_info": lead_info,
                "lead_captured": True,
                "lead_in_progress": False,
                "sources": [],
            }

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
            "lead_in_progress": True,
            "sources": [],
        }

    prefix = ""
    if invalid_email:
        prefix = "That email address format doesn't look quite right (please check for '@' and domain, e.g. name@example.com). "
    elif lead_info.get("name") and len(missing) < 3:
        if not lead_info.get("email") and not lead_info.get("platform"):
            prefix = f"Great, {lead_info['name']}! "
        elif not lead_info.get("email") and lead_info.get("platform"):
            prefix = f"Great, {lead_info['name']}! Got your {lead_info['platform']} channel. "
        elif lead_info.get("email") and not lead_info.get("platform"):
            prefix = f"Great, {lead_info['name']}! "
    elif not lead_info.get("name") and lead_info.get("platform"):
        prefix = f"Got it, {lead_info['platform']}! "

    if not any(lead_info.get(f) for f in ("name", "email", "platform")) and not invalid_email:
        prefix = "Awesome, I'd love to help you get started on the Pro plan! I just need a few quick details. "

    field_questions = {
        "name": "could I get your name",
        "email": "what's the best email to reach you at",
        "platform": "which platform do you create content on (YouTube, Instagram, TikTok, etc.)",
    }
    asks = [field_questions[f] for f in missing]
    if len(asks) == 1:
        question = asks[0][0].upper() + asks[0][1:] + "?"
    elif len(asks) == 2:
        question = asks[0][0].upper() + asks[0][1:] + ", and " + asks[1] + "?"
    else:
        question = asks[0][0].upper() + asks[0][1:] + ", " + asks[1] + ", and " + asks[2] + "?"

    reply = (prefix + question).strip()
    return {
        "messages": [AIMessage(content=reply)],
        "lead_info": lead_info,
        "lead_captured": already_captured,
        "lead_in_progress": True,
        "sources": [],
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
        return {"messages": [AIMessage(content=reply)], "lead_captured": True, "lead_in_progress": False, "lead_info": lead_info, "sources": []}

    reply = (
        "Thanks for your interest! Our team wants to follow up personally before "
        "finalizing this — someone will reach out to go over the details with you."
    )
    return {"messages": [AIMessage(content=reply)], "lead_captured": False, "lead_in_progress": False, "lead_info": {}, "sources": []}


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

