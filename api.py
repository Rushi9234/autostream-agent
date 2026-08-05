"""
FastAPI wrapper around the LangGraph agent.

I built this to learn how to put a LangGraph agent behind an HTTP API instead
of only a CLI REPL — the same shape a WhatsApp/Slack webhook or a hosted chat
widget would need: given a thread_id and a message, run one turn of the graph
and return the reply.

/chat also surfaces the human-in-the-loop step: once a lead's name/email/
platform are all collected, the graph pauses at escalate_node instead of
capturing it immediately (see agent.py). This endpoint reports that as
pending_approval=true; a separate call to /approve resumes the graph with a
reviewer's decision.

Run with: uvicorn api:app --reload
Then:     curl -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" \\
              -d '{"thread_id": "demo-1", "message": "Hi, tell me about your pricing."}'
"""

import os
import secrets

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from agent import build_graph, get_pending_lead_approval, resume_with_lead_decision

load_dotenv()

app = FastAPI(title="AutoStream Agent API")

# /approve is the "human" half of the human-in-the-loop flow -- without this
# check, anyone who knows or guesses a thread_id could approve or reject
# someone else's lead, since thread_id alone was the only thing gating it.
# If ADMIN_TOKEN isn't set (e.g. a quick local demo), generate one at startup
# and log it once, rather than silently leaving the endpoint wide open.
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN")
if not ADMIN_TOKEN:
    ADMIN_TOKEN = secrets.token_urlsafe(24)
    print(
        f"No ADMIN_TOKEN set in the environment -- generated one for this run: {ADMIN_TOKEN}\n"
        "Set ADMIN_TOKEN in .env for a stable value across restarts. "
        "Pass it as the X-Admin-Token header when calling /approve."
    )


def require_admin_token(x_admin_token: str | None = Header(default=None)):
    if not x_admin_token or not secrets.compare_digest(x_admin_token, ADMIN_TOKEN):
        raise HTTPException(status_code=401, detail="Missing or invalid X-Admin-Token header")


# Default on, since this is a public demo with no real reviewer behind it --
# set AUTO_APPROVE_DEMO_LEADS=false in .env for a deployment where /approve
# should be the only way a lead ever gets captured.
AUTO_APPROVE_DEMO_LEADS = os.getenv("AUTO_APPROVE_DEMO_LEADS", "true").lower() != "false"

# Serves the custom chat UI (static/index.html + style.css + script.js) --
# built instead of relying on Gradio's default theme so this doesn't read as
# a generic scaffolded chatbot. /chat and /approve below are what it talks to.
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def index():
    return FileResponse("static/index.html")

# The graph (and its SQLite checkpointer connection) is built once per process
# and reused across requests — rebuilding it per-request would reopen the
# checkpoint DB on every call for no benefit.
_graph = build_graph()


class ChatRequest(BaseModel):
    thread_id: str
    message: str


class ChatResponse(BaseModel):
    thread_id: str
    reply: str
    intent: str
    lead_captured: bool
    pending_approval: bool
    pending_lead: dict | None = None


class ApproveRequest(BaseModel):
    thread_id: str
    approved: bool


def _to_response(thread_id, result, graph):
    config = {"configurable": {"thread_id": thread_id}}
    pending = get_pending_lead_approval(graph, config)
    return ChatResponse(
        thread_id=thread_id,
        reply=result["messages"][-1].content,
        intent=result.get("intent", ""),
        lead_captured=result.get("lead_captured", False),
        pending_approval=pending is not None,
        pending_lead=pending["lead_info"] if pending else None,
    )


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """Run one turn of the agent for the given thread_id and return its reply."""
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="message must not be empty")

    config = {"configurable": {"thread_id": request.thread_id}}
    result = _graph.invoke(
        {"messages": [HumanMessage(content=request.message)]},
        config=config,
    )

    # This public demo has no admin panel calling /approve for real, so it
    # auto-approves server-side instead. This used to be a second fetch()
    # from static/script.js straight to /approve -- but /approve now requires
    # ADMIN_TOKEN (see below), and a browser-side client obviously can't hold
    # that secret without exposing it to every visitor. Doing the auto-approve
    # here, in-process, needs no token: it's server code calling server code,
    # not an unauthenticated network request pretending to be a reviewer.
    if AUTO_APPROVE_DEMO_LEADS and get_pending_lead_approval(_graph, config) is not None:
        result = resume_with_lead_decision(_graph, config, approved=True)

    return _to_response(request.thread_id, result, _graph)


@app.post("/approve", response_model=ChatResponse, dependencies=[Depends(require_admin_token)])
def approve(request: ApproveRequest):
    """
    Resume a thread that's paused waiting on lead approval.

    This is the "human" half of the human-in-the-loop flow — call this from
    whatever a real reviewer/salesperson uses (an admin panel, a Slack
    button, etc.) once they've looked at pending_lead from a /chat response.
    Requires the X-Admin-Token header (see require_admin_token above) --
    without it, anyone who knew or guessed a thread_id could approve/reject
    someone else's lead.
    """
    config = {"configurable": {"thread_id": request.thread_id}}
    if get_pending_lead_approval(_graph, config) is None:
        raise HTTPException(
            status_code=409,
            detail=f"thread_id {request.thread_id!r} has no lead pending approval",
        )

    result = resume_with_lead_decision(_graph, config, request.approved)
    return _to_response(request.thread_id, result, _graph)


@app.get("/health")
def health():
    return {"status": "ok"}
