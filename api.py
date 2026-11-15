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

import json
import logging
import os
import secrets
import sqlite3
import time
from collections import defaultdict

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from agent import build_graph, get_pending_lead_approval, resume_with_lead_decision

load_dotenv()

app = FastAPI(title="AutoStream Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:8000",
        "http://localhost:8000",
        "http://127.0.0.1:3000",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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


_rate_limit_records = defaultdict(list)
MAX_REQUESTS_PER_MINUTE = 60


def check_rate_limit(request: Request):
    client_ip = request.client.host if (request.client and request.client.host) else "127.0.0.1"
    now = time.time()
    window_start = now - 60.0

    # Prune inactive IP records periodically to prevent memory exhaustion
    if len(_rate_limit_records) > 200:
        stale_ips = [ip for ip, timestamps in _rate_limit_records.items() if not timestamps or timestamps[-1] < window_start]
        for ip in stale_ips:
            del _rate_limit_records[ip]

    recent = [t for t in _rate_limit_records[client_ip] if t > window_start]
    if len(recent) >= MAX_REQUESTS_PER_MINUTE:
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Please wait a minute.")
    recent.append(now)
    _rate_limit_records[client_ip] = recent


AUTO_APPROVE_DEMO_LEADS = os.getenv("AUTO_APPROVE_DEMO_LEADS", "true").lower() != "false"


def dispatch_lead_webhook(lead_info: dict, event_type: str = "lead.captured"):
    """
    Dispatch lead webhook notification.
    Logs event to agent.log and POSTs to WEBHOOK_URL if configured.
    """
    import urllib.request
    payload = {
        "event": event_type,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "lead": lead_info,
    }
    api_logger = logging.getLogger("autostream.webhook")
    api_logger.info("Webhook event dispatched: %s for %s", event_type, lead_info.get("email"))
    webhook_url = os.getenv("WEBHOOK_URL")
    if webhook_url:
        try:
            req = urllib.request.Request(
                webhook_url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(req, timeout=3)
        except Exception as e:
            api_logger.warning("Failed to POST webhook to %s: %s", webhook_url, e)


app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def index():
    return FileResponse("static/index.html")


@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)


@app.get("/api/client-config")
def get_client_config():
    """Returns safe public metadata for web studio sessions."""
    return {"status": "online", "service": "AutoStream AI Studio"}


_graph = build_graph()


class ChatRequest(BaseModel):
    thread_id: str = Field(..., min_length=1, max_length=128)
    message: str = Field(..., min_length=1, max_length=2000)


class ChatResponse(BaseModel):
    thread_id: str
    reply: str
    intent: str
    lead_captured: bool
    lead_info: dict | None = None
    pending_approval: bool
    pending_lead: dict | None = None
    sources: list[dict] | None = None


class ApproveRequest(BaseModel):
    thread_id: str = Field(..., min_length=1, max_length=128)
    approved: bool


class HistoryMessage(BaseModel):
    role: str
    content: str


class HistoryResponse(BaseModel):
    thread_id: str
    messages: list[HistoryMessage]


def _to_response(thread_id, result, graph):
    config = {"configurable": {"thread_id": thread_id}}
    pending = get_pending_lead_approval(graph, config)
    return ChatResponse(
        thread_id=thread_id,
        reply=result["messages"][-1].content,
        intent=result.get("intent", ""),
        lead_captured=result.get("lead_captured", False),
        lead_info=result.get("lead_info", {}),
        pending_approval=pending is not None,
        pending_lead=pending["lead_info"] if pending else None,
        sources=result.get("sources", []),
    )


@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(check_rate_limit)])
def chat(request: ChatRequest):
    """Run one turn of the agent for the given thread_id and return its reply."""
    if not request.message or not request.message.strip():
        raise HTTPException(status_code=400, detail="message must not be empty or whitespace-only")

    config = {"configurable": {"thread_id": request.thread_id}}
    result = _graph.invoke(
        {"messages": [HumanMessage(content=request.message)]},
        config=config,
    )

    if AUTO_APPROVE_DEMO_LEADS and get_pending_lead_approval(_graph, config) is not None:
        result = resume_with_lead_decision(_graph, config, approved=True)

    if result.get("lead_captured"):
        dispatch_lead_webhook(result.get("lead_info", {}), "lead.captured")

    return _to_response(request.thread_id, result, _graph)


@app.post("/chat/stream", dependencies=[Depends(check_rate_limit)])
def chat_stream(request: ChatRequest):
    """Run one turn of the agent and stream the reply tokens via Server-Sent Events (SSE)."""
    if not request.message or not request.message.strip():
        raise HTTPException(status_code=400, detail="message must not be empty or whitespace-only")

    config = {"configurable": {"thread_id": request.thread_id}}

    def event_generator():
        try:
            result = _graph.invoke(
                {"messages": [HumanMessage(content=request.message)]},
                config=config,
            )

            if AUTO_APPROVE_DEMO_LEADS and get_pending_lead_approval(_graph, config) is not None:
                result = resume_with_lead_decision(_graph, config, approved=True)

            if result.get("lead_captured"):
                dispatch_lead_webhook(result.get("lead_info", {}), "lead.captured")

            full_reply = result["messages"][-1].content

            words = full_reply.split(" ")
            for i, word in enumerate(words):
                suffix = " " if i < len(words) - 1 else ""
                token = word + suffix
                yield f"event: token\ndata: {json.dumps({'token': token})}\n\n"
                time.sleep(0.015)

            meta = _to_response(request.thread_id, result, _graph)
            yield f"event: meta\ndata: {json.dumps(meta.model_dump())}\n\n"
            yield f"event: done\ndata: {json.dumps({'status': 'done'})}\n\n"
        except Exception as exc:
            yield f"event: error\ndata: {json.dumps({'error': str(exc)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/chat/history", response_model=HistoryResponse)
def get_chat_history(thread_id: str):
    """Retrieve existing conversation message history for a given thread_id."""
    if not thread_id or not thread_id.strip():
        raise HTTPException(status_code=400, detail="thread_id must not be empty or whitespace-only")

    config = {"configurable": {"thread_id": thread_id}}
    snapshot = _graph.get_state(config)
    messages = []
    if snapshot and snapshot.values and "messages" in snapshot.values:
        for m in snapshot.values["messages"]:
            role = "user" if isinstance(m, HumanMessage) else "assistant"
            messages.append(HistoryMessage(role=role, content=m.content))

    return HistoryResponse(thread_id=thread_id, messages=messages)


@app.post("/approve", response_model=ChatResponse, dependencies=[Depends(require_admin_token), Depends(check_rate_limit)])
def approve(request: ApproveRequest):
    """Resume a thread that's paused waiting on lead approval (Admin endpoint requiring X-Admin-Token)."""
    config = {"configurable": {"thread_id": request.thread_id}}
    if get_pending_lead_approval(_graph, config) is None:
        raise HTTPException(
            status_code=409,
            detail=f"thread_id {request.thread_id!r} has no lead pending approval",
        )

    result = resume_with_lead_decision(_graph, config, request.approved)
    if result.get("lead_captured"):
        dispatch_lead_webhook(result.get("lead_info", {}), "lead.approved")
    return _to_response(request.thread_id, result, _graph)


@app.post("/confirm-lead", response_model=ChatResponse, dependencies=[Depends(check_rate_limit)])
def confirm_lead(request: ApproveRequest):
    """
    Client-facing confirmation endpoint for users to verify their own pending lead details.
    Guarded by thread existence and pending state without requiring master ADMIN_TOKEN.
    """
    config = {"configurable": {"thread_id": request.thread_id}}
    if get_pending_lead_approval(_graph, config) is None:
        raise HTTPException(
            status_code=409,
            detail=f"thread_id {request.thread_id!r} has no lead pending approval",
        )

    result = resume_with_lead_decision(_graph, config, request.approved)
    if result.get("lead_captured"):
        dispatch_lead_webhook(result.get("lead_info", {}), "lead.confirmed")
    return _to_response(request.thread_id, result, _graph)


@app.get("/api/admin/verify", dependencies=[Depends(require_admin_token)])
def verify_admin_token():
    """Verify admin token validity. Returns 200 OK if valid, 401 if invalid."""
    return {"status": "authenticated", "role": "admin"}


@app.get("/api/leads", dependencies=[Depends(check_rate_limit)])
def get_captured_leads(x_admin_token: str | None = Header(default=None)):
    """Retrieve stored leads for the AI Daily Brief. Unauthenticated requests receive masked emails."""
    from tools import list_leads, _mask_email
    try:
        raw_leads = list_leads()
        is_admin = bool(x_admin_token and secrets.compare_digest(x_admin_token, ADMIN_TOKEN))
        leads = []
        for lead in raw_leads:
            l = dict(lead)
            if not is_admin:
                l["email"] = _mask_email(l.get("email", ""))
            leads.append(l)
        return {"leads": leads, "count": len(leads)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to fetch leads: {exc}")


@app.get("/api/leads/export", dependencies=[Depends(check_rate_limit)])
def export_leads_csv(token: str | None = None, x_admin_token: str | None = Header(default=None)):
    """Export all captured leads as a RFC 4180 CSV file (Admin only via header or token param)."""
    auth_token = x_admin_token or token
    if not auth_token or not secrets.compare_digest(auth_token, ADMIN_TOKEN):
        raise HTTPException(status_code=401, detail="Missing or invalid admin token")

    from tools import list_leads
    import io
    import csv

    leads = list_leads()
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["ID", "Captured At", "Name", "Email", "Platform"])
    for lead in leads:
        writer.writerow([
            lead.get("id", ""),
            lead.get("captured_at", ""),
            lead.get("name", ""),
            lead.get("email", ""),
            lead.get("platform", ""),
        ])
    csv_bytes = output.getvalue().encode("utf-8")
    return Response(
        content=csv_bytes,
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename=autostream_leads_{int(time.time())}.csv"
        },
    )


@app.get("/api/threads", dependencies=[Depends(check_rate_limit)])
def get_recent_threads(limit: int = 10, thread_ids: str | None = None):
    """
    Retrieve list of recent conversation threads.
    If thread_ids comma-separated string is passed, scopes strictly to those user threads.
    """
    if not os.path.exists("checkpoints.sqlite"):
        return {"threads": []}

    try:
        conn = sqlite3.connect("checkpoints.sqlite")
        cursor = conn.cursor()

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_checkpoints_thread_id ON checkpoints(thread_id)")

        filter_ids = [tid.strip() for tid in thread_ids.split(",") if tid.strip()] if thread_ids else []

        if filter_ids:
            placeholders = ",".join("?" for _ in filter_ids)
            cursor.execute(
                f"""
                SELECT thread_id, metadata
                FROM checkpoints
                WHERE metadata IS NOT NULL AND thread_id IN ({placeholders})
                GROUP BY thread_id
                ORDER BY MAX(rowid) DESC
                LIMIT ?
                """,
                (*filter_ids, limit)
            )
        else:
            cursor.execute(
                """
                SELECT thread_id, metadata
                FROM checkpoints
                WHERE metadata IS NOT NULL
                GROUP BY thread_id
                ORDER BY MAX(rowid) DESC
                LIMIT ?
                """,
                (limit,)
            )
        rows = cursor.fetchall()
        conn.close()

        result = []
        for tid, meta_bytes in rows:
            title = "Conversation"
            if meta_bytes:
                try:
                    meta_str = meta_bytes.decode("utf-8") if isinstance(meta_bytes, bytes) else str(meta_bytes)
                    meta_json = json.loads(meta_str)
                    writes = meta_json.get("writes", {})
                    start_node = writes.get("__start__", {})
                    msgs = start_node.get("messages", [])
                    if msgs and isinstance(msgs, list):
                        content = msgs[0].get("kwargs", {}).get("content", "")
                        if content:
                            title = content[:38] + ("..." if len(content) > 38 else "")
                except Exception:
                    pass

            if title == "Conversation":
                title = f"Chat {tid[:8]}"

            result.append({"thread_id": tid, "title": title})

        return {"threads": result}
    except Exception as exc:
        return {"threads": []}


@app.get("/health")
def health():
    return {"status": "ok"}



