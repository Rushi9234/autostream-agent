# AutoStream Agent

**Demo video:** https://drive.google.com/drive/folders/1rtzM5f_DhSS6CN9Krr4fgQyCOE-T1xGP?usp=sharing

A conversational AI agent I built to learn how to design a real multi-turn,
stateful LLM agent instead of a single prompt-in/text-out wrapper — intent
routing, retrieval-augmented answering, structured multi-turn slot-filling,
and a persisted tool call, all as an explicit **LangGraph** state machine.

The agent sits in front of **AutoStream**, a reference/demo SaaS product for
automated video editing (pricing, features, and policies are all sample data
in `knowledge_base/autostream_kb.md` — the point of this project is the agent
architecture, not a real business). It:

- Classifies each incoming message into `casual`, `inquiry`, or `high_intent`
- Answers product questions using **RAG** over a local knowledge base (FAISS)
- Runs a guided, multi-turn signup flow that extracts name/email/platform,
  then **pauses for human approval** (LangGraph's `interrupt()`) before the
  lead-capture tool ever fires — see "Human-in-the-Loop" below
- Persists both the conversation state and captured leads to disk (SQLite),
  so a restart doesn't lose in-progress conversations or captured leads
- Logs every node transition (which node ran, resulting intent, latency) to
  `agent.log`, and masks PII in anything that hits stdout/logs

Built with **LangGraph**, **Gemini 2.5 Flash-Lite**, and **FAISS**. Originally
built against Gemini 1.5 Flash; migrated to 2.5 Flash-Lite after Google
retired the 1.5 model family.

## Try it

```bash
uvicorn api:app --reload
```

then open **http://127.0.0.1:8000** for the custom chat UI (`static/`) — a
small vanilla HTML/CSS/JS front end built specifically for this project
instead of a framework's default theme (see "Frontend" below), served
directly by `api.py`.

Or skip the UI and hit the API directly:

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"thread_id": "demo-1", "message": "Hi, tell me about your pricing."}'
```

There's also a Gradio UI (`python app.py`, http://127.0.0.1:7860) and a plain
CLI REPL (`python main.py`) if you'd rather not run a browser front end at
all.

## How to Run Locally

You need Python 3.9+ and a free Gemini API key from
[Google AI Studio](https://aistudio.google.com/app/apikey).

**1. Clone the repo and enter the folder**

```bash
git clone https://github.com/Rushi9234/autostream-agent.git
cd autostream-agent
```

**2. Create a virtual environment and install dependencies**

```bash
python -m venv .venv
source .venv/bin/activate          # On Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

**3. Add your Gemini API key**

```bash
cp .env.example .env               # On Windows: copy .env.example .env
```

Open the `.env` file and paste your key after `GOOGLE_API_KEY=`.

**4. Run one of the front-ends above.**

For a quick end-to-end check without typing anything, there's also a scripted
walkthrough:

```bash
python demo.py
```

which runs a fixed example conversation, auto-approves the resulting lead
(see "Human-in-the-Loop Lead Review" below for what that means), and ends
with something like:

```
[Auto-approving demo lead: Alex <alex@example.com>]
Assistant [lead_review]: Perfect, you're all set, Alex! ...
```

That row is now a real one in `leads.db` — query it with `python tools.py`.

> **Free-tier note:** Gemini's free tier caps `gemini-2.5-flash-lite` at a
> small number of requests per day. `demo.py` sleeps between turns to help
> stay under the per-minute limit, but the *daily* cap is separate and much
> harder to work around — if you see `429 ResourceExhausted` errors, that's
> the daily quota, not a bug in this project.

## Project Structure

```
autostream-agent/
├── main.py                   # CLI REPL front-end (incl. lead-approval prompt)
├── app.py                    # Gradio chat UI (local or Hugging Face Spaces)
├── api.py                    # FastAPI wrapper (POST /chat, POST /approve, serves static/)
├── static/                   # custom chat UI (vanilla HTML/CSS/JS, no framework)
│   ├── index.html
│   ├── style.css
│   └── script.js
├── demo.py                   # scripted end-to-end walkthrough
├── eval_intents.py           # intent-classifier accuracy + confusion matrix
├── agent.py                  # LangGraph: state, nodes, routing, HITL, logging
├── rag.py                    # FAISS index (persisted to disk) and retrieval
├── tools.py                  # SQLite-backed lead capture
├── prompts.py                # prompt templates
├── tests/                    # pytest unit tests (no live API calls)
├── knowledge_base/
│   └── autostream_kb.md      # sample pricing, features, policies
├── .github/workflows/
│   └── tests.yml             # CI: runs pytest on every push/PR
├── requirements.txt
├── requirements-dev.txt      # pytest, black, ruff
├── Dockerfile
├── .env.example
├── .gitignore
└── README.md
```

## Frontend

`static/` is a small, dependency-free HTML/CSS/JS chat UI served directly by
`api.py` (`GET /` → `static/index.html`, `/static/*` → the CSS/JS) — no
framework, no build step, no Gradio default theme.

The design is grounded in what AutoStream actually is (automated video
editing) rather than a generic "AI chatbot" look: a dark, flat panel
aesthetic borrowed from NLE software (Premiere/Resolve/Final Cut), a coral
"record dot" instead of a blue status icon, monospace timecodes on every
message, and a timeline-tick divider instead of a plain rule. Both a light
and a dark theme are defined (`prefers-color-scheme`, plus a manual toggle in
the top bar) — the dark theme is closer to how an actual editing suite looks,
the light one for anyone who'd rather not.

It talks to `/chat` and `/approve` with plain `fetch()` calls — open
`static/script.js` to see the whole client; there's nothing hidden behind a
bundler.

## Architecture

The agent is a **LangGraph state machine**. Every turn enters a single
`route_turn` node which classifies the user's latest message into one of
three intents (`casual`, `inquiry`, `high_intent`). Based on that intent, a
conditional edge routes the turn to exactly one handler node:

- `casual_node` — replies briefly to greetings and small talk.
- `rag_node` — does a FAISS similarity search over the markdown knowledge
  base and produces a grounded answer (the prompt explicitly refuses to
  invent prices, features, or policies it can't find in the retrieved
  context).
- `lead_node` — extracts any name/email/platform values from the user's
  message, merges them with whatever's already collected, and either asks
  for what's still missing or, once all three are present, hands off to...
- `escalate_node` — pauses the graph with LangGraph's `interrupt()` and waits
  for a human decision before the lead-capture tool ever fires. See
  "Human-in-the-Loop Lead Review" below.

One detail that isn't obvious from a basic LangGraph tutorial: if we're
already mid-way through collecting lead details, `route_turn` skips
re-classification entirely and stays on the lead path. Without that guard, a
bare reply like `"Alex"` (just a name) gets classified as casual small talk
and the signup flow silently breaks.

**Why LangGraph over a plain prompt loop or AutoGen:** this is a
single-agent workflow with conditional branching and one tool trigger — no
need for multiple agents coordinating with each other (which is what AutoGen
is built for). LangGraph's explicit state schema and graph structure make the
routing logic inspectable and testable in a way a single big prompt isn't.

**State management:** state is a `TypedDict` with four fields — `messages`,
`intent`, `lead_info`, `lead_captured`. Conversation memory is handled by
LangGraph's `SqliteSaver` checkpointer, keyed by a `thread_id` per chat
session, so multi-turn memory (and in-progress signups) survive a process
restart. This replaced the original `MemorySaver`, which kept everything in
process RAM only.

## Human-in-the-Loop Lead Review

Once `lead_node` has collected name, email, and platform, the graph doesn't
call the capture tool itself — it pauses at `escalate_node` via LangGraph's
`interrupt()` and waits for an explicit approve/reject decision. This matters
because "if all 3 fields are present, save it" means the *agent* alone
decides who counts as a customer; a real deployment (or a resume line
claiming "agentic AI") should have a human in that loop for anything that
writes to a real CRM or bills a real card.

While paused, the state (including the pending `lead_info`) is safely
checkpointed to `checkpoints.sqlite` — the process can restart and the thread
resumes exactly where it left off.

- **CLI (`main.py`)**: prompts you right there in the terminal — `Approve
  this lead? [y/n]`.
- **API (`api.py`)**: `/chat` returns `pending_approval: true` and the
  `pending_lead` details instead of finalizing anything; call `POST /approve`
  with `{"thread_id": ..., "approved": true/false}` to resume — this is the
  shape a real admin panel or Slack-button reviewer would call.
- **Gradio (`app.py`)**: auto-approves, since a public chat widget has no
  reviewer UI behind it — see the code comment there for why that's an
  honest simplification, not a silent gap.
- **`demo.py`**: auto-approves too, so the scripted walkthrough completes
  unattended, but it goes through the exact same `interrupt()`/resume path.

## What Changed From the First Version

This started as a smaller exercise and I kept coming back to harden it once
the core routing/RAG/extraction logic was working. The main upgrades:

- **Lead capture is real now.** `tools.py` used to just `print()` a string.
  It now writes to a SQLite table (`leads.db`) via `mock_lead_capture()` —
  kept the original function name since `agent.py` calls it unchanged, but
  it's a real persisted row now, not a mock.
- **Conversation memory survives a restart.** Swapped `MemorySaver` (RAM
  only) for `SqliteSaver` (`checkpoints.sqlite`).
- **The FAISS index is cached to disk.** `rag.py` used to rebuild (and
  re-embed) the whole knowledge base on every process start. It now saves the
  index under `index/` and only rebuilds when the KB file's content hash
  changes.
- **LLM calls don't crash the whole turn anymore.** Every call into Gemini
  goes through `agent.invoke_llm()`, which retries transient failures with
  backoff and returns `None` on total failure so each node can fall back to a
  safe default instead of raising.
- **There's an actual evaluation artifact.** `eval_intents.py` runs ~30
  hand-labeled example messages through the intent classifier and reports
  accuracy plus a confusion matrix — see below for how to run it.
- **Three front-ends now, not one.** `main.py` (CLI) was the only way to try
  this. Added `api.py` (FastAPI, for wiring into WhatsApp/Slack/anything
  webhook-based) and `app.py` (Gradio, for a link anyone can click).
- **Basic dev tooling.** `requirements-dev.txt` (pytest/black/ruff) and a
  `tests/` directory covering the routing guard, JSON extraction, retry
  logic, and lead persistence — none of which need a live API key to run.
- **Human-in-the-loop lead review.** The capture tool no longer fires
  automatically — `escalate_node` pauses the graph with `interrupt()` and
  waits for an approve/reject decision. See the dedicated section above.
- **Observability.** Every node logs its name, resulting intent, and latency
  to `agent.log` (see `agent.log_node`); PII (emails) is masked in anything
  that hits stdout/logs, even though the full value is still stored in
  `leads.db`.
- **CI.** `.github/workflows/tests.yml` runs the full `pytest` suite on every
  push/PR — no API key/secret needed since the tests never call a live LLM.

## Evaluating the Intent Classifier

```bash
python eval_intents.py
```

This runs ~30 hand-labeled example messages (10 per label) through the same
classification path `route_turn` uses, and prints accuracy plus a confusion
matrix, e.g.:

```
Confusion matrix (rows = expected, columns = predicted):
                    casual     inquiry high_intent
casual                  10           0           0
inquiry                  0          10           0
high_intent               0           0          10
```

Note: Gemini's free tier caps requests per day, and this script makes ~30
calls in one run — if you hit a `429`, it's the daily quota, not a failure of
the script. Run it once you've got quota headroom and read off the actual
numbers; I'd rather point you at the script than paste a number here that
might not match what you see with your own key/quota.

## Running Tests

```bash
pip install -r requirements-dev.txt
pytest tests/
```

The test suite covers the routing guard, JSON extraction from LLM output,
the LLM retry/fallback logic, and SQLite lead persistence — all without
hitting a live API, so it runs the same with or without a configured key.

## Deploying

**FastAPI, anywhere that runs Python:**

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

**Docker:**

```bash
docker build -t autostream-agent .
docker run -p 8000:8000 --env-file .env \
  -v autostream-leads:/app/leads.db \
  -v autostream-checkpoints:/app/checkpoints.sqlite \
  -v autostream-index:/app/index \
  autostream-agent
```

(those volumes keep captured leads, conversation state, and the FAISS index
across container restarts — without them every container start is a clean
slate)

**Gradio, on Hugging Face Spaces:** push this repo to a new Space with the
Gradio SDK, set `GOOGLE_API_KEY` as a Space secret, and `app.py` is picked up
automatically as the entry point.

## Extending to a Real Channel (e.g. WhatsApp)

To put this behind the WhatsApp Cloud API (webhook-based):

1. **Reuse `api.py`'s shape**, but add two endpoints instead of one:
   - `GET /webhook` for Meta's verification handshake — echo back
     `hub.challenge` if `hub.verify_token` matches what's configured in the
     Meta app dashboard.
   - `POST /webhook` to receive each incoming message (JSON with the
     sender's phone number and message text).
2. **Use the phone number as the `thread_id`.** `api.py` already threads a
   `thread_id` through to the graph's checkpointer — using the sender's
   number as that ID gives every user their own persistent conversation with
   no agent-code changes.
3. **Reply via the WhatsApp Cloud API's messages endpoint**
   (`https://graph.facebook.com/v20.0/<phone_number_id>/messages`) with the
   access token in the `Authorization` header.
4. **Acknowledge the webhook immediately** (200 response) and process the
   message in a background task, since Meta retries webhooks that respond
   slowly.

## What's Deliberately Out of Scope (for now)

- **Streaming responses** — every reply currently blocks until the full LLM
  response is back.
- **Vector DB beyond FAISS** — FAISS is enough at this scale; a thin
  retriever interface with a Chroma/Qdrant backend would be the next step if
  the knowledge base grew.
- **A real reviewer UI for lead approval** — `api.py`'s `/approve` endpoint
  is the right shape for this, but there's no admin panel calling it yet;
  today that's the CLI prompt, or auto-approval in the demo/Gradio front-ends.

## Notes

- The knowledge base is a single markdown file
  (`knowledge_base/autostream_kb.md`), split on markdown headers (H1/H2/H3)
  so pricing, features, and policies become separate retrievable chunks.
  It's intentionally simple/synthetic — swapping in a real document set would
  only touch `rag.py`.
- The lead-capture tool is guarded: it only fires once name, email, and
  platform have all been collected. Until then the agent keeps asking for
  whatever's missing.
- Temperature is set to 0.2 across all nodes to keep replies consistent.
