# AutoStream - Social-to-Lead Agentic Workflow
**Demo video:** https://drive.google.com/drive/folders/1rtzM5f_DhSS6CN9Krr4fgQyCOE-T1xGP?usp=sharing

This is my submission for the ServiceHive ML Internship assignment. It is a
conversational AI agent for a fictional SaaS product called **AutoStream**
(automated video editing for content creators). The agent classifies user
intent, answers product questions using a local knowledge base (RAG), and
captures high-intent leads by calling a mock backend tool.

Built with **LangGraph**, **Gemini 2.5 Flash-Lite**, and **FAISS**.

> **Note on the LLM choice:** The assignment spec listed Gemini 1.5 Flash, but
> Google has retired the entire 1.5 model family (it now returns a 404 from
> the API). This project uses **Gemini 2.5 Flash-Lite** instead, which is
> Google's current free-tier successor in the same family — cheap, fast, and
> more than capable for intent classification and lightweight RAG.

## Features

- Intent classification into `casual`, `inquiry`, and `high_intent`
- RAG pipeline over a local markdown knowledge base (pricing, features, policies)
- Lead capture tool that only fires after collecting name, email, and platform
- Memory across multiple conversation turns using LangGraph's `MemorySaver`

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

**4. Run the chat**

```bash
python main.py
```

You can try the example conversation from the assignment:

```
You: Hi, tell me about your pricing.
You: I want to try the Pro plan for my YouTube channel.
You: My name is Alex, email is alex@example.com
```

Once the agent has all three fields (name, email, platform), it will call
the mock tool and you should see:

```
Lead captured successfully: Alex, alex@example.com, YouTube
```

There is also a `demo.py` script that runs the assignment's example
conversation automatically (useful for recording the demo video):

```bash
python demo.py
```

## Project Structure

```
autostream-agent/
├── main.py                   # CLI chat loop
├── demo.py                   # runs the spec example conversation
├── agent.py                  # LangGraph: state, nodes, routing
├── rag.py                    # FAISS index and retrieval
├── tools.py                  # mock_lead_capture function
├── prompts.py                # prompt templates
├── knowledge_base/
│   └── autostream_kb.md      # pricing, features, policies
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Architecture Explanation

The agent is built as a **LangGraph state machine**. Every turn enters a
single `route_turn` node which classifies the user's latest message into
one of three intents (`casual`, `inquiry`, `high_intent`). Based on that
intent, a conditional edge routes the turn to exactly one handler node:

- `casual_node` - replies briefly to greetings and small talk.
- `rag_node` - does a FAISS similarity search over the markdown knowledge
  base and produces a grounded answer.
- `lead_node` - extracts any name/email/platform values from the user's
  message. If all three are now collected, it calls `mock_lead_capture()`.
  Otherwise it asks for whatever is still missing.

One important detail: if we're already mid-way through collecting lead
details, the router skips classification and stays on the lead path.
Without this, a short reply like "Alex" (just a name) gets misclassified
as casual and the signup flow breaks.

**Why LangGraph over AutoGen:** AutoGen is designed for multi-agent
conversations where multiple agents coordinate with each other. This
assignment is a single-agent workflow with conditional branching and one
tool trigger, so LangGraph's explicit state schema and graph structure
fit the problem more naturally.

**State management:** The state is a `TypedDict` with four fields:
`messages`, `intent`, `lead_info`, and `lead_captured`. Conversation memory
is handled by LangGraph's `MemorySaver` checkpointer, keyed by a unique
`thread_id` for each chat session. This gives us the 5-6 turn memory the
assignment requires without extra work.

## WhatsApp Deployment

To integrate this agent with WhatsApp, I would use the **WhatsApp Cloud
API**, which is webhook-based. The plan:

1. **Wrap the agent in a web server** (e.g. FastAPI) with two endpoints:
   - `GET /webhook` for Meta's verification handshake. Meta sends
     `hub.verify_token` and `hub.challenge` query parameters. If the token
     matches the one we configured in the Meta app dashboard, we echo back
     the challenge.
   - `POST /webhook` to receive every incoming user message. The payload is
     JSON containing the sender's phone number and the message text.

2. **Use the phone number as the `thread_id`.** This is the key part. In
   `main.py` we already pass a `thread_id` in the LangGraph config. If we
   use the sender's phone number as that ID, every user gets their own
   persistent conversation state automatically, without changing any agent
   code.

3. **Send the reply back** by calling the WhatsApp Cloud API's messages
   endpoint (`https://graph.facebook.com/v20.0/<phone_number_id>/messages`)
   with the access token in the Authorization header.

4. **For production**, I would also replace `MemorySaver` with a persistent
   checkpointer (like `SqliteSaver`) so conversation state survives server
   restarts, and acknowledge the webhook with a 200 response immediately
   while processing the message in a background task (since Meta retries
   webhooks if the response is slow).

## Notes

- The knowledge base is stored as a simple markdown file in
  `knowledge_base/autostream_kb.md`. It's split on markdown headers
  (H1/H2/H3) so pricing, features, and policies become separate chunks.
- The lead-capture tool is guarded: it will only fire when all three
  fields (name, email, platform) have been collected. Until then, the
  agent keeps asking for the missing ones.
- Temperature is set to 0.2 to keep replies consistent.
