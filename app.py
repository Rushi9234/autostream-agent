"""
Gradio chat UI for the AutoStream agent.

The CLI (main.py) and API (api.py) both need something else installed/run to
try, which is a real barrier for anyone who isn't the person who built this.
This file exists so the whole agent — routing, RAG, and the lead-capture flow
— can be tried from a browser with nothing installed, including on Hugging
Face Spaces' free tier.

Run locally with: python app.py
To deploy: push this repo to a Hugging Face Space (SDK: Gradio) and set the
GOOGLE_API_KEY secret in the Space settings.
"""

import uuid

import gradio as gr
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from agent import build_graph, get_pending_lead_approval, resume_with_lead_decision

load_dotenv()

_graph = build_graph()


def respond(message, history, thread_id):
    # Each browser session gets its own thread_id (see gr.State below), so the
    # SqliteSaver checkpointer keeps every visitor's conversation separate.
    config = {"configurable": {"thread_id": thread_id}}
    result = _graph.invoke(
        {"messages": [HumanMessage(content=message)]},
        config=config,
    )
    reply = result["messages"][-1].content

    # This chat widget has no admin view for a human to approve/reject leads
    # (that's what api.py's /approve endpoint is for), so this public demo
    # auto-approves instead of leaving every visitor's signup stuck pending
    # forever. A real deployment would route this to an actual reviewer.
    pending = get_pending_lead_approval(_graph, config)
    if pending:
        result = resume_with_lead_decision(_graph, config, approved=True)
        reply = result["messages"][-1].content

    return reply


with gr.Blocks(title="AutoStream Agent") as demo:
    gr.Markdown(
        "# AutoStream Agent\n"
        "A LangGraph agent for a demo SaaS product. Ask about pricing/features, "
        "say hi, or say you'd like to sign up to see the lead-capture flow."
    )
    # Callable default -> Gradio generates a fresh UUID per browser session
    # instead of every visitor sharing one thread_id.
    thread_state = gr.State(lambda: str(uuid.uuid4()))
    chatbot = gr.ChatInterface(
        fn=respond,
        additional_inputs=[thread_state],
        examples=[
            ["Hi, tell me about your pricing."],
            ["What's your refund policy?"],
            ["I want to try the Pro plan for my YouTube channel."],
        ],
    )

if __name__ == "__main__":
    demo.launch()
