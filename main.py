"""
CLI entry point for the AutoStream agent.

This is the simplest possible front-end for the graph in agent.py — a plain
REPL loop, mainly so I could test the routing/RAG/lead-capture logic by hand
before wiring up the FastAPI (api.py) and Gradio (app.py) front-ends.

Run `python main.py` to start a REPL chat. Type 'exit' or 'quit' to leave.
"""

import os
import uuid

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from agent import build_graph, get_pending_lead_approval, resume_with_lead_decision


def main():
    load_dotenv()

    provider = os.getenv("LLM_PROVIDER", "gemini").lower().strip()

    # Figure out which key the selected provider needs.
    required_keys = {
        "gemini": ["GOOGLE_API_KEY"],
        "openai": ["OPENAI_API_KEY"],
        "anthropic": ["ANTHROPIC_API_KEY", "GOOGLE_API_KEY"],  # Anthropic uses Gemini embeddings
    }.get(provider, ["GOOGLE_API_KEY"])

    missing = [k for k in required_keys if not os.getenv(k)]
    if missing:
        print(
            f"ERROR: LLM_PROVIDER={provider!r} needs these env vars, "
            f"but they aren't set: {missing}\n"
            "Copy .env.example to .env and fill in the required key(s)."
        )
        return

    graph = build_graph()

    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    print("=" * 60)
    print("  AutoStream Assistant")
    print("  Ask about pricing, features, or say hi!")
    print("  (type 'exit' to quit)")
    print("=" * 60)

    first_turn = True

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        if first_turn:
            initial_state = {
                "messages": [HumanMessage(content=user_input)],
                "intent": "",
                "lead_info": {},
                "lead_captured": False,
            }
            result = graph.invoke(initial_state, config=config)
            first_turn = False
        else:
            result = graph.invoke(
                {"messages": [HumanMessage(content=user_input)]},
                config=config,
            )

        # The assistant's most recent reply is the last message in state.
        reply = result["messages"][-1].content
        intent = result.get("intent", "")
        print(f"\nAssistant [{intent}]: {reply}")

        # If the graph paused at escalate_node, play the human-reviewer role
        # right here (in a real deployment this would be a separate person
        # using api.py's /approve endpoint, not the same terminal).
        pending = get_pending_lead_approval(graph, config)
        if pending:
            lead_info = pending["lead_info"]
            print("\n--- Human review required before this lead is captured ---")
            print(f"  Name: {lead_info['name']}")
            print(f"  Email: {lead_info['email']}")
            print(f"  Platform: {lead_info['platform']}")
            decision = input("  Approve this lead? [y/n]: ").strip().lower()
            approved = decision.startswith("y")

            result = resume_with_lead_decision(graph, config, approved)
            reply = result["messages"][-1].content
            print(f"\nAssistant [lead_review]: {reply}")


if __name__ == "__main__":
    main()
