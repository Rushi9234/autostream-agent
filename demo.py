"""
Runs a canonical example conversation end-to-end automatically.

Useful for a quick smoke test of the whole graph (routing -> RAG -> lead
capture) without typing each turn by hand, or for recording a demo clip.
"""

import os
import time
import uuid

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from agent import build_graph, get_pending_lead_approval, resume_with_lead_decision


SCRIPT = [
    "Hey there!",
    "Hi, tell me about your pricing.",
    "That sounds good, I want to try the Pro plan for my YouTube channel.",
    "I'm Alex, my email is alex@example.com",
]


def main():
    load_dotenv()

    if not os.getenv("GOOGLE_API_KEY"):
        print("ERROR: GOOGLE_API_KEY not set. See .env.example.")
        return

    graph = build_graph()
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    print("=" * 60)
    print("  AutoStream Agent - Automated Demo")
    print("=" * 60)

    first_turn = True
    result = None
    for user_input in SCRIPT:
        # Small delay to stay under the Gemini free tier rate limit.
        time.sleep(5)
        print(f"\nYou: {user_input}")

        if first_turn:
            state = {
                "messages": [HumanMessage(content=user_input)],
                "intent": "",
                "lead_info": {},
                "lead_captured": False,
            }
            result = graph.invoke(state, config=config)
            first_turn = False
        else:
            result = graph.invoke(
                {"messages": [HumanMessage(content=user_input)]},
                config=config,
            )

        intent = result.get("intent", "")
        print(f"\nAssistant [{intent}]: {result['messages'][-1].content}")

        # This script always auto-approves so the demo runs unattended, but
        # it goes through the real interrupt()/resume path — a live deployment
        # would swap this for an actual human clicking approve/reject.
        pending = get_pending_lead_approval(graph, config)
        if pending:
            lead_info = pending["lead_info"]
            print(f"\n[Auto-approving demo lead: {lead_info['name']} <{lead_info['email']}>]")
            result = resume_with_lead_decision(graph, config, approved=True)
            print(f"\nAssistant [lead_review]: {result['messages'][-1].content}")

    print("\n" + "=" * 60)
    print("  Demo complete. Final state:")
    print(f"  lead_info     = {result.get('lead_info')}")
    print(f"  lead_captured = {result.get('lead_captured')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
