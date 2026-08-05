"""
Small evaluation harness for the intent classifier in agent.route_turn.

I wanted evidence that the classifier actually works, not just a vibe from
manual testing, so this hand-labels a set of example messages and reports
accuracy plus a confusion matrix. It's deliberately tiny (a few dozen
examples) — the point is a repeatable check, not a benchmark suite.

Run with: python eval_intents.py
"""

import os

from dotenv import load_dotenv

from agent import invoke_llm
from prompts import INTENT_CLASSIFIER_PROMPT

LABELS = ("casual", "inquiry", "high_intent")

# (message, expected_label). History is intentionally empty for all of these —
# each one should be classifiable from the message alone.
EXAMPLES = [
    ("Hey there!", "casual"),
    ("Hi, how are you?", "casual"),
    ("Good morning", "casual"),
    ("Thanks, that's helpful", "casual"),
    ("Thank you so much!", "casual"),
    ("Bye, talk soon", "casual"),
    ("Nice, appreciate it", "casual"),
    ("lol okay", "casual"),
    ("Just saying hi", "casual"),
    ("What's up", "casual"),
    ("How much does the Pro plan cost?", "inquiry"),
    ("What's included in the Basic plan?", "inquiry"),
    ("Do you support 4K exports?", "inquiry"),
    ("What's your refund policy?", "inquiry"),
    ("Is there 24/7 support?", "inquiry"),
    ("What's the difference between Basic and Pro?", "inquiry"),
    ("Can I get a refund after 10 days?", "inquiry"),
    ("Does the Basic plan include captions?", "inquiry"),
    ("How many videos can I export per month on Pro?", "inquiry"),
    ("What platforms do you support?", "inquiry"),
    ("I want to sign up for the Pro plan", "high_intent"),
    ("Sign me up please", "high_intent"),
    ("I'd like to try AutoStream for my YouTube channel", "high_intent"),
    ("Let's get started, I want the Pro plan", "high_intent"),
    ("I want to buy the Basic plan", "high_intent"),
    ("Can you set me up with an account?", "high_intent"),
    ("I'm ready to subscribe", "high_intent"),
    ("Yes, I want to try the Pro plan for my TikTok channel", "high_intent"),
    ("Get me registered for Pro", "high_intent"),
    ("I want in, sign me up for Basic", "high_intent"),
]


def classify(message):
    prompt = INTENT_CLASSIFIER_PROMPT.format(history="(no prior messages)", user_message=message)
    raw = invoke_llm(prompt)
    if raw is None:
        return "casual"  # matches route_turn's fallback-on-failure behavior
    raw = raw.lower()
    if "high_intent" in raw:
        return "high_intent"
    if "inquiry" in raw:
        return "inquiry"
    return "casual"


def run_eval():
    load_dotenv()
    if not os.getenv("GOOGLE_API_KEY"):
        print("ERROR: GOOGLE_API_KEY not set. See .env.example.")
        return

    matrix = {actual: {predicted: 0 for predicted in LABELS} for actual in LABELS}
    correct = 0
    misses = []

    for message, expected in EXAMPLES:
        predicted = classify(message)
        matrix[expected][predicted] += 1
        if predicted == expected:
            correct += 1
        else:
            misses.append((message, expected, predicted))

    total = len(EXAMPLES)
    accuracy = correct / total

    print("=" * 60)
    print(f"  Intent classifier eval — {correct}/{total} correct ({accuracy:.1%})")
    print("=" * 60)

    print("\nConfusion matrix (rows = expected, columns = predicted):")
    header = " " * 14 + "".join(f"{label:>12}" for label in LABELS)
    print(header)
    for actual in LABELS:
        row = "".join(f"{matrix[actual][predicted]:>12}" for predicted in LABELS)
        print(f"{actual:<14}{row}")

    if misses:
        print("\nMisclassified examples:")
        for message, expected, predicted in misses:
            print(f'  "{message}" -> expected {expected}, got {predicted}')
    else:
        print("\nNo misclassifications.")


if __name__ == "__main__":
    run_eval()
