"""Map text to 12 broad contexts, without putting raw text in the policy table."""

import re


LENGTH_BUCKETS = ("short", "medium", "long")
INTENT_CATEGORIES = ("general", "question", "explanation", "how_to")
ALL_STATES = tuple(
    f"{length}|{intent}" for length in LENGTH_BUCKETS for intent in INTENT_CATEGORIES
)


def generate_state(message: str) -> str:
    if not isinstance(message, str) or not message.strip():
        raise ValueError("A state requires a non-empty message.")
    cleaned = message.strip()
    text = cleaned.casefold()
    length = "short" if len(cleaned) <= 80 else "medium" if len(cleaned) <= 240 else "long"
    # Ordered rules: a request for steps is more specific than a generic question.
    if re.search(r"\b(how\s+(to|do|can|should)|steps?|tutorial)\b", text):
        intent = "how_to"
    elif re.search(r"\b(explain|why|describe|difference|compare)\b", text):
        intent = "explanation"
    elif "?" in text or re.match(r"^(what|who|when|where|which|is|are|can|does)\b", text):
        intent = "question"
    else:
        intent = "general"
    return f"{length}|{intent}"
