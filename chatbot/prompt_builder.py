"""The chosen bandit action changes the actual system prompt sent to Ollama."""

from collections.abc import Mapping, Sequence

from database.models import Action


STRATEGY_INSTRUCTIONS = {
    Action.SHORT: "Give a short, direct answer in one to three sentences. Avoid extra detail unless essential.",
    Action.DETAILED: "Give a detailed explanation. Explain the reasoning in a few clear paragraphs or numbered steps, within about 220 words.",
    Action.EXAMPLE: "Explain using one concrete, appropriate example. Connect the example to the answer, within about 180 words.",
}


def build_messages(
    user_message: str,
    action: Action,
    history: Sequence[Mapping] = (),
    history_char_budget: int = 4000,
) -> list[dict[str, str]]:
    if not user_message.strip():
        raise ValueError("The message cannot be empty.")
    if history_char_budget < 0:
        raise ValueError("History budget cannot be negative.")
    action = Action(action)
    system = (
        "You are a helpful local chatbot. Answer the user's question accurately and in their language. "
        "If you do not know, say so. Do not invent facts. "
        "Follow the requested response style without mentioning the strategy or feedback system.\n"
        f"Response style ({action.name}): {STRATEGY_INSTRUCTIONS[action]}"
    )
    # Keep recent complete messages rather than silently slicing their text.
    recent = []
    used = 0
    for message in reversed(history):
        if message["role"] not in ("user", "assistant"):
            continue
        content = str(message["content"])
        if used + len(content) > history_char_budget:
            break
        recent.append({"role": message["role"], "content": content})
        used += len(content)
    recent.reverse()
    # Do not start history with an assistant answer whose question was dropped.
    while recent and recent[0]["role"] != "user":
        recent.pop(0)
    return [
        {"role": "system", "content": system},
        *recent,
        {"role": "user", "content": user_message.strip()},
    ]
