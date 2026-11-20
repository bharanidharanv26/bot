"""Only explicit, recognized ratings are rewards. Missing ratings are not zero."""


REWARD_MAP = {
    "Very helpful": 2,
    "Helpful": 1,
    "Neutral": 0,
    "Not helpful": -1,
}


def reward_for_feedback(label: str) -> int:
    if not isinstance(label, str) or label not in REWARD_MAP:
        raise ValueError("Choose Very helpful, Helpful, Neutral, or Not helpful.")
    return REWARD_MAP[label]
