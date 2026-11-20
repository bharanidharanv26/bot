"""Shared records and the SQLite schema (no ORM dependency needed)."""

from dataclasses import dataclass
from enum import IntEnum


class Action(IntEnum):
    SHORT = 0
    DETAILED = 1
    EXAMPLE = 2


@dataclass(frozen=True)
class ActionSelection:
    state: str
    action: Action
    q_value: float
    epsilon: float
    explored: bool
    learning_enabled: bool
    learning_run_id: int


@dataclass(frozen=True)
class FeedbackResult:
    accepted: bool
    learned: bool
    reward: int
    message: str
    q_before: float | None = None
    q_after: float | None = None


# Rewards belong to feedback. The UNIQUE constraint is the final defense against
# duplicate clicks, including submissions from two browser sessions at once.
SCHEMA = """
CREATE TABLE IF NOT EXISTS learning_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);
INSERT OR IGNORE INTO learning_runs (id) VALUES (1);

CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL REFERENCES conversations(id),
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL CHECK (length(trim(content)) > 0),
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE TABLE IF NOT EXISTS response_strategies (
    response_id INTEGER PRIMARY KEY REFERENCES messages(id),
    user_message_id INTEGER NOT NULL UNIQUE REFERENCES messages(id),
    state TEXT NOT NULL,
    action INTEGER NOT NULL CHECK (action IN (0, 1, 2)),
    q_at_selection REAL NOT NULL,
    epsilon REAL NOT NULL CHECK (epsilon >= 0 AND epsilon <= 1),
    explored INTEGER NOT NULL CHECK (explored IN (0, 1)),
    learning_enabled INTEGER NOT NULL CHECK (learning_enabled IN (0, 1)),
    learning_run_id INTEGER NOT NULL REFERENCES learning_runs(id),
    model TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    response_id INTEGER NOT NULL UNIQUE REFERENCES response_strategies(response_id),
    label TEXT NOT NULL,
    reward INTEGER NOT NULL,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    CHECK (
        (label = 'Very helpful' AND reward = 2) OR
        (label = 'Helpful' AND reward = 1) OR
        (label = 'Neutral' AND reward = 0) OR
        (label = 'Not helpful' AND reward = -1)
    )
);

CREATE TABLE IF NOT EXISTS state_action_values (
    learning_run_id INTEGER NOT NULL REFERENCES learning_runs(id),
    state TEXT NOT NULL,
    action INTEGER NOT NULL CHECK (action IN (0, 1, 2)),
    q_value REAL NOT NULL CHECK (q_value >= -1 AND q_value <= 2),
    update_count INTEGER NOT NULL CHECK (update_count > 0),
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    PRIMARY KEY (learning_run_id, state, action)
);

CREATE TABLE IF NOT EXISTS learning_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    feedback_id INTEGER NOT NULL UNIQUE REFERENCES feedback(id),
    learning_run_id INTEGER NOT NULL REFERENCES learning_runs(id),
    state TEXT NOT NULL,
    action INTEGER NOT NULL CHECK (action IN (0, 1, 2)),
    reward INTEGER NOT NULL CHECK (reward IN (-1, 0, 1, 2)),
    alpha REAL NOT NULL CHECK (alpha > 0 AND alpha <= 1),
    q_before REAL NOT NULL,
    q_after REAL NOT NULL,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id, id);
CREATE INDEX IF NOT EXISTS idx_strategies_run ON response_strategies(learning_run_id);
CREATE INDEX IF NOT EXISTS idx_events_run ON learning_events(learning_run_id, id);
"""
