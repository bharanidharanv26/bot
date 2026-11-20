"""Short-lived connections keep SQLite safe across Streamlit reruns and threads."""

from contextlib import contextmanager
from pathlib import Path
import sqlite3
from typing import Iterator
from uuid import uuid4

from database.models import ActionSelection, SCHEMA


class Database:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=15)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(SCHEMA)

    @staticmethod
    def active_run(connection: sqlite3.Connection) -> int:
        return int(connection.execute("SELECT MAX(id) FROM learning_runs").fetchone()[0])

    def current_run_id(self) -> int:
        with self.connection() as connection:
            return self.active_run(connection)

    def reset_learning(self) -> int:
        """A fresh run has implicit zero Q-values; previous history is retained."""
        with self.connection() as connection:
            cursor = connection.execute("INSERT INTO learning_runs DEFAULT VALUES")
            return int(cursor.lastrowid)

    def create_conversation(self) -> str:
        conversation_id = str(uuid4())
        with self.connection() as connection:
            connection.execute("INSERT INTO conversations (id) VALUES (?)", (conversation_id,))
        return conversation_id

    def list_conversations(self) -> list[dict]:
        with self.connection() as connection:
            rows = connection.execute(
                """SELECT c.id, c.created_at,
                          (SELECT substr(m.content, 1, 45) FROM messages m
                           WHERE m.conversation_id = c.id AND m.role = 'user'
                           ORDER BY m.id LIMIT 1) AS title
                   FROM conversations c ORDER BY c.created_at DESC, c.rowid DESC"""
            ).fetchall()
        return [dict(row) for row in rows]

    def save_interaction(
        self,
        conversation_id: str,
        user_message: str,
        response: str,
        selection: ActionSelection,
        model: str,
    ) -> int:
        """Store only successful generations, as one complete, atomic chat turn."""
        if not user_message.strip() or not response.strip() or not model.strip():
            raise ValueError("A chat turn needs a user message, response, and model name.")
        with self.connection() as connection:
            user_id = connection.execute(
                "INSERT INTO messages (conversation_id, role, content) VALUES (?, 'user', ?)",
                (conversation_id, user_message.strip()),
            ).lastrowid
            response_id = connection.execute(
                "INSERT INTO messages (conversation_id, role, content) VALUES (?, 'assistant', ?)",
                (conversation_id, response.strip()),
            ).lastrowid
            connection.execute(
                """INSERT INTO response_strategies
                   (response_id, user_message_id, state, action, q_at_selection, epsilon,
                    explored, learning_enabled, learning_run_id, model)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    response_id, user_id, selection.state, int(selection.action),
                    selection.q_value, selection.epsilon, int(selection.explored),
                    int(selection.learning_enabled), selection.learning_run_id, model,
                ),
            )
            return int(response_id)

    def get_messages(self, conversation_id: str) -> list[dict]:
        with self.connection() as connection:
            rows = connection.execute(
                """SELECT m.*, s.state, s.action, s.q_at_selection, s.epsilon,
                          s.explored, s.learning_enabled, s.learning_run_id, s.model,
                          f.label AS feedback_label, f.reward,
                          e.id AS learning_event_id, e.q_before, e.q_after
                   FROM messages m
                   LEFT JOIN response_strategies s ON s.response_id = m.id
                   LEFT JOIN feedback f ON f.response_id = m.id
                   LEFT JOIN learning_events e ON e.feedback_id = f.id
                   WHERE m.conversation_id = ? ORDER BY m.id""",
                (conversation_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_q_values(self, state: str, run_id: int | None = None) -> dict[int, float]:
        with self.connection() as connection:
            connection.execute("BEGIN")
            run_id = self.active_run(connection) if run_id is None else run_id
            rows = connection.execute(
                "SELECT action, q_value FROM state_action_values WHERE learning_run_id = ? AND state = ?",
                (run_id, state),
            ).fetchall()
        values = {0: 0.0, 1: 0.0, 2: 0.0}
        values.update({row["action"]: row["q_value"] for row in rows})
        return values
