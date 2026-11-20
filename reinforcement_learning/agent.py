"""Epsilon-greedy contextual bandit with persistent, feedback-only updates."""

import math
import random
import sqlite3

from config import Settings
from database.db import Database
from database.models import Action, ActionSelection
from reinforcement_learning.state import ALL_STATES


class ContextualBandit:
    def __init__(self, database: Database, settings: Settings, rng: random.Random | None = None):
        self.database = database
        self.settings = settings
        self.rng = rng if rng is not None else random.Random()

    def effective_epsilon(self, feedback_count: int, learning_enabled: bool = True) -> float:
        if feedback_count < 0:
            raise ValueError("Feedback count cannot be negative.")
        if not learning_enabled:
            return 0.0
        return max(
            self.settings.min_epsilon,
            self.settings.epsilon / (1 + feedback_count / self.settings.decay_feedback),
        )

    def select_action(self, state: str, learning_enabled: bool = True) -> ActionSelection:
        if state not in ALL_STATES:
            raise ValueError("Unknown conversation state.")
        # Read the run and estimates from the same snapshot if another tab resets.
        with self.database.connection() as connection:
            connection.execute("BEGIN")
            run_id = self.database.active_run(connection)
            rows = connection.execute(
                """SELECT action, q_value, update_count FROM state_action_values
                   WHERE learning_run_id = ? AND state = ?""",
                (run_id, state),
            ).fetchall()
        values = {action: 0.0 for action in Action}
        values.update({Action(row["action"]): row["q_value"] for row in rows})
        count = sum(row["update_count"] for row in rows)
        epsilon = self.effective_epsilon(count, learning_enabled)
        explored = self.rng.random() < epsilon
        if explored:
            action = self.rng.choice(list(Action))
        else:
            best_value = max(values.values())
            # Random tie-breaking avoids always choosing SHORT at a cold start.
            action = self.rng.choice([action for action, value in values.items() if value == best_value])
        return ActionSelection(state, action, values[action], epsilon, explored, learning_enabled, run_id)

    @staticmethod
    def updated_value(q_old: float, reward: int, alpha: float) -> float:
        if isinstance(reward, bool) or reward not in (-1, 0, 1, 2):
            raise ValueError("Reward must be -1, 0, 1, or 2.")
        if not math.isfinite(q_old) or not -1 <= q_old <= 2 or not 0 < alpha <= 1:
            raise ValueError("Invalid Q-value or learning rate.")
        return q_old + alpha * (reward - q_old)

    def _learn_from_feedback(self, connection: sqlite3.Connection, feedback_id: int) -> tuple[float, float]:
        """Internal: caller must hold the feedback transaction's write lock.

        Fetch the reward and exact action from stored feedback, never from a UI
        field or the currently selected action. The event's unique feedback_id
        also makes accidental repeated application fail and roll back.
        """
        row = connection.execute(
            """SELECT f.reward, s.state, s.action, s.learning_run_id, s.learning_enabled
               FROM feedback f JOIN response_strategies s ON s.response_id = f.response_id
               WHERE f.id = ?""",
            (feedback_id,),
        ).fetchone()
        if row is None or not row["learning_enabled"]:
            raise ValueError("Learning requires eligible, stored feedback.")
        if row["learning_run_id"] != self.database.active_run(connection):
            raise ValueError("Feedback belongs to a previous learning run.")
        key = (row["learning_run_id"], row["state"], row["action"])
        old = connection.execute(
            """SELECT q_value FROM state_action_values
               WHERE learning_run_id = ? AND state = ? AND action = ?""",
            key,
        ).fetchone()
        q_before = float(old["q_value"]) if old else 0.0
        q_after = self.updated_value(q_before, row["reward"], self.settings.alpha)
        connection.execute(
            """INSERT INTO state_action_values
               (learning_run_id, state, action, q_value, update_count) VALUES (?, ?, ?, ?, 1)
               ON CONFLICT (learning_run_id, state, action) DO UPDATE SET
                   q_value = excluded.q_value, update_count = update_count + 1,
                   updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')""",
            (*key, q_after),
        )
        connection.execute(
            """INSERT INTO learning_events
               (feedback_id, learning_run_id, state, action, reward, alpha, q_before, q_after)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (feedback_id, *key, row["reward"], self.settings.alpha, q_before, q_after),
        )
        return q_before, q_after

    def inspect_policy(self, learning_enabled: bool = True) -> list[dict]:
        with self.database.connection() as connection:
            connection.execute("BEGIN")
            run_id = self.database.active_run(connection)
            stored = connection.execute(
                "SELECT * FROM state_action_values WHERE learning_run_id = ?", (run_id,)
            ).fetchall()
        lookup = {(row["state"], row["action"]): row for row in stored}
        policy = []
        for state in ALL_STATES:
            counts = [lookup.get((state, int(action))) for action in Action]
            total = sum(row["update_count"] for row in counts if row is not None)
            epsilon = self.effective_epsilon(total, learning_enabled)
            values = {
                action: lookup[(state, int(action))]["q_value"]
                if (state, int(action)) in lookup else 0.0 for action in Action
            }
            best = max(values.values())
            ties = sum(value == best for value in values.values())
            for action in Action:
                record = lookup.get((state, int(action)))
                probability = epsilon / len(Action)
                if values[action] == best:
                    probability += (1 - epsilon) / ties
                policy.append({
                    "State": state,
                    "Strategy": action.name,
                    "Q-value": values[action],
                    "Updates": record["update_count"] if record else 0,
                    "Epsilon": epsilon,
                    "Selection probability": probability,
                })
        return policy
