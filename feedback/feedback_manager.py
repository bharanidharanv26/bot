"""Feedback insertion and learning share ONE atomic database transaction."""

from database.db import Database
from database.models import FeedbackResult
from reinforcement_learning.agent import ContextualBandit
from reinforcement_learning.reward import reward_for_feedback


class FeedbackManager:
    def __init__(self, database: Database, agent: ContextualBandit):
        self.database = database
        self.agent = agent

    def submit(self, response_id: int, label: str, learning_enabled: bool = True) -> FeedbackResult:
        reward = reward_for_feedback(label)  # Validate before writing anything.
        with self.database.connection() as connection:
            # Serialize writers before checking duplicates or reading Q-values.
            connection.execute("BEGIN IMMEDIATE")
            strategy = connection.execute(
                "SELECT * FROM response_strategies WHERE response_id = ?", (response_id,)
            ).fetchone()
            if strategy is None:
                raise ValueError("Feedback must refer to an existing chatbot response.")
            previous = connection.execute(
                "SELECT reward FROM feedback WHERE response_id = ?", (response_id,)
            ).fetchone()
            if previous:
                return FeedbackResult(False, False, previous["reward"], "Feedback already recorded; no second update.")
            feedback_id = connection.execute(
                "INSERT INTO feedback (response_id, label, reward) VALUES (?, ?, ?)",
                (response_id, label, reward),
            ).lastrowid
            if strategy["learning_run_id"] != self.database.active_run(connection):
                return FeedbackResult(True, False, reward, "Feedback saved for the previous run; current policy unchanged.")
            if not learning_enabled or not strategy["learning_enabled"]:
                return FeedbackResult(True, False, reward, "Feedback saved; learning was paused. It will not be replayed later.")
            before, after = self.agent._learn_from_feedback(connection, int(feedback_id))
            return FeedbackResult(True, True, reward, "Feedback saved and policy updated once.", before, after)
