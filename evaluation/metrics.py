"""Unrated responses count as responses, but never enter reward averages."""

import pandas as pd

from database.db import Database
from database.models import Action


def load_interactions(database: Database, current_run_only: bool = True) -> pd.DataFrame:
    with database.connection() as connection:
        connection.execute("BEGIN")
        run_id = database.active_run(connection)
        query = """SELECT s.response_id, s.state, s.action, s.learning_run_id, s.model,
                          s.epsilon, s.explored, s.learning_enabled,
                          s.created_at AS response_at,
                          f.id AS feedback_id, f.label, f.reward, f.created_at AS feedback_at,
                          e.id AS learning_event_id, e.q_before, e.q_after, e.alpha
                   FROM response_strategies s
                   LEFT JOIN feedback f ON f.response_id = s.response_id
                   LEFT JOIN learning_events e ON e.feedback_id = f.id"""
        params = ()
        if current_run_only:
            query += " WHERE s.learning_run_id = ?"
            params = (run_id,)
        frame = pd.read_sql_query(query + " ORDER BY s.response_id", connection, params=params)
    frame["strategy"] = frame["action"].map({int(action): action.name for action in Action})
    frame["reward"] = pd.to_numeric(frame["reward"], errors="coerce")
    frame["feedback_at"] = pd.to_datetime(frame["feedback_at"], utc=True)
    return frame


def summary_metrics(interactions: pd.DataFrame) -> dict:
    rewards = interactions["reward"].dropna()
    return {
        "responses": len(interactions),
        "feedback_submissions": int(interactions["feedback_id"].notna().sum()),
        "average_reward": float(rewards.mean()) if not rewards.empty else None,
        "learning_updates": int(interactions["learning_event_id"].notna().sum()),
    }


def strategy_metrics(interactions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for action in Action:
        subset = interactions.loc[interactions["action"] == int(action)]
        rewards = subset["reward"].dropna()
        rows.append({
            "Strategy": action.name,
            "Selections": len(subset),
            "Feedback": int(subset["feedback_id"].notna().sum()),
            "Average reward": float(rewards.mean()) if not rewards.empty else None,
        })
    return pd.DataFrame(rows)


def reward_progress(interactions: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    if window < 1:
        raise ValueError("The rolling window must be at least 1.")
    rated = interactions.loc[interactions["feedback_id"].notna()].copy()
    # Order by rating time, not response time: users can rate old messages later.
    rated = rated.sort_values(["feedback_at", "feedback_id"]).reset_index(drop=True)
    progress = rated[["feedback_at", "reward"]].copy()
    progress["feedback_number"] = range(1, len(progress) + 1)
    progress["cumulative_average"] = progress["reward"].expanding().mean()
    progress["rolling_average"] = progress["reward"].rolling(window, min_periods=1).mean()
    return progress
