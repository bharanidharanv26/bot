import io

import pytest

from database.models import Action
from evaluation.metrics import load_interactions, reward_progress, strategy_metrics, summary_metrics
from evaluation.visualizations import average_reward_chart


def test_empty_evaluation_does_not_invent_a_reward(database):
    interactions = load_interactions(database)
    assert summary_metrics(interactions) == {
        "responses": 0, "feedback_submissions": 0, "average_reward": None, "learning_updates": 0,
    }
    assert reward_progress(interactions).empty
    assert strategy_metrics(interactions)["Average reward"].isna().all()


def test_unrated_responses_are_excluded_from_reward_averages(database, manager, make_turn):
    manager.submit(make_turn(Action.SHORT), "Very helpful")
    manager.submit(make_turn(Action.EXAMPLE), "Not helpful")
    make_turn(Action.DETAILED)
    interactions = load_interactions(database)
    assert summary_metrics(interactions) == {
        "responses": 3, "feedback_submissions": 2, "average_reward": 0.5, "learning_updates": 2,
    }
    strategies = strategy_metrics(interactions).set_index("Strategy")
    assert strategies.loc["DETAILED", "Selections"] == 1
    assert strategies.loc["DETAILED", "Feedback"] == 0
    assert strategies.loc["SHORT", "Average reward"] == 2
    assert strategies.loc["EXAMPLE", "Average reward"] == -1


def test_progress_uses_feedback_order_and_known_arithmetic(database, manager, make_turn):
    first_response = make_turn()
    second_response = make_turn()
    manager.submit(second_response, "Very helpful")
    manager.submit(first_response, "Not helpful")
    with database.connection() as connection:
        connection.execute("UPDATE feedback SET created_at = '2026-01-01T10:00:00.000Z' WHERE response_id = ?", (second_response,))
        connection.execute("UPDATE feedback SET created_at = '2026-01-02T10:00:00.000Z' WHERE response_id = ?", (first_response,))
    progress = reward_progress(load_interactions(database), window=1)
    assert progress["reward"].tolist() == [2, -1]
    assert progress["cumulative_average"].tolist() == [2, 0.5]
    assert progress["rolling_average"].tolist() == [2, -1]


def test_paused_neutral_feedback_counts_but_is_not_a_learning_event(database, manager, make_turn):
    manager.submit(make_turn(), "Neutral", learning_enabled=False)
    summary = summary_metrics(load_interactions(database))
    assert summary["average_reward"] == 0
    assert summary["feedback_submissions"] == 1
    assert summary["learning_updates"] == 0


def test_evaluation_separates_reset_runs(database, manager, make_turn):
    manager.submit(make_turn(), "Very helpful")
    database.reset_learning()
    assert summary_metrics(load_interactions(database))["responses"] == 0
    manager.submit(make_turn(), "Not helpful")
    assert summary_metrics(load_interactions(database))["average_reward"] == -1
    assert summary_metrics(load_interactions(database, current_run_only=False))["average_reward"] == 0.5


@pytest.mark.parametrize("with_feedback", [False, True])
def test_chart_renders_without_a_display(database, manager, make_turn, with_feedback):
    if with_feedback:
        manager.submit(make_turn(), "Helpful")
    figure = average_reward_chart(load_interactions(database))
    output = io.BytesIO()
    figure.savefig(output, format="png")
    assert output.getvalue().startswith(b"\x89PNG")
