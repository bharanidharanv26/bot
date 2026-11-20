from concurrent.futures import ThreadPoolExecutor
import sqlite3
from threading import Barrier

import pytest

from database.models import Action
from reinforcement_learning.reward import reward_for_feedback


@pytest.mark.parametrize("label, reward", [
    ("Very helpful", 2), ("Helpful", 1), ("Neutral", 0), ("Not helpful", -1),
])
def test_reward_mapping(label, reward):
    assert reward_for_feedback(label) == reward


@pytest.mark.parametrize("label", [None, "", "good", 2])
def test_invalid_feedback_is_rejected(label):
    with pytest.raises(ValueError):
        reward_for_feedback(label)


def test_feedback_records_exact_response_action_and_event(database, manager, make_turn):
    first = make_turn(Action.DETAILED)
    second = make_turn(Action.EXAMPLE)
    result = manager.submit(first, "Very helpful")
    assert result.accepted and result.learned and result.reward == 2
    with database.connection() as connection:
        feedback = connection.execute("SELECT * FROM feedback").fetchone()
        event = connection.execute("SELECT * FROM learning_events").fetchone()
    assert feedback["response_id"] == first != second
    assert event["action"] == Action.DETAILED
    assert event["q_before"] == 0 and event["q_after"] == pytest.approx(0.4)
    assert event["feedback_id"] == feedback["id"]
    assert event["created_at"].endswith("Z")


def test_duplicate_feedback_cannot_update_twice(database, manager, make_turn):
    response_id = make_turn()
    manager.submit(response_id, "Very helpful")
    duplicate = manager.submit(response_id, "Not helpful")
    assert not duplicate.accepted and not duplicate.learned
    assert duplicate.reward == 2
    assert database.get_q_values("short|question")[0] == pytest.approx(0.4)
    with database.connection() as connection:
        assert connection.execute("SELECT COUNT(*) FROM feedback").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM learning_events").fetchone()[0] == 1
        assert connection.execute("SELECT update_count FROM state_action_values").fetchone()[0] == 1


def test_database_unique_constraint_also_blocks_duplicates(database, manager, make_turn):
    response_id = make_turn()
    manager.submit(response_id, "Helpful")
    with pytest.raises(sqlite3.IntegrityError), database.connection() as connection:
        connection.execute(
            "INSERT INTO feedback (response_id, label, reward) VALUES (?, 'Helpful', 1)", (response_id,),
        )


def test_concurrent_duplicate_submissions_are_serialized(database, manager, make_turn):
    response_id = make_turn()
    barrier = Barrier(4)

    def submit():
        barrier.wait(timeout=10)
        return manager.submit(response_id, "Very helpful")

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(submit) for _ in range(4)]
        results = [future.result(timeout=20) for future in futures]
    assert sum(result.accepted for result in results) == 1
    assert sum(result.learned for result in results) == 1
    assert database.get_q_values("short|question")[0] == pytest.approx(0.4)


def test_failed_update_rolls_back_feedback_q_value_and_event(database, manager, agent, make_turn, monkeypatch):
    response_id = make_turn()
    original = agent._learn_from_feedback

    def fail_after_writes(connection, feedback_id):
        original(connection, feedback_id)
        raise RuntimeError("Simulated failure before commit")

    monkeypatch.setattr(agent, "_learn_from_feedback", fail_after_writes)
    with pytest.raises(RuntimeError):
        manager.submit(response_id, "Very helpful")
    with database.connection() as connection:
        for table in ("feedback", "state_action_values", "learning_events"):
            assert connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0
    monkeypatch.setattr(agent, "_learn_from_feedback", original)
    assert manager.submit(response_id, "Very helpful").learned


@pytest.mark.parametrize("enabled_at_selection, enabled_at_feedback", [(False, True), (True, False), (False, False)])
def test_paused_feedback_is_saved_but_never_replayed(database, manager, make_turn, enabled_at_selection, enabled_at_feedback):
    response_id = make_turn(learning_enabled=enabled_at_selection)
    result = manager.submit(response_id, "Helpful", learning_enabled=enabled_at_feedback)
    assert result.accepted and not result.learned
    assert not manager.submit(response_id, "Helpful", learning_enabled=True).learned
    assert database.get_q_values("short|question") == {0: 0, 1: 0, 2: 0}
    with database.connection() as connection:
        assert connection.execute("SELECT COUNT(*) FROM feedback").fetchone()[0] == 1


def test_neutral_is_valid_and_moves_positive_estimate_toward_zero(database, manager, make_turn):
    manager.submit(make_turn(), "Very helpful")
    result = manager.submit(make_turn(), "Neutral")
    assert result.learned and result.reward == 0
    assert result.q_before == pytest.approx(0.4)
    assert result.q_after == pytest.approx(0.32)


def test_bad_or_missing_feedback_does_not_change_database(database, manager, make_turn):
    response_id = make_turn()
    with pytest.raises(ValueError):
        manager.submit(response_id, "not a rating")
    with pytest.raises(ValueError):
        manager.submit(response_id - 1, "Helpful")  # A user message, not an answer.
    with pytest.raises(ValueError):
        manager.submit(99999, "Helpful")
    with database.connection() as connection:
        assert connection.execute("SELECT COUNT(*) FROM feedback").fetchone()[0] == 0


def test_reset_retains_history_and_old_feedback_cannot_train_new_run(database, manager, make_turn):
    manager.submit(make_turn(), "Very helpful")
    delayed_response = make_turn(Action.EXAMPLE)
    old_run = database.current_run_id()
    new_run = database.reset_learning()
    assert new_run > old_run
    assert database.get_q_values("short|question") == {0: 0, 1: 0, 2: 0}
    assert database.get_q_values("short|question", old_run)[0] == pytest.approx(0.4)
    delayed = manager.submit(delayed_response, "Very helpful")
    assert delayed.accepted and not delayed.learned
    fresh = manager.submit(make_turn(Action.DETAILED), "Helpful")
    assert fresh.learned and fresh.q_after == pytest.approx(0.2)
    with database.connection() as connection:
        assert connection.execute("SELECT COUNT(*) FROM feedback").fetchone()[0] == 3
        assert connection.execute("SELECT COUNT(*) FROM learning_events").fetchone()[0] == 2
        assert connection.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 6
