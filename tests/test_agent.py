from dataclasses import replace
import random

import pytest

from database.db import Database
from database.models import Action
from reinforcement_learning.agent import ContextualBandit
from reinforcement_learning.state import ALL_STATES, generate_state


def test_epsilon_zero_exploits_best_action(database, settings, manager, make_turn):
    manager.submit(make_turn(Action.DETAILED), "Very helpful")
    agent = ContextualBandit(database, replace(settings, epsilon=0, min_epsilon=0), random.Random(1))
    decisions = [agent.select_action("short|question") for _ in range(50)]
    assert all(choice.action == Action.DETAILED and not choice.explored for choice in decisions)


def test_epsilon_one_explores_all_actions(database, settings, manager, make_turn):
    manager.submit(make_turn(Action.DETAILED), "Very helpful")
    agent = ContextualBandit(database, replace(settings, epsilon=1, min_epsilon=1), random.Random(2))
    decisions = [agent.select_action("short|question") for _ in range(150)]
    assert all(choice.explored for choice in decisions)
    assert {choice.action for choice in decisions} == set(Action)
    counts = [sum(choice.action == action for choice in decisions) for action in Action]
    assert all(25 < count < 75 for count in counts)


def test_greedy_ties_are_random_not_always_short(database, settings):
    agent = ContextualBandit(database, replace(settings, epsilon=0, min_epsilon=0), random.Random(3))
    assert {agent.select_action("short|question").action for _ in range(50)} == set(Action)


@pytest.mark.parametrize("reward, expected", [(2, 0.8), (1, 0.6), (0, 0.4), (-1, 0.2)])
def test_incremental_equation(reward, expected):
    assert ContextualBandit.updated_value(0.5, reward, 0.2) == pytest.approx(expected)


def test_documented_sequence_changes_only_rated_pair(database, manager, make_turn):
    for label, expected in [("Very helpful", 0.4), ("Helpful", 0.52), ("Not helpful", 0.216)]:
        result = manager.submit(make_turn(Action.DETAILED), label)
        assert result.q_after == pytest.approx(expected)
    values = database.get_q_values("short|question")
    assert values[Action.DETAILED] == pytest.approx(0.216)
    assert values[Action.SHORT] == values[Action.EXAMPLE] == 0
    assert database.get_q_values("long|general") == {0: 0, 1: 0, 2: 0}


def test_selection_and_missing_feedback_do_not_learn(database, agent, make_turn):
    make_turn()
    for _ in range(20):
        assert agent.select_action("short|question").epsilon == 0.2
    with database.connection() as connection:
        for table in ("feedback", "state_action_values", "learning_events"):
            assert connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0


def test_epsilon_decays_per_state_only_with_real_updates(agent, manager, make_turn):
    assert agent.effective_epsilon(0) == pytest.approx(0.2)
    assert agent.effective_epsilon(20) == pytest.approx(0.1)
    assert agent.effective_epsilon(60) == pytest.approx(0.05)
    assert agent.effective_epsilon(1000) == pytest.approx(0.05)
    manager.submit(make_turn(), "Helpful")
    assert agent.select_action("short|question").epsilon == pytest.approx(0.2 / 1.05)
    assert agent.select_action("long|question").epsilon == 0.2
    assert agent.select_action("short|question", learning_enabled=False).epsilon == 0


def test_learning_survives_new_database_and_agent(settings, manager, make_turn):
    manager.submit(make_turn(Action.EXAMPLE), "Very helpful")
    reopened = Database(settings.db_path)
    reopened.initialize()
    new_agent = ContextualBandit(reopened, replace(settings, epsilon=0, min_epsilon=0))
    selection = new_agent.select_action("short|question")
    assert selection.action == Action.EXAMPLE
    assert selection.q_value == pytest.approx(0.4)
    policy = new_agent.inspect_policy()
    assert sum(row["Updates"] for row in policy) == 1


def test_policy_probabilities_match_epsilon_greedy(agent, manager, make_turn):
    manager.submit(make_turn(Action.EXAMPLE), "Helpful")
    policy = agent.inspect_policy()
    assert len(policy) == 36
    for state in ALL_STATES:
        rows = [row for row in policy if row["State"] == state]
        assert sum(row["Selection probability"] for row in rows) == pytest.approx(1)
    rows = [row for row in policy if row["State"] == "short|question"]
    epsilon = 0.2 / 1.05
    assert rows[2]["Selection probability"] == pytest.approx(1 - epsilon + epsilon / 3)
    assert rows[0]["Selection probability"] == pytest.approx(epsilon / 3)


@pytest.mark.parametrize("message, expected", [
    ("Hello", "short|general"),
    ("What is SQLite?", "short|question"),
    ("WHY is SQLite useful?", "short|explanation"),
    ("How do I explain a list?", "short|how_to"),
    ("Please describe sorting", "short|explanation"),
    ("a" * 80, "short|general"),
    ("a" * 81, "medium|general"),
    ("a" * 240, "medium|general"),
    ("a" * 241, "long|general"),
    ("  Hello  ", "short|general"),
    ("你好", "short|general"),
    ("İ" * 80, "short|general"),
])
def test_state_generation(message, expected):
    assert generate_state(message) == expected


@pytest.mark.parametrize("message", ["", " \n\t ", None])
def test_invalid_state_input(message):
    with pytest.raises(ValueError):
        generate_state(message)


def test_policy_rejects_unknown_state(agent):
    with pytest.raises(ValueError):
        agent.select_action("contains a person's name")


@pytest.mark.parametrize("overrides", [
    {"alpha": 0}, {"alpha": float("nan")}, {"epsilon": 1.1},
    {"epsilon": 0.01, "min_epsilon": 0.05}, {"decay_feedback": 0},
    {"timeout_seconds": float("inf")},
])
def test_invalid_configuration_is_rejected(settings, overrides):
    with pytest.raises(ValueError):
        replace(settings, **overrides)
