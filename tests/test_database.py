from dataclasses import replace
import sqlite3

import pytest

from database.db import Database
from database.models import Action


def test_initialization_creates_schema_and_is_repeatable(database):
    database.initialize()
    with database.connection() as connection:
        names = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    assert {
        "conversations", "messages", "response_strategies", "feedback",
        "state_action_values", "learning_events", "learning_runs",
    } <= names
    assert database.current_run_id() == 1
    database.reset_learning()
    database.initialize()
    assert database.current_run_id() == 2


def test_conversation_and_turn_round_trip(database, agent, settings):
    conversation = database.create_conversation()
    selection = replace(agent.select_action("short|question"), action=Action.EXAMPLE)
    response_id = database.save_interaction(conversation, "What's SQL?", "SQL queries databases.", selection, settings.model)
    messages = database.get_messages(conversation)
    assert [message["role"] for message in messages] == ["user", "assistant"]
    assert messages[0]["content"] == "What's SQL?"
    assert messages[1]["id"] == response_id
    assert messages[1]["action"] == Action.EXAMPLE
    assert messages[1]["state"] == "short|question"
    assert messages[1]["feedback_label"] is None
    assert messages[1]["created_at"].endswith("Z")
    assert database.list_conversations()[0]["title"] == "What's SQL?"
    reopened = Database(settings.db_path)
    assert reopened.get_messages(conversation) == messages


def test_parameterized_sql_preserves_quotes_as_text(database, agent, settings):
    conversation = database.create_conversation()
    text = "Robert'); DROP TABLE messages; --"
    database.save_interaction(conversation, text, "Quoted text is data.", agent.select_action("short|general"), settings.model)
    assert database.get_messages(conversation)[0]["content"] == text


@pytest.mark.parametrize("invalid_conversation, invalid_run", [(True, False), (False, True)])
def test_foreign_key_failure_rolls_back_whole_turn(database, agent, settings, invalid_conversation, invalid_run):
    conversation = "missing" if invalid_conversation else database.create_conversation()
    selection = agent.select_action("short|general")
    if invalid_run:
        selection = replace(selection, learning_run_id=99999)
    with pytest.raises(sqlite3.IntegrityError):
        database.save_interaction(conversation, "Hello", "Hi", selection, settings.model)
    with database.connection() as connection:
        assert connection.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM response_strategies").fetchone()[0] == 0


def test_empty_generation_is_not_saved(database, agent, settings):
    conversation = database.create_conversation()
    with pytest.raises(ValueError):
        database.save_interaction(conversation, "Hello", "  ", agent.select_action("short|general"), settings.model)
    assert database.get_messages(conversation) == []


def test_reset_during_generation_keeps_original_run(database, agent, manager, settings):
    selection = agent.select_action("short|general")
    database.reset_learning()
    conversation = database.create_conversation()
    response_id = database.save_interaction(conversation, "Hello", "Hi", selection, settings.model)
    result = manager.submit(response_id, "Very helpful")
    assert result.accepted and not result.learned
    assert database.get_messages(conversation)[1]["learning_run_id"] == selection.learning_run_id
