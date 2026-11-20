from unittest.mock import Mock

import pytest
from streamlit.testing.v1 import AppTest

from chatbot.llm import ModelStatus, OllamaClient, OllamaError
from config import PROJECT_ROOT
from evaluation.metrics import load_interactions, summary_metrics


@pytest.fixture
def ui(database, settings, monkeypatch):
    monkeypatch.setenv("BOT_DB_PATH", str(settings.db_path))
    monkeypatch.setenv("BOT_MODEL", "qwen2.5:1.5b")
    monkeypatch.setenv("BOT_ALPHA", "0.2")
    monkeypatch.setenv("BOT_EPSILON", "0.2")
    monkeypatch.setenv("BOT_MIN_EPSILON", "0.05")
    monkeypatch.setattr(OllamaClient, "check_connection", lambda self: ModelStatus(True, "Available locally"))
    monkeypatch.setattr(OllamaClient, "generate", Mock(return_value="A Python list stores an ordered sequence of items."))
    return AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=30).run()


def button(ui, label):
    return next(item for item in ui.button if item.label == label)


def test_chat_feedback_and_reload_persist(ui, database):
    assert not ui.exception
    assert ui.title[0].value == "Mini RL Chatbot"
    ui.chat_input[0].set_value("What is a Python list?").run()
    assert not ui.exception
    assert len(ui.chat_message) == 2
    assert summary_metrics(load_interactions(database))["learning_updates"] == 0
    button(ui, "Very helpful").click().run()
    assert not ui.exception
    assert summary_metrics(load_interactions(database))["average_reward"] == 2
    assert summary_metrics(load_interactions(database))["learning_updates"] == 1
    assert not any(item.label == "Very helpful" for item in ui.button)
    reloaded = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=30).run()
    assert not reloaded.exception
    assert len(reloaded.chat_message) == 2
    assert summary_metrics(load_interactions(database))["learning_updates"] == 1
    button(reloaded, "Learning details").click().run()
    assert not reloaded.exception
    assert len(reloaded.dataframe) == 2


def test_unavailable_model_can_retry_without_fake_turns(ui, database, monkeypatch):
    generator = Mock(side_effect=OllamaError("Start Ollama with ollama serve."))
    monkeypatch.setattr(OllamaClient, "generate", generator)
    ui.chat_input[0].set_value("Hello").run()
    assert not ui.exception
    assert any("ollama serve" in item.value for item in ui.error)
    assert summary_metrics(load_interactions(database))["responses"] == 0
    assert not any(item.label == "Very helpful" for item in ui.button)
    generator.side_effect = None
    generator.return_value = "Hello! How can I help?"
    button(ui, "Retry response").click().run()
    assert not ui.exception and not ui.error
    assert len(ui.chat_message) == 2
    assert summary_metrics(load_interactions(database))["responses"] == 1
    assert summary_metrics(load_interactions(database))["learning_updates"] == 0


def test_pause_and_reset_controls_do_not_erase_chat(ui, database):
    ui.toggle[0].set_value(False).run()
    ui.chat_input[0].set_value("What is SQLite?").run()
    button(ui, "Helpful").click().run()
    assert not ui.exception
    assert summary_metrics(load_interactions(database))["feedback_submissions"] == 1
    assert summary_metrics(load_interactions(database))["learning_updates"] == 0
    ui.toggle[0].set_value(True).run()
    ui.chat_input[0].set_value("What is a database?").run()
    button(ui, "Very helpful").click().run()
    assert summary_metrics(load_interactions(database))["learning_updates"] == 1
    button(ui, "Reset learning").click().run()
    assert not ui.exception
    assert database.current_run_id() == 2
    assert len(ui.chat_message) == 4
    assert summary_metrics(load_interactions(database))["responses"] == 0
    assert summary_metrics(load_interactions(database, current_run_only=False))["responses"] == 2


def test_new_conversation_keeps_saved_chat(ui, database):
    ui.chat_input[0].set_value("Hello").run()
    original_id = ui.radio[0].value
    button(ui, "New conversation").click().run()
    assert not ui.exception and len(ui.chat_message) == 0
    assert len(database.list_conversations()) == 2
    ui.radio[0].set_value(original_id).run()
    assert not ui.exception and len(ui.chat_message) == 2
