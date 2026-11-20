"""Every test gets isolated storage; fixture ratings never enter real history."""

from dataclasses import replace
import random

import pytest

from config import Settings
from database.db import Database
from database.models import Action
from feedback.feedback_manager import FeedbackManager
from reinforcement_learning.agent import ContextualBandit
from reinforcement_learning.state import generate_state


@pytest.fixture
def settings(tmp_path):
    return Settings(db_path=tmp_path / "test.sqlite3")


@pytest.fixture
def database(settings):
    database = Database(settings.db_path)
    database.initialize()
    return database


@pytest.fixture
def agent(database, settings):
    return ContextualBandit(database, settings, random.Random(42))


@pytest.fixture
def manager(database, agent):
    return FeedbackManager(database, agent)


@pytest.fixture
def make_turn(database, agent, settings):
    conversation_id = database.create_conversation()

    def create(action=Action.SHORT, message="What is Python?", learning_enabled=True):
        selection = agent.select_action(generate_state(message), learning_enabled)
        selection = replace(selection, action=action, q_value=database.get_q_values(selection.state)[int(action)])
        return database.save_interaction(
            conversation_id, message, "A mocked language-model response.", selection, settings.model,
        )

    return create
