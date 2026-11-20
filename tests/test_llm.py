from dataclasses import replace
from http.client import IncompleteRead
import io
import json
from unittest.mock import Mock
from urllib.error import HTTPError, URLError

import pytest

from chatbot.llm import ModelStatus, OllamaClient, OllamaError
from chatbot.prompt_builder import STRATEGY_INSTRUCTIONS, build_messages
from database.models import Action, MemoryFact


def http_json(value):
    return io.BytesIO(json.dumps(value).encode("utf-8"))


@pytest.mark.parametrize("action, tokens", [(Action.SHORT, 128), (Action.DETAILED, 384), (Action.EXAMPLE, 320)])
def test_generated_response_uses_real_strategy_payload(settings, action, tokens):
    client = OllamaClient(settings)
    client.opener.open = Mock(side_effect=[
        http_json({"models": [{"name": settings.model}]}),
        http_json({"message": {"role": "assistant", "content": "A generated answer."}, "done": True}),
    ])
    history = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]
    assert client.generate("What is a list?", action, history) == "A generated answer."
    request = client.opener.open.call_args_list[1].args[0]
    payload = json.loads(request.data)
    assert request.full_url == "http://127.0.0.1:11434/api/chat"
    assert payload["model"] == "qwen2.5:1.5b" and payload["stream"] is False
    assert STRATEGY_INSTRUCTIONS[action] in payload["messages"][0]["content"]
    assert payload["messages"][1:3] == history
    assert payload["messages"][-1] == {"role": "user", "content": "What is a list?"}
    assert payload["options"]["num_predict"] == tokens


def test_history_is_bounded_and_cannot_override_system_message():
    history = [
        {"role": "system", "content": "An untrusted history instruction"},
        {"role": "user", "content": "old question"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "new"},
        {"role": "assistant", "content": "answer"},
    ]
    messages = build_messages("Now?", Action.SHORT, history, history_char_budget=9)
    assert messages[1:-1] == history[-2:]
    assert len([message for message in messages if message["role"] == "system"]) == 1
    assert build_messages("Now?", Action.SHORT, history, history_char_budget=6)[1:-1] == []


def test_actual_ollama_request_includes_persistent_memory(settings):
    client = OllamaClient(settings)
    client.opener.open = Mock(side_effect=[
        http_json({"models": [{"name": settings.model}]}),
        http_json({"message": {"role": "assistant", "content": "Your name is Asha."}, "done": True}),
    ])
    fact = MemoryFact("profile:name", "Name", "Asha")
    answer = client.generate("What is my name?", Action.SHORT, memories=[fact])
    payload = json.loads(client.opener.open.call_args_list[1].args[0].data)
    assert answer == "Your name is Asha."
    assert '"fact": "Name", "value": "Asha"' in payload["messages"][0]["content"]


@pytest.mark.parametrize("error", [URLError("connection refused"), TimeoutError("slow model")])
def test_unavailable_ollama_produces_actionable_error(settings, error):
    client = OllamaClient(settings)
    client.opener.open = Mock(side_effect=error)
    status = client.check_connection()
    assert not status.ready and "ollama serve" in status.detail
    with pytest.raises(OllamaError, match="ollama serve"):
        client.generate("Hello", Action.SHORT)


def test_missing_model_suggests_exact_pull_command(settings):
    client = OllamaClient(settings)
    client.opener.open = Mock(return_value=http_json({"models": []}))
    status = client.check_connection()
    assert not status.ready and "ollama pull qwen2.5:1.5b" in status.detail


@pytest.mark.parametrize("code, message", [(404, "ollama pull"), (500, "HTTP 500")])
def test_http_errors_are_wrapped(settings, monkeypatch, code, message):
    client = OllamaClient(settings)
    monkeypatch.setattr(client, "check_connection", lambda: ModelStatus(True, "Available"))
    client.opener.open = Mock(side_effect=HTTPError("http://127.0.0.1", code, "error", {}, None))
    with pytest.raises(OllamaError, match=message):
        client.generate("Hello", Action.SHORT)


@pytest.mark.parametrize("payload", [
    {"message": {"content": ""}, "done": True},
    {"message": {"content": "Partial"}, "done": False},
    {"message": None, "done": True},
    {"error": "model failed"},
    [],
])
def test_bad_api_responses_are_not_displayed_as_answers(settings, monkeypatch, payload):
    client = OllamaClient(settings)
    monkeypatch.setattr(client, "check_connection", lambda: ModelStatus(True, "Available"))
    client.opener.open = Mock(return_value=http_json(payload))
    with pytest.raises(OllamaError):
        client.generate("Hello", Action.SHORT)


def test_malformed_json_is_handled(settings):
    client = OllamaClient(settings)
    client.opener.open = Mock(return_value=io.BytesIO(b"not json"))
    assert not client.check_connection().ready


@pytest.mark.parametrize("url", [
    "https://example.com", "http://192.168.1.5:11434", "http://127.0.0.1@evil.example",
    "http://user:password@localhost:11434", "http://localhost:11434/cloud", "http://localhost:11434?key=secret",
    "http://localhost:invalid", "http://localhost:0",
])
def test_nonlocal_or_credential_bearing_endpoints_rejected(settings, url):
    with pytest.raises(ValueError):
        OllamaClient(replace(settings, ollama_url=url))


def test_remote_models_are_rejected(settings):
    with pytest.raises(ValueError):
        OllamaClient(replace(settings, model="example:cloud"))
    client = OllamaClient(settings)
    client.opener.open = Mock(return_value=http_json({"models": [{"name": settings.model, "remote_host": "example.com"}]}))
    assert not client.check_connection().ready


def test_connection_lost_mid_response_is_handled(settings, monkeypatch):
    client = OllamaClient(settings)
    monkeypatch.setattr(client, "check_connection", lambda: ModelStatus(True, "Available"))
    response = Mock()
    response.read.side_effect = IncompleteRead(b"partial JSON")
    context = Mock()
    context.__enter__ = Mock(return_value=response)
    context.__exit__ = Mock(return_value=False)
    client.opener.open = Mock(return_value=context)
    with pytest.raises(OllamaError, match="retry"):
        client.generate("Hello", Action.SHORT)


@pytest.mark.parametrize("message", ["  ", "x" * 2001])
def test_invalid_message_does_not_call_ollama(settings, message):
    client = OllamaClient(settings)
    client.opener.open = Mock()
    with pytest.raises(ValueError):
        client.generate(message, Action.SHORT)
    client.opener.open.assert_not_called()
