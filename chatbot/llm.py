"""Ollama's local HTTP API via Python's standard library; no cloud fallback."""

from dataclasses import dataclass
from http.client import HTTPException
import json
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from chatbot.prompt_builder import build_messages
from config import Settings
from database.models import Action


class OllamaError(RuntimeError):
    """An actionable model error that the UI can display without crashing."""


@dataclass(frozen=True)
class ModelStatus:
    ready: bool
    detail: str


class _NoRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class OllamaClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        url = urlsplit(settings.ollama_url)
        # Accessing .port also validates malformed/non-numeric port strings.
        if url.port is not None and not 1 <= url.port <= 65535:
            raise ValueError("BOT_OLLAMA_URL must use a port between 1 and 65535.")
        if (
            url.scheme != "http" or url.hostname not in {"localhost", "127.0.0.1", "::1"}
            or url.username or url.password or url.query or url.fragment or url.path not in ("", "/")
        ):
            raise ValueError("BOT_OLLAMA_URL must be a local HTTP address, such as http://127.0.0.1:11434.")
        if "cloud" in settings.model.casefold():
            raise ValueError("This Phase 1 application supports local Ollama models only.")
        self.base_url = settings.ollama_url.rstrip("/")
        # Ignore OS proxies and reject redirects: chat text stays on loopback.
        self.opener = build_opener(ProxyHandler({}), _NoRedirects())

    def _request(self, endpoint: str, payload: dict | None = None, timeout: float | None = None) -> dict:
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = Request(self.base_url + endpoint, data=data, headers={"Content-Type": "application/json"})
        try:
            with self.opener.open(request, timeout=timeout or self.settings.timeout_seconds) as response:
                result = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            if exc.code == 404:
                raise OllamaError(f"Model not found. Run: ollama pull {self.settings.model}") from exc
            raise OllamaError(f"Ollama returned HTTP {exc.code}. Check Ollama's logs and available memory, then retry.") from exc
        except (TimeoutError, URLError, OSError, HTTPException) as exc:
            raise OllamaError(
                "Cannot reach Ollama, or generation timed out. Start Ollama (ollama serve), "
                "check the local address, and retry. For a slow first load, increase BOT_TIMEOUT_SECONDS."
            ) from exc
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise OllamaError("Ollama returned an unreadable response. Check or restart the local server.") from exc
        if not isinstance(result, dict):
            raise OllamaError("Ollama returned an unexpected response format.")
        if result.get("error"):
            raise OllamaError("Ollama could not complete the request. Check its logs and installed model.")
        return result

    def check_connection(self) -> ModelStatus:
        try:
            result = self._request("/api/tags", timeout=3)
            models = result.get("models")
            if not isinstance(models, list):
                raise OllamaError("Ollama returned an invalid model list.")
            name = self.settings.model
            wanted = name if ":" in name else name + ":latest"
            for model in models:
                if not isinstance(model, dict):
                    continue
                if model.get("name") in {name, wanted} or model.get("model") in {name, wanted}:
                    if model.get("remote_host") or model.get("remote_model"):
                        raise OllamaError("The selected model uses remote inference. Choose a downloaded local model.")
                    return ModelStatus(True, f"{name} is available locally.")
            return ModelStatus(False, f"Model missing. Run: ollama pull {name}")
        except OllamaError as exc:
            return ModelStatus(False, str(exc))

    def generate(self, user_message: str, action: Action, history=(), *, memories=(), memory_update_labels=()) -> str:
        if not user_message.strip() or len(user_message.strip()) > self.settings.max_message_chars:
            raise ValueError(f"Enter between 1 and {self.settings.max_message_chars} characters.")
        status = self.check_connection()
        if not status.ready:
            raise OllamaError(status.detail)
        action = Action(action)
        messages = build_messages(
            user_message, action, history, self.settings.history_char_budget,
            memories=memories, memory_update_labels=memory_update_labels,
        )
        result = self._request("/api/chat", {
            "model": self.settings.model,
            "messages": messages,
            "stream": False,
            "keep_alive": "5m",
            "options": {
                "temperature": 0.5,
                "num_ctx": 4096,
                "num_predict": {Action.SHORT: 128, Action.DETAILED: 384, Action.EXAMPLE: 320}[action],
            },
        })
        message = result.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content.strip() or result.get("done") is not True:
            raise OllamaError("Ollama returned an empty or incomplete answer. Please retry.")
        return content.strip()
