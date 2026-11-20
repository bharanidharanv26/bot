"""Small, explicit settings. Environment variables are optional overrides."""

from dataclasses import dataclass
import math
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Settings:
    db_path: Path = PROJECT_ROOT / "data" / "bot.sqlite3"
    ollama_url: str = "http://127.0.0.1:11434"
    model: str = "qwen2.5:1.5b"
    alpha: float = 0.2
    epsilon: float = 0.2
    min_epsilon: float = 0.05
    decay_feedback: int = 20
    timeout_seconds: float = 180
    max_message_chars: int = 2000
    history_char_budget: int = 4000

    def __post_init__(self) -> None:
        if not 0 < self.alpha <= 1:
            raise ValueError("BOT_ALPHA must be greater than 0 and at most 1.")
        if not 0 <= self.min_epsilon <= self.epsilon <= 1:
            raise ValueError("Use 0 <= BOT_MIN_EPSILON <= BOT_EPSILON <= 1.")
        if self.decay_feedback < 1:
            raise ValueError("BOT_DECAY_FEEDBACK must be a positive integer.")
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("BOT_TIMEOUT_SECONDS must be a positive finite number.")
        if self.max_message_chars < 1 or self.history_char_budget < 0:
            raise ValueError("Message limits must be positive; history may be zero.")
        if not self.model.strip():
            raise ValueError("BOT_MODEL cannot be empty.")


def load_settings() -> Settings:
    """Relative database paths are resolved inside this project, not the shell's cwd."""
    db_path = Path(os.getenv("BOT_DB_PATH", "data/bot.sqlite3")).expanduser()
    if not db_path.is_absolute():
        db_path = PROJECT_ROOT / db_path
    return Settings(
        db_path=db_path,
        ollama_url=os.getenv("BOT_OLLAMA_URL", "http://127.0.0.1:11434"),
        model=os.getenv("BOT_MODEL", "qwen2.5:1.5b"),
        alpha=float(os.getenv("BOT_ALPHA", "0.2")),
        epsilon=float(os.getenv("BOT_EPSILON", "0.2")),
        min_epsilon=float(os.getenv("BOT_MIN_EPSILON", "0.05")),
        decay_feedback=int(os.getenv("BOT_DECAY_FEEDBACK", "20")),
        timeout_seconds=float(os.getenv("BOT_TIMEOUT_SECONDS", "180")),
    )
