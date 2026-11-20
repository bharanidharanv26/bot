"""Small, local long-term memory: explicit facts, not guessed user attributes.

Common first-person statements are recognized with rules. "Remember that ..."
also accepts arbitrary notes. No second model call or embedding model is needed.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import re

from database.db import Database
from database.models import MemoryFact


MEMORY_CONTEXT_CHARS = 3000
LABEL_ALIASES = {
    "name": "Name", "full name": "Name",
    "city": "Location", "location": "Location", "current city": "Location",
    "age": "Age", "job": "Occupation", "occupation": "Occupation",
    "home town": "Hometown", "hometown": "Hometown",
    "favourite color": "Favorite color", "favourite colour": "Favorite color", "favorite colour": "Favorite color",
}
STOP_WORDS = frozenset("a an and are as at be can do does for from how i in is it me my of on or please remember that the this to what when where which who with you your".split())
SECRET_LABEL = re.compile(r"\b(password|passcode|api[ _-]?key|access[ _-]?token|private[ _-]?key|secret[ _-]?key|pin[ _-]?code)\b", re.I)


@dataclass(frozen=True)
class MemoryPlan:
    context: tuple[MemoryFact, ...]
    updates: tuple[MemoryFact, ...]
    history: tuple[Mapping, ...]
    revision: int


def make_fact(label: str, value: str) -> MemoryFact:
    label = re.sub(r"\s+", " ", label).strip().strip(" .:;")
    label = re.sub(r"^(my|our|the)\s+", "", label, flags=re.I)
    value = value.strip().strip(" .;")
    if SECRET_LABEL.search(label) or re.search(r"-----BEGIN .*PRIVATE KEY-----|\bsk-[A-Za-z0-9]{20,}", value):
        raise ValueError("Passwords and API keys cannot be saved as memories.")
    label = LABEL_ALIASES.get(label.casefold(), label[:1].upper() + label[1:])
    key = "profile:" + re.sub(r"[^\w]+", "_", label.casefold()).strip("_")
    return MemoryFact(key, label, value)


def _note(text: str) -> MemoryFact:
    text = text.strip().strip(" .;")
    if SECRET_LABEL.search(text):
        raise ValueError("Passwords and API keys cannot be saved as memories.")
    normalized = re.sub(r"\s+", " ", text.casefold())
    key = "note:" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:24]
    return MemoryFact(key, "Note", text)


def _statement_fact(statement: str, explicit: bool) -> MemoryFact | None:
    statement = statement.strip().strip(" .;")
    if not statement or statement.endswith("?") or "```" in statement:
        return None
    if re.match(r"^(?:what i (?:said|told)|(?:me|this|that|it|everything|anything)$)", statement, re.I):
        return None
    # Quoted examples, questions, conditionals, and instructions do not match
    # these anchored first-person patterns and are not inferred as profile data.
    patterns = (
        (r"my\s+name\s+is\s+(.+)", "Name"),
        (r"call\s+me\s+(.+)", "Name"),
        (r"i\s+live\s+in\s+(.+)", "Location"),
        (r"i(?:\s+am|'m|’m)\s+from\s+(.+)", "Hometown"),
        (r"i(?:\s+am|'m|’m)\s+(\d{1,3})\s+years?\s+old", "Age"),
        (r"i\s+work\s+as\s+(?:an?\s+)?(.+)", "Occupation"),
        (r"i(?:\s+am|'m|’m)\s+(?:an?\s+)((?:[\w-]+\s+)*(?:student|engineer|developer|teacher|doctor|nurse|designer|researcher|programmer|artist|writer|manager|accountant|photographer))", "Occupation"),
        (r"i\s+(?:study|am studying)\s+at\s+(.+)", "School"),
        (r"i(?:\s+am|'m|’m)\s+(?:learning|studying)\s+(.+)", "Learning"),
    )
    for pattern, label in patterns:
        if match := re.fullmatch(pattern, statement, re.I):
            value = match.group(1).strip()
            if re.match(r"(?:not|no longer|unknown|a secret)\b", value, re.I):
                return None
            return make_fact(label, value)
    # Covers facts such as "My dog's name is Luna" and "My exam is on Monday".
    subject = r"my\s+(.{1,80}?)" if not explicit else r"(.{1,80}?)"
    if match := re.fullmatch(subject + r"\s+(?:is|are)\s+(.+)", statement, re.I):
        value = match.group(2).strip()
        if not re.match(r"(?:not|no longer|unknown)\b", value, re.I):
            return make_fact(match.group(1), value)
        return None
    if explicit or re.match(r"^i\s+(?:like|love|enjoy|prefer|have|own)\b", statement, re.I):
        return _note(statement)
    return None


def extract_facts(message: str) -> tuple[MemoryFact, ...]:
    """Extract only direct statements; asking a question never creates a fact."""
    if not isinstance(message, str) or not message.strip():
        return ()
    text = message.strip()
    if re.search(r"\b(?:do not|don't|don’t)\s+(?:save|remember|store)\b", text, re.I):
        return ()
    explicit = re.match(r"^(?:please\s+)?remember(?:\s+that)?(?:\s*[:,]\s*|\s+)(.+)$", text, re.I | re.S)
    if explicit:
        text = explicit.group(1).strip()
    text = re.sub(r"^(?:actually|correction)[,:]?\s+", "", text, flags=re.I)
    statements = re.split(
        r"(?<=[.!?;])\s+|\n+|,\s*(?=(?:my|i)\b)|\s+and\s+(?=(?:my|i)\b)", text, flags=re.I,
    )
    facts = {}
    for statement in statements:
        try:
            fact = _statement_fact(statement, explicit=explicit is not None)
        except ValueError:
            continue
        if fact is not None:
            facts[fact.key] = fact  # The latest value in this message wins.
    return tuple(facts.values())


def _words(text: str) -> set[str]:
    words = set(re.findall(r"[^\W_]+", text.casefold())) - STOP_WORDS
    return words | {word[:-1] for word in words if len(word) > 4 and word.endswith("s")}


def select_memories(message: str, facts: Sequence[MemoryFact], char_budget: int = MEMORY_CONTEXT_CHARS) -> tuple[MemoryFact, ...]:
    """Prefer relevant labels, then values; keep names and recent facts useful."""
    query = _words(message)
    if query & {"live", "living", "city", "based"}:
        query.add("location")
    if query & {"old", "years"}:
        query.add("age")
    if query & {"job", "work", "profession"}:
        query.add("occupation")
    if query & {"called", "call"}:
        query.add("name")

    def score(fact: MemoryFact) -> int:
        return 4 * len(query & _words(fact.label)) + len(query & _words(fact.value)) + (2 if fact.key == "profile:name" else 0)

    selected = []
    used = 2  # JSON list brackets.
    for fact in sorted(facts, key=score, reverse=True):
        size = len(json.dumps({"fact": fact.label, "value": fact.value}, ensure_ascii=False)) + 2
        if used + size <= char_budget:
            selected.append(fact)
            used += size
    return tuple(selected)


class MemoryManager:
    def __init__(self, database: Database):
        self.database = database

    def import_existing_facts(self) -> int:
        """One-time upgrade: use past USER statements, never assistant guesses.

        The completion flag prevents deleted facts from returning on a restart.
        Existing manually saved facts take precedence over imported history.
        """
        with self.database.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if connection.execute("SELECT history_imported FROM memory_settings WHERE id = 1").fetchone()[0]:
                return 0
            existing = {row[0] for row in connection.execute("SELECT memory_key FROM user_memories")}
            candidates = {}
            for row in connection.execute("SELECT id, content FROM messages WHERE role = 'user' ORDER BY id"):
                for fact in extract_facts(row["content"]):
                    if fact.key not in existing:
                        candidates[fact.key] = (fact, row["id"])
            for fact, source_id in sorted(candidates.values(), key=lambda item: item[1]):
                self.database.store_memory(connection, fact, source_id)
            connection.execute(
                "UPDATE memory_settings SET history_imported = 1, revision = revision + ? WHERE id = 1",
                (int(bool(candidates)),),
            )
            return len(candidates)

    def prepare(self, message: str, history: Sequence[Mapping] = ()) -> MemoryPlan:
        snapshot = self.database.memory_snapshot()
        stored = [MemoryFact(row["memory_key"], row["label"], row["value"]) for row in snapshot["facts"]]
        updates = extract_facts(message)
        previous = {fact.key: fact for fact in stored}
        corrected = any(fact.key in previous and previous[fact.key].value != fact.value for fact in updates)
        combined = [*updates, *(fact for fact in stored if fact.key not in {new.key for new in updates})]
        # Earlier visible chats remain stored. Removing them from the prompt
        # prevents a corrected/deleted value from resurfacing through history.
        recent = () if corrected else tuple(
            message for message in history if message.get("id", 0) > snapshot["history_after_id"]
        )
        return MemoryPlan(select_memories(message, combined), updates, recent, snapshot["revision"])

    def save_fact(self, label: str, value: str) -> None:
        self.database.save_memory(make_fact(label, value))

    def update_fact(self, memory_id: int, value: str) -> None:
        records = self.database.memory_snapshot()["facts"]
        row = next((record for record in records if record["id"] == memory_id), None)
        if row is None:
            raise ValueError("This fact is no longer available. Reopen Saved memory.")
        fact = _note(value) if row["memory_key"].startswith("note:") else make_fact(row["label"], value)
        self.database.update_memory(memory_id, fact)
