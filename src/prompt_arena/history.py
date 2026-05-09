# ABOUTME: Append-only history of prompt evaluations.
# ABOUTME: One JSON line per (prompt, split, model) eval — enables caching and audit.

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def prompt_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def _now_iso() -> str:
    """ISO 8601 UTC timestamp with microseconds — sortable, stable across same-second writes."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")


def log_evaluation(history_path: Path, entry: dict) -> None:
    """Append a single evaluation record to the history JSONL."""
    history_path.parent.mkdir(parents=True, exist_ok=True)
    record = {"timestamp": _now_iso(), **entry}
    with history_path.open("a") as f:
        f.write(json.dumps(record) + "\n")


def lookup_eval(history_path: Path, prompt_hash_value: str, split: str, model: str) -> dict | None:
    """Return the most recent matching eval, or None."""
    if not history_path.exists():
        return None
    matching: list[dict] = []
    with history_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (rec.get("prompt_hash") == prompt_hash_value
                    and rec.get("split") == split
                    and rec.get("model") == model):
                matching.append(rec)
    return matching[-1] if matching else None


def history_table(history_path: Path) -> list[dict]:
    """Return all logged evals (most recent first)."""
    if not history_path.exists():
        return []
    rows: list[dict] = []
    with history_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    rows.sort(key=lambda r: r.get("timestamp", ""), reverse=True)
    return rows
