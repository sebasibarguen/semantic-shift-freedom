# ABOUTME: Abstract provider interface for the LLM council.
# ABOUTME: Each vendor (Claude, GPT, Gemini) implements submit + collect.

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class CouncilLabel:
    """One model's verdict on one sentence."""
    sentence_id: str
    label: str          # one of LABEL_VALUES, or "error" / "missing"
    rationale: str
    confidence: float   # [0, 1]; -1 if missing
    raw_provider_id: str  # e.g. "claude:claude-opus-4-7"
    error: str | None = None


class BaseCouncilProvider(ABC):
    """Vendor-agnostic interface. Each provider knows how to:
       1. submit a list of (sentence_id, record) pairs as a batch
       2. collect results back as a list of CouncilLabel"""

    name: str               # short identifier, e.g. "claude"
    model: str              # provider-specific model id
    cost_per_sentence: float  # ballpark $/sentence for budgeting

    @abstractmethod
    def submit(self, records: list[dict], state_path: Path) -> str:
        """Submit a batch. Persist any provider-specific state to state_path
        (batch id, file id, etc.). Return a string handle suitable for
        passing back to collect()."""

    @abstractmethod
    def collect(self, handle: str) -> list[CouncilLabel]:
        """Wait for the batch to finish and return one CouncilLabel per
        record (in the same order they were submitted; missing entries
        emitted as label='error')."""

    @abstractmethod
    def estimate_cost(self, n: int) -> float:
        """Ballpark dollar cost for n sentences."""
