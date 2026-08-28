# ABOUTME: Claude provider for the council. Uses Opus 4.7 via Batches API.
# ABOUTME: Forced tool-use guarantees structured {label, rationale, confidence} output.

import json
import time
from pathlib import Path

import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

from ..prompt import SYSTEM_PROMPT, format_user_message
from ..schema import CLAUDE_TOOL, LABEL_VALUES
from .base import BaseCouncilProvider, CouncilLabel

# Batches API pricing for Opus 4.7 (50% off list): $2.50 in, $12.50 out per 1M tokens.
# Per-sentence ~700 in + 100 out tokens.
_OPUS_INPUT_PER_M = 2.50
_OPUS_OUTPUT_PER_M = 12.50
_TOKENS_IN_PER_SENTENCE = 700
_TOKENS_OUT_PER_SENTENCE = 100

POLL_INTERVAL_SEC = 20


class ClaudeProvider(BaseCouncilProvider):
    name = "claude"
    cost_per_sentence = (
        _TOKENS_IN_PER_SENTENCE / 1_000_000 * _OPUS_INPUT_PER_M
        + _TOKENS_OUT_PER_SENTENCE / 1_000_000 * _OPUS_OUTPUT_PER_M
    )

    def __init__(self, model: str = "claude-opus-4-7", max_tokens: int = 400):
        self.model = model
        self.max_tokens = max_tokens
        self._client = anthropic.Anthropic()
        # Index → record mapping is preserved by positional custom_id ("i-NNNNNN")
        # and the recorded sentence_ids, so duplicate-id corpora work.
        self._submitted_ids: list[str] = []

    # ------------------------------------------------------------------

    def estimate_cost(self, n: int) -> float:
        return n * self.cost_per_sentence

    def submit(self, records: list[dict], state_path: Path) -> str:
        self._submitted_ids = [r["id"] for r in records]
        requests = [
            Request(
                custom_id=f"i-{i:06d}",
                params=MessageCreateParamsNonStreaming(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    system=SYSTEM_PROMPT,
                    tools=[CLAUDE_TOOL],
                    tool_choice={"type": "tool", "name": "classify_liberty"},
                    messages=[{"role": "user", "content": format_user_message(r)}],
                ),
            )
            for i, r in enumerate(records)
        ]
        print(f"[claude] submitting {len(requests):,} requests to Batches API ({self.model})...")
        batch = self._client.messages.batches.create(requests=requests)
        print(f"[claude] batch_id={batch.id} status={batch.processing_status}")

        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps({
            "provider": self.name,
            "model": self.model,
            "batch_id": batch.id,
            "submitted_ids": self._submitted_ids,
        }))
        return batch.id

    def collect(self, handle: str) -> list[CouncilLabel]:
        # If submit() wasn't called in this process, fall back to reading the state file
        # — but normal flow keeps the ids in memory.
        ids = self._submitted_ids
        n = len(ids)
        provider_id = f"{self.name}:{self.model}"

        # Poll until done
        consecutive_errors = 0
        while True:
            try:
                batch = self._client.messages.batches.retrieve(handle)
                consecutive_errors = 0
            except Exception as e:
                consecutive_errors += 1
                print(f"[claude] poll error ({consecutive_errors}): {e}", flush=True)
                if consecutive_errors >= 10:
                    raise
                time.sleep(min(POLL_INTERVAL_SEC * consecutive_errors, 120))
                continue
            counts = batch.request_counts
            print(
                f"[claude] [{time.strftime('%H:%M:%S')}] {batch.processing_status} | "
                f"processing={counts.processing} succeeded={counts.succeeded} "
                f"errored={counts.errored} canceled={counts.canceled} expired={counts.expired}",
                flush=True,
            )
            if batch.processing_status == "ended":
                break
            time.sleep(POLL_INTERVAL_SEC)

        # Index results by custom_id
        by_index: dict[int, CouncilLabel] = {}
        for result in self._client.messages.batches.results(handle):
            try:
                idx = int(result.custom_id.split("-", 1)[1])
            except (IndexError, ValueError):
                continue
            sid = ids[idx] if idx < len(ids) else result.custom_id
            label_obj = self._parse_result(result, sid, provider_id)
            by_index[idx] = label_obj

        # Fill in gaps with errors so order matches submission
        return [
            by_index.get(i, CouncilLabel(
                sentence_id=ids[i] if i < len(ids) else f"i-{i}",
                label="error",
                rationale="",
                confidence=-1.0,
                raw_provider_id=provider_id,
                error="result missing from batch",
            ))
            for i in range(n)
        ]

    # ------------------------------------------------------------------

    def _parse_result(self, result, sentence_id: str, provider_id: str) -> CouncilLabel:
        rtype = result.result.type
        if rtype != "succeeded":
            err_msg = rtype
            if rtype == "errored" and getattr(result.result, "error", None):
                outer = result.result.error
                inner = getattr(outer, "error", None)
                if inner and getattr(inner, "message", None):
                    err_msg = f"{getattr(inner, 'type', 'error')}: {inner.message}"
                elif getattr(outer, "message", None):
                    err_msg = outer.message
                else:
                    err_msg = getattr(outer, "type", "error")
            return CouncilLabel(
                sentence_id=sentence_id,
                label="error",
                rationale="",
                confidence=-1.0,
                raw_provider_id=provider_id,
                error=str(err_msg),
            )

        msg = result.result.message
        tool_block = next(
            (b for b in msg.content if b.type == "tool_use" and b.name == "classify_liberty"),
            None,
        )
        if tool_block is None:
            return CouncilLabel(
                sentence_id=sentence_id,
                label="error",
                rationale="",
                confidence=-1.0,
                raw_provider_id=provider_id,
                error="no tool_use block",
            )

        inp = tool_block.input or {}
        label = inp.get("label", "error")
        if label not in LABEL_VALUES:
            label = "error"
        rationale = (inp.get("rationale") or "").strip()
        try:
            confidence = float(inp.get("confidence", -1.0))
        except (TypeError, ValueError):
            confidence = -1.0
        if confidence >= 0:  # clamp valid values to [0, 1]
            confidence = max(0.0, min(1.0, confidence))
        return CouncilLabel(
            sentence_id=sentence_id,
            label=label,
            rationale=rationale,
            confidence=confidence,
            raw_provider_id=provider_id,
        )
