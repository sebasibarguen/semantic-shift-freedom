# ABOUTME: Google Gemini provider for the council. Uses Gemini 3.1 preview via batch mode.
# ABOUTME: Forces structured JSON output via response_schema.

import json
import os
import time
from pathlib import Path

from google import genai
from google.genai import types

from ..prompt import SYSTEM_PROMPT, format_user_message
from ..schema import COUNCIL_OUTPUT_SCHEMA, LABEL_VALUES
from .base import BaseCouncilProvider, CouncilLabel


# Batch pricing 50% off list. Conservative estimate; adjust when published.
_INPUT_PER_M = 0.625
_OUTPUT_PER_M = 5.0
_TOKENS_IN_PER_SENTENCE = 700
_TOKENS_OUT_PER_SENTENCE = 100

POLL_INTERVAL_SEC = 30


class GeminiProvider(BaseCouncilProvider):
    name = "gemini"
    cost_per_sentence = (
        _TOKENS_IN_PER_SENTENCE / 1_000_000 * _INPUT_PER_M
        + _TOKENS_OUT_PER_SENTENCE / 1_000_000 * _OUTPUT_PER_M
    )

    def __init__(self, model: str = "gemini-3.1-pro-preview"):
        self.model = model
        api_key = os.environ.get("GOOGLE_AI_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_AI_API_KEY not set")
        self._client = genai.Client(api_key=api_key)
        self._submitted_ids: list[str] = []

    # ------------------------------------------------------------------

    def estimate_cost(self, n: int) -> float:
        return n * self.cost_per_sentence

    def submit(self, records: list[dict], state_path: Path) -> str:
        """Inline batch creation with InlinedRequest objects."""
        self._submitted_ids = [r["id"] for r in records]

        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=_to_openapi_schema(COUNCIL_OUTPUT_SCHEMA),
            temperature=0.0,
        )

        inlined = [
            types.InlinedRequest(
                model=self.model,
                contents=[
                    types.Content(
                        role="user",
                        parts=[types.Part(text=format_user_message(rec))],
                    )
                ],
                config=config,
                metadata={"key": f"i-{i:06d}"},
            )
            for i, rec in enumerate(records)
        ]

        print(f"[gemini] submitting batch ({len(inlined):,} requests, model={self.model})...")
        batch = self._client.batches.create(
            model=self.model,
            src=inlined,
        )
        print(f"[gemini] batch_name={batch.name} state={batch.state}")

        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps({
            "provider": self.name,
            "model": self.model,
            "batch_name": batch.name,
            "submitted_ids": self._submitted_ids,
        }))
        return batch.name

    def collect(self, handle: str) -> list[CouncilLabel]:
        ids = self._submitted_ids
        n = len(ids)
        provider_id = f"{self.name}:{self.model}"

        terminal_ok = {"JOB_STATE_SUCCEEDED"}
        terminal_bad = {"JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"}

        def _state_name(s) -> str:
            """Normalize whether SDK returns enum or string."""
            return getattr(s, "name", None) or str(s).split(".")[-1]

        while True:
            batch = self._client.batches.get(name=handle)
            state = _state_name(batch.state)
            print(f"[gemini] [{time.strftime('%H:%M:%S')}] state={state}", flush=True)
            if state in terminal_ok or state in terminal_bad:
                break
            time.sleep(POLL_INTERVAL_SEC)

        final_state = _state_name(batch.state)
        if final_state not in terminal_ok:
            err = getattr(batch, "error", None)
            return [
                CouncilLabel(
                    sentence_id=ids[i] if i < len(ids) else f"i-{i}",
                    label="error", rationale="", confidence=-1.0,
                    raw_provider_id=provider_id,
                    error=f"batch {final_state}: {err}",
                )
                for i in range(n)
            ]

        # Per-request results live under batch.dest.inlined_responses for inlined batches.
        by_index: dict[int, CouncilLabel] = {}
        dest = getattr(batch, "dest", None)
        responses = getattr(dest, "inlined_responses", None) if dest else None
        if responses is None:
            # Some SDK versions expose under batch.responses
            responses = getattr(batch, "responses", []) or []

        for resp in responses:
            # metadata is a dict on InlinedResponse: {"key": "i-NNNNNN"}
            metadata = getattr(resp, "metadata", None) or {}
            key = metadata.get("key") if isinstance(metadata, dict) else getattr(metadata, "key", None)
            idx = self._index_from_key(key)
            if idx is None:
                continue
            sid = ids[idx] if idx < len(ids) else key
            by_index[idx] = self._parse_response(resp, sid, provider_id)

        return [
            by_index.get(i, CouncilLabel(
                sentence_id=ids[i] if i < len(ids) else f"i-{i}",
                label="error", rationale="", confidence=-1.0,
                raw_provider_id=provider_id,
                error="result missing from batch",
            ))
            for i in range(n)
        ]

    # ------------------------------------------------------------------

    def _index_from_key(self, key: str | None) -> int | None:
        if not key:
            return None
        try:
            return int(key.split("-", 1)[1])
        except (IndexError, ValueError):
            return None

    def _parse_response(self, resp, sentence_id: str, provider_id: str) -> CouncilLabel:
        # Per-request error
        err = getattr(resp, "error", None)
        if err:
            return CouncilLabel(
                sentence_id=sentence_id, label="error", rationale="", confidence=-1.0,
                raw_provider_id=provider_id, error=str(err),
            )

        # Extract the GenerateContentResponse
        gcr = getattr(resp, "response", None)
        if gcr is None:
            return CouncilLabel(
                sentence_id=sentence_id, label="error", rationale="", confidence=-1.0,
                raw_provider_id=provider_id, error="no response object",
            )

        text = getattr(gcr, "text", None)
        if not text:
            # Fallback: dig into candidates
            cands = getattr(gcr, "candidates", []) or []
            if cands:
                content = getattr(cands[0], "content", None)
                parts = getattr(content, "parts", []) if content else []
                text = "".join(getattr(p, "text", "") or "" for p in parts)

        if not text:
            return CouncilLabel(
                sentence_id=sentence_id, label="error", rationale="", confidence=-1.0,
                raw_provider_id=provider_id, error="empty response",
            )

        try:
            payload = json.loads(text)
        except json.JSONDecodeError as e:
            return CouncilLabel(
                sentence_id=sentence_id, label="error", rationale="", confidence=-1.0,
                raw_provider_id=provider_id, error=f"json parse: {e}",
            )

        label = payload.get("label", "error")
        if label not in LABEL_VALUES:
            label = "error"
        rationale = (payload.get("rationale") or "").strip()
        try:
            confidence = float(payload.get("confidence", -1.0))
        except (TypeError, ValueError):
            confidence = -1.0
        if confidence >= 0:
            confidence = max(0.0, min(1.0, confidence))
        return CouncilLabel(
            sentence_id=sentence_id, label=label, rationale=rationale,
            confidence=confidence, raw_provider_id=provider_id,
        )


def _to_openapi_schema(schema: dict) -> dict:
    """Strip JSON-Schema-only fields that Gemini's response_schema rejects.

    Gemini accepts an OpenAPI 3.0 subset, not full JSON Schema:
    - additionalProperties: not supported (must be removed)
    - $schema, $defs, $ref: not in this codebase but would also be invalid
    """
    if isinstance(schema, dict):
        cleaned = {k: _to_openapi_schema(v) for k, v in schema.items()
                   if k not in ("additionalProperties", "$schema", "$defs", "$ref")}
        return cleaned
    if isinstance(schema, list):
        return [_to_openapi_schema(v) for v in schema]
    return schema
