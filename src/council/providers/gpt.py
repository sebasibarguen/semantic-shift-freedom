# ABOUTME: OpenAI provider for the council. Uses GPT 5.5 via Batch API.
# ABOUTME: Forces structured JSON output via response_format json_schema.

import io
import json
import time
from pathlib import Path

from openai import OpenAI

from ..prompt import SYSTEM_PROMPT, format_user_message
from ..schema import COUNCIL_OUTPUT_SCHEMA, LABEL_VALUES
from .base import BaseCouncilProvider, CouncilLabel

# Batch API pricing (50% off list). GPT-5.5 pricing not yet definitively
# known — using a conservative Sonnet-class estimate; adjust when published.
_INPUT_PER_M = 1.25
_OUTPUT_PER_M = 10.0
_TOKENS_IN_PER_SENTENCE = 700
_TOKENS_OUT_PER_SENTENCE = 100

POLL_INTERVAL_SEC = 30


class GPTProvider(BaseCouncilProvider):
    name = "gpt"
    cost_per_sentence = (
        _TOKENS_IN_PER_SENTENCE / 1_000_000 * _INPUT_PER_M
        + _TOKENS_OUT_PER_SENTENCE / 1_000_000 * _OUTPUT_PER_M
    )

    def __init__(
        self,
        model: str = "gpt-5.5",
        max_completion_tokens: int = 400,
    ):
        self.model = model
        self.max_completion_tokens = max_completion_tokens
        self._client = OpenAI()
        self._submitted_ids: list[str] = []

    # ------------------------------------------------------------------

    def estimate_cost(self, n: int) -> float:
        return n * self.cost_per_sentence

    def submit(self, records: list[dict], state_path: Path) -> str:
        self._submitted_ids = [r["id"] for r in records]

        # Build JSONL: one line per request, OpenAI Batch input format.
        lines = []
        for i, rec in enumerate(records):
            body = {
                "model": self.model,
                "max_completion_tokens": self.max_completion_tokens,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": format_user_message(rec)},
                ],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "classify_liberty",
                        "strict": True,
                        "schema": _strict_schema(COUNCIL_OUTPUT_SCHEMA),
                    },
                },
            }
            lines.append(json.dumps({
                "custom_id": f"i-{i:06d}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body,
            }))

        jsonl_bytes = ("\n".join(lines) + "\n").encode("utf-8")
        print(f"[gpt] uploading batch input ({len(jsonl_bytes):,} bytes, {len(lines):,} requests)...")

        upload = self._client.files.create(
            file=("council_batch.jsonl", io.BytesIO(jsonl_bytes), "application/jsonl"),
            purpose="batch",
        )
        print(f"[gpt] file_id={upload.id}")

        batch = self._client.batches.create(
            input_file_id=upload.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        print(f"[gpt] batch_id={batch.id} status={batch.status}")

        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps({
            "provider": self.name,
            "model": self.model,
            "input_file_id": upload.id,
            "batch_id": batch.id,
            "submitted_ids": self._submitted_ids,
        }))
        return batch.id

    def collect(self, handle: str) -> list[CouncilLabel]:
        ids = self._submitted_ids
        n = len(ids)
        provider_id = f"{self.name}:{self.model}"

        # Poll
        terminal_ok = {"completed"}
        terminal_bad = {"failed", "expired", "cancelled"}
        consecutive_errors = 0
        while True:
            try:
                batch = self._client.batches.retrieve(handle)
                consecutive_errors = 0
            except Exception as e:
                consecutive_errors += 1
                print(f"[gpt] poll error ({consecutive_errors}): {e}", flush=True)
                if consecutive_errors >= 10:
                    raise  # 10 consecutive failures = give up
                time.sleep(min(POLL_INTERVAL_SEC * consecutive_errors, 120))
                continue
            counts = batch.request_counts
            print(
                f"[gpt] [{time.strftime('%H:%M:%S')}] {batch.status} | "
                f"completed={counts.completed}/{counts.total} failed={counts.failed}",
                flush=True,
            )
            if batch.status in terminal_ok or batch.status in terminal_bad:
                break
            time.sleep(POLL_INTERVAL_SEC)

        # Output file may be None if the whole batch failed
        out_id = batch.output_file_id
        err_id = batch.error_file_id

        by_index: dict[int, CouncilLabel] = {}

        if out_id:
            content = self._client.files.content(out_id).read()
            for line in content.decode("utf-8").splitlines():
                if not line.strip():
                    continue
                obj = json.loads(line)
                idx = self._index_from_custom_id(obj.get("custom_id", ""))
                if idx is None:
                    continue
                sid = ids[idx] if idx < len(ids) else obj.get("custom_id", "")
                by_index[idx] = self._parse_response(obj, sid, provider_id)

        # Capture errored requests
        if err_id:
            err_content = self._client.files.content(err_id).read()
            for line in err_content.decode("utf-8").splitlines():
                if not line.strip():
                    continue
                obj = json.loads(line)
                idx = self._index_from_custom_id(obj.get("custom_id", ""))
                if idx is None or idx in by_index:
                    continue
                sid = ids[idx] if idx < len(ids) else obj.get("custom_id", "")
                err_msg = (
                    (obj.get("error") or {}).get("message")
                    or (obj.get("response", {}).get("body", {}).get("error") or {}).get("message")
                    or "errored in batch output"
                )
                by_index[idx] = CouncilLabel(
                    sentence_id=sid, label="error", rationale="", confidence=-1.0,
                    raw_provider_id=provider_id, error=err_msg,
                )

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

    def _index_from_custom_id(self, cid: str) -> int | None:
        try:
            return int(cid.split("-", 1)[1])
        except (IndexError, ValueError):
            return None

    def _parse_response(self, obj: dict, sentence_id: str, provider_id: str) -> CouncilLabel:
        # Top-level error
        if obj.get("error"):
            return CouncilLabel(
                sentence_id=sentence_id, label="error", rationale="", confidence=-1.0,
                raw_provider_id=provider_id, error=str(obj["error"]),
            )
        body = obj.get("response", {}).get("body", {})
        if not body or body.get("error"):
            return CouncilLabel(
                sentence_id=sentence_id, label="error", rationale="", confidence=-1.0,
                raw_provider_id=provider_id, error=str(body.get("error", "no body")),
            )
        choices = body.get("choices") or []
        if not choices:
            return CouncilLabel(
                sentence_id=sentence_id, label="error", rationale="", confidence=-1.0,
                raw_provider_id=provider_id, error="no choices in response",
            )
        message = choices[0].get("message") or {}
        # Refusal handling
        if message.get("refusal"):
            return CouncilLabel(
                sentence_id=sentence_id, label="error", rationale="", confidence=-1.0,
                raw_provider_id=provider_id, error=f"refusal: {message['refusal']}",
            )
        text = message.get("content") or ""
        if not text:
            return CouncilLabel(
                sentence_id=sentence_id, label="error", rationale="", confidence=-1.0,
                raw_provider_id=provider_id, error="empty content",
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


def _strict_schema(schema: dict) -> dict:
    """OpenAI strict json_schema requires every property to be in `required`
    and `additionalProperties: false`. Our base schema already complies; this
    is a defensive copy in case the schema gets relaxed elsewhere."""
    s = json.loads(json.dumps(schema))  # deep copy
    if s.get("type") == "object":
        s.setdefault("additionalProperties", False)
        s.setdefault("required", list((s.get("properties") or {}).keys()))
    return s
