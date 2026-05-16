# ABOUTME: Evaluates a candidate prompt by running the small model on a split,
# ABOUTME: computing agreement vs council gold, F1, confusion matrix, and sample errors.

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

from ..council.schema import LABEL_VALUES, CLAUDE_TOOL
from .history import log_evaluation, lookup_eval, prompt_hash


@dataclass
class EvalResult:
    prompt_hash: str
    prompt_path: str
    split: str
    model: str
    n: int
    n_skipped: int
    accuracy: float
    per_class_f1: dict[str, float]
    per_class_precision: dict[str, float]
    per_class_recall: dict[str, float]
    confusion_matrix: dict[str, dict[str, int]]
    cached: bool = False
    sample_errors: list[dict] | None = None


# Same per-sentence formatter as classify_liberty
def _format_user_message(record: dict) -> str:
    year = record.get("year")
    speaker = record.get("speaker")
    party = record.get("party")
    bits = [str(year) if year else None, speaker, party]
    header = ", ".join(b for b in bits if b)
    sentence = (record.get("sentence") or "").strip()
    head = f"Sentence ({header}): {sentence}" if header else f"Sentence: {sentence}"
    return f"{head}\n\nClassify this sentence."


def _run_haiku_batch(
    client: anthropic.Anthropic,
    system_prompt: str,
    records: list[dict],
    model: str,
    max_tokens: int = 300,
    poll_interval_sec: int = 20,
) -> list[dict]:
    """Submit one Haiku batch, return list of {label, rationale} aligned with records."""
    import time

    requests = [
        Request(
            custom_id=f"i-{i:06d}",
            params=MessageCreateParamsNonStreaming(
                model=model,
                max_tokens=max_tokens,
                system=system_prompt,
                tools=[CLAUDE_TOOL],
                tool_choice={"type": "tool", "name": "classify_liberty"},
                messages=[{"role": "user", "content": _format_user_message(r)}],
            ),
        )
        for i, r in enumerate(records)
    ]
    print(f"[arena] submitting {len(requests):,} requests to Batches API ({model})...")
    batch = client.messages.batches.create(requests=requests)
    print(f"[arena] batch_id={batch.id}")

    while True:
        b = client.messages.batches.retrieve(batch.id)
        if b.processing_status == "ended":
            break
        print(f"[arena] [{time.strftime('%H:%M:%S')}] {b.processing_status} | "
              f"processing={b.request_counts.processing} succeeded={b.request_counts.succeeded} "
              f"errored={b.request_counts.errored}")
        time.sleep(poll_interval_sec)

    by_index: dict[int, dict] = {}
    for result in client.messages.batches.results(batch.id):
        try:
            idx = int(result.custom_id.split("-", 1)[1])
        except (IndexError, ValueError):
            continue
        if result.result.type != "succeeded":
            by_index[idx] = {"label": "error", "rationale": ""}
            continue
        msg = result.result.message
        tool_block = next(
            (b for b in msg.content if b.type == "tool_use" and b.name == "classify_liberty"),
            None,
        )
        if tool_block is None:
            by_index[idx] = {"label": "error", "rationale": ""}
            continue
        inp = tool_block.input or {}
        label = inp.get("label") if inp.get("label") in LABEL_VALUES else "error"
        by_index[idx] = {"label": label, "rationale": (inp.get("rationale") or "").strip()}

    return [by_index.get(i, {"label": "error", "rationale": ""}) for i in range(len(records))]


def compute_metrics(predicted: list[str], gold: list[str], labels: list[str]) -> dict:
    """Per-class precision/recall/F1 + confusion matrix + accuracy.
    Both lists must be same length. Skips entries with predicted == 'error'.
    """
    assert len(predicted) == len(gold)
    n = len(predicted)
    n_skipped = sum(1 for p in predicted if p == "error")

    valid_pred, valid_gold = [], []
    for p, g in zip(predicted, gold):
        if p == "error":
            continue
        valid_pred.append(p)
        valid_gold.append(g)

    correct = sum(1 for p, g in zip(valid_pred, valid_gold) if p == g)
    accuracy = correct / len(valid_pred) if valid_pred else 0.0

    cm: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for p, g in zip(valid_pred, valid_gold):
        cm[p][g] += 1

    per_p, per_r, per_f1 = {}, {}, {}
    for label in labels:
        tp = cm[label].get(label, 0)
        fp = sum(cm[label].get(other, 0) for other in labels if other != label)
        fn = sum(cm[other].get(label, 0) for other in labels if other != label)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_p[label] = round(precision, 4)
        per_r[label] = round(recall, 4)
        per_f1[label] = round(f1, 4)

    cm_serializable = {row: dict(d) for row, d in cm.items()}
    return {
        "n": n,
        "n_skipped": n_skipped,
        "accuracy": round(accuracy, 4),
        "per_class_precision": per_p,
        "per_class_recall": per_r,
        "per_class_f1": per_f1,
        "confusion_matrix": cm_serializable,
    }


def evaluate_prompt(
    prompt_path: Path,
    records: list[dict],
    split: str,
    model: str,
    history_path: Path,
    use_cache: bool = True,
    sample_errors: int = 30,
) -> EvalResult:
    """Run a single eval. Returns EvalResult with cache hit info."""
    prompt_text = Path(prompt_path).read_text()
    pid = prompt_hash(prompt_text)

    cached = None
    if use_cache:
        cached = lookup_eval(history_path, pid, split, model)
        # Cache hit is only valid if the sample size matches (otherwise the
        # underlying gold dataset has changed — pilot → full, etc.)
        if cached is not None and cached.get("n") != len(records):
            cached = None

    if cached is not None:
        return EvalResult(
            prompt_hash=pid,
            prompt_path=str(prompt_path),
            split=split,
            model=model,
            n=cached.get("n", 0),
            n_skipped=cached.get("n_skipped", 0),
            accuracy=cached.get("accuracy", 0.0),
            per_class_f1=cached.get("per_class_f1", {}),
            per_class_precision=cached.get("per_class_precision", {}),
            per_class_recall=cached.get("per_class_recall", {}),
            confusion_matrix=cached.get("confusion_matrix", {}),
            cached=True,
            sample_errors=cached.get("sample_errors"),
        )

    client = anthropic.Anthropic()
    predictions = _run_haiku_batch(client, prompt_text, records, model=model)

    pred_labels = [p["label"] for p in predictions]
    gold_labels = [r["gold_label"] for r in records]

    metrics = compute_metrics(pred_labels, gold_labels, list(LABEL_VALUES))

    # Sample errors for inspection
    errors = []
    for r, pred in zip(records, predictions):
        if pred["label"] == "error" or pred["label"] == r["gold_label"]:
            continue
        errors.append({
            "id": r["id"],
            "year": r.get("year"),
            "sentence": r.get("sentence"),
            "gold_label": r["gold_label"],
            "predicted_label": pred["label"],
            "predicted_rationale": pred.get("rationale"),
            "council_rationales": r.get("rationales", [])[:3],
        })
    errors = errors[:sample_errors]

    entry = {
        "prompt_hash": pid,
        "prompt_path": str(prompt_path),
        "split": split,
        "model": model,
        "sample_errors": errors,
        **metrics,
    }
    log_evaluation(history_path, entry)

    return EvalResult(
        prompt_hash=pid,
        prompt_path=str(prompt_path),
        split=split,
        model=model,
        n=metrics["n"],
        n_skipped=metrics["n_skipped"],
        accuracy=metrics["accuracy"],
        per_class_f1=metrics["per_class_f1"],
        per_class_precision=metrics["per_class_precision"],
        per_class_recall=metrics["per_class_recall"],
        confusion_matrix=metrics["confusion_matrix"],
        sample_errors=errors,
    )
