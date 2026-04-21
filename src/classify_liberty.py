# ABOUTME: LLM classifier for Berlin's positive/negative liberty using Claude Haiku 4.5.
# ABOUTME: One sentence per request via the Message Batches API with strict tool-use output.

"""
Classify Hansard / EEBO sentences according to Isaiah Berlin's "Two Concepts of Liberty."

Design:
    - one request per sentence (independent reasoning)
    - Message Batches API (50% cheaper, async, cap 100K/batch)
    - strict tool-use: forces a structured {rationale, label} output, no text parsing
    - year/speaker/party passed into the user message for context

Usage:
    # Evaluate against the 100-sentence Opus comparison set
    uv run python src/classify_liberty.py --eval

    # Classify one decade file (overwrites methods.llm in place)
    uv run python src/classify_liberty.py --input web/data/sentences_1980s.json

    # Resume a previously-submitted batch
    uv run python src/classify_liberty.py --resume msgbatch_XXXXX --output <file>
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request
from dotenv import load_dotenv

load_dotenv()

MODEL = "claude-haiku-4-5"
POLL_INTERVAL_SEC = 20

SYSTEM_PROMPT = """You classify individual sentences from UK Parliamentary debates (Hansard, 1803-2025) according to Isaiah Berlin's "Two Concepts of Liberty."

CRITICAL RULE: "freedom of X" (speech, press, religion, debate, conscience, navigation, contract, trade, association, expression, movement) is almost always NEGATIVE liberty — it names freedom from interference with X. Do NOT classify these as positive liberty even when the surface grammar uses "to".

NEGATIVE LIBERTY (freedom FROM interference):
- "freedom of speech/press/religion/conscience/debate/expression" = negative
- "freedom from want/fear/oppression/torture" = negative
- "civil liberties", "personal liberty" re: detention, habeas corpus = negative
- "freedom of contract/trade/navigation/movement" = negative
- "restrict/curtail/infringe freedom" = negative (discussing removal of non-interference)
- rhetorical invocations of liberty as a cause opposed to tyranny = negative

POSITIVE LIBERTY (capacity/empowerment TO act):
- "freedom to choose one's school/doctor" = positive (enabled capacity)
- "freedom to innovate/compete/provide services" = positive (opportunity)
- "grant liberty to [country/body] to govern itself" = positive (empowerment)
- "financial freedom", self-sufficiency = positive
- welfare/education enabling people to do things = positive
- local authorities given "freedom to raise funding" = positive

AMBIGUOUS (genuine mixed or under-specified):
- sentences that explicitly present both enabling and constraining aspects
- bare "freedom" / "liberty" without enough context to decide
- both readings are defensible and the sentence does not disambiguate

OTHER (not making a claim about liberty-as-a-value):
- parliamentary procedure: "at liberty to speak", "took the liberty of"
- proper nouns and company/act names ("Liberty Steel", named Acts)
- contents/index entries, lists of page numbers, truncated headers
- the word appears incidentally without substantive use

Guidance:
- Reason about what is being freed and from-or-to what. The grammatical "from"/"to" is a hint, not a rule.
- Prefer "ambiguous" over guessing when a single-sentence excerpt genuinely does not disambiguate.
- Always call the classify_liberty tool exactly once per request."""

TOOL_DEFINITION = {
    "name": "classify_liberty",
    "description": "Record the classification of the sentence according to Berlin's distinction.",
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "rationale": {
                "type": "string",
                "description": "One sentence: what is being freed, and from-or-to what? Name the specific object of freedom in the sentence.",
            },
            "label": {
                "type": "string",
                "enum": ["positive_liberty", "negative_liberty", "ambiguous", "other"],
            },
        },
        "required": ["rationale", "label"],
        "additionalProperties": False,
    },
}


def format_user_message(record: dict) -> str:
    """Build the per-sentence user message with context fields."""
    parts = []
    year = record.get("year")
    speaker = record.get("speaker")
    party = record.get("party")
    header_bits = []
    if year:
        header_bits.append(str(year))
    if speaker:
        header_bits.append(speaker)
    if party:
        header_bits.append(party)
    header = f"({', '.join(header_bits)})" if header_bits else ""
    sentence = record["sentence"].strip()
    parts.append(f"Sentence {header}: {sentence}" if header else f"Sentence: {sentence}")
    parts.append("\nClassify this sentence.")
    return "".join(parts)


def build_request(record: dict, max_tokens: int = 300) -> Request:
    """Turn a sentence record into a Batch API Request."""
    return Request(
        custom_id=record["id"],
        params=MessageCreateParamsNonStreaming(
            model=MODEL,
            max_tokens=max_tokens,
            system=SYSTEM_PROMPT,
            tools=[TOOL_DEFINITION],
            tool_choice={"type": "tool", "name": "classify_liberty"},
            messages=[{"role": "user", "content": format_user_message(record)}],
        ),
    )


def submit_batch(client: anthropic.Anthropic, records: list[dict]) -> str:
    """Submit one request per record, return the batch ID."""
    print(f"Building {len(records):,} requests...")
    requests = [build_request(r) for r in records]

    print("Submitting batch...")
    batch = client.messages.batches.create(requests=requests)
    print(f"Batch ID: {batch.id}")
    print(f"Status: {batch.processing_status}")
    return batch.id


def wait_for_batch(client: anthropic.Anthropic, batch_id: str):
    """Poll until the batch is done."""
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        counts = batch.request_counts
        status = batch.processing_status
        print(
            f"[{time.strftime('%H:%M:%S')}] {status} | "
            f"processing={counts.processing} succeeded={counts.succeeded} "
            f"errored={counts.errored} canceled={counts.canceled} expired={counts.expired}"
        )
        if status == "ended":
            return batch
        time.sleep(POLL_INTERVAL_SEC)


def collect_results(client: anthropic.Anthropic, batch_id: str) -> dict[str, dict]:
    """Fetch results, return {custom_id: {label, rationale, error?}}."""
    results: dict[str, dict] = {}
    succeeded = errored = 0
    for result in client.messages.batches.results(batch_id):
        cid = result.custom_id
        rtype = result.result.type
        if rtype == "succeeded":
            msg = result.result.message
            tool_block = next(
                (b for b in msg.content if b.type == "tool_use" and b.name == "classify_liberty"),
                None,
            )
            if tool_block is None:
                results[cid] = {"label": "error", "rationale": "no tool_use block in response"}
                errored += 1
                continue
            inp = tool_block.input
            results[cid] = {
                "label": inp.get("label", "error"),
                "rationale": inp.get("rationale", ""),
            }
            succeeded += 1
        elif rtype == "errored":
            results[cid] = {
                "label": "error",
                "rationale": f"api_error: {result.result.error.type}",
            }
            errored += 1
        else:  # canceled | expired
            results[cid] = {"label": "error", "rationale": rtype}
            errored += 1
    print(f"Collected {succeeded:,} succeeded, {errored:,} errored.")
    return results


def classify(records: list[dict], state_file: Path | None = None) -> dict[str, dict]:
    """Submit, wait, collect. Writes batch_id to state_file if provided."""
    client = anthropic.Anthropic()
    batch_id = submit_batch(client, records)
    if state_file:
        state_file.parent.mkdir(parents=True, exist_ok=True)
        state_file.write_text(json.dumps({"batch_id": batch_id}))
        print(f"Saved state to {state_file}")
    wait_for_batch(client, batch_id)
    return collect_results(client, batch_id)


# ---------- modes ---------------------------------------------------------

def run_eval():
    """Classify the 100 sentences in opus_vs_haiku.json, compare vs. Opus."""
    project_root = Path(__file__).parent.parent
    eval_path = project_root / "outputs" / "opus_vs_haiku.json"
    out_path = project_root / "outputs" / "haiku_v2_eval.json"
    state_path = project_root / "outputs" / ".batch_state_eval.json"

    eval_data = json.loads(eval_path.read_text())
    print(f"Loaded {len(eval_data)} eval sentences from {eval_path.name}")

    # Minimal records — opus_vs_haiku.json lacks speaker/party; that's fine.
    records = [
        {"id": d["id"], "sentence": d["sentence"], "year": d.get("year")}
        for d in eval_data
    ]
    results = classify(records, state_file=state_path)

    # Merge v2 results back
    for d in eval_data:
        res = results.get(d["id"], {"label": "error", "rationale": "missing"})
        d["haiku_v2"] = res["label"]
        d["haiku_v2_rationale"] = res["rationale"]

    out_path.write_text(json.dumps(eval_data, indent=2))
    print(f"Wrote {out_path}")
    print_eval_report(eval_data)


def print_eval_report(eval_data: list[dict]):
    """Confusion matrix vs. Opus for both Haiku v1 and v2."""
    labels = ["positive_liberty", "negative_liberty", "ambiguous", "other", "error"]

    def confusion(pred_key: str):
        cm: dict[tuple[str, str], int] = defaultdict(int)
        for d in eval_data:
            pred = d.get(pred_key, "error")
            truth = d.get("opus", "error")
            cm[(pred, truth)] += 1
        return cm

    def agreement(pred_key: str) -> float:
        total = len(eval_data)
        agree = sum(1 for d in eval_data if d.get(pred_key) == d.get("opus"))
        return agree / total * 100

    def print_cm(title: str, pred_key: str):
        cm = confusion(pred_key)
        print(f"\n{title}")
        print("=" * len(title))
        print(f"Agreement vs Opus: {agreement(pred_key):.1f}%  ({len(eval_data)} sentences)")
        print()
        header = "pred \\ opus".ljust(18) + "  ".join(l[:7].rjust(7) for l in labels)
        print(header)
        for row in labels:
            vals = [str(cm.get((row, col), 0)).rjust(7) for col in labels]
            print(row[:16].ljust(18) + "  ".join(vals))

    print_cm("HAIKU v1 (20-batch, text parse)", "haiku")
    print_cm("HAIKU v2 (per-sentence, tool-use, rationale)", "haiku_v2")

    # Bucket-level diff
    print("\nWhere v2 DIFFERS from v1:")
    diffs = [d for d in eval_data if d.get("haiku") != d.get("haiku_v2")]
    print(f"  {len(diffs)} of {len(eval_data)} sentences changed label")

    # Of those, how many moved toward Opus?
    moved_to_opus = sum(1 for d in diffs if d.get("haiku_v2") == d.get("opus"))
    moved_away = sum(
        1 for d in diffs
        if d.get("haiku") == d.get("opus") and d.get("haiku_v2") != d.get("opus")
    )
    print(f"  → agreed w/ Opus: {moved_to_opus}")
    print(f"  → disagreed w/ Opus (regression): {moved_away}")


def run_input_file(input_path: Path):
    """Classify every sentence in a file, merge results into methods.llm."""
    records = json.loads(input_path.read_text())
    if not isinstance(records, list):
        sys.exit("Input file must be a JSON list of sentence records")
    print(f"Loaded {len(records)} records from {input_path}")

    state_path = input_path.parent / f".batch_state_{input_path.stem}.json"
    results = classify(records, state_file=state_path)

    for r in records:
        res = results.get(r["id"], {"label": "error", "rationale": "missing"})
        r.setdefault("methods", {})
        r["methods"]["llm"] = {
            "label": res["label"],
            "rationale": res["rationale"],
            "model": MODEL,
            "version": "v2_batched_tooluse",
        }

    input_path.write_text(json.dumps(records, separators=(",", ":")))
    print(f"Wrote {len(records)} records back to {input_path}")


def run_resume(batch_id: str, output_path: Path):
    """Fetch results for an already-completed batch and write them to a file."""
    client = anthropic.Anthropic()
    wait_for_batch(client, batch_id)
    results = collect_results(client, batch_id)
    output_path.write_text(json.dumps(results, indent=2))
    print(f"Wrote {len(results)} results to {output_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--eval", action="store_true", help="Run against outputs/opus_vs_haiku.json")
    mode.add_argument("--input", type=Path, help="Classify sentences in this JSON file in place")
    mode.add_argument("--resume", type=str, help="Resume a previously-submitted batch ID")
    p.add_argument("--output", type=Path, help="Output path (required with --resume)")
    args = p.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY not set (put it in .env or export it)")

    if args.eval:
        run_eval()
    elif args.input:
        run_input_file(args.input)
    elif args.resume:
        if not args.output:
            sys.exit("--output is required with --resume")
        run_resume(args.resume, args.output)


if __name__ == "__main__":
    main()
