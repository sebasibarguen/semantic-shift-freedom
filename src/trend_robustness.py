# ABOUTME: Robustness diagnostics for the Method 1 positive-liberty share trend.
# ABOUTME: Per-word split, topic-composition adjustment, corpus-handoff control, classifier-error-over-time.

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from src.liberty_trends import (
    get_llm_label,
    load_records,
    normal_two_sided_p,
    weighted_linear_trend,
    wilson_interval,
)

# Historic Hansard records carry a source_file like "S3V0117P0"; ParlParse
# records (1910s onward) have an empty source_file. The two corpora are
# stitched at the 1908/1919 handoff, so source segment is a clean dummy for
# the transcription-style change (third-person summary -> verbatim).
HANDOFF_DECADE = 1910


def decade_of(year: int) -> int:
    return (int(year) // 10) * 10


def source_segment(record: dict) -> str:
    """Which source corpus a sentence came from."""
    src = (record.get("source_file") or "").strip()
    return "parlparse" if src == "" else "historic_hansard"


def primary_domain(record: dict) -> str:
    """Topic proxy: the most-represented domain lexicon in the sentence."""
    domains = record.get("methods", {}).get("domains", {}) or {}
    if not domains:
        return "untagged"
    # Deterministic tie-break: highest count, then alphabetical.
    return sorted(domains.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def primary_share(positive: int, negative: int) -> float | None:
    """positive / (positive + negative); None when no substantive labels."""
    denom = positive + negative
    return positive / denom if denom > 0 else None


def _decade_pos_neg(records, key=lambda r: True) -> dict[int, Counter]:
    """Per-decade positive/negative counts over records passing `key`."""
    counts: dict[int, Counter] = defaultdict(Counter)
    for r in records:
        year = r.get("year")
        if year is None or not key(r):
            continue
        label = get_llm_label(r)
        if label in ("positive_liberty", "negative_liberty"):
            counts[decade_of(year)][label] += 1
    return counts


def _trend_from_counts(counts: dict[int, Counter], min_denominator: int) -> dict | None:
    points = []
    for decade in sorted(counts):
        pos = counts[decade]["positive_liberty"]
        neg = counts[decade]["negative_liberty"]
        share = primary_share(pos, neg)
        if share is None or (pos + neg) < min_denominator:
            continue
        points.append((decade, share, pos + neg))
    return weighted_linear_trend(points)


# --- Diagnostic A: per-word trend split -------------------------------------

def per_word_trends(records, min_denominator: int) -> dict:
    """Recompute the trend separately for 'freedom' and 'liberty'.

    A pooled trend can rise purely because liberty (negative-skewed) thins out
    relative to freedom over time. If both per-word slopes are positive, the
    pooled result is not that frequency-mix artifact.
    """
    out: dict = {"by_word": {}, "frequency_mix": {}}
    for word in ("freedom", "liberty"):
        counts = _decade_pos_neg(records, key=lambda r, w=word: r.get("word") == w)
        out["by_word"][word] = _trend_from_counts(counts, min_denominator)

    # Share of substantive (pos+neg) sentences that use 'liberty', by decade.
    word_counts: dict[int, Counter] = defaultdict(Counter)
    for r in records:
        year = r.get("year")
        if year is None or get_llm_label(r) not in ("positive_liberty", "negative_liberty"):
            continue
        word_counts[decade_of(year)][r.get("word")] += 1
    for decade in sorted(word_counts):
        c = word_counts[decade]
        total = c["freedom"] + c["liberty"]
        out["frequency_mix"][str(decade)] = {
            "freedom": c["freedom"],
            "liberty": c["liberty"],
            "liberty_share": round(c["liberty"] / total, 4) if total else None,
        }
    return out


# --- Diagnostic B: topic-composition adjustment -----------------------------

def composition_adjusted_trend(records, min_domain_denominator: int,
                               min_denominator: int) -> dict:
    """Direct standardization of the positive share on debate topic.

    Holds the topic mix fixed at its pooled distribution so the trend reflects
    within-topic change, not a shifting parliamentary agenda. Also reports the
    raw trend and each topic's own slope for comparison.
    """
    # (decade, domain) -> pos/neg counts.
    cell: dict[tuple[int, str], Counter] = defaultdict(Counter)
    domain_totals: Counter = Counter()
    for r in records:
        year = r.get("year")
        if year is None:
            continue
        label = get_llm_label(r)
        if label not in ("positive_liberty", "negative_liberty"):
            continue
        dom = primary_domain(r)
        cell[(decade_of(year), dom)][label] += 1
        domain_totals[dom] += 1

    domains = sorted(domain_totals)
    grand = sum(domain_totals.values())
    pooled_weights = {d: domain_totals[d] / grand for d in domains}

    decades = sorted({d for (d, _) in cell})
    std_points = []
    raw_points = []
    for decade in decades:
        # Within-decade per-domain shares, plus the raw aggregate.
        dec_pos = dec_neg = 0
        avail = {}
        for dom in domains:
            c = cell[(decade, dom)]
            pos, neg = c["positive_liberty"], c["negative_liberty"]
            dec_pos += pos
            dec_neg += neg
            if pos + neg >= min_domain_denominator:
                avail[dom] = pos / (pos + neg)
        if dec_pos + dec_neg < min_denominator:
            continue
        raw_points.append((decade, dec_pos / (dec_pos + dec_neg), dec_pos + dec_neg))

        wsum = sum(pooled_weights[d] for d in avail)
        if wsum > 0:
            std = sum((pooled_weights[d] / wsum) * p for d, p in avail.items())
            std_points.append((decade, std, dec_pos + dec_neg))

    # Each topic's own secular slope.
    within_domain = {}
    for dom in domains:
        counts = {d: cell[(d, dom)] for d in decades}
        within_domain[dom] = {
            "pooled_weight": round(pooled_weights[dom], 4),
            "trend": _trend_from_counts(counts, min_domain_denominator),
        }

    return {
        "raw_trend": weighted_linear_trend(raw_points),
        "composition_adjusted_trend": weighted_linear_trend(std_points),
        "within_domain": within_domain,
    }


# --- Diagnostic C: corpus-handoff level-shift -------------------------------

def _solve(a: list[list[float]], b: list[float]) -> list[float]:
    """Gaussian elimination with partial pivoting for a small dense system."""
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) < 1e-12:
            raise ValueError("singular system")
        m[col], m[piv] = m[piv], m[col]
        for r in range(n):
            if r == col:
                continue
            f = m[r][col] / m[col][col]
            m[r] = [m[r][k] - f * m[col][k] for k in range(n + 1)]
    return [m[i][n] / m[i][i] for i in range(n)]


def _weighted_ols(rows: list[tuple[list[float], float, float]]) -> dict:
    """Weighted OLS. rows = (x_vector, y, weight). Returns coefs + std errors."""
    p = len(rows[0][0])
    xtx = [[0.0] * p for _ in range(p)]
    xty = [0.0] * p
    wsum = 0.0
    for x, y, w in rows:
        wsum += w
        for i in range(p):
            xty[i] += w * x[i] * y
            for j in range(p):
                xtx[i][j] += w * x[i] * x[j]
    beta = _solve(xtx, xty)
    # Residual variance and coefficient standard errors.
    rss = sum(w * (y - sum(b * xi for b, xi in zip(beta, x, strict=True))) ** 2 for x, y, w in rows)
    dof = len(rows) - p
    sigma2 = rss / dof if dof > 0 else 0.0
    inv = _invert(xtx)
    se = [(sigma2 * inv[i][i]) ** 0.5 if sigma2 > 0 else 0.0 for i in range(p)]
    return {"beta": beta, "se": se, "dof": dof}


def _invert(a: list[list[float]]) -> list[list[float]]:
    n = len(a)
    m = [row[:] + [1.0 if i == j else 0.0 for j in range(n)] for i, row in enumerate(a)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(m[r][col]))
        m[col], m[piv] = m[piv], m[col]
        d = m[col][col]
        m[col] = [v / d for v in m[col]]
        for r in range(n):
            if r == col:
                continue
            f = m[r][col]
            m[r] = [m[r][k] - f * m[col][k] for k in range(2 * n)]
    return [row[n:] for row in m]


def handoff_controlled_trend(records, min_denominator: int) -> dict:
    """Trend within each source corpus, plus a pooled fit with a source dummy.

    If the post-1890s break is really the 1909 corpus handoff, the within-source
    slopes will be flat and the dummy will absorb the jump. The dummy-model
    slope is the secular trend net of the level shift.
    """
    out: dict = {"by_source": {}}
    decade_counts = _decade_pos_neg(records)

    for seg in ("historic_hansard", "parlparse"):
        counts = _decade_pos_neg(records, key=lambda r, s=seg: source_segment(r) == s)
        out["by_source"][seg] = _trend_from_counts(counts, min_denominator)

    # Pooled dummy model: share ~ x_century + post_handoff.
    rows = []
    decades = []
    for decade in sorted(decade_counts):
        pos = decade_counts[decade]["positive_liberty"]
        neg = decade_counts[decade]["negative_liberty"]
        if pos + neg < min_denominator:
            continue
        decades.append((decade, pos / (pos + neg), pos + neg))
    if len(decades) >= 4:
        mean_dec = sum(d for d, _, _ in decades) / len(decades)
        rows = [
            ([1.0, (d - mean_dec) / 100, 1.0 if d >= HANDOFF_DECADE else 0.0], y, n)
            for d, y, n in decades
        ]
        fit = _weighted_ols(rows)
        slope, dummy = fit["beta"][1], fit["beta"][2]
        slope_se, dummy_se = fit["se"][1], fit["se"][2]
        z_slope = slope / slope_se if slope_se > 0 else 0.0
        z_dummy = dummy / dummy_se if dummy_se > 0 else 0.0
        out["source_dummy_model"] = {
            "n_decades": len(decades),
            "slope_per_century_net_of_handoff": round(slope, 6),
            "slope_se": round(slope_se, 6),
            "slope_z": round(z_slope, 3),
            "slope_p_approx": round(normal_two_sided_p(z_slope), 6),
            "handoff_level_shift": round(dummy, 6),
            "handoff_z": round(z_dummy, 3),
            "handoff_p_approx": round(normal_two_sided_p(z_dummy), 6),
        }
    return out


# --- Diagnostic D: classifier error over time -------------------------------

def classifier_error_over_time(eval_path: Path, ref_key: str, hyp_key: str) -> dict:
    """Agreement and directional bias of the classifier vs a reference, by era.

    Time-correlated error manufactures or masks trend; a flat disagreement rate
    means the trend survives any constant error level. Directional bias =
    P(hyp says positive | ref says negative) - P(hyp says negative | ref says
    positive); a drift in this across eras is the dangerous case.
    """
    rows = json.loads(eval_path.read_text())
    rows = [r for r in rows if r.get(ref_key) and r.get(hyp_key) and r.get("year")]

    def era(year: int) -> str:
        return "pre_1909_historic" if year < HANDOFF_DECADE else "post_1909_parlparse"

    buckets: dict[str, list] = defaultdict(list)
    for r in rows:
        buckets[era(r["year"])].append(r)
    buckets["all"] = rows

    result = {"eval_file": eval_path.name, "ref": ref_key, "hyp": hyp_key, "by_era": {}}
    for name, group in buckets.items():
        n = len(group)
        if n == 0:
            continue
        agree = sum(1 for r in group if r[ref_key] == r[hyp_key])
        # Substantive-only confusion for directional bias.
        sub = [r for r in group if r[ref_key] in ("positive_liberty", "negative_liberty")]
        ref_neg = [r for r in sub if r[ref_key] == "negative_liberty"]
        ref_pos = [r for r in sub if r[ref_key] == "positive_liberty"]
        false_pos = sum(1 for r in ref_neg if r[hyp_key] == "positive_liberty")
        false_neg = sum(1 for r in ref_pos if r[hyp_key] == "negative_liberty")
        fp_rate = false_pos / len(ref_neg) if ref_neg else None
        fn_rate = false_neg / len(ref_pos) if ref_pos else None
        result["by_era"][name] = {
            "n": n,
            "agreement": round(agree / n, 4),
            "agreement_ci": wilson_interval(agree, n),
            "n_substantive": len(sub),
            "false_positive_rate": round(fp_rate, 4) if fp_rate is not None else None,
            "false_negative_rate": round(fn_rate, 4) if fn_rate is not None else None,
            "directional_bias": (
                round(fp_rate - fn_rate, 4)
                if fp_rate is not None and fn_rate is not None else None
            ),
        }
    return result


def run_analysis(data_dir: Path, eval_path: Path, min_denominator: int = 30,
                 min_domain_denominator: int = 30) -> dict:
    records = load_records(data_dir)
    return {
        "metadata": {
            "source": str(data_dir),
            "total_records": len(records),
            "min_denominator": min_denominator,
            "min_domain_denominator": min_domain_denominator,
        },
        "per_word": per_word_trends(records, min_denominator),
        "composition": composition_adjusted_trend(
            records, min_domain_denominator, min_denominator
        ),
        "handoff": handoff_controlled_trend(records, min_denominator),
        "classifier_error": classifier_error_over_time(
            eval_path, "opus", "haiku_v2"
        ) if eval_path.exists() else None,
    }


def _fmt(trend: dict | None) -> str:
    if not trend:
        return "insufficient data"
    return (
        f"slope/century={trend['slope_per_century']:+.4f} "
        f"(z={trend['z']}, p≈{trend['p_value_approx']}), "
        f"{trend['first_decade']}={trend['first_proportion']:.3f} -> "
        f"{trend['last_decade']}={trend['last_proportion']:.3f}"
    )


def print_summary(results: dict) -> None:
    print("=" * 72)
    print("METHOD 1 ROBUSTNESS DIAGNOSTICS")
    print("=" * 72)

    print("\nA. Per-word trend (Simpson's-paradox check)")
    for word, trend in results["per_word"]["by_word"].items():
        print(f"  {word:8s}: {_fmt(trend)}")
    mix = results["per_word"]["frequency_mix"]
    if mix:
        first, last = min(mix), max(mix)
        print(f"  liberty share of substantive: {first}={mix[first]['liberty_share']} "
              f"-> {last}={mix[last]['liberty_share']}")

    print("\nB. Topic-composition adjustment")
    print(f"  raw         : {_fmt(results['composition']['raw_trend'])}")
    print(f"  standardized: {_fmt(results['composition']['composition_adjusted_trend'])}")
    print("  within-domain slopes/century:")
    for dom, d in sorted(results["composition"]["within_domain"].items(),
                         key=lambda kv: -kv[1]["pooled_weight"]):
        t = d["trend"]
        s = f"{t['slope_per_century']:+.4f} (z={t['z']})" if t else "n/a"
        print(f"    {dom:24s} w={d['pooled_weight']:.3f}  {s}")

    print("\nC. Corpus-handoff control")
    for seg, trend in results["handoff"]["by_source"].items():
        print(f"  {seg:18s}: {_fmt(trend)}")
    dm = results["handoff"].get("source_dummy_model")
    if dm:
        print(f"  dummy model : slope net of handoff="
              f"{dm['slope_per_century_net_of_handoff']:+.4f} "
              f"(z={dm['slope_z']}, p≈{dm['slope_p_approx']}); "
              f"level shift={dm['handoff_level_shift']:+.4f} (z={dm['handoff_z']})")

    ce = results.get("classifier_error")
    if ce:
        print("\nD. Classifier error over time (haiku_v2 vs opus, "
              f"{ce['eval_file']})")
        for era, e in ce["by_era"].items():
            print(f"  {era:22s} n={e['n']:3d} agree={e['agreement']:.3f} "
                  f"dir-bias={e['directional_bias']}")


def main() -> None:
    project_root = Path(__file__).parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=project_root / "web" / "data")
    parser.add_argument("--eval", type=Path,
                        default=project_root / "outputs" / "haiku_v2_eval.json")
    parser.add_argument("--output", type=Path,
                        default=project_root / "outputs" / "trend_robustness.json")
    parser.add_argument("--min-denominator", type=int, default=30)
    parser.add_argument("--min-domain-denominator", type=int, default=30)
    args = parser.parse_args()

    results = run_analysis(args.data_dir, args.eval, args.min_denominator,
                           args.min_domain_denominator)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
    print_summary(results)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
