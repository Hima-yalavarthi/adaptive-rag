"""Paired comparison of two evaluation runs on the same questions.

Reports the mean difference with a paired bootstrap 95% CI for each metric,
and an exact McNemar test on exact-match wins/losses.
"""

from __future__ import annotations

import argparse
import json
import random
from math import comb
from pathlib import Path

METRICS = (
    "em",
    "f1",
    "retrieval_recall",
    "all_gold_retrieved",
    "citation_precision",
    "confidently_wrong",
)


def confidently_wrong(row: dict) -> float:
    if row.get("confidently_wrong") is not None:
        return float(row["confidently_wrong"])
    insufficient = str(row.get("prediction", "")).strip().lower() == "insufficient evidence"
    return float((not insufficient) and (not row["em"]))


def metric_value(row: dict, key: str) -> float:
    if key == "confidently_wrong":
        return confidently_wrong(row)
    return float(row[key])


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def paired_bootstrap(diffs: list[float], iterations: int = 10_000, seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(sum(rng.choice(diffs) for _ in range(n)) / n for _ in range(iterations))
    return means[int(0.025 * iterations)], means[int(0.975 * iterations) - 1]


def mcnemar_exact(wins: int, losses: int) -> float:
    """Two-sided exact binomial test on discordant pairs."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    tail = sum(comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def compare(a_rows: list[dict], b_rows: list[dict]) -> dict:
    if [r["id"] for r in a_rows] != [r["id"] for r in b_rows]:
        raise ValueError("Runs must cover the same questions in the same order.")
    out: dict = {"n": len(a_rows), "metrics": {}}
    for m in METRICS:
        if m not in a_rows[0] and m != "confidently_wrong":
            continue
        diffs = [metric_value(b, m) - metric_value(a, m) for a, b in zip(a_rows, b_rows)]
        lo, hi = paired_bootstrap(diffs)
        out["metrics"][m] = {
            "a": round(sum(metric_value(r, m) for r in a_rows) / len(a_rows), 4),
            "b": round(sum(metric_value(r, m) for r in b_rows) / len(b_rows), 4),
            "diff": round(sum(diffs) / len(diffs), 4),
            "ci95": [round(lo, 4), round(hi, 4)],
            "excludes_zero": bool(lo > 0 or hi < 0),
        }
    wins = sum(1 for a, b in zip(a_rows, b_rows) if not a["em"] and b["em"])
    losses = sum(1 for a, b in zip(a_rows, b_rows) if a["em"] and not b["em"])
    out["em_mcnemar"] = {"b_wins": wins, "b_losses": losses, "p_value": round(mcnemar_exact(wins, losses), 4)}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("a", type=Path, help="Per-question JSONL for system A (e.g. baseline).")
    parser.add_argument("b", type=Path, help="Per-question JSONL for system B (e.g. reranked).")
    parser.add_argument("--out", type=Path, default=None, help="Optional JSON output path.")
    args = parser.parse_args()

    result = compare(load_rows(args.a), load_rows(args.b))
    print(f"{args.b.stem} vs. {args.a.stem}  (n={result['n']}, paired)\n")
    print(f"  {'metric':<20} {'A':>7} {'B':>7} {'B-A':>7}   95% CI")
    for m, r in result["metrics"].items():
        print(f"  {m:<20} {r['a']:>7.1%} {r['b']:>7.1%} {r['diff']:>+7.1%}   [{r['ci95'][0]:+.1%}, {r['ci95'][1]:+.1%}]")
    mc = result["em_mcnemar"]
    print(f"\n  EM flips: B fixed {mc['b_wins']}, B broke {mc['b_losses']}  (exact McNemar p = {mc['p_value']:.3f})")
    if args.out:
        args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
