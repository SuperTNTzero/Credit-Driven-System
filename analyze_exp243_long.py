"""Paired statistical analysis for EXP243-long."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parent
CONTRASTS = (
    ("shuffled_credit", "promotions", "credit changes persistent structure"),
    ("shuffled_credit", "postformation_success", "credit improves downstream function"),
    ("no_phi", "recovery_success", "persistent organization supports repair"),
    ("no_compile", "complex_postformation_success", "compilation expands fixed-budget reach"),
    ("program_memory", "transfer_success", "organized reuse exceeds direct memory"),
    ("phase_reset", "success_rate", "persistence exceeds phase-local state"),
)


def exact_sign_flip(differences: list[float]) -> float:
    nonzero = [value for value in differences if abs(value) > 1e-12]
    observed = abs(mean(differences))
    if not nonzero:
        return 1.0
    if len(nonzero) > 22:
        return float("nan")
    extreme = 0
    total = 2 ** len(nonzero)
    zeros = len(differences) - len(nonzero)
    for mask in range(total):
        signed = [
            value if mask & (1 << index) else -value
            for index, value in enumerate(nonzero)
        ]
        if abs(sum(signed) / len(differences)) >= observed - 1e-12:
            extreme += 1
    return extreme / total


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    location = (len(ordered) - 1) * q
    lower = math.floor(location)
    upper = math.ceil(location)
    if lower == upper:
        return ordered[lower]
    weight = location - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def deterministic_bootstrap_ci(differences: list[float]) -> tuple[float, float]:
    # A fixed RNG seed makes the ordinary paired bootstrap reproducible.
    n = len(differences)
    rng = random.Random(243)
    samples = [
        mean(differences[rng.randrange(n)] for _ in range(n))
        for _ in range(10000)
    ]
    return percentile(samples, 0.025), percentile(samples, 0.975)


def holm_adjust(rows: list[dict]) -> None:
    finite = sorted(
        ((index, row["exact_p"]) for index, row in enumerate(rows) if math.isfinite(row["exact_p"])),
        key=lambda item: item[1],
    )
    running = 0.0
    m = len(finite)
    for rank, (index, p_value) in enumerate(finite):
        adjusted = min(1.0, (m - rank) * p_value)
        running = max(running, adjusted)
        rows[index]["holm_p"] = running


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", default=str(ROOT / "results" / "EXP243_long_closed_loop.json")
    )
    parser.add_argument(
        "--output", default=str(ROOT / "results" / "EXP243_long_analysis.json")
    )
    args = parser.parse_args()
    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    lookup = {
        (row["method"], row["seed"]): row for row in data["final_rows"]
    }
    seeds = sorted({row["seed"] for row in data["final_rows"]})
    rows = []
    for baseline, field, claim in CONTRASTS:
        differences = [
            float(lookup[("full_loop", seed)][field])
            - float(lookup[(baseline, seed)][field])
            for seed in seeds
        ]
        low, high = deterministic_bootstrap_ci(differences)
        rows.append(
            {
                "claim": claim,
                "baseline": baseline,
                "metric": field,
                "paired_seeds": len(seeds),
                "full_mean": mean(float(lookup[("full_loop", seed)][field]) for seed in seeds),
                "baseline_mean": mean(float(lookup[(baseline, seed)][field]) for seed in seeds),
                "mean_difference": mean(differences),
                "difference_ci95": [low, high],
                "wins_ties_losses": [
                    sum(value > 1e-12 for value in differences),
                    sum(abs(value) <= 1e-12 for value in differences),
                    sum(value < -1e-12 for value in differences),
                ],
                "exact_p": exact_sign_flip(differences),
                "holm_p": None,
            }
        )
    holm_adjust(rows)
    output = {
        "experiment": "EXP243-long",
        "primary_contrasts": rows,
        "interpretation_rule": (
            "A mechanism claim requires a positive paired effect and Holm-adjusted "
            "p < 0.05; boolean engineering gates alone are not statistical evidence."
        ),
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
