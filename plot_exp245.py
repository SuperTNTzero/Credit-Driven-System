"""Analyze and visualize EXP245 probe lifecycle dynamics."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
SOURCE = RESULTS / "EXP245_probe_lifecycle.json"
data = json.loads(SOURCE.read_text(encoding="utf-8"))
records = data["records"]

POLICIES = tuple(data["policies"])
PHASES = ("formation", "shift", "revisit")
LABELS = {
    "random_reset": "Random reset",
    "greedy_reset": "Greedy reset",
    "persistent_fifo": "Full persistence",
    "shuffled_lifecycle": "Random lifecycle",
    "credit_lifecycle": "Credit lifecycle",
    "oracle_lifecycle": "Oracle lifecycle",
}
COLORS = {
    "random_reset": "#6c7a89",
    "greedy_reset": "#d9a441",
    "persistent_fifo": "#8d6e63",
    "shuffled_lifecycle": "#c95d63",
    "credit_lifecycle": "#4c956c",
    "oracle_lifecycle": "#417aa8",
}


def rows(policy: str, phase: str | None = None) -> list[dict]:
    return [
        row
        for row in records
        if row["policy"] == policy and (phase is None or row["phase"] == phase)
    ]


def avg(values) -> float:
    values = list(values)
    return float(np.mean(values)) if values else 0.0


def exact_sign_flip(policy_a: str, policy_b: str, phase: str | None = None) -> dict:
    differences = []
    for seed in range(data["seeds"]):
        a = [row["success"] for row in rows(policy_a, phase) if row["seed"] == seed]
        b = [row["success"] for row in rows(policy_b, phase) if row["seed"] == seed]
        differences.append(avg(a) - avg(b))
    nonzero = [difference for difference in differences if abs(difference) > 1e-12]
    observed = abs(avg(differences))
    if not nonzero:
        p_value = 1.0
    else:
        extreme = 0
        total = 2 ** len(nonzero)
        for mask in range(total):
            signed = [
                value if mask & (1 << index) else -value
                for index, value in enumerate(nonzero)
            ]
            if abs(avg(signed)) >= observed - 1e-12:
                extreme += 1
        p_value = extreme / total
    return {
        "policy_a": policy_a,
        "policy_b": policy_b,
        "phase": phase or "all",
        "mean_success_difference": avg(differences),
        "two_sided_exact_p": p_value,
        "paired_seeds": len(differences),
    }


policy_summary = []
for policy in POLICIES:
    policy_rows = rows(policy)
    item = {
        "policy": policy,
        "success": avg(row["success"] for row in policy_rows),
        "queries_per_task": avg(row["probe_queries"] for row in policy_rows),
        "generated_per_task": avg(row["probe_generated"] for row in policy_rows),
        "final_archive": next(
            row["final_archive_mean"]
            for row in data["lifecycle_totals"]
            if row["policy"] == policy
        ),
    }
    for phase in PHASES:
        phase_rows = rows(policy, phase)
        item[f"{phase}_success"] = avg(row["success"] for row in phase_rows)
        item[f"{phase}_generated"] = avg(row["probe_generated"] for row in phase_rows)
    item["revisit_minus_formation"] = item["revisit_success"] - item["formation_success"]
    policy_summary.append(item)

analysis = {
    "experiment": "EXP245-analysis",
    "source": SOURCE.name,
    "policy_summary": policy_summary,
    "paired_seed_sign_flip": [
        exact_sign_flip("credit_lifecycle", comparator, phase)
        for comparator in (
            "random_reset",
            "greedy_reset",
            "persistent_fifo",
            "shuffled_lifecycle",
        )
        for phase in (None, "formation", "revisit")
    ],
    "lifecycle_totals": data["lifecycle_totals"],
    "headline": {
        "credit_revisit_success": next(
            item["revisit_success"] for item in policy_summary if item["policy"] == "credit_lifecycle"
        ),
        "random_reset_revisit_success": next(
            item["revisit_success"] for item in policy_summary if item["policy"] == "random_reset"
        ),
        "credit_revisit_generated": next(
            item["revisit_generated"] for item in policy_summary if item["policy"] == "credit_lifecycle"
        ),
        "random_reset_revisit_generated": next(
            item["revisit_generated"] for item in policy_summary if item["policy"] == "random_reset"
        ),
    },
}
(RESULTS / "EXP245_probe_lifecycle_analysis.json").write_text(
    json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8"
)


fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.7), constrained_layout=True)
stream_indices = range(len(data["stream"]))
selected = ("random_reset", "persistent_fifo", "shuffled_lifecycle", "credit_lifecycle", "oracle_lifecycle")

for policy in selected:
    values = [
        avg(row["success"] for row in rows(policy) if row["stream_index"] == index)
        for index in stream_indices
    ]
    axes[0, 0].plot(stream_indices, values, "o-", linewidth=1.8, markersize=4, color=COLORS[policy], label=LABELS[policy])
axes[0, 0].axvline(4.5, color="#999999", linestyle="--", linewidth=1)
axes[0, 0].axvline(9.5, color="#999999", linestyle="--", linewidth=1)
axes[0, 0].set(title="A  Discovery across the task stream", xlabel="Task index", ylabel="Exact discovery rate", ylim=(-0.04, 1.05), xticks=list(stream_indices))
axes[0, 0].legend(frameon=False, fontsize=7, ncol=2)

x = np.arange(len(POLICIES))
width = 0.25
for offset, phase, color in zip((-width, 0, width), PHASES, ("#4c956c", "#d9a441", "#417aa8")):
    values = [next(item[f"{phase}_success"] for item in policy_summary if item["policy"] == policy) for policy in POLICIES]
    axes[0, 1].bar(x + offset, values, width=width, label=phase.title(), color=color)
axes[0, 1].set(title="B  Accuracy by environment phase", ylabel="Exact discovery rate", ylim=(0, 1.0), xticks=x, xticklabels=[LABELS[p] for p in POLICIES])
axes[0, 1].tick_params(axis="x", labelrotation=28, labelsize=7)
axes[0, 1].legend(frameon=False, fontsize=8)

generated = [next(item["generated_per_task"] for item in policy_summary if item["policy"] == policy) for policy in POLICIES]
queries = [next(item["queries_per_task"] for item in policy_summary if item["policy"] == policy) for policy in POLICIES]
axes[0, 2].bar(x - 0.18, generated, width=0.36, color="#c95d63", label="New probes")
axes[0, 2].bar(x + 0.18, queries, width=0.36, color="#6c7a89", label="Queries")
axes[0, 2].set(title="C  Probe cost per task", ylabel="Count", xticks=x, xticklabels=[LABELS[p] for p in POLICIES])
axes[0, 2].tick_params(axis="x", labelrotation=28, labelsize=7)
axes[0, 2].legend(frameon=False, fontsize=8)

lifecycle_policies = ("persistent_fifo", "shuffled_lifecycle", "credit_lifecycle", "oracle_lifecycle")
events = ("births_mean", "reactivations_mean", "eliminations_mean")
event_labels = ("Births", "Reactivations", "Eliminations")
bottom = np.zeros(len(lifecycle_policies))
for event, label, color in zip(events, event_labels, ("#d9a441", "#417aa8", "#c95d63")):
    values = np.array([
        next(row[event] for row in data["lifecycle_totals"] if row["policy"] == policy)
        for policy in lifecycle_policies
    ])
    axes[1, 0].bar(range(len(lifecycle_policies)), values, bottom=bottom, color=color, label=label)
    bottom += values
axes[1, 0].set(title="D  Lifecycle events over 15 tasks", ylabel="Mean cumulative events", xticks=range(len(lifecycle_policies)), xticklabels=[LABELS[p] for p in lifecycle_policies])
axes[1, 0].tick_params(axis="x", labelrotation=25, labelsize=8)
axes[1, 0].legend(frameon=False, fontsize=8)

for policy in lifecycle_policies:
    archive = [avg(row["archived_probe_count"] for row in rows(policy) if row["stream_index"] == index) for index in stream_indices]
    axes[1, 1].plot(stream_indices, archive, "o-", linewidth=1.8, markersize=4, color=COLORS[policy], label=LABELS[policy])
axes[1, 1].axvline(4.5, color="#999999", linestyle="--", linewidth=1)
axes[1, 1].axvline(9.5, color="#999999", linestyle="--", linewidth=1)
axes[1, 1].set(title="E  Archive growth and pruning", xlabel="Task index", ylabel="Archived probe units", xticks=list(stream_indices))
axes[1, 1].legend(frameon=False, fontsize=7)

for policy in selected:
    item = next(item for item in policy_summary if item["policy"] == policy)
    axes[1, 2].scatter(item["generated_per_task"], item["success"], s=90, color=COLORS[policy], label=LABELS[policy])
    axes[1, 2].text(item["generated_per_task"] + 0.05, item["success"] + 0.005, LABELS[policy], fontsize=7)
axes[1, 2].set(title="F  Generation cost versus discovery", xlabel="New probe units per task", ylabel="Overall exact discovery rate", ylim=(0.15, 0.65))

fig.suptitle("EXP245: probe units with birth, retirement, reactivation, and elimination", fontsize=14, fontweight="bold")
output = RESULTS / "EXP245_probe_lifecycle.png"
fig.savefig(output, dpi=190)
print(RESULTS / "EXP245_probe_lifecycle_analysis.json")
print(output)
