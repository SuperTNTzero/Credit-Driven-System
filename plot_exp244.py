"""Analyze and visualize EXP244 complexity scaling."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
SOURCE = RESULTS / "EXP244_complexity_scaling.json"
data = json.loads(SOURCE.read_text(encoding="utf-8"))
summary = data["summary"]
records = data["records"]
HIGH_BUDGET = max(data["budgets"])


def summary_row(method: str, budget: int, family: str, degree: int) -> dict:
    return next(
        row
        for row in summary
        if row["method"] == method
        and row["budget"] == budget
        and row["family"] == family
        and row["degree"] == degree
    )


def critical_row(method: str, budget: int, family: str) -> dict:
    return next(
        row
        for row in data["critical_degrees"]
        if row["method"] == method
        and row["budget"] == budget
        and row["family"] == family
    )


def exact_mcnemar(method_a: str, method_b: str, degree: int) -> dict:
    subset = {
        (row["method"], row["seed"]): bool(row["success"])
        for row in records
        if row["budget"] == HIGH_BUDGET
        and row["family"] == "recursive_reuse"
        and row["degree"] == degree
        and row["method"] in (method_a, method_b)
    }
    seeds = sorted({seed for _, seed in subset})
    b = sum(subset[(method_a, seed)] and not subset[(method_b, seed)] for seed in seeds)
    c = sum(subset[(method_b, seed)] and not subset[(method_a, seed)] for seed in seeds)
    discordant = b + c
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(discordant, k) for k in range(min(b, c) + 1)) / (2**discordant)
        p_value = min(1.0, 2.0 * tail)
    return {"degree": degree, "a_only": b, "b_only": c, "discordant": discordant, "exact_p": p_value}


def cascade_analysis() -> list[dict]:
    phi = [
        row
        for row in records
        if row["method"] == "phi_recursive"
        and row["budget"] == HIGH_BUDGET
        and row["family"] == "recursive_reuse"
    ]
    by_seed = {(row["seed"], row["degree"]): bool(row["success"]) for row in phi}
    output = []
    for degree in range(2, 6):
        previous_success = [seed for seed in range(data["seeds"]) if by_seed[(seed, degree - 1)]]
        previous_failure = [seed for seed in range(data["seeds"]) if not by_seed[(seed, degree - 1)]]
        output.append(
            {
                "degree": degree,
                "p_success_given_previous_success": (
                    sum(by_seed[(seed, degree)] for seed in previous_success) / len(previous_success)
                    if previous_success
                    else None
                ),
                "p_success_given_previous_failure": (
                    sum(by_seed[(seed, degree)] for seed in previous_failure) / len(previous_failure)
                    if previous_failure
                    else None
                ),
                "previous_success_n": len(previous_success),
                "previous_failure_n": len(previous_failure),
            }
        )
    return output


def paired_auc_permutation(method_a: str, method_b: str) -> dict:
    lookup = {
        (row["method"], row["seed"], row["degree"]): float(row["success"])
        for row in records
        if row["budget"] == HIGH_BUDGET
        and row["family"] == "recursive_reuse"
        and row["method"] in (method_a, method_b)
    }
    differences = []
    for seed in range(data["seeds"]):
        auc_a = np.mean([lookup[(method_a, seed, degree)] for degree in range(1, 6)])
        auc_b = np.mean([lookup[(method_b, seed, degree)] for degree in range(1, 6)])
        differences.append(float(auc_a - auc_b))
    nonzero = [difference for difference in differences if difference != 0.0]
    observed = abs(float(np.mean(differences)))
    if not nonzero:
        p_value = 1.0
    else:
        extreme = 0
        total = 2 ** len(nonzero)
        for mask in range(total):
            signed = [value if mask & (1 << index) else -value for index, value in enumerate(nonzero)]
            if abs(float(np.mean(signed))) >= observed - 1e-12:
                extreme += 1
        p_value = extreme / total
    return {
        "method_a": method_a,
        "method_b": method_b,
        "mean_auc_difference": float(np.mean(differences)),
        "two_sided_exact_p": p_value,
        "paired_seeds": len(differences),
    }


phi_records = [
    row
    for row in records
    if row["method"] == "phi_recursive"
    and row["budget"] == HIGH_BUDGET
    and row["family"] == "recursive_reuse"
]
retrieval = np.array([row["retrieval_recall"] for row in phi_records], dtype=float)
success = np.array([row["success"] for row in phi_records], dtype=float)
retrieval_success_correlation = float(np.corrcoef(retrieval, success)[0, 1])

paired = {
    comparator: [exact_mcnemar("phi_recursive", comparator, degree) for degree in range(1, 6)]
    for comparator in ("direct_memory", "phi_static", "phi_shuffled")
}

analysis = {
    "experiment": "EXP244-analysis",
    "source": SOURCE.name,
    "high_budget": HIGH_BUDGET,
    "retrieval_success_correlation": retrieval_success_correlation,
    "paired_exact_mcnemar": paired,
    "paired_auc_permutation": [
        paired_auc_permutation("phi_recursive", comparator)
        for comparator in ("direct_memory", "phi_static", "phi_shuffled")
    ],
    "cascade": cascade_analysis(),
    "headline": {
        "flat_critical_degree": critical_row("flat_active", HIGH_BUDGET, "recursive_reuse")["critical_degree"],
        "direct_memory_critical_degree": critical_row("direct_memory", HIGH_BUDGET, "recursive_reuse")["critical_degree"],
        "phi_critical_degree": critical_row("phi_recursive", HIGH_BUDGET, "recursive_reuse")["critical_degree"],
        "oracle_critical_degree": critical_row("oracle_recursive", HIGH_BUDGET, "recursive_reuse")["critical_degree"],
        "phi_reuse_auc": critical_row("phi_recursive", HIGH_BUDGET, "recursive_reuse")["success_auc"],
        "phi_control_auc": critical_row("phi_recursive", HIGH_BUDGET, "low_reuse_control")["success_auc"],
        "phi_degree5_success": summary_row("phi_recursive", HIGH_BUDGET, "recursive_reuse", 5)["success_rate"],
        "shuffled_degree5_success": summary_row("phi_shuffled", HIGH_BUDGET, "recursive_reuse", 5)["success_rate"],
        "phi_degree5_archive_size": summary_row("phi_recursive", HIGH_BUDGET, "recursive_reuse", 5)["archive_size_mean"],
        "phi_degree5_expanded_size": summary_row("phi_recursive", HIGH_BUDGET, "recursive_reuse", 5)["expanded_size"],
        "phi_degree5_call_cost": summary_row("phi_recursive", HIGH_BUDGET, "recursive_reuse", 5)["solution_search_cost_mean"],
        "phi_degree1_retrieval_candidates": summary_row("phi_recursive", HIGH_BUDGET, "recursive_reuse", 1)["retrieval_candidates_mean"],
        "phi_degree5_retrieval_candidates": summary_row("phi_recursive", HIGH_BUDGET, "recursive_reuse", 5)["retrieval_candidates_mean"],
        "phi_degree5_retrieval_to_main_budget": summary_row("phi_recursive", HIGH_BUDGET, "recursive_reuse", 5)["retrieval_candidates_mean"] / HIGH_BUDGET,
    },
}
(RESULTS / "EXP244_complexity_analysis.json").write_text(
    json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8"
)


colors = {
    "flat_active": "#6c7a89",
    "direct_memory": "#d9a441",
    "phi_static": "#8d6e63",
    "phi_shuffled": "#c95d63",
    "phi_recursive": "#4c956c",
    "oracle_recursive": "#417aa8",
}
labels = {
    "flat_active": "Flat active",
    "direct_memory": "Direct memory",
    "phi_static": "Static macros",
    "phi_shuffled": "Shuffled credit",
    "phi_recursive": "Dynamic Phi",
    "oracle_recursive": "Oracle retrieval",
}
selected_methods = tuple(colors)
degrees = range(1, 6)

fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.6), constrained_layout=True)

for method in selected_methods:
    values = [summary_row(method, HIGH_BUDGET, "recursive_reuse", degree)["success_rate"] for degree in degrees]
    axes[0, 0].plot(degrees, values, "o-", linewidth=2, color=colors[method], label=labels[method])
axes[0, 0].set(title="A  Discovery under recursive reuse", xlabel="Polynomial degree", ylabel="Exact discovery rate", xticks=list(degrees), ylim=(-0.04, 1.05))
axes[0, 0].legend(frameon=False, fontsize=7, ncol=2)

heat_methods = ("flat_active", "direct_memory", "phi_static", "phi_shuffled", "phi_recursive", "oracle_recursive")
heat = np.array([
    [summary_row(method, HIGH_BUDGET, "recursive_reuse", degree)["success_rate"] for degree in degrees]
    for method in heat_methods
])
image = axes[0, 1].imshow(heat, vmin=0, vmax=1, cmap="viridis", aspect="auto")
axes[0, 1].set(title="B  High-budget complexity boundary", xlabel="Polynomial degree", xticks=range(5), xticklabels=list(degrees), yticks=range(len(heat_methods)), yticklabels=[labels[item] for item in heat_methods])
for row in range(len(heat_methods)):
    for column in range(5):
        axes[0, 1].text(column, row, f"{heat[row, column]:.2f}", ha="center", va="center", color="white" if heat[row, column] < 0.58 else "black", fontsize=8)
fig.colorbar(image, ax=axes[0, 1], shrink=0.8, label="Success rate")

for method in ("direct_memory", "phi_shuffled", "phi_recursive", "oracle_recursive"):
    auc = [critical_row(method, budget, "recursive_reuse")["success_auc"] for budget in data["budgets"]]
    axes[0, 2].plot(data["budgets"], auc, "o-", linewidth=2, color=colors[method], label=labels[method])
axes[0, 2].set(title="C  Candidate-budget scaling", xlabel="Candidate evaluations per task", ylabel="Success AUC across degree", ylim=(-0.03, 1.05), xticks=data["budgets"])
axes[0, 2].legend(frameon=False, fontsize=8)

x = np.arange(len(selected_methods))
reuse_auc = [critical_row(method, HIGH_BUDGET, "recursive_reuse")["success_auc"] for method in selected_methods]
control_auc = [critical_row(method, HIGH_BUDGET, "low_reuse_control")["success_auc"] for method in selected_methods]
axes[1, 0].bar(x - 0.18, reuse_auc, width=0.36, label="Recursive reuse", color="#4c956c")
axes[1, 0].bar(x + 0.18, control_auc, width=0.36, label="Low-reuse control", color="#d9a441")
axes[1, 0].set(title="D  Benefit is structure-specific", ylabel="Success AUC", ylim=(0, 1.05), xticks=x, xticklabels=[labels[item] for item in selected_methods])
axes[1, 0].tick_params(axis="x", labelrotation=30, labelsize=7)
axes[1, 0].legend(frameon=False, fontsize=8)

phi_points = [summary_row("phi_recursive", HIGH_BUDGET, "recursive_reuse", degree) for degree in degrees]
for row in phi_points:
    axes[1, 1].scatter(row["retrieval_recall_mean"], row["success_rate"], s=90, color="#4c956c")
    axes[1, 1].text(row["retrieval_recall_mean"] + 0.012, row["success_rate"] + 0.012, f"d={row['degree']}", fontsize=8)
axes[1, 1].set(title=f"E  Retrieval predicts success (r={retrieval_success_correlation:.2f})", xlabel="Mean responsibility retrieval recall", ylabel="Exact discovery rate", xlim=(0, 1.05), ylim=(-0.03, 1.05))

expanded = [row["expanded_size"] for row in phi_points]
call_cost = [row["solution_search_cost_mean"] for row in phi_points]
axes[1, 2].plot(degrees, expanded, "o-", linewidth=2, label="Expanded target size", color="#c95d63")
axes[1, 2].plot(degrees, call_cost, "s-", linewidth=2, label="Phi solution call cost", color="#417aa8")
axes[1, 2].set(title="F  Consolidated calls compress execution", xlabel="Polynomial degree", ylabel="Nodes / call cost", xticks=list(degrees), ylim=(0, max(expanded) * 1.1))
axes[1, 2].legend(frameon=False, fontsize=8)

fig.suptitle("EXP244: complexity scaling from linear to nonlinear symbolic discovery", fontsize=14, fontweight="bold")
output = RESULTS / "EXP244_complexity_scaling.png"
fig.savefig(output, dpi=190)
print(RESULTS / "EXP244_complexity_analysis.json")
print(output)
