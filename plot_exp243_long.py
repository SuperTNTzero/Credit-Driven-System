"""Visualize the formal EXP243-long result."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
METHODS = (
    "full_loop",
    "shuffled_credit",
    "no_phi",
    "no_compile",
    "program_memory",
    "phase_reset",
)
COLORS = {
    "full_loop": "#147d64",
    "shuffled_credit": "#d97706",
    "no_phi": "#2563a6",
    "no_compile": "#a33a52",
    "program_memory": "#6b7280",
    "phase_reset": "#7c4aa5",
}
LABELS = {
    "full_loop": "Full loop",
    "shuffled_credit": "Shuffled credit",
    "no_phi": "No persistent Phi",
    "no_compile": "No compile",
    "program_memory": "Program memory",
    "phase_reset": "Phase reset",
}


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def mean_ci(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    center = float(array.mean())
    if len(array) < 2:
        return center, 0.0
    return center, float(1.96 * array.std(ddof=1) / math.sqrt(len(array)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", default=str(ROOT / "results" / "EXP243_long_closed_loop.json")
    )
    parser.add_argument(
        "--output", default=str(ROOT / "results" / "EXP243_long_closed_loop.png")
    )
    args = parser.parse_args()
    data = load(Path(args.input))
    summary = {row["method"]: row for row in data["summary"]}
    final = defaultdict(dict)
    for row in data["final_rows"]:
        final[row["method"]][row["seed"]] = row

    fig, axes = plt.subplots(2, 3, figsize=(17, 9.5), constrained_layout=True)
    fig.suptitle(
        f"EXP243-long: one persistent organization loop ({data['seeds']} paired seeds)",
        fontsize=16,
        fontweight="bold",
    )

    ax = axes[0, 0]
    by_task = defaultdict(lambda: defaultdict(list))
    task_order = [item["name"] for item in data["stream"]]
    for row in data["task_rows"]:
        if row["method"] in METHODS:
            by_task[row["method"]][row["task"]].append(float(row["success"]))
    x = np.arange(len(task_order))
    for method in METHODS:
        values = [np.mean(by_task[method][task]) for task in task_order]
        ax.plot(x, values, marker="o", linewidth=1.8, markersize=4, color=COLORS[method], label=LABELS[method])
    ax.set_xticks(x, [str(i + 1) for i in x])
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Event-stream task index")
    ax.set_ylabel("Success probability")
    ax.set_title("A. Performance through the uninterrupted stream")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8, ncol=2, loc="lower left")

    ax = axes[0, 1]
    fields = ("formation_success", "transfer_success", "recovery_success", "late_revisit_success")
    phase_labels = ("Formation", "Transfer", "Recovery", "Late/revisit")
    width = 0.12
    base = np.arange(len(fields))
    for index, method in enumerate(METHODS):
        centers = []
        errors = []
        for field in fields:
            values = [row[field] for row in final[method].values()]
            center, error = mean_ci(values)
            centers.append(center)
            errors.append(error)
        ax.bar(base + (index - 2.5) * width, centers, width, yerr=errors, capsize=2, color=COLORS[method], label=LABELS[method])
    ax.set_xticks(base, phase_labels)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Mean success (95% CI)")
    ax.set_title("B. Where the loop helps")
    ax.grid(axis="y", alpha=0.25)

    ax = axes[0, 2]
    mechanism_fields = ("promotions", "active_units", "lifecycle_event_coverage")
    mechanism_labels = ("Promotions / 10", "Active units / 12", "Lifecycle coverage")
    base = np.arange(len(mechanism_fields))
    for index, method in enumerate(METHODS):
        values = [
            summary[method]["promotions"] / 10.0,
            summary[method]["active_units"] / 12.0,
            summary[method]["lifecycle_event_coverage"],
        ]
        ax.bar(base + (index - 2.5) * width, values, width, color=COLORS[method])
    ax.set_xticks(base, mechanism_labels, rotation=10)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Normalized value")
    ax.set_title("C. Persistent structure and lifecycle")
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1, 0]
    damage_fields = ("damage_drop", "recovered")
    damage_labels = ("Specific damage", "Recovered")
    base = np.arange(2)
    for index, method in enumerate(METHODS):
        centers = []
        errors = []
        for field in damage_fields:
            values = [float(row[field]) for row in final[method].values()]
            center, error = mean_ci(values)
            centers.append(center)
            errors.append(error)
        ax.bar(base + (index - 2.5) * width, centers, width, yerr=errors, capsize=2, color=COLORS[method])
    ax.set_xticks(base, damage_labels)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Probability (95% CI)")
    ax.set_title("D. Causal damage and feedback repair")
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1, 1]
    for method in METHODS:
        x_value = summary[method]["candidate_evaluations"]
        y_value = summary[method]["postformation_success"]
        ax.scatter(x_value, y_value, s=85, color=COLORS[method], edgecolor="white", linewidth=0.8)
        ax.annotate(LABELS[method], (x_value, y_value), xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Mean candidate evaluations")
    ax.set_ylabel("Post-formation success")
    ax.set_ylim(0, 1.02)
    ax.set_title("E. Function under explicit search cost")
    ax.grid(alpha=0.25)

    ax = axes[1, 2]
    gate_names = list(data["gates"])
    values = [1 if data["gates"][name] else 0 for name in gate_names]
    labels = [name.replace("_", " ") for name in gate_names]
    colors = ["#147d64" if value else "#b42335" for value in values]
    positions = np.arange(len(labels))
    ax.barh(positions, values, color=colors)
    ax.set_yticks(positions, labels, fontsize=8)
    ax.set_xlim(0, 1.02)
    ax.set_xticks((0, 1), ("Fail", "Pass"))
    ax.invert_yaxis()
    ax.set_title(f"F. Pre-registered mechanism gates ({sum(values)}/{len(values)})")
    ax.grid(axis="x", alpha=0.25)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    print(output)


if __name__ == "__main__":
    main()
