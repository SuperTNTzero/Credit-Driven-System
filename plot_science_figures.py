"""Generate the four main composite figures for the journal manuscript.

The figures deliberately separate mechanism, flagship causal evidence,
complexity scaling, and cross-domain boundary evidence.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
OUT = RESULTS

COLORS = {
    "ink": "#1f2933",
    "muted": "#52606d",
    "grid": "#d7dee5",
    "phi": "#006d77",
    "credit": "#c44536",
    "memory": "#52606d",
    "control": "#9aa5b1",
    "oracle": "#6a4c93",
    "success": "#2f855a",
    "warning": "#c77700",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 8.5,
    "axes.linewidth": 0.8,
    "figure.dpi": 140,
    "savefig.dpi": 260,
    "mathtext.fontset": "stixsans",
})


def load(name: str):
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def style(ax, title=None, xlabel=None, ylabel=None):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(COLORS["grid"])
    ax.tick_params(colors=COLORS["muted"], labelsize=8, length=3, width=0.7)
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.65, alpha=0.75)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, loc="left", fontsize=10.5, fontweight="bold", color=COLORS["ink"], pad=8)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=8, color=COLORS["muted"])
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8, color=COLORS["muted"])


def save(fig, name: str):
    fig.savefig(OUT / name, dpi=260, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)


def figure1():
    fig = plt.figure(figsize=(11.5, 6.3))
    gs = fig.add_gridspec(2, 1, height_ratios=[2.25, 0.82], hspace=0.16)
    ax = fig.add_subplot(gs[0])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    nodes = [
        (0.06, 0.60, 0.13, 0.24, "W", "capacity / operators", "#dbeafe"),
        (0.25, 0.60, 0.13, 0.24, "q", "instant responsibility", "#fee2e2"),
        (0.44, 0.60, 0.13, 0.24, "y", "action / output", "#fef3c7"),
        (0.63, 0.60, 0.13, 0.24, "c", "credit from feedback", "#ffedd5"),
        (0.82, 0.60, 0.13, 0.24, r"$\Phi$", "persistent organization", "#ccfbf1"),
        (0.63, 0.17, 0.13, 0.24, "U", "validated unit", "#ede9fe"),
        (0.82, 0.17, 0.13, 0.24, r"$W'$", "compiled capability", "#dbeafe"),
    ]
    for x, y, w, h, label, desc, color in nodes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                                    facecolor=color, edgecolor=COLORS["ink"], linewidth=1.2))
        ax.text(x + w / 2, y + h * 0.62, label, ha="center", va="center", fontsize=22,
                fontweight="bold", color=COLORS["ink"])
        ax.text(x + w / 2, y + h * 0.22, desc, ha="center", va="center", fontsize=7.5,
                color=COLORS["muted"])

    arrows = [
        ((0.19, 0.72), (0.25, 0.72), "call"),
        ((0.38, 0.72), (0.44, 0.72), "route"),
        ((0.57, 0.72), (0.63, 0.72), "feedback"),
        ((0.76, 0.72), (0.82, 0.72), "update"),
        ((0.875, 0.60), (0.70, 0.41), "consolidate"),
        ((0.76, 0.29), (0.82, 0.29), "compile"),
        ((0.885, 0.41), (0.885, 0.60), "persist", "#0b7285"),
    ]
    for item in arrows:
        start, end, text = item[:3]
        color = item[3] if len(item) == 4 else COLORS["muted"]
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=13,
                                     linewidth=1.5, color=color, connectionstyle="arc3,rad=0.0"))
        mx, my = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2
        ax.text(mx, my + 0.035, text, ha="center", va="center", fontsize=7, color=color)
    ax.add_patch(FancyArrowPatch((0.885, 0.60), (0.885, 0.41), arrowstyle="-|>", mutation_scale=13,
                                 linewidth=1.5, color=COLORS["phi"], linestyle="--"))
    ax.text(0.90, 0.50, "cross-prediction", rotation=90, fontsize=7, color=COLORS["phi"], va="center")

    ax.text(0.06, 0.94, "A  Minimal organization loop", fontsize=14, fontweight="bold", color=COLORS["ink"])
    ax.text(0.06, 0.88, "W determines what can be executed; $\\Phi$ determines which capability is organized and called now.",
            fontsize=9, color=COLORS["muted"])

    ax2 = fig.add_subplot(gs[1])
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 1)
    ax2.axis("off")
    ax2.text(0.01, 0.88, "B  Distinct time scales", fontsize=10.5, fontweight="bold", color=COLORS["ink"])
    for y, label, left, right, color in [
        (0.62, "fast: execution / q", 2.15, 4.1, "#f0a202"),
        (0.40, "medium: credit / $\\Phi$", 2.15, 6.5, COLORS["phi"]),
        (0.18, "slow: validated unit / $W'$", 2.15, 9.35, COLORS["oracle"]),
    ]:
        ax2.text(0.01, y, label, fontsize=8, color=COLORS["muted"], va="center")
        ax2.plot([left, right], [y, y], color=color, linewidth=6, solid_capstyle="round")
        ax2.text(right + 0.12, y, "short" if y == 0.62 else "persistent" if y == 0.40 else "compiled",
                 fontsize=7.5, color=color, va="center")
    ax2.text(9.5, 0.86, "feedback changes organization, not just parameters", ha="right", fontsize=8.5,
             color=COLORS["credit"])
    fig.suptitle("Figure 1 | Organization is a dynamical state between capacity and behavior", x=0.04, ha="left",
                 fontsize=13.5, fontweight="bold", color=COLORS["ink"])
    save(fig, "figure1_minimal_organization_loop.png")


def summary_map(exp):
    return {row["method"]: row for row in exp["summary"]}


def figure2():
    exp = load("EXP243_long_closed_loop.json")
    analysis = load("EXP243_long_analysis.json")
    sm = summary_map(exp)
    fig = plt.figure(figsize=(13, 8.3))
    gs = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.28)

    ax = fig.add_subplot(gs[0, 0])
    full = next(h for h in exp["histories"] if h["method"] == "full_loop" and h["seed"] == 0)
    clocks = sorted({e["clock"] for e in full["events"]})
    counts = {k: [] for k in ("promote", "verify", "credit", "task_success")}
    for clock in clocks:
        events = [e for e in full["events"] if e["clock"] <= clock]
        for k in counts:
            counts[k].append(sum(e["kind"] == k for e in events))
    ax.plot(clocks, counts["promote"], color=COLORS["phi"], linewidth=2.5, label="promoted units")
    ax.plot(clocks, counts["verify"], color=COLORS["memory"], linewidth=1.8, label="verification calls")
    ax.plot(clocks, counts["credit"], color=COLORS["credit"], linewidth=1.8, label="credit assignments")
    ax.scatter(clocks, counts["task_success"], color=COLORS["success"], s=18, label="successful tasks", zorder=3)
    for x, label in [(4, "formation"), (23, "damage"), (25, "recovery"), (28, "revisit")]:
        ax.axvline(x, color=COLORS["grid"], linewidth=0.9)
        ax.text(x + 0.25, max(counts["verify"]) * 0.96, label, fontsize=7, color=COLORS["muted"], rotation=90, va="top")
    style(ax, "A  Credit turns an event stream into a library", "event clock", "cumulative events")
    ax.legend(frameon=False, fontsize=7, ncol=2, loc="upper left")

    ax = fig.add_subplot(gs[0, 1])
    method_order = ["full_loop", "shuffled_credit", "no_phi", "phase_reset", "no_compile", "program_memory"]
    metric_order = [("formation_success", "formation"), ("transfer_success", "transfer"),
                    ("recovery_success", "recovery"), ("postformation_success", "post-formation")]
    mat = np.array([[sm[m].get(k, np.nan) for k, _ in metric_order] for m in method_order])
    im = ax.imshow(mat, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(metric_order)), [n for _, n in metric_order], rotation=25, ha="right", fontsize=8)
    ax.set_yticks(range(len(method_order)), [m.replace("_", " ") for m in method_order], fontsize=8)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if mat[i, j] > 0.62 else COLORS["ink"])
    ax.set_title("B  Same stream, different organization state", loc="left", fontsize=11, fontweight="bold", color=COLORS["ink"], pad=10)
    ax.set_xlabel("behavioral phase", fontsize=8, color=COLORS["muted"])

    ax = fig.add_subplot(gs[1, 0])
    contrasts = analysis["primary_contrasts"]
    labels = ["credit\nvs shuffled", "persistent $\\Phi$\nvs no $\\Phi$", "compile\nvs no compile",
              "organized reuse\nvs program memory", "persistent\nvs phase reset"]
    idx = [0, 2, 3, 4, 5]
    vals = [contrasts[i]["mean_difference"] for i in idx]
    lows = [contrasts[i]["difference_ci95"][0] for i in idx]
    highs = [contrasts[i]["difference_ci95"][1] for i in idx]
    y = np.arange(len(labels))
    ax.errorbar(vals, y, xerr=[np.array(vals) - lows, np.array(highs) - np.array(vals)], fmt="o",
                color=COLORS["credit"], ecolor=COLORS["credit"], capsize=3, linewidth=1.5)
    ax.axvline(0, color=COLORS["muted"], linewidth=0.9)
    ax.set_yticks(y, labels, fontsize=8)
    ax.set_xlim(-0.05, 0.85)
    style(ax, "C  Paired causal contrasts", "difference in success / rate", None)
    ax.invert_yaxis()

    ax = fig.add_subplot(gs[1, 1])
    method_colors = {
        "full_loop": COLORS["phi"], "shuffled_credit": COLORS["warning"], "no_phi": COLORS["memory"],
        "phase_reset": COLORS["control"], "no_compile": COLORS["credit"], "program_memory": COLORS["oracle"],
    }
    short_names = {"full_loop": "full", "shuffled_credit": "shuffle", "no_phi": "no $\\Phi$",
                   "phase_reset": "reset", "no_compile": "no compile", "program_memory": "memory"}
    for method in method_order:
        row = sm[method]
        x_val, y_val = row["postformation_success"], row["recovery_success"]
        ax.scatter(x_val, y_val, s=58, color=method_colors[method], edgecolor="white", linewidth=0.8, zorder=3)
        offsets = {
            "program_memory": (-0.055, 0.085), "no_compile": (0.018, 0.085),
            "no_phi": (0.012, 0.025), "phase_reset": (0.012, 0.025),
            "shuffled_credit": (0.012, 0.018), "full_loop": (0.015, 0.015),
        }
        dx, dy = offsets[method]
        ax.text(x_val + dx, y_val + dy, short_names[method], fontsize=7.5, color=COLORS["ink"])
    ax.axvline(0.5, color=COLORS["grid"], linewidth=0.8, linestyle="--")
    ax.axhline(0.5, color=COLORS["grid"], linewidth=0.8, linestyle="--")
    ax.text(0.03, 0.96, "high post-formation + high recovery", transform=ax.transAxes, fontsize=7, color=COLORS["muted"], va="top")
    ax.set_xlim(0.12, 1.0)
    ax.set_ylim(-0.02, 1.0)
    style(ax, "D  Function and repair move together", "post-formation success", "recovery success")
    fig.suptitle("Figure 2 | EXP243-long: credit shapes a persistent, repairable theory library", x=0.04, ha="left",
                 fontsize=13.5, fontweight="bold", color=COLORS["ink"])
    save(fig, "figure2_exp243_long_flagship.png")


def figure3():
    exp = load("EXP244_complexity_scaling.json")
    rows = [r for r in exp["summary"] if r["budget"] == 2400 and r["family"] == "recursive_reuse"]
    methods = ["direct_memory", "phi_static", "phi_shuffled", "phi_recursive", "oracle_recursive"]
    labels = {"direct_memory": "direct memory", "phi_static": "static $\\Phi$", "phi_shuffled": "shuffled $\\Phi$",
              "phi_recursive": "dynamic $\\Phi$", "oracle_recursive": "oracle"}
    colors = {"direct_memory": COLORS["memory"], "phi_static": COLORS["control"], "phi_shuffled": COLORS["warning"],
              "phi_recursive": COLORS["phi"], "oracle_recursive": COLORS["oracle"]}
    fig = plt.figure(figsize=(13, 7.8))
    gs = fig.add_gridspec(2, 2, hspace=0.36, wspace=0.28)

    ax = fig.add_subplot(gs[0, 0])
    for method in methods:
        rr = sorted([r for r in rows if r["method"] == method], key=lambda r: r["degree"])
        line, = ax.plot([r["degree"] for r in rr], [r["success_rate"] for r in rr], marker="o", linewidth=2,
                color=colors[method], label=labels[method])
        if method == "phi_recursive":
            means = np.array([r["success_rate"] for r in rr])
            spread = np.array([r.get("success_std", 0.0) for r in rr])
            ax.fill_between([r["degree"] for r in rr], np.maximum(0, means - spread), np.minimum(1, means + spread),
                            color=colors[method], alpha=0.12, linewidth=0)
    ax.axvline(2, color=COLORS["phi"], linestyle="--", linewidth=1, alpha=0.7)
    ax.text(2.08, 0.08, "critical degree = 2", fontsize=7.5, color=COLORS["phi"], fontweight="bold")
    style(ax, "A  Recursive reuse shifts the complexity boundary", "composition degree", "success rate")
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_ylim(-0.03, 1.05)
    ax.legend(frameon=False, fontsize=7, ncol=2, loc="lower left", handlelength=2.3, columnspacing=1.0)

    ax = fig.add_subplot(gs[0, 1])
    rr = sorted([r for r in rows if r["method"] == "phi_recursive"], key=lambda r: r["degree"])
    degrees = [r["degree"] for r in rr]
    candidates = [r["retrieval_candidates_mean"] for r in rr]
    calls = [r["solution_search_cost_mean"] for r in rr]
    ax2 = ax.twinx()
    ax.plot(degrees, candidates, marker="o", color=COLORS["phi"], linewidth=2, label="retrieval candidates")
    ax2.plot(degrees, calls, marker="s", color=COLORS["credit"], linewidth=1.8, label="solution call cost")
    style(ax, "B  Organization trades search for retrieval", "composition degree", "retrieval candidates")
    ax2.set_ylabel("solution call cost", fontsize=8, color=COLORS["muted"])
    ax2.tick_params(labelsize=8, colors=COLORS["muted"])
    ax.set_xticks(degrees)
    lines, labs = ax.get_legend_handles_labels(); lines2, labs2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labs + labs2, frameon=False, fontsize=7, loc="upper left")

    ax = fig.add_subplot(gs[1, 0])
    degree5 = [r for r in rows if r["degree"] == 5]
    x = np.arange(len(methods))
    vals = [next(r["success_rate"] for r in degree5 if r["method"] == m) for m in methods]
    bars = ax.bar(x, vals, color=[colors[m] for m in methods], width=0.68)
    ax.set_xticks(x, [labels[m] for m in methods], rotation=22, ha="right", fontsize=8)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.025, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_ylim(0, 1.08)
    style(ax, "C  At degree 5, retrieval is not enumeration", "method", "success rate")
    ax.text(0.02, 0.05, "same recursive task family, same 2,400 main-search budget", transform=ax.transAxes,
            fontsize=7, color=COLORS["muted"])

    ax = fig.add_subplot(gs[1, 1])
    metrics = ["dynamic $\\Phi$", "shuffled $\\Phi$", "direct memory", "static $\\Phi$"]
    auc = [0.5833, 0.2667, 0.2000, 0.2000]
    bars = ax.barh(np.arange(len(metrics)), auc, color=[COLORS["phi"], COLORS["warning"], COLORS["memory"], COLORS["control"]])
    ax.set_yticks(np.arange(len(metrics)), metrics, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 0.7)
    for b, v in zip(bars, auc):
        ax.text(v + 0.015, b.get_y() + b.get_height() / 2, f"{v:.3f}", va="center", fontsize=8)
    style(ax, "D  Reuse advantage is selective, not universal", "AUC across degrees 1--5", None)
    ax.text(0.02, 0.04, "dynamic $\\Phi$ vs direct memory: paired exact p = 0.001", transform=ax.transAxes, fontsize=7, color=COLORS["muted"])
    fig.suptitle("Figure 3 | EXP244: recursive organization changes the scaling regime", x=0.04, ha="left",
                 fontsize=13.5, fontweight="bold", color=COLORS["ink"])
    save(fig, "figure3_exp244_complexity_boundary.png")


def nested_mean(obj, key):
    try:
        return float(obj[key]["mean"])
    except (KeyError, TypeError):
        return float("nan")


def mini_method(summary, method):
    value = summary[method]
    return value.get("mean", value)


def figure4():
    mini = load("EXP120_minigrid_phi_persistent_organization.json")["summary"]
    robot = load(ROOT / "robot_phi_exp166" / "results" / "EXP166_robot_skill_reuse.json") if (ROOT / "robot_phi_exp166" / "results" / "EXP166_robot_skill_reuse.json").exists() else load("robot_phi_exp166/results/EXP166_robot_skill_reuse.json")
    lifecycle = load("EXP245_probe_lifecycle_analysis.json")["policy_summary"]
    mini_methods = ["phi_organization", "content_kv", "fifo_kv", "last_strategy"]
    mini_labels = {"phi_organization": "$\\Phi$ organization", "content_kv": "content-KV", "fifo_kv": "FIFO-KV", "last_strategy": "last strategy"}
    mini_colors = {"phi_organization": COLORS["phi"], "content_kv": COLORS["memory"], "fifo_kv": COLORS["control"], "last_strategy": COLORS["warning"]}
    fig = plt.figure(figsize=(13, 8.0))
    gs = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.28)

    ax = fig.add_subplot(gs[0, 0])
    phase_keys = ["calibration", "interference", "transfer", "pre_damage", "post_damage", "recovery"]
    phase_labels = ["cal", "interf", "transfer", "pre", "damage", "recover"]
    x = np.arange(len(phase_keys))
    for method in ["phi_organization", "content_kv"]:
        vals = [mini[method]["phase_success_rate"].get(k, np.nan) for k in phase_keys]
        ax.plot(x, vals, marker="o", linewidth=2, label=mini_labels[method], color=mini_colors[method])
    ax.set_xticks(x, phase_labels)
    ax.set_ylim(0, 1.05)
    style(ax, "A  MiniGrid: persistence has an effect, but content-KV wins", "phase", "success rate")
    ax.legend(frameon=False, fontsize=8)
    ax.text(0.02, 0.04, "small enumerable layouts remain a direct-memory regime", transform=ax.transAxes, fontsize=7, color=COLORS["muted"])

    ax = fig.add_subplot(gs[0, 1])
    dmg = robot["summary"]["damage_curves"]
    robot_methods = [("phi_credit", "$\\Phi$ + credit", COLORS["phi"]), ("phi_shuffled", "shuffled $\\Phi$", COLORS["warning"]),
                     ("w_only_dispatch", "$W$-only dispatch", COLORS["memory"]), ("oracle", "oracle", COLORS["oracle"])]
    for method, label, color in robot_methods:
        curve = dmg.get(method, [])
        if not curve:
            continue
        ax.plot([p["episode"] for p in curve], [p["success"]["mean"] for p in curve], marker="o", linewidth=2, label=label, color=color)
    ax.set_ylim(-0.05, 1.05)
    style(ax, "B  MuJoCo: credit changes post-damage recovery", "episode after damage", "success rate")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    ax.text(0.02, 0.10, "pre-damage performance is saturated; the informative contrast is recovery", transform=ax.transAxes, fontsize=7, color=COLORS["muted"])

    ax = fig.add_subplot(gs[1, 0])
    policies = {r["policy"]: r for r in lifecycle}
    order = ["random_reset", "persistent_fifo", "credit_lifecycle", "oracle_lifecycle"]
    policy_labels = {"random_reset": "random", "persistent_fifo": "persistent", "credit_lifecycle": "credit", "oracle_lifecycle": "oracle"}
    policy_colors = {"random_reset": COLORS["memory"], "persistent_fifo": COLORS["control"],
                     "credit_lifecycle": COLORS["credit"], "oracle_lifecycle": COLORS["oracle"]}
    for policy in order:
        row = policies[policy]
        ax.scatter(row["generated_per_task"], row["success"], s=72, color=policy_colors[policy],
                   edgecolor="white", linewidth=0.8, zorder=3)
        ax.text(row["generated_per_task"] + 0.07, row["success"] + 0.008, policy_labels[policy], fontsize=7.5, color=COLORS["ink"])
    ax.annotate("lower generation cost\nwithout higher success", xy=(policies["credit_lifecycle"]["generated_per_task"], policies["credit_lifecycle"]["success"]),
                xytext=(3.6, 0.48), fontsize=7, color=COLORS["credit"],
                arrowprops={"arrowstyle": "-", "color": COLORS["credit"], "lw": 0.8})
    ax.set_xlim(0.7, 5.9)
    ax.set_ylim(0.24, 0.50)
    style(ax, "C  EXP245 reveals a cost--ability trade-off", "new probes per task", "overall success")
    ax.text(0.02, 0.04, "credit lifecycle: 2.73 new probes/task, 0.35 success", transform=ax.transAxes, fontsize=7, color=COLORS["muted"])

    ax = fig.add_subplot(gs[1, 1])
    evidence = np.array([
        [1.0, 1.0, 1.0, 0.0],
        [1.0, 0.35, 0.65, 0.0],
        [1.0, 1.0, 1.0, 0.0],
        [1.0, 0.35, 0.0, 1.0],
    ])
    row_labels = ["arithmetic", "MiniGrid", "MuJoCo", "probe lifecycle"]
    col_labels = ["persistent\nstate", "relation\nreuse", "targeted\nrecovery", "lifecycle\ncost"]
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("evidence", ["#f0f3f5", "#b7d9d8", COLORS["phi"]])
    ax.imshow(evidence, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(4), col_labels, fontsize=7.5)
    ax.set_yticks(range(4), row_labels, fontsize=8)
    for i in range(evidence.shape[0]):
        for j in range(evidence.shape[1]):
            value = evidence[i, j]
            label = "strong" if value == 1 else "partial" if value else "not tested"
            ax.text(j, i, label, ha="center", va="center", fontsize=7,
                    color="white" if value > 0.65 else COLORS["muted"])
    ax.set_title("D  What the bridge experiments actually establish", loc="left", fontsize=10.5, fontweight="bold", color=COLORS["ink"], pad=8)
    ax.set_xlabel("evidence type; qualitative synthesis of tested dimensions", fontsize=7.5, color=COLORS["muted"])
    fig.suptitle("Figure 4 | Generality is a mechanism pattern with explicit boundary conditions", x=0.04, ha="left",
                 fontsize=13.5, fontweight="bold", color=COLORS["ink"])
    save(fig, "figure4_cross_domain_lifecycle.png")


if __name__ == "__main__":
    figure1()
    figure2()
    figure3()
    figure4()
    print("generated four main figures in", OUT)
