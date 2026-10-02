"""Create the publication overview figure for organization dynamics."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parent
PNG = ROOT / "results" / "organization_loop_overview.png"
PDF = ROOT / "results" / "organization_loop_overview.pdf"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.linewidth": 0.8,
    "mathtext.fontset": "stixsans",
})

fig, ax = plt.subplots(figsize=(14.2, 7.4))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

COLORS = {
    "ink": "#1f2933",
    "muted": "#52606d",
    "line": "#5b6873",
    "capability": "#2f6f9f",
    "feedback": "#c4454d",
    "organization": "#23845c",
    "candidate": "#c58b1a",
    "budget": "#80675a",
    "fast_bg": "#f2f5f7",
    "medium_bg": "#edf7f1",
    "slow_bg": "#fff7e6",
}


def box(x, y, w, h, title, subtitle, color, fill="#ffffff"):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.009,rounding_size=0.010",
        linewidth=1.65, edgecolor=color, facecolor=fill, zorder=4,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h * 0.63, title, ha="center", va="center",
            fontsize=12.0, weight="bold", color=COLORS["ink"], zorder=5)
    ax.text(x + w / 2, y + h * 0.27, subtitle, ha="center", va="center",
            fontsize=8.2, color=COLORS["muted"], zorder=5)
    return (x, y, w, h)


def left(item):
    x, y, _, h = item
    return x, y + h / 2


def right(item):
    x, y, w, h = item
    return x + w, y + h / 2


def top(item):
    x, y, w, h = item
    return x + w / 2, y + h


def bottom(item):
    x, y, w, _ = item
    return x + w / 2, y


def arrow(start, end, color=None, width=1.55, dashed=False, curve=0.0,
          label=None, label_xy=None, zorder=3):
    color = color or COLORS["line"]
    patch = FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=12.5,
        linewidth=width, color=color, linestyle="--" if dashed else "-",
        connectionstyle=f"arc3,rad={curve}", shrinkA=2, shrinkB=2, zorder=zorder,
    )
    ax.add_patch(patch)
    if label and label_xy:
        ax.text(*label_xy, label, ha="center", va="center", fontsize=7.8,
                color=color, bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.1}, zorder=5)


ax.axhspan(0.64, 0.90, color=COLORS["fast_bg"], zorder=0)
ax.axhspan(0.355, 0.605, color=COLORS["medium_bg"], zorder=0)
ax.axhspan(0.095, 0.315, color=COLORS["slow_bg"], zorder=0)
ax.text(0.026, 0.868, "FAST", fontsize=9.5, weight="bold", color=COLORS["line"])
ax.text(0.026, 0.844, "execution and feedback", fontsize=7.8, color=COLORS["muted"])
ax.text(0.026, 0.573, "MEDIUM", fontsize=9.5, weight="bold", color=COLORS["organization"])
ax.text(0.026, 0.549, "persistent organization and selection", fontsize=7.8, color=COLORS["muted"])
ax.text(0.026, 0.282, "SLOW", fontsize=9.5, weight="bold", color=COLORS["candidate"])
ax.text(0.026, 0.258, "verified consolidation and capability change", fontsize=7.8, color=COLORS["muted"])

w, h = 0.125, 0.105
w_t = box(0.065, 0.705, w, h, r"$W_t$", "capability support", COLORS["capability"])
q_t = box(0.235, 0.705, w, h, r"$q_t$", "instant responsibility", COLORS["line"])
y_t = box(0.405, 0.705, w, h, r"$y_t$", "behavior / proposal", COLORS["line"])
r_t = box(0.575, 0.705, w, h, r"$r_{t+d}$", "outcome feedback", COLORS["feedback"])
c_t = box(0.745, 0.705, w, h, r"$c_t$", "attributed outcome", COLORS["feedback"])

phi_t = box(0.235, 0.415, w, h, r"$\Phi_t$", "persistent organization", COLORS["organization"])
candidate_t = box(0.405, 0.415, w, h, r"$\mathcal{C}_t$", "candidate relations", COLORS["candidate"])
gate_t = box(0.575, 0.415, w, h, r"$\mathcal{V}$ + gates", "validity, function, cause", COLORS["candidate"])
unit_t = box(0.745, 0.415, w, h, r"$U_t$", "effective units", COLORS["organization"])

budget_t = box(0.065, 0.145, 0.18, 0.09, r"$B_t$", "compute / memory / verification", COLORS["budget"])
compile_t = box(0.475, 0.145, 0.15, 0.09, r"$\mathcal{A}$", "verified consolidation", COLORS["candidate"])
w_next = box(0.745, 0.145, w, 0.09, r"$W_{t+1}$", "updated callable support", COLORS["capability"])

for source, target in ((w_t, q_t), (q_t, y_t), (y_t, r_t), (r_t, c_t),
                       (phi_t, candidate_t), (candidate_t, gate_t), (gate_t, unit_t),
                       (compile_t, w_next)):
    arrow(right(source), left(target))

arrow(bottom(unit_t), top(compile_t), color=COLORS["organization"], curve=0.12,
      label="verified reuse", label_xy=(0.705, 0.325))

cx, cy = bottom(c_t)
px, py = phi_t[0] + 0.035, phi_t[1] + phi_t[3]
ax.plot([cx, cx, px], [cy, 0.615, 0.615], color=COLORS["feedback"], linewidth=1.55, zorder=2)
arrow((px, 0.615), (px, py), color=COLORS["feedback"])
ax.text(0.575, 0.622, "feedback updates organization", ha="center", va="center", fontsize=7.8,
        color=COLORS["feedback"], bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.0})

responsibility_x = phi_t[0] + phi_t[2] - 0.035
arrow((responsibility_x, phi_t[1] + phi_t[3]), (responsibility_x, q_t[1]),
      color=COLORS["organization"], label="biases responsibility", label_xy=(0.315, 0.625))

bx, by = right(budget_t)
arrow((bx, by), left(compile_t), color=COLORS["budget"], width=1.2, dashed=True,
      label="shared resource constraint", label_xy=(0.36, by + 0.027))
ax.plot([0.445, 0.445], [by, bottom(candidate_t)[1] - 0.012], color=COLORS["budget"],
        linewidth=1.0, linestyle="--", zorder=1)
ax.plot([0.615, 0.615], [by, bottom(gate_t)[1] - 0.012], color=COLORS["budget"],
        linewidth=1.0, linestyle="--", zorder=1)

wx, wy = bottom(w_next)
old_x, old_y = bottom(w_t)
ax.plot([wx, wx, old_x], [wy, 0.070, 0.070], color=COLORS["capability"], linewidth=1.55, zorder=2)
arrow((old_x, 0.070), (old_x, old_y), color=COLORS["capability"])
ax.text(0.49, 0.058, "next organization cycle", ha="center", va="center", fontsize=7.8,
        color=COLORS["capability"], bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.0})

ax.text(0.895, 0.785, r"Operational signature of $\Phi$", fontsize=9.2, weight="bold",
        color=COLORS["organization"], ha="left")
ax.text(0.895, 0.745, "feedback-updated\ncross-prediction persistent\nindependently intervenable",
        fontsize=7.8, color=COLORS["muted"], ha="left", va="top", linespacing=1.35)
ax.text(0.895, 0.475, "Organization is a causal role,", fontsize=8.7, weight="bold",
        color=COLORS["ink"], ha="left")
ax.text(0.895, 0.443, "not an architecture name.", fontsize=7.8, color=COLORS["muted"], ha="left")

fig.suptitle("A minimal dynamical loop for organization formation", fontsize=16, weight="bold", y=0.976,
             color=COLORS["ink"])
ax.text(0.5, 0.927,
        r"Capability $W$ constrains what is possible; credit-updated $\Phi$ changes what is selected, retained, and consolidated.",
        ha="center", va="center", fontsize=9.5, color=COLORS["muted"])

PNG.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(PNG, dpi=260, bbox_inches="tight", facecolor="white", edgecolor="none")
fig.savefig(PDF, bbox_inches="tight", facecolor="white", edgecolor="none")
plt.close(fig)
print(PNG)
print(PDF)
