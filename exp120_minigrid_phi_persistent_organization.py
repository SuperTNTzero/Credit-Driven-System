# -*- coding: utf-8 -*-
"""EXP120: persistent organization state in unlabeled MiniGrid.

This is a controlled MiniGrid strategy-organization benchmark.  A frozen
capability library supplies reliable policies for several official MiniGrid
families.  The selector sees only the first partial observation; it does not
receive the environment name, mission, task label, or privileged map.  It
chooses a capability, executes the episode, and receives only terminal
success/failure before the next episode.

The experiment targets the paper's stronger claim about Phi:

    feedback -> persistent organization state -> context-conditioned reuse

The main controls have the same six slots and the same 21-float context key:

* phi_organization: prototype-routed utility state with credit updates;
* content_kv: robust content-addressed key/value records;
* fifo_kv: direct recent-event memory with the same state budget;
* last_strategy: one-step recency/cache control;
* random and oracle reference policies.

The evaluation includes held-out layout transfer, distractor interference,
state damage, and recovery.  It is a mechanism-level MiniGrid experiment,
not an end-to-end RL or SOTA benchmark.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict

import gymnasium as gym
import matplotlib.pyplot as plt
import minigrid  # noqa: F401 - registers official environments
import numpy as np

import exp59_minigrid_architecture_comparison as obs_base
import exp62_minigrid_fair_phi_attention as mg


EXPERIMENT = "EXP120"
SEEDS = list(range(1200, 1220))
FAMILIES = tuple(mg.ENV_IDS.keys())
STRATEGIES = ("nav", "keydoor", "safe", "corridor", "memory")
ORACLE_STRATEGY = {
    "nav": "nav",
    "door": "keydoor",
    "lava": "safe",
    "corridor": "corridor",
    "memory": "memory",
    "unlock": "keydoor",
}
STRATEGY_FAMILY = {
    "nav": "nav",
    "keydoor": "door",
    "safe": "lava",
    "corridor": "corridor",
    "memory": "memory",
}
DISPLAY = {
    "phi_organization": "Phi organization",
    "content_kv": "Content KV",
    "fifo_kv": "FIFO-KV",
    "last_strategy": "Last strategy",
    "random": "Random",
    "oracle": "Oracle",
}
METHODS = tuple(DISPLAY)
CONTEXT_DIM = 21
NUM_STRATEGIES = len(STRATEGIES)
SLOTS = 6
STATE_FLOATS = SLOTS * (CONTEXT_DIM + NUM_STRATEGIES)
RESULT_JSON = os.path.join("results", "EXP120_minigrid_phi_persistent_organization.json")
RESULT_PNG = os.path.join("results", "EXP120_minigrid_phi_persistent_organization.png")
STRUCTURE_PNG = os.path.join("results", "EXP120_structure_intervention.png")


def _normalize(value):
    value = np.asarray(value, dtype=np.float64)
    return value / (np.linalg.norm(value) + 1e-8)


def _context(obs):
    """Mission-free 21-float partial-observation signature."""
    key = obs_base._obs_key(obs)
    return _normalize(np.asarray(key, dtype=np.float64))


def _context_key(obs):
    return tuple(np.asarray(obs_base._obs_key(obs), dtype=np.int16).tolist())


def _softmax(logits, temperature=0.18):
    values = np.asarray(logits, dtype=np.float64) / temperature
    values -= values.max()
    weights = np.exp(values)
    return weights / (weights.sum() + 1e-8)


class OrganizationState:
    """Common interface for prediction-time read and post-feedback update."""

    dynamic_state_floats = STATE_FLOATS

    def choose(self, context):
        raise NotImplementedError

    def update(self, context, slot, strategy, success):
        raise NotImplementedError

    def damage(self, rng):
        raise NotImplementedError

    def state_vector(self):
        raise NotImplementedError


class PhiOrganization(OrganizationState):
    """Prototype-routed, credit-updated organizational state.

    The prototype and utility state are persistent across episodes.  `choose`
    is read-only.  Only `update` changes Phi after terminal feedback arrives.
    """

    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)
        self.prototypes = self.rng.normal(size=(SLOTS, CONTEXT_DIM))
        self.prototypes /= np.linalg.norm(self.prototypes, axis=1, keepdims=True)
        self.utility = np.zeros((SLOTS, NUM_STRATEGIES), dtype=np.float64)
        self.counts = np.zeros(SLOTS, dtype=np.int64)
        self.update_count = 0
        self.route_history = []
        self.strategy_history = []

    def _route(self, context):
        scores = self.prototypes @ context
        return int(np.argmax(scores)), scores

    def choose(self, context):
        slot, route_scores = self._route(context)
        strategy_scores = self.utility[slot].copy()
        # Exploration is only a selector decision; it does not mutate state.
        explore = 0.26 * np.exp(-self.update_count / 48.0) + 0.035
        if self.rng.random() < explore or np.max(np.abs(strategy_scores)) < 0.05:
            strategy = int(self.rng.integers(NUM_STRATEGIES))
        else:
            strategy = int(np.argmax(strategy_scores))
        self.route_history.append(slot)
        self.strategy_history.append(strategy)
        return strategy, slot, float(route_scores[slot])

    def update(self, context, slot, strategy, success):
        reward = 1.0 if success else -1.0
        # Credit updates both the context prototype and the strategy utility.
        # A failure still supplies information, but with smaller structural
        # plasticity so one bad episode does not erase a stable route.
        proto_lr = 0.18 if success else 0.07
        self.prototypes[slot] = _normalize(
            (1.0 - proto_lr) * self.prototypes[slot] + proto_lr * context
        )
        self.utility[slot] *= 0.985
        self.utility[slot, strategy] += 0.38 * reward
        self.utility[slot] = np.clip(self.utility[slot], -2.0, 2.0)
        self.counts[slot] += 1
        self.update_count += 1

    def damage(self, rng):
        slots = rng.choice(SLOTS, size=SLOTS // 2, replace=False)
        self.utility[slots] *= 0.0
        self.prototypes[slots] = rng.normal(size=(len(slots), CONTEXT_DIM))
        self.prototypes[slots] /= np.linalg.norm(self.prototypes[slots], axis=1, keepdims=True)
        return [int(x) for x in slots]

    def state_vector(self):
        return np.concatenate([self.prototypes.ravel(), self.utility.ravel()])


class ContentKV(OrganizationState):
    """A strong prototype/content-addressed KV baseline with equal capacity."""

    def __init__(self, seed, fifo=False):
        self.rng = np.random.default_rng(seed)
        self.keys = np.zeros((SLOTS, CONTEXT_DIM), dtype=np.float64)
        self.values = np.zeros((SLOTS, NUM_STRATEGIES), dtype=np.float64)
        self.valid = np.zeros(SLOTS, dtype=bool)
        self.age = np.zeros(SLOTS, dtype=np.int64)
        self.clock = 0
        self.fifo = fifo

    def _nearest(self, context):
        if not self.valid.any():
            return None, -1.0
        scores = self.keys @ context
        slot = int(np.argmax(scores))
        return slot, float(scores[slot])

    def choose(self, context):
        slot, similarity = self._nearest(context)
        if slot is None or similarity < 0.82:
            return int(self.rng.integers(NUM_STRATEGIES)), -1, similarity
        values = self.values[slot]
        if np.max(np.abs(values)) < 0.05:
            strategy = int(self.rng.integers(NUM_STRATEGIES))
        else:
            strategy = int(np.argmax(values))
        return strategy, slot, similarity

    def update(self, context, slot, strategy, success):
        reward = 1.0 if success else -1.0
        if slot < 0 or not self.valid[slot] or (not self.fifo and self.keys[slot] @ context < 0.82):
            if self.fifo:
                slot = int(np.argmin(self.age))
            else:
                invalid = np.flatnonzero(~self.valid)
                slot = int(invalid[0]) if len(invalid) else int(np.argmin(self.age))
            self.valid[slot] = True
            self.keys[slot] = context
            self.values[slot] = 0.0
        if self.fifo:
            # Direct event memory does not consolidate neighboring contexts.
            self.keys[slot] = context
            self.values[slot] = 0.0
        else:
            self.keys[slot] = _normalize(0.85 * self.keys[slot] + 0.15 * context)
        self.values[slot] *= 0.985
        self.values[slot, strategy] += 0.38 * reward
        self.values[slot] = np.clip(self.values[slot], -2.0, 2.0)
        self.clock += 1
        self.age[slot] = self.clock

    def damage(self, rng):
        slots = rng.choice(SLOTS, size=SLOTS // 2, replace=False)
        self.values[slots] *= 0.0
        self.keys[slots] = rng.normal(size=(len(slots), CONTEXT_DIM))
        self.keys[slots] /= np.linalg.norm(self.keys[slots], axis=1, keepdims=True)
        return [int(x) for x in slots]

    def state_vector(self):
        return np.concatenate([self.keys.ravel(), self.values.ravel()])


class LastStrategy(OrganizationState):
    dynamic_state_floats = 1

    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)
        self.last = int(self.rng.integers(NUM_STRATEGIES))

    def choose(self, context):
        del context
        return self.last, 0, 1.0

    def update(self, context, slot, strategy, success):
        del context, slot
        if success:
            self.last = strategy

    def damage(self, rng):
        self.last = int(rng.integers(NUM_STRATEGIES))
        return [0]

    def state_vector(self):
        return np.asarray([self.last], dtype=np.float64)


def _make_state(method, seed):
    if method == "phi_organization":
        return PhiOrganization(seed)
    if method == "content_kv":
        return ContentKV(seed, fifo=False)
    if method == "fifo_kv":
        return ContentKV(seed, fifo=True)
    if method == "last_strategy":
        return LastStrategy(seed)
    return None


def _schedule(seed):
    """Return phase/family pairs with distinct held-out layout seeds."""
    rng = np.random.default_rng(seed)
    stream = []
    calibration_order = list(FAMILIES)
    for block in range(4):
        order = calibration_order.copy()
        if block % 2:
            rng.shuffle(order)
        stream.extend(("calibration", family) for family in order)
    for block in range(4):
        order = list(FAMILIES)
        rng.shuffle(order)
        stream.extend(("interference", family) for family in order)
    # These are new layout seeds, not new labels.  The first item of each
    # family is the zero-shot transfer probe before its feedback update.
    for block in range(2):
        order = list(FAMILIES)
        rng.shuffle(order)
        stream.extend(("transfer", family) for family in order)
    for block in range(2):
        order = list(FAMILIES)
        rng.shuffle(order)
        stream.extend(("pre_damage", family) for family in order)
    for block in range(2):
        order = list(FAMILIES)
        rng.shuffle(order)
        stream.extend(("post_damage", family) for family in order)
    for block in range(4):
        order = list(FAMILIES)
        rng.shuffle(order)
        stream.extend(("recovery", family) for family in order)
    return stream


def _episode_seed(seed, episode, phase):
    offsets = {
        "calibration": 100000,
        "interference": 200000,
        "transfer": 300000,
        "pre_damage": 400000,
        "post_damage": 500000,
        "recovery": 600000,
    }
    return int(offsets[phase] + seed * 1000 + episode)


def _probe_context(family, env_seed):
    env = gym.make(mg.ENV_IDS[family])
    obs, _ = env.reset(seed=env_seed)
    context = _context(obs)
    key = _context_key(obs)
    env.close()
    return context, key


def _run_capability(family, strategy, env_seed):
    env = gym.make(mg.ENV_IDS[family])
    obs, _ = env.reset(seed=env_seed)
    total = 0.0
    steps = 0
    for steps in range(env.unwrapped.max_steps):
        if strategy == "memory" and family != "memory":
            # MemoryS7's success/failure coordinate interface does not exist
            # in other environments.  An inapplicable capability is a normal
            # failed selection, represented by a simple forward policy.
            action = int(env.unwrapped.actions.forward)
        else:
            action = mg._expert_action(env, STRATEGY_FAMILY[strategy])
        obs, reward, terminated, truncated, _ = env.step(action)
        total += float(reward)
        if terminated or truncated:
            break
    env.close()
    return bool(total > 0.0), steps + 1


def _run_method(method, seed):
    state = _make_state(method, seed + 70000)
    rng = np.random.default_rng(seed + 80000)
    records = []
    transfer_seen = set()
    damage_done = False
    pre_damage_vector = None
    post_interference_vector = None
    for episode, (phase, family) in enumerate(_schedule(seed)):
        if phase == "post_damage" and not damage_done:
            pre_damage_vector = state.state_vector().copy() if state is not None else np.zeros(1)
            damaged_slots = state.damage(rng) if state is not None else []
            damage_done = True
        env_seed = _episode_seed(seed, episode, phase)
        context, context_key = _probe_context(family, env_seed)
        oracle = STRATEGIES.index(ORACLE_STRATEGY[family])
        if method == "oracle":
            strategy, slot, similarity = oracle, 0, 1.0
        elif method == "random":
            strategy, slot, similarity = int(rng.integers(NUM_STRATEGIES)), -1, 0.0
        else:
            strategy, slot, similarity = state.choose(context)
        success, steps = _run_capability(family, STRATEGIES[strategy], env_seed)
        if state is not None:
            state.update(context, slot, strategy, success)
        is_zero_shot = phase == "transfer" and family not in transfer_seen
        if phase == "transfer":
            transfer_seen.add(family)
        records.append({
            "episode": episode,
            "phase": phase,
            "family": family,
            "context_key": context_key,
            "strategy": STRATEGIES[strategy],
            "oracle_strategy": STRATEGIES[oracle],
            "selection_correct": int(strategy == oracle),
            "success": int(success),
            "steps": int(steps),
            "slot": int(slot),
            "similarity": float(similarity),
            "zero_shot": int(is_zero_shot),
        })
        if phase == "interference" and episode == max(
            index for index, item in enumerate(_schedule(seed)) if item[0] == "interference"
        ):
            post_interference_vector = state.state_vector().copy() if state is not None else np.zeros(1)
    if state is not None and post_interference_vector is None:
        post_interference_vector = state.state_vector().copy()
    if state is not None and pre_damage_vector is None:
        pre_damage_vector = state.state_vector().copy()
    return records, state, {
        "damaged_slots": damaged_slots if damage_done else [],
        "pre_damage_vector": pre_damage_vector,
        "post_interference_vector": post_interference_vector,
    }


def _cosine(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.dot(a, b) / ((np.linalg.norm(a) + 1e-8) * (np.linalg.norm(b) + 1e-8)))


def _phase_rate(records, phase, field="success"):
    rows = [r for r in records if r["phase"] == phase]
    return float(np.mean([r[field] for r in rows])) if rows else 0.0


def _recovery_steps(records):
    rows = [r for r in records if r["phase"] == "recovery"]
    for index in range(len(rows) - 1):
        if rows[index]["success"] and rows[index + 1]["success"]:
            return index + 1
    return len(rows) + 1


def _summarize(method, records, state, audit):
    by_family = {
        family: float(np.mean([r["success"] for r in records if r["family"] == family]))
        for family in FAMILIES
    }
    selected = [r["selection_correct"] for r in records]
    slots = [r["slot"] for r in records if r["slot"] >= 0]
    churn = sum(
        int(records[i]["strategy"] != records[i - 1]["strategy"])
        for i in range(1, len(records))
    )
    state_persistence = _cosine(audit["pre_damage_vector"], audit["post_interference_vector"])
    return {
        "method": method,
        "overall_success_rate": float(np.mean([r["success"] for r in records])),
        "overall_selection_accuracy": float(np.mean(selected)),
        "phase_success_rate": {
            phase: _phase_rate(records, phase) for phase in (
                "calibration", "interference", "transfer", "pre_damage", "post_damage", "recovery"
            )
        },
        "phase_selection_accuracy": {
            phase: _phase_rate(records, phase, "selection_correct") for phase in (
                "calibration", "interference", "transfer", "pre_damage", "post_damage", "recovery"
            )
        },
        "family_success_rate": by_family,
        "zero_shot_transfer_success": float(np.mean([r["success"] for r in records if r["zero_shot"]])) if any(r["zero_shot"] for r in records) else 0.0,
        "interference_retention": _phase_rate(records, "pre_damage") - _phase_rate(records, "interference"),
        "damage_drop": _phase_rate(records, "pre_damage") - _phase_rate(records, "post_damage"),
        "recovery_steps": int(_recovery_steps(records)),
        "strategy_churn": int(churn),
        "state_persistence_cosine": state_persistence,
        "slot_usage": {str(slot): int(slots.count(slot)) for slot in sorted(set(slots))},
        "dynamic_state_floats": int(state.dynamic_state_floats if state is not None else 0),
        "damaged_slots": audit["damaged_slots"],
    }


def _plot(raw):
    names = list(METHODS)
    x = np.arange(len(names))
    metrics = [
        ("overall_success_rate", "Overall success", (0.0, 1.05)),
        ("zero_shot_transfer_success", "Held-out layout transfer", (0.0, 1.05)),
        ("state_persistence_cosine", "State persistence", (-0.05, 1.05)),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    colors = ["#d97706", "#0f766e", "#64748b", "#94a3b8", "#dc2626", "#2563eb"]
    for ax, (key, title, ylim) in zip(axes[0], metrics):
        means = [np.mean([row[key] for row in raw[name]]) for name in names]
        stds = [np.std([row[key] for row in raw[name]], ddof=1) for name in names]
        ax.bar(x, means, yerr=stds, capsize=3, color=colors)
        ax.set_title(title)
        ax.set_ylim(*ylim)
        ax.set_xticks(x, [DISPLAY[name] for name in names], rotation=22, ha="right")
        ax.grid(axis="y", alpha=0.25)
    phase_names = ("pre_damage", "post_damage", "recovery")
    phase_labels = ("Pre-damage", "Post-damage", "Recovery")
    width = 0.13
    for j, (phase, label) in enumerate(zip(phase_names, phase_labels)):
        values = [np.mean([row["phase_success_rate"][phase] for row in raw[name]]) for name in names]
        axes[1, 0].bar(x + (j - 1) * width, values, width, label=label)
    axes[1, 0].set_title("Damage and recovery")
    axes[1, 0].set_ylim(0, 1.05)
    axes[1, 0].set_xticks(x, [DISPLAY[name] for name in names], rotation=22, ha="right")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(axis="y", alpha=0.25)
    recovery = [np.mean([row["recovery_steps"] for row in raw[name]]) for name in names]
    axes[1, 1].bar(x, recovery, color=colors)
    axes[1, 1].set_title("Episodes to two consecutive recoveries")
    axes[1, 1].set_xticks(x, [DISPLAY[name] for name in names], rotation=22, ha="right")
    axes[1, 1].grid(axis="y", alpha=0.25)
    retention = [np.mean([row["interference_retention"] for row in raw[name]]) for name in names]
    axes[1, 2].bar(x, retention, color=colors)
    axes[1, 2].set_title("Pre-damage minus interference loss")
    axes[1, 2].axhline(0.0, color="black", linewidth=0.8)
    axes[1, 2].set_xticks(x, [DISPLAY[name] for name in names], rotation=22, ha="right")
    axes[1, 2].grid(axis="y", alpha=0.25)
    fig.suptitle("EXP120 MiniGrid: persistent organization state under hidden task switching")
    fig.tight_layout()
    fig.savefig(RESULT_PNG, dpi=180)
    plt.close(fig)

    # A compact qualitative view of the mechanism-level intervention.
    phi_rows = raw["phi_organization"]
    kv_rows = raw["content_kv"]
    labels = ["pre-damage", "post-damage", "recovery"]
    phi_values = [np.mean([r["phase_success_rate"][p] for r in phi_rows]) for p in phase_names]
    kv_values = [np.mean([r["phase_success_rate"][p] for r in kv_rows]) for p in phase_names]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.plot(labels, phi_values, marker="o", linewidth=2.5, label="Phi organization", color="#d97706")
    ax.plot(labels, kv_values, marker="o", linewidth=2.5, label="Content KV", color="#0f766e")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("success rate")
    ax.set_title("EXP120: structure damage and recovery")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(STRUCTURE_PNG, dpi=180)
    plt.close(fig)


def run_exp120(seeds=SEEDS):
    os.makedirs("results", exist_ok=True)
    raw = {method: [] for method in METHODS}
    traces = {}
    for method in METHODS:
        for seed in seeds:
            records, state, audit = _run_method(method, seed)
            raw[method].append(_summarize(method, records, state, audit))
            if seed == seeds[0]:
                traces[method] = records
    _plot(raw)
    summary = {}
    for method in METHODS:
        rows = raw[method]
        numeric = (
            "overall_success_rate", "overall_selection_accuracy",
            "zero_shot_transfer_success", "interference_retention", "damage_drop",
            "recovery_steps", "strategy_churn", "state_persistence_cosine",
            "dynamic_state_floats",
        )
        summary[method] = {
            "mean": {key: float(np.mean([row[key] for row in rows])) for key in numeric},
            "std": {key: float(np.std([row[key] for row in rows], ddof=1)) for key in numeric if key != "dynamic_state_floats"},
            "phase_success_rate": {
                phase: float(np.mean([row["phase_success_rate"][phase] for row in rows]))
                for phase in ("calibration", "interference", "transfer", "pre_damage", "post_damage", "recovery")
            },
            "phase_selection_accuracy": {
                phase: float(np.mean([row["phase_selection_accuracy"][phase] for row in rows]))
                for phase in ("calibration", "interference", "transfer", "pre_damage", "post_damage", "recovery")
            },
            "family_success_rate": {
                family: float(np.mean([row["family_success_rate"][family] for row in rows]))
                for family in FAMILIES
            },
            "per_seed": rows,
        }
    result = {
        "experiment": EXPERIMENT,
        "title": "MiniGrid hidden-task persistent organization and structural recovery",
        "protocol": {
            "envs": dict(mg.ENV_IDS),
            "families": list(FAMILIES),
            "strategies": list(STRATEGIES),
            "seeds": list(seeds),
            "episodes_per_seed": len(_schedule(seeds[0])),
            "mission_visible_to_selector": False,
            "task_label_visible_to_selector": False,
            "feedback": "terminal success/failure only, after each episode",
            "capability_library": "fixed MiniGrid expert policies used only to isolate organization",
            "held_out_transfer": "new environment seeds in transfer phase; first episode per family is zero-shot",
            "interference": "four mixed family blocks before transfer and revisit",
            "damage": "half of persistent slots are reset before post_damage",
            "matched_dynamic_state_floats": STATE_FLOATS,
            "limitations": [
                "not end-to-end RL",
                "capability policies are privileged and frozen",
                "partial observation signature is a coarse engineered encoder",
                "oracle is an upper bound, not a learnable competitor",
            ],
        },
        "state_budget": {
            "phi_organization": STATE_FLOATS,
            "content_kv": STATE_FLOATS,
            "fifo_kv": STATE_FLOATS,
            "last_strategy": 1,
            "random": 0,
            "oracle": 0,
        },
        "summary": summary,
        "trace_seed": traces,
    }
    with open(RESULT_JSON, "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    return result


if __name__ == "__main__":
    result = run_exp120()
    for method in METHODS:
        mean = result["summary"][method]["mean"]
        print(
            DISPLAY[method],
            "overall=", round(mean["overall_success_rate"], 4),
            "transfer=", round(mean["zero_shot_transfer_success"], 4),
            "recovery_steps=", round(mean["recovery_steps"], 3),
        )
