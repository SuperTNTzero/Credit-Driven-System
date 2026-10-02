# -*- coding: utf-8 -*-
"""EXP56: MiniGrid non-stationary strategy organization.

The environment is official MiniGrid, but the experiment isolates the
organization question from visual representation learning.  A frozen set of
four privileged capability policies is available to every method.  The model
does not receive mission text, environment id, or a task label.  It sees only
the first partial observation and chooses one capability for the episode.
After the episode, success/failure updates the online organization state.

This is intentionally a strategy-selection benchmark, not an end-to-end
MiniGrid SOTA benchmark.  It asks whether Phi can reuse a small capability
library under task switching with less churn than direct episodic memory.
"""
from __future__ import annotations

import json
import os
from collections import Counter, deque

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import minigrid  # noqa: F401 - registers the official environments


SEEDS = [0, 1, 2, 3, 4]
ENV_IDS = {
    "nav": "MiniGrid-Empty-5x5-v0",
    "door": "MiniGrid-DoorKey-5x5-v0",
    "lava": "MiniGrid-LavaGapS5-v0",
    "corridor": "MiniGrid-KeyCorridorS3R1-v0",
}
EXPERTS = ("nav", "keydoor", "safe", "corridor")
ORACLE_EXPERT = {"nav": "nav", "door": "keydoor", "lava": "safe", "corridor": "corridor"}
DISPLAY = {
    "uniform": "Uniform selector",
    "recency": "Recency state",
    "direct_memory": "Direct strategy memory",
    "phi": "Phi organization",
    "oracle": "Oracle task label",
}
RESULT_JSON = os.path.join("results", "EXP56_minigrid_strategy_organization.json")
RESULT_PNG = os.path.join("results", "EXP56_minigrid_strategy_organization.png")


def _obj_at(env, pos):
    obj = env.unwrapped.grid.get(int(pos[0]), int(pos[1]))
    return obj


def _objects(env):
    out = []
    for y in range(env.unwrapped.height):
        for x in range(env.unwrapped.width):
            obj = _obj_at(env, (x, y))
            if obj is not None:
                out.append((obj, (x, y)))
    return out


def _find(env, kinds):
    for obj, pos in _objects(env):
        if getattr(obj, "type", None) in kinds:
            yield obj, pos


def _front_pos(env):
    pos = np.asarray(env.unwrapped.agent_pos, dtype=int)
    direction = np.asarray([(1, 0), (0, 1), (-1, 0), (0, -1)])[env.unwrapped.agent_dir]
    return tuple((pos + direction).tolist())


def _carrying(env, kind):
    carried = env.unwrapped.carrying
    return carried is not None and getattr(carried, "type", None) == kind


def _passable(obj, avoid_lava):
    if obj is None:
        return True
    kind = getattr(obj, "type", None)
    if kind in ("goal", "key", "ball", "box"):
        return True
    if kind == "lava":
        return not avoid_lava
    if kind == "door":
        return bool(getattr(obj, "is_open", False))
    return False


def _bfs(env, target, avoid_lava=True):
    """Return a shortest coordinate path using only observable grid semantics."""
    start = tuple(map(int, env.unwrapped.agent_pos))
    target = tuple(map(int, target))
    if start == target:
        return [start]
    queue = deque([start])
    parent = {start: None}
    while queue:
        cur = queue.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nxt = (cur[0] + dx, cur[1] + dy)
            if not (0 <= nxt[0] < env.unwrapped.width and 0 <= nxt[1] < env.unwrapped.height):
                continue
            if nxt in parent:
                continue
            obj = _obj_at(env, nxt)
            if not _passable(obj, avoid_lava):
                continue
            parent[nxt] = cur
            if nxt == target:
                path = [nxt]
                while path[-1] != start:
                    path.append(parent[path[-1]])
                return list(reversed(path))
            queue.append(nxt)
    return [start]


def _target(env, strategy):
    goals = list(_find(env, {"goal"}))
    goal = goals[0][1] if goals else tuple(map(int, env.unwrapped.agent_pos))
    if strategy == "nav":
        return goal, True
    if strategy == "safe":
        return goal, True
    if strategy == "corridor":
        if not _carrying(env, "key"):
            keys = list(_find(env, {"key"}))
            if keys:
                key_path = _bfs(env, keys[0][1], avoid_lava=True)
                if len(key_path) > 1 or tuple(map(int, env.unwrapped.agent_pos)) == tuple(keys[0][1]):
                    return keys[0][1], True
        closed_doors = [pos for door, pos in _find(env, {"door"})
                        if not getattr(door, "is_open", False)
                        and (not getattr(door, "is_locked", False) or _carrying(env, "key"))]
        if closed_doors:
            return _adjacent_interaction_target(env, closed_doors[0]), True
        balls = list(_find(env, {"ball"}))
        if balls:
            return balls[0][1], True
        return goal, True
    # keydoor: acquire key, then open the door, then reach the goal.
    if not _carrying(env, "key"):
        keys = list(_find(env, {"key"}))
        if keys:
            return keys[0][1], True
    doors = list(_find(env, {"door"}))
    for door, pos in doors:
        if not getattr(door, "is_open", False):
            return _adjacent_interaction_target(env, pos), True
    return goal, True


def _adjacent_interaction_target(env, pos):
    """Choose a passable square from which a door can be toggled."""
    candidates = []
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        candidate = (int(pos[0]) + dx, int(pos[1]) + dy)
        if not (0 <= candidate[0] < env.unwrapped.width and
                0 <= candidate[1] < env.unwrapped.height):
            continue
        if _passable(_obj_at(env, candidate), avoid_lava=True):
            candidates.append(candidate)
    if not candidates:
        return tuple(pos)
    current = tuple(map(int, env.unwrapped.agent_pos))
    reachable = [(p, _bfs(env, p, avoid_lava=True)) for p in candidates]
    reachable = [(p, path) for p, path in reachable if path[-1] == p]
    if reachable:
        return min(reachable, key=lambda item: len(item[1]))[0]
    return current if current in candidates else candidates[0]


def _expert_action(env, strategy):
    """A fixed capability policy.  It can observe the grid for isolation only."""
    action = env.unwrapped.actions
    corridor_phase = getattr(env.unwrapped, "_corridor_phase", None)
    if strategy == "corridor" and corridor_phase == "turn_for_drop":
        if int(env.unwrapped.agent_dir) != 2:
            return int(action.left)
        env.unwrapped._corridor_phase = "turn_back"
        return int(action.drop)
    if strategy == "corridor" and corridor_phase == "turn_back":
        if int(env.unwrapped.agent_dir) != 0:
            return int(action.right)
        env.unwrapped._corridor_phase = None
    front = _front_pos(env)
    front_obj = None
    if 0 <= front[0] < env.unwrapped.width and 0 <= front[1] < env.unwrapped.height:
        front_obj = _obj_at(env, front)

    if strategy in ("keydoor", "corridor") and front_obj is not None:
        if getattr(front_obj, "type", None) == "key" and not _carrying(env, "key"):
            return int(action.pickup)
        if (getattr(front_obj, "type", None) == "door"
                and not getattr(front_obj, "is_open", False)
                and (not getattr(front_obj, "is_locked", False) or _carrying(env, "key"))):
            return int(action.toggle)

    if strategy in ("keydoor", "corridor"):
        for door, door_pos in _find(env, {"door"}):
            if getattr(door, "is_open", False):
                continue
            if getattr(door, "is_locked", False) and not _carrying(env, "key"):
                continue
            here = tuple(map(int, env.unwrapped.agent_pos))
            delta = (int(door_pos[0]) - here[0], int(door_pos[1]) - here[1])
            if abs(delta[0]) + abs(delta[1]) == 1:
                desired = {(1, 0): 0, (0, 1): 1, (-1, 0): 2, (0, -1): 3}[delta]
                current = int(env.unwrapped.agent_dir)
                turn = (desired - current) % 4
                if turn == 0:
                    return int(action.toggle)
                return int(action.right if turn == 1 else action.left)
            break
    if strategy == "corridor" and front_obj is not None:
        if getattr(front_obj, "type", None) == "ball":
            if _carrying(env, "key"):
                # Drop the key behind the agent, then return to the ball.
                env.unwrapped._corridor_phase = "turn_for_drop"
                return int(action.left)
            return int(action.pickup)

    target, avoid_lava = _target(env, strategy)
    path = _bfs(env, target, avoid_lava=avoid_lava)
    if len(path) <= 1:
        return int(action.forward)
    here = tuple(map(int, env.unwrapped.agent_pos))
    nxt = path[1]
    desired = {(1, 0): 0, (0, 1): 1, (-1, 0): 2, (0, -1): 3}[(nxt[0] - here[0], nxt[1] - here[1])]
    current = int(env.unwrapped.agent_dir)
    delta = (desired - current) % 4
    if delta == 0:
        return int(action.forward)
    if delta == 1:
        return int(action.right)
    return int(action.left)


def _obs_features(obs):
    """Mission-free coarse signature from the official partial observation."""
    image = np.asarray(obs["image"], dtype=np.int64)
    object_hist = np.bincount(image[:, :, 0].ravel(), minlength=11)[:11]
    color_hist = np.bincount(image[:, :, 1].ravel(), minlength=6)[:6]
    state_hist = np.bincount(image[:, :, 2].ravel(), minlength=3)[:3]
    feat = np.concatenate([object_hist, color_hist, state_hist, [int(obs["direction"])]])
    return feat.astype(np.float64)


def _context_key(obs):
    feat = _obs_features(obs)
    # Keep the key independent of mission text and exact coordinates.
    return tuple(feat.astype(np.int16).tolist())


def _family_key(obs):
    feat = _obs_features(obs)
    # Coarse buckets allow recurrence across different maps/seeds.
    return tuple(np.concatenate([feat[:11], feat[11:17], [feat[-1]]]).astype(np.int16).tolist())


class Selector:
    def __init__(self, kind, seed, phi_lr=0.25, memory_capacity=12):
        self.kind = kind
        self.rng = np.random.default_rng(seed)
        self.phi_lr = phi_lr
        self.memory_capacity = memory_capacity
        self.phi = np.zeros((1, len(EXPERTS)), dtype=np.float64)
        self.global_phi = np.zeros(len(EXPERTS), dtype=np.float64)
        self.contexts = []
        self.context_index = {}
        self.memory = deque(maxlen=memory_capacity)
        self.last_expert = None
        self.last_success = None
        self.churn = 0
        self.updates = 0

    def _row(self, key):
        if key not in self.context_index:
            self.context_index[key] = len(self.contexts)
            self.contexts.append(key)
            self.phi = np.vstack([self.phi, np.zeros((1, len(EXPERTS)))])
        return self.context_index[key]

    def choose(self, obs, family, oracle=None):
        if self.kind == "oracle":
            return EXPERTS.index(ORACLE_EXPERT[oracle])
        if self.kind == "uniform":
            chosen = int(self.rng.integers(len(EXPERTS)))
        elif self.kind == "recency":
            chosen = 0 if self.last_expert is None else self.last_expert
        elif self.kind == "direct_memory":
            key = _family_key(obs)
            matches = [e for k, e in reversed(self.memory) if k == key]
            chosen = matches[0] if matches else int(self.rng.integers(len(EXPERTS)))
        else:
            key = _family_key(obs)
            row = self._row(key)
            scores = self.phi[row] + 0.35 * self.global_phi
            # Small exploration keeps feedback available for previously bad choices.
            if self.rng.random() < 0.08:
                chosen = int(self.rng.integers(len(EXPERTS)))
            else:
                best = np.flatnonzero(scores == scores.max())
                chosen = int(self.rng.choice(best))
        if self.last_expert is not None and chosen != self.last_expert:
            self.churn += 1
        self.last_expert = chosen
        return chosen

    def update(self, obs, expert, success):
        if self.kind == "oracle":
            return
        reward = 1.0 if success else -1.0
        if self.kind == "recency":
            if success:
                self.last_expert = expert
            return
        if self.kind == "direct_memory":
            if success:
                self.memory.append((_family_key(obs), expert))
            self.updates += 1
            return
        if self.kind == "phi":
            row = self._row(_family_key(obs))
            target = 1.0 if success else -1.0
            self.phi[row, expert] = (1.0 - self.phi_lr) * self.phi[row, expert] + self.phi_lr * target
            self.global_phi[expert] = 0.98 * self.global_phi[expert] + 0.02 * reward
            self.updates += 1


def _run_episode(env_name, strategy, seed, max_steps=None):
    env = gym.make(ENV_IDS[env_name])
    obs, _ = env.reset(seed=int(seed))
    # The mission remains in the Gym observation for compatibility, but is not
    # passed to the selector or the capability policies.
    steps = int(max_steps or env.unwrapped.max_steps)
    terminated = truncated = False
    for step in range(steps):
        action = _expert_action(env, strategy)
        obs, reward, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    success = bool(terminated and float(reward) > 0.0)
    env.close()
    return success, step + 1, obs


def _schedule(seed):
    rng = np.random.default_rng(seed)
    phases = [
        ("calibration", ["nav", "door", "lava", "corridor"]),
        ("switch", ["lava", "corridor", "door", "nav"]),
        ("revisit", ["door", "lava", "nav", "corridor"]),
    ]
    stream = []
    for phase, order in phases:
        # Blocks create measurable within-block adaptation without exposing a label.
        for block in range(5):
            local = order.copy()
            if block % 2:
                rng.shuffle(local)
            stream.extend((phase, family) for family in local)
    return stream


def _run_method(method, seed):
    selector = Selector(method, seed + 1000)
    records = []
    for episode, (phase, family) in enumerate(_schedule(seed)):
        env_seed = 10000 + seed * 1000 + episode
        # Selector receives only the first partial image and direction.
        probe = gym.make(ENV_IDS[family])
        obs, _ = probe.reset(seed=env_seed)
        expert = selector.choose(obs, family, oracle=family)
        probe.close()
        success, steps, final_obs = _run_episode(family, EXPERTS[expert], env_seed)
        selector.update(obs, expert, success)
        records.append({
            "episode": episode,
            "phase": phase,
            "family": family,
            "expert": EXPERTS[expert],
            "success": int(success),
            "steps": steps,
        })
    return records, selector


def _first_success(records, phase):
    vals = [r for r in records if r["phase"] == phase]
    for idx, item in enumerate(vals):
        if item["success"]:
            return idx + 1
    return len(vals) + 1


def _summarize(records, selector):
    phases = {}
    for phase in sorted(set(r["phase"] for r in records), key=("calibration", "switch", "revisit").index):
        rows = [r for r in records if r["phase"] == phase]
        phases[phase] = {
            "success_rate": float(np.mean([r["success"] for r in rows])),
            "mean_steps": float(np.mean([r["steps"] for r in rows])),
            "episodes_to_first_success": _first_success(records, phase),
        }
    by_family = {}
    for family in ENV_IDS:
        rows = [r for r in records if r["family"] == family]
        by_family[family] = float(np.mean([r["success"] for r in rows]))
    return {
        "overall_success_rate": float(np.mean([r["success"] for r in records])),
        "phase_metrics": phases,
        "family_success_rate": by_family,
        "strategy_churn": int(selector.churn),
        "online_updates": int(selector.updates),
        "phi_contexts": int(len(selector.contexts)),
        # Direct memory stores the 18-value coarse context key plus the
        # selected expert index for each entry; do not count it as two floats.
        "dynamic_state_floats": int(selector.phi.size + selector.global_phi.size) if selector.kind == "phi" else int(selector.memory_capacity * 19 if selector.kind == "direct_memory" else 1),
    }


def _plot(all_results):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    names = list(DISPLAY)
    x = np.arange(len(names))
    overall = [np.mean([r["overall_success_rate"] for r in all_results[n]]) for n in names]
    churn = [np.mean([r["strategy_churn"] for r in all_results[n]]) for n in names]
    revisit = [np.mean([r["phase_metrics"]["revisit"]["success_rate"] for r in all_results[n]]) for n in names]
    axes[0].bar(x, overall, color="#3b82f6")
    axes[0].set_title("Overall success")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("rate")
    axes[1].bar(x, revisit, color="#10b981")
    axes[1].set_title("Cycle revisit success")
    axes[1].set_ylim(0, 1.05)
    axes[2].bar(x, churn, color="#f59e0b")
    axes[2].set_title("Strategy churn")
    for ax in axes:
        ax.set_xticks(x, [DISPLAY[n].replace(" ", "\n") for n in names], rotation=0)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("EXP56 MiniGrid: unlabeled strategy organization")
    fig.tight_layout()
    fig.savefig(RESULT_PNG, dpi=170)
    plt.close(fig)


def run_exp56(seeds=SEEDS):
    os.makedirs("results", exist_ok=True)
    all_results = {method: [] for method in DISPLAY}
    traces = {method: [] for method in DISPLAY}
    for method in DISPLAY:
        for seed in seeds:
            records, selector = _run_method(method, seed)
            all_results[method].append(_summarize(records, selector))
            if seed == seeds[0]:
                traces[method] = records
    _plot(all_results)
    summary = {}
    for method, rows in all_results.items():
        summary[method] = {
            "mean": {key: float(np.mean([row[key] for row in rows])) for key in ("overall_success_rate", "strategy_churn", "dynamic_state_floats")},
            "std": {key: float(np.std([row[key] for row in rows], ddof=1)) for key in ("overall_success_rate", "strategy_churn")},
            "phase_mean": {
                phase: float(np.mean([row["phase_metrics"][phase]["success_rate"] for row in rows]))
                for phase in ("calibration", "switch", "revisit")
            },
            "family_mean": {
                family: float(np.mean([row["family_success_rate"][family] for row in rows]))
                for family in ENV_IDS
            },
        }
    output = {
        "experiment": "EXP56",
        "title": "Official MiniGrid non-stationary strategy organization",
        "protocol": {
            "envs": ENV_IDS,
            "seeds": list(seeds),
            "experts": list(EXPERTS),
            "mission_visible_to_selector": False,
            "task_label_visible_to_selector": False,
            "observation": "official 7x7x3 partial image plus direction; mission omitted",
            "episodes_per_seed": len(_schedule(seeds[0])),
            "phases": ["calibration", "switch", "revisit"],
            "limitation": "frozen privileged capability policies isolate organization from representation learning",
        },
        "summary": summary,
        "trace_seed_0": traces,
    }
    with open(RESULT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    return output


if __name__ == "__main__":
    result = run_exp56()
    for method, values in result["summary"].items():
        print(method, values["mean"]["overall_success_rate"], values["phase_mean"])
