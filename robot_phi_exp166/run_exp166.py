#!/usr/bin/env python3
"""EXP166: structured skill reuse and damage repair in MetaWorld reach-v1."""
from __future__ import annotations

import argparse
import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from continualworld.envs import get_single_env


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
ROLES = ("x+", "x-", "y+", "y-", "z+", "z-")
ROLE_COUNT = len(ROLES)
MODULE_COUNT = 12
STATE_BUDGET = ROLE_COUNT * MODULE_COUNT * 2
METHODS = (
    "phi_credit",
    "phi_shuffled",
    "w_only_dispatch",
    "task_kv",
    "recent_task",
    "oracle",
)
COLORS = {
    "phi_credit": "#287271",
    "phi_shuffled": "#C8553D",
    "w_only_dispatch": "#6C757D",
    "task_kv": "#7A5195",
    "recent_task": "#D4A72C",
    "oracle": "#1D3557",
}
LABELS = {
    "phi_credit": "Phi-credit",
    "phi_shuffled": "Phi-shuffled",
    "w_only_dispatch": "W-only dispatcher",
    "task_kv": "Task-KV",
    "recent_task": "Recent-task state",
    "oracle": "Oracle",
}


def role_for_axis(axis: int, value: float) -> int:
    return 2 * axis + (1 if value < 0 else 0)


def task_key(delta: np.ndarray) -> np.ndarray:
    key = np.zeros(ROLE_COUNT, dtype=np.float32)
    for axis, value in enumerate(delta):
        if abs(value) > 1e-8:
            key[role_for_axis(axis, value)] = 1.0
    return key


def key_string(key: np.ndarray) -> str:
    return "".join(str(int(value)) for value in key)


def make_tasks(seed: int) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    signed_displacements = ((0.140, -0.140), (0.200, -0.110), (0.120, -0.120))
    singles = []
    for axis in range(3):
        for direction in range(2):
            delta = np.zeros(3, dtype=np.float32)
            delta[axis] = signed_displacements[axis][direction]
            singles.append(delta)

    pairs = []
    for first, second in ((0, 1), (0, 2), (1, 2)):
        for first_direction in range(2):
            for second_direction in range(2):
                delta = np.zeros(3, dtype=np.float32)
                delta[first] = signed_displacements[first][first_direction]
                delta[second] = signed_displacements[second][second_direction]
                pairs.append(delta)

    rng = np.random.default_rng(166_100 + seed)
    seen_pairs, heldout_pairs = [], []
    for block in range(3):
        indices = np.arange(4 * block, 4 * block + 4)
        rng.shuffle(indices)
        seen_pairs.extend([pairs[index] for index in indices[:2]])
        heldout_pairs.extend([pairs[index] for index in indices[2:]])

    triples = []
    for x_direction in range(2):
        for y_direction in range(2):
            for z_direction in range(2):
                triples.append(np.array([
                    signed_displacements[0][x_direction],
                    signed_displacements[1][y_direction],
                    signed_displacements[2][z_direction],
                ], dtype=np.float32))
    return singles + seen_pairs, heldout_pairs, triples


@dataclass
class SkillLibrary:
    vectors: np.ndarray
    primary_by_role: np.ndarray
    backup_by_role: np.ndarray

    @classmethod
    def create(cls, seed: int) -> "SkillLibrary":
        vectors = []
        semantic = []
        for role in range(ROLE_COUNT):
            axis = role // 2
            sign = 1.0 if role % 2 == 0 else -1.0
            primary = np.zeros(3, dtype=np.float32)
            primary[axis] = sign
            backup = np.zeros(3, dtype=np.float32)
            backup[axis] = 0.76 * sign
            other = (axis + 1) % 3
            backup[other] = (0.035 if role % 2 == 0 else -0.035)
            vectors.extend((primary, backup))
            semantic.extend(((role, "primary"), (role, "backup")))
        rng = np.random.default_rng(166_200 + seed)
        order = rng.permutation(MODULE_COUNT)
        shuffled = np.asarray(vectors, dtype=np.float32)[order]
        primary = np.zeros(ROLE_COUNT, dtype=np.int64)
        backup = np.zeros(ROLE_COUNT, dtype=np.int64)
        for module, source_index in enumerate(order):
            role, kind = semantic[int(source_index)]
            if kind == "primary":
                primary[role] = module
            else:
                backup[role] = module
        return cls(shuffled, primary, backup)

    def output(self, module: int, damaged_module: Optional[int]) -> np.ndarray:
        if damaged_module is not None and int(module) == int(damaged_module):
            return np.zeros(3, dtype=np.float32)
        return self.vectors[int(module)].copy()


class PhiController:
    def __init__(
        self,
        scores: Optional[np.ndarray] = None,
        counts: Optional[np.ndarray] = None,
        seed: int = 0,
        shuffled_credit: bool = False,
        frozen: bool = False,
        recent: bool = False,
    ):
        self.rng = np.random.default_rng(166_300 + seed)
        self.scores = np.zeros((ROLE_COUNT, MODULE_COUNT), dtype=np.float64) if scores is None else scores.copy()
        self.counts = np.zeros((ROLE_COUNT, MODULE_COUNT), dtype=np.float64) if counts is None else counts.copy()
        self.foundation_scores = self.scores.copy()
        self.foundation_counts = self.counts.copy()
        self.shuffled_credit = shuffled_credit
        self.frozen = frozen
        self.recent = recent
        self.last_key = None
        self.route_log = []

    def seal_foundation(self) -> None:
        self.foundation_scores = self.scores.copy()
        self.foundation_counts = self.counts.copy()

    def begin_task(self, key: np.ndarray) -> None:
        current = key_string(key)
        if self.recent and self.last_key is not None and current != self.last_key:
            self.scores = self.foundation_scores.copy()
            self.counts = self.foundation_counts.copy()
        self.last_key = current

    def select(self, role: int, learn: bool) -> int:
        del learn
        best = np.flatnonzero(self.scores[role] >= np.max(self.scores[role]) - 1e-12)
        module = int(self.rng.choice(best))
        self.route_log.append((role, module))
        return module

    def update(self, role: int, module: int, quality: float, learn: bool) -> None:
        if not learn or self.frozen:
            return
        write_role = (role + 1) % ROLE_COUNT if self.shuffled_credit else role
        foundation_owner = int(np.argmax(self.foundation_scores[write_role]))
        # Runtime credit is event-triggered. A failed incumbent loses
        # responsibility; a successful alternative is consolidated. Correct
        # routes are otherwise left unchanged so physical lag near the goal
        # cannot erode a stable organization.
        if int(module) == foundation_owner and quality < 0.05:
            self.scores[write_role, module] = max(
                -1.5, self.scores[write_role, module] - 0.35
            )
            self.counts[write_role, module] += 1.0
        elif int(module) != foundation_owner and quality > 0.30:
            target = min(1.5, max(float(quality), self.foundation_scores[write_role, module] + 0.10))
            self.scores[write_role, module] = max(self.scores[write_role, module], target)
            self.counts[write_role, module] += 1.0

    @property
    def state_floats(self) -> int:
        return int(self.scores.size + self.counts.size)


class OracleController:
    def __init__(self, library: SkillLibrary):
        self.library = library
        self.route_log = []

    def begin_task(self, key: np.ndarray) -> None:
        del key

    def select(self, role: int, learn: bool, damaged_module: Optional[int] = None) -> int:
        del learn
        primary = int(self.library.primary_by_role[role])
        module = int(self.library.backup_by_role[role]) if primary == damaged_module else primary
        self.route_log.append((role, module))
        return module

    def update(self, role: int, module: int, quality: float, learn: bool) -> None:
        del role, module, quality, learn

    @property
    def state_floats(self) -> int:
        return 0


class TaskKVController:
    def __init__(self, capacity: int, seed: int):
        self.capacity = int(capacity)
        self.rng = np.random.default_rng(166_400 + seed)
        self.entries: Dict[str, dict] = {}
        self.clock = 0
        self.current_entry = None
        self.route_log = []

    def add_seed_entry(self, key: np.ndarray, route: np.ndarray, backup_route: np.ndarray) -> None:
        name = key_string(key)
        if name not in self.entries and len(self.entries) >= self.capacity:
            oldest = min(self.entries, key=lambda item: self.entries[item]["age"])
            del self.entries[oldest]
        self.entries[name] = {
            "key": key.copy(),
            "route": route.copy(),
            "backup_route": backup_route.copy(),
            "confidence": np.where(route >= 0, 1.0, 0.0).astype(np.float64),
            "tried": [set([int(value)]) if value >= 0 else set() for value in route],
            "age": self.clock,
        }
        self.clock += 1

    def _nearest(self, key: np.ndarray) -> Optional[dict]:
        if not self.entries:
            return None
        values = list(self.entries.values())
        distances = np.array([np.mean(np.abs(item["key"] - key)) for item in values])
        candidates = np.flatnonzero(distances <= np.min(distances) + 1e-12)
        return values[int(self.rng.choice(candidates))]

    def begin_task(self, key: np.ndarray, learn: bool = False) -> None:
        name = key_string(key)
        if name in self.entries:
            self.current_entry = self.entries[name]
            self.current_entry["age"] = self.clock
            self.clock += 1
            return
        nearest = self._nearest(key)
        if nearest is None:
            route = np.full(ROLE_COUNT, -1, dtype=np.int64)
            backup_route = np.full(ROLE_COUNT, -1, dtype=np.int64)
            confidence = np.zeros(ROLE_COUNT, dtype=np.float64)
        else:
            route = nearest["route"].copy()
            backup_route = nearest["backup_route"].copy()
            confidence = nearest["confidence"].copy()
        transient = {
            "key": key.copy(), "route": route, "backup_route": backup_route,
            "confidence": confidence,
            "tried": [set([int(value)]) if value >= 0 else set() for value in route],
            "age": self.clock,
        }
        if learn:
            if len(self.entries) >= self.capacity:
                oldest = min(self.entries, key=lambda item: self.entries[item]["age"])
                del self.entries[oldest]
            self.entries[name] = transient
            self.clock += 1
        self.current_entry = transient

    def select(self, role: int, learn: bool) -> int:
        route = self.current_entry["route"]
        module = int(route[role])
        if module < 0:
            module = int(self.rng.integers(MODULE_COUNT))
            if learn:
                route[role] = module
                self.current_entry["tried"][role].add(module)
        self.route_log.append((role, module))
        return module

    def update(self, role: int, module: int, quality: float, learn: bool) -> None:
        if not learn:
            return
        entry = self.current_entry
        entry["confidence"][role] = 0.70 * entry["confidence"][role] + 0.30 * quality
        if quality < 0.18:
            tried = entry["tried"][role]
            tried.add(int(module))
            backup = int(entry["backup_route"][role])
            if backup >= 0 and backup not in tried:
                entry["route"][role] = backup
                tried.add(backup)
            elif module != backup:
                available = np.array([index for index in range(MODULE_COUNT) if index not in tried])
                if len(available) == 0:
                    tried.clear()
                    available = np.arange(MODULE_COUNT)
                entry["route"][role] = int(self.rng.choice(available))
        else:
            entry["route"][role] = int(module)

    @property
    def state_floats(self) -> int:
        return int(len(self.entries) * 24)


def prepare_episode(env, delta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    observation = np.asarray(env.reset(), dtype=np.float32)
    start = observation[:3].copy()
    target = start + np.asarray(delta, dtype=np.float32)
    raw = env.unwrapped
    raw._target_pos = target.copy()
    raw.maxReachDist = float(np.linalg.norm(target - start))
    raw._set_pos_site("goal_reach", target.copy())
    observation = np.asarray(raw._get_obs(), dtype=np.float32)
    return observation, target


def run_episode(
    env,
    controller,
    library: SkillLibrary,
    delta: np.ndarray,
    learn: bool,
    damaged_module: Optional[int],
    max_steps: int = 45,
    force_role_module: Optional[Tuple[int, int]] = None,
) -> dict:
    observation, target = prepare_episode(env, delta)
    key = task_key(delta)
    if isinstance(controller, TaskKVController):
        controller.begin_task(key, learn=learn)
    else:
        controller.begin_task(key)
    start_distance = float(np.linalg.norm(target - observation[:3]))
    selected_by_role = {role: [] for role in range(ROLE_COUNT)}
    qualities_by_role = {role: [] for role in range(ROLE_COUNT)}
    success = False
    steps = 0
    for step in range(max_steps):
        hand = observation[:3]
        error = target - hand
        distance = float(np.linalg.norm(error))
        if distance <= 0.05:
            success = True
            steps = step
            break
        selected = []
        action_xyz = np.zeros(3, dtype=np.float32)
        for axis, value in enumerate(error):
            if abs(value) <= 0.012:
                continue
            role = role_for_axis(axis, float(value))
            if force_role_module is not None and role == force_role_module[0]:
                module = int(force_role_module[1])
            elif isinstance(controller, OracleController):
                module = controller.select(role, learn, damaged_module=damaged_module)
            else:
                module = controller.select(role, learn)
            action_xyz += library.output(module, damaged_module)
            selected.append((role, module, axis, float(value)))
            selected_by_role[role].append(module)
        action = np.zeros(4, dtype=np.float32)
        action[:3] = np.clip(action_xyz, -1.0, 1.0)
        next_observation, _, _, info = env.step(action)
        next_observation = np.asarray(next_observation, dtype=np.float32)
        next_error = target - next_observation[:3]
        for role, module, axis, previous_error in selected:
            progress = abs(previous_error) - abs(float(next_error[axis]))
            quality = float(np.clip(progress / 0.008, -1.0, 1.0))
            controller.update(role, module, quality, learn)
            qualities_by_role[role].append(quality)
        observation = next_observation
        steps = step + 1
        if float(info.get("success", 0.0)) > 0.5 or np.linalg.norm(target - observation[:3]) <= 0.05:
            success = True
            break
    final_distance = float(np.linalg.norm(target - observation[:3]))
    return {
        "success": float(success),
        "steps": int(steps),
        "start_distance": start_distance,
        "final_distance": final_distance,
        "progress_fraction": float(np.clip(1.0 - final_distance / max(start_distance, 1e-8), -1.0, 1.0)),
        "selected_by_role": {str(role): values for role, values in selected_by_role.items()},
        "quality_by_role": {
            str(role): float(np.mean(values)) if values else 0.0
            for role, values in qualities_by_role.items()
        },
    }


def mean_metric(records: Sequence[dict], name: str) -> float:
    return float(np.mean([record[name] for record in records])) if records else float("nan")


def foundation_state(env, library: SkillLibrary, tasks: Sequence[np.ndarray], seed: int) -> PhiController:
    del tasks
    controller = PhiController(seed=seed)
    magnitudes = np.array([0.080, 0.095, 0.075], dtype=np.float32)
    probe_steps = 12
    for role in range(ROLE_COUNT):
        axis = role // 2
        sign = 1.0 if role % 2 == 0 else -1.0
        delta = np.zeros(3, dtype=np.float32)
        delta[axis] = sign * magnitudes[axis]
        baseline_observation, _ = prepare_episode(env, delta)
        baseline_start = baseline_observation[:3].copy()
        zero_action = np.zeros(4, dtype=np.float32)
        for _ in range(probe_steps):
            baseline_observation, _, _, _ = env.step(zero_action)
            baseline_observation = np.asarray(baseline_observation, dtype=np.float32)
        baseline_displacement = baseline_observation[:3] - baseline_start
        for module in range(MODULE_COUNT):
            observation, _ = prepare_episode(env, delta)
            start = observation[:3].copy()
            action = np.zeros(4, dtype=np.float32)
            action[:3] = np.clip(library.output(module, None), -1.0, 1.0)
            for _ in range(probe_steps):
                observation, _, _, _ = env.step(action)
                observation = np.asarray(observation, dtype=np.float32)
            displacement = observation[:3] - start - baseline_displacement
            desired = sign * float(displacement[axis])
            orthogonal = float(np.linalg.norm(np.delete(displacement, axis)))
            reference = 0.010 * probe_steps
            quality = np.clip(desired / reference - 0.35 * orthogonal / reference, -1.5, 1.5)
            controller.scores[role, module] = float(quality)
            controller.counts[role, module] = float(probe_steps)
    controller.seal_foundation()
    return controller


def route_from_scores(scores: np.ndarray, key: np.ndarray) -> np.ndarray:
    del key
    route = np.full(ROLE_COUNT, -1, dtype=np.int64)
    for role in range(ROLE_COUNT):
        route[role] = int(np.argmax(scores[role]))
    return route


def backup_route_from_scores(scores: np.ndarray, key: np.ndarray) -> np.ndarray:
    del key
    route = np.full(ROLE_COUNT, -1, dtype=np.int64)
    for role in range(ROLE_COUNT):
        order = np.argsort(scores[role])
        route[role] = int(order[-2])
    return route


def evaluate_tasks(env, controller, library, tasks, damaged_module, force=None) -> List[dict]:
    return [
        run_episode(env, controller, library, delta, False, damaged_module, force_role_module=force)
        for delta in tasks
    ]


def run_seed(seed: int, damage_episodes: int) -> dict:
    np.random.seed(seed)
    env = get_single_env("reach-v1", one_hot_idx=0, one_hot_len=1)
    if hasattr(env, "seed"):
        env.seed(seed)
    library = SkillLibrary.create(seed)
    seen, heldout_pairs, triples = make_tasks(seed)
    foundation = foundation_state(env, library, seen, seed)
    base_scores = foundation.scores.copy()
    base_counts = foundation.counts.copy()

    controllers = {
        "phi_credit": PhiController(base_scores, base_counts, seed=seed),
        "phi_shuffled": PhiController(base_scores, base_counts, seed=seed + 11, shuffled_credit=True),
        "w_only_dispatch": PhiController(base_scores, base_counts, seed=seed + 22, frozen=True),
        "recent_task": PhiController(base_scores, base_counts, seed=seed + 33, recent=True),
        "oracle": OracleController(library),
    }
    for name in ("phi_credit", "phi_shuffled", "w_only_dispatch", "recent_task"):
        controllers[name].seal_foundation()

    kv = TaskKVController(capacity=STATE_BUDGET // 24, seed=seed)
    for delta in seen:
        key = task_key(delta)
        kv.add_seed_entry(
            key,
            route_from_scores(base_scores, key),
            backup_route_from_scores(base_scores, key),
        )
    controllers["task_kv"] = kv

    unseen_tasks = heldout_pairs + triples
    pre_damage = {}
    for method in METHODS:
        records = evaluate_tasks(env, controllers[method], library, unseen_tasks, None)
        pre_damage[method] = {
            "success": mean_metric(records, "success"),
            "final_distance": mean_metric(records, "final_distance"),
            "progress_fraction": mean_metric(records, "progress_fraction"),
        }

    damaged_role = 0
    # Damage the module that actually owns x+ after formation, rather than an
    # implementation-labelled "primary" that the learned dispatcher may not use.
    damaged_module = int(np.argmax(base_scores[damaged_role]))
    probe = next(delta for delta in seen if np.count_nonzero(delta) == 2 and delta[0] > 0)
    immediate = {}
    damage_curves = {}
    role0_before = {}
    role0_after = {}
    for method in METHODS:
        controller = controllers[method]
        immediate_record = run_episode(env, controller, library, probe, False, damaged_module)
        immediate[method] = immediate_record
        if hasattr(controller, "scores"):
            role0_before[method] = controller.scores[damaged_role].tolist()
        curve = []
        for episode in range(damage_episodes):
            record = run_episode(env, controller, library, probe, True, damaged_module)
            curve.append({"episode": episode, **record})
        damage_curves[method] = curve
        if hasattr(controller, "scores"):
            role0_after[method] = controller.scores[damaged_role].tolist()

    cross_shared = [delta for delta in unseen_tasks if delta[0] > 0 and key_string(task_key(delta)) != key_string(task_key(probe))]
    cross_control = [delta for delta in unseen_tasks if delta[0] < 0]
    post_damage = {}
    for method in METHODS:
        shared = evaluate_tasks(env, controllers[method], library, cross_shared, damaged_module)
        control = evaluate_tasks(env, controllers[method], library, cross_control, damaged_module)
        post_damage[method] = {
            "shared_success": mean_metric(shared, "success"),
            "shared_final_distance": mean_metric(shared, "final_distance"),
            "control_success": mean_metric(control, "success"),
            "control_final_distance": mean_metric(control, "final_distance"),
            "specificity": mean_metric(shared, "success") - mean_metric(control, "success"),
        }

    phi_clone = copy.deepcopy(controllers["phi_credit"])
    normal = evaluate_tasks(env, phi_clone, library, cross_shared, damaged_module)
    forced = evaluate_tasks(
        env, phi_clone, library, cross_shared, damaged_module,
        force=(damaged_role, damaged_module),
    )
    intervention = {
        "normal_success": mean_metric(normal, "success"),
        "forced_damaged_route_success": mean_metric(forced, "success"),
        "causal_drop": mean_metric(normal, "success") - mean_metric(forced, "success"),
    }

    foundation_best = np.argmax(base_scores, axis=1)
    foundation_role_accuracy = float(np.mean([
        foundation_best[role] in {library.primary_by_role[role], library.backup_by_role[role]}
        for role in range(ROLE_COUNT)
    ]))
    state = {
        method: {
            "dynamic_state_floats": (
                0 if method == "w_only_dispatch" else int(controllers[method].state_floats)
            ),
            "fixed_dispatch_floats": 72 if method == "w_only_dispatch" else 0,
        }
        for method in METHODS
    }
    try:
        env.close()
    except Exception:
        pass
    return {
        "seed": seed,
        "foundation_role_accuracy": foundation_role_accuracy,
        "foundation_best_module_by_role": foundation_best.tolist(),
        "foundation_scores": base_scores.tolist(),
        "module_primary_by_role": library.primary_by_role.tolist(),
        "module_backup_by_role": library.backup_by_role.tolist(),
        "damaged_role": damaged_role,
        "damaged_module": damaged_module,
        "probe_key": key_string(task_key(probe)),
        "pre_damage": pre_damage,
        "damage_immediate": immediate,
        "damage_curves": damage_curves,
        "post_damage": post_damage,
        "intervention": intervention,
        "role0_scores_before": role0_before,
        "role0_scores_after": role0_after,
        "state": state,
        "task_counts": {
            "seen": len(seen), "heldout_pairs": len(heldout_pairs),
            "unseen_triples": len(triples), "cross_shared": len(cross_shared),
            "cross_control": len(cross_control),
        },
    }


def mean_ci(values: Sequence[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(array))
    if len(array) < 2:
        return {"mean": mean, "low": mean, "high": mean, "std": 0.0}
    std = float(np.std(array, ddof=1))
    critical = {
        1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
        6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
        16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
        24: 2.064, 29: 2.045,
    }
    degrees = len(array) - 1
    if degrees in critical:
        t_value = critical[degrees]
    elif degrees < 24:
        lower = max(key for key in critical if key < degrees)
        upper = min(key for key in critical if key > degrees)
        fraction = (degrees - lower) / (upper - lower)
        t_value = critical[lower] + fraction * (critical[upper] - critical[lower])
    else:
        t_value = 1.96 if degrees >= 60 else 2.00
    half = float(t_value * std / math.sqrt(len(array)))
    return {"mean": mean, "low": mean - half, "high": mean + half, "std": std}


def summarize(seeds: Sequence[dict], damage_episodes: int) -> dict:
    summary = {
        "foundation_role_accuracy": mean_ci([row["foundation_role_accuracy"] for row in seeds]),
        "pre_damage": {}, "damage_immediate": {}, "damage_curves": {},
        "post_damage": {}, "intervention": {}, "paired": {},
    }
    for method in METHODS:
        summary["pre_damage"][method] = {
            metric: mean_ci([row["pre_damage"][method][metric] for row in seeds])
            for metric in ("success", "final_distance", "progress_fraction")
        }
        summary["damage_immediate"][method] = {
            metric: mean_ci([row["damage_immediate"][method][metric] for row in seeds])
            for metric in ("success", "final_distance", "progress_fraction")
        }
        summary["damage_curves"][method] = [
            {
                "episode": episode,
                "success": mean_ci([row["damage_curves"][method][episode]["success"] for row in seeds]),
                "final_distance": mean_ci([row["damage_curves"][method][episode]["final_distance"] for row in seeds]),
            }
            for episode in range(damage_episodes)
        ]
        summary["post_damage"][method] = {
            metric: mean_ci([row["post_damage"][method][metric] for row in seeds])
            for metric in ("shared_success", "shared_final_distance", "control_success", "control_final_distance", "specificity")
        }
    summary["intervention"] = {
        metric: mean_ci([row["intervention"][metric] for row in seeds])
        for metric in ("normal_success", "forced_damaged_route_success", "causal_drop")
    }
    for baseline in ("phi_shuffled", "w_only_dispatch", "task_kv", "recent_task"):
        summary["paired"][baseline] = {
            "unseen_success_difference": mean_ci([
                row["pre_damage"]["phi_credit"]["success"] - row["pre_damage"][baseline]["success"]
                for row in seeds
            ]),
            "cross_shared_difference": mean_ci([
                row["post_damage"]["phi_credit"]["shared_success"] - row["post_damage"][baseline]["shared_success"]
                for row in seeds
            ]),
        }
    return summary


def plot(payload: dict, output: Path) -> None:
    summary = payload["summary"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    positions = np.arange(len(METHODS))
    pre = [summary["pre_damage"][method]["success"]["mean"] for method in METHODS]
    pre_low = [pre[index] - summary["pre_damage"][method]["success"]["low"] for index, method in enumerate(METHODS)]
    pre_high = [summary["pre_damage"][method]["success"]["high"] - pre[index] for index, method in enumerate(METHODS)]
    axes[0, 0].bar(positions, pre, color=[COLORS[m] for m in METHODS], yerr=[pre_low, pre_high], capsize=3)
    axes[0, 0].set_xticks(positions, [LABELS[m] for m in METHODS], rotation=28, ha="right")
    axes[0, 0].set_ylim(0, 1.05)
    axes[0, 0].set_ylabel("Success rate")
    axes[0, 0].set_title("Held-out skill compositions before damage")

    recovery = np.array([
        [item["success"]["mean"] for item in summary["damage_curves"][method]]
        for method in METHODS
    ])
    image = axes[0, 1].imshow(recovery, vmin=0.0, vmax=1.0, cmap="viridis", aspect="auto")
    axes[0, 1].set_xticks(np.arange(recovery.shape[1]), np.arange(1, recovery.shape[1] + 1))
    axes[0, 1].set_yticks(np.arange(len(METHODS)), [LABELS[m] for m in METHODS])
    axes[0, 1].set_xlabel("Feedback episode after local damage")
    axes[0, 1].set_ylabel("Method")
    axes[0, 1].set_title("Responsibility reallocation after damage")
    fig.colorbar(image, ax=axes[0, 1], fraction=0.046, pad=0.04, label="Success rate")

    width = 0.36
    shared = [summary["post_damage"][method]["shared_success"]["mean"] for method in METHODS]
    control = [summary["post_damage"][method]["control_success"]["mean"] for method in METHODS]
    axes[1, 0].bar(positions - width / 2, shared, width, color=[COLORS[m] for m in METHODS], label="Shares damaged role")
    axes[1, 0].bar(positions + width / 2, control, width, color="none", edgecolor=[COLORS[m] for m in METHODS], hatch="//", label="Unrelated control")
    axes[1, 0].set_xticks(positions, [LABELS[m] for m in METHODS], rotation=28, ha="right")
    axes[1, 0].set_ylim(0, 1.05)
    axes[1, 0].set_ylabel("Zero-feedback transfer success")
    axes[1, 0].set_title("Cross-task repair transfer and specificity")
    axes[1, 0].legend(fontsize=8)

    state = [payload["seeds"][0]["state"][method]["dynamic_state_floats"] for method in METHODS]
    axes[1, 1].bar(positions, state, color=[COLORS[m] for m in METHODS])
    axes[1, 1].axhline(STATE_BUDGET, color="#333333", linestyle="--", linewidth=1, label="Budget")
    axes[1, 1].set_xticks(positions, [LABELS[m] for m in METHODS], rotation=28, ha="right")
    axes[1, 1].set_ylabel("Persistent dynamic-state floats")
    axes[1, 1].set_title("State budget after experiment")
    axes[1, 1].legend(fontsize=8)

    fig.suptitle("EXP166: robot skill composition, damage repair, and structural reuse", fontsize=15)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--damage-episodes", type=int, default=8)
    parser.add_argument("--output", default=str(RESULTS / "EXP166_robot_skill_reuse.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = tuple(int(value) for value in args.seeds.split(",") if value.strip())
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in seeds:
        rows.append(run_seed(seed, args.damage_episodes))
        print("seed", seed, "complete", flush=True)
    payload = {
        "experiment": "EXP166_robot_skill_reuse",
        "environment": "MetaWorld reach-v1 in ContinualWorld MuJoCo 2.0 container",
        "config": {
            "seeds": list(seeds), "damage_episodes": args.damage_episodes,
            "roles": list(ROLES), "modules": MODULE_COUNT,
            "state_budget_floats": STATE_BUDGET,
        },
        "summary": summarize(rows, args.damage_episodes),
        "seeds": rows,
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    figure = output.with_suffix(".png")
    plot(payload, figure)
    print("saved", output, flush=True)
    print("saved", figure, flush=True)


if __name__ == "__main__":
    main()
