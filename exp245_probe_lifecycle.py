"""EXP245: probes as budgeted units with a lifecycle.

The symbolic discovery engine and macro organization are fixed. Probe
policies differ in whether diagnostic input coordinates can accumulate credit,
remain active, retire to an archive, reactivate after a regime change, and be
eliminated. All policies share active slots, query budget, candidate budget,
and the exact polynomial verifier.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev

from exp244_complexity_scaling import (
    DOMAIN,
    ONE,
    PRIMITIVES,
    W,
    Expr,
    Macro,
    Task,
    add_expr,
    benchmark_tasks,
    candidate_score,
    degree,
    eval_poly,
    foundation_macros,
    mul_expr,
    retrieve_macros,
)


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
POLICIES = (
    "random_reset",
    "greedy_reset",
    "persistent_fifo",
    "shuffled_lifecycle",
    "credit_lifecycle",
    "oracle_lifecycle",
)


@dataclass
class ProbeUnit:
    point: tuple[int, int, int, int]
    credit: float
    birth: int
    last_used: int
    uses: int = 0
    active: bool = True
    reactivations: int = 0
    marginal_sum: float = 0.0


def disagreement(candidates: list[Expr], point: tuple[int, int, int, int]) -> float:
    top = candidates[: min(32, len(candidates))]
    if len(top) < 2:
        return 0.0
    values = [eval_poly(candidate.poly, point) for candidate in top]
    center = mean(values)
    variance = mean((value - center) ** 2 for value in values)
    return math.log1p(variance) + len(set(values)) / len(values)


def oracle_elimination(
    candidates: list[Expr], target: Expr, point: tuple[int, int, int, int]
) -> float:
    top = candidates[: min(48, len(candidates))]
    if not top:
        return 0.0
    target_value = eval_poly(target.poly, point)
    return mean(float(eval_poly(candidate.poly, point) != target_value) for candidate in top)


class ProbeManager:
    def __init__(self, policy: str, slots: int, query_budget: int):
        self.policy = policy
        self.slots = slots
        self.query_budget = query_budget
        self.active: dict[tuple[int, int, int, int], ProbeUnit] = {}
        self.archive: dict[tuple[int, int, int, int], ProbeUnit] = {}
        self.clock = 0
        self.births = 0
        self.retirements = 0
        self.reactivations = 0
        self.eliminations = 0
        self.task_queries = 0
        self.task_generated = 0

    @property
    def persistent(self) -> bool:
        return self.policy not in ("random_reset", "greedy_reset")

    def _birth(self, point: tuple[int, int, int, int]) -> ProbeUnit:
        unit = ProbeUnit(point, credit=0.25, birth=self.clock, last_used=self.clock)
        self.births += 1
        self.task_generated += 1
        return unit

    def start_task(
        self,
        shared_initial: list[tuple[int, int, int, int]],
    ) -> list[tuple[int, int, int, int]]:
        self.clock += 1
        self.task_queries = 0
        self.task_generated = 0
        if not self.persistent:
            self.active = {}
            self.archive = {}

        # A lifecycle needs both inheritance and turnover. Keeping every slot
        # indefinitely is retained as the persistent_fifo ablation; learned
        # lifecycle policies preserve a stable core and free half the slots
        # for environmental sampling at each task boundary.
        if self.persistent and self.policy != "persistent_fifo" and self.active:
            units = list(self.active.values())
            if self.policy == "shuffled_lifecycle":
                random.Random(self.clock * 104729).shuffle(units)
                retained = units[: max(1, self.slots // 2)]
            else:
                retained = sorted(
                    units,
                    key=lambda unit: (-unit.credit, -unit.marginal_sum, unit.point),
                )[: max(1, self.slots // 2)]
            retained_points = {unit.point for unit in retained}
            for point in list(self.active):
                if point in retained_points:
                    continue
                unit = self.active.pop(point)
                unit.active = False
                self.archive[point] = unit
                self.retirements += 1

        if self.persistent and len(self.active) > self.slots:
            ordered = sorted(self.active.values(), key=lambda unit: (-unit.credit, unit.point))
            self.active = {unit.point: unit for unit in ordered[: self.slots]}

        for point in shared_initial:
            if len(self.active) >= self.slots:
                break
            if point in self.active:
                continue
            if point in self.archive:
                unit = self.archive.pop(point)
                unit.active = True
                unit.reactivations += 1
                self.reactivations += 1
            else:
                unit = self._birth(point)
            self.active[point] = unit

        # Persistent managers may enter a task with a full inherited bank.
        for unit in self.active.values():
            unit.last_used = self.clock
            unit.uses += 1
        self.task_queries += len(self.active)
        return list(self.active)

    def _candidate_points(
        self,
        used_this_task: set[tuple[int, int, int, int]],
        rng: random.Random,
    ) -> list[tuple[int, int, int, int]]:
        archived = [point for point in self.archive if point not in used_this_task]
        available = [
            point
            for point in DOMAIN
            if point not in used_this_task and point not in self.active
        ]
        fresh = rng.sample(available, min(70, len(available)))
        if self.policy in ("credit_lifecycle", "shuffled_lifecycle", "oracle_lifecycle"):
            archived.sort(key=lambda point: (-self.archive[point].credit, point))
            return list(dict.fromkeys(archived[:24] + fresh))
        return fresh

    def replace(
        self,
        ranked_candidates: list[Expr],
        target: Expr,
        used_this_task: set[tuple[int, int, int, int]],
        rng: random.Random,
    ) -> tuple[int, int, int, int] | None:
        if self.task_queries >= self.query_budget:
            return None
        options = self._candidate_points(used_this_task, rng)
        if not options:
            return None

        if self.policy == "random_reset":
            point = rng.choice(options)
            utility = disagreement(ranked_candidates, point)
        elif self.policy == "oracle_lifecycle":
            point = max(options, key=lambda item: oracle_elimination(ranked_candidates, target, item))
            utility = oracle_elimination(ranked_candidates, target, point)
        else:
            scored = []
            shuffled_credit = [self.archive[point].credit for point in options if point in self.archive]
            rng.shuffle(shuffled_credit)
            shuffled_index = 0
            for option in options:
                diagnostic = disagreement(ranked_candidates, option)
                stored = self.archive[option].credit if option in self.archive else 0.0
                if self.policy == "shuffled_lifecycle" and option in self.archive:
                    stored = shuffled_credit[shuffled_index]
                    shuffled_index += 1
                persistence = 0.18 * stored if "lifecycle" in self.policy else 0.0
                generation_cost = 0.04 if option not in self.archive else 0.0
                scored.append((diagnostic + persistence - generation_cost, diagnostic, option))
            scored.sort(key=lambda item: (-item[0], item[2]))
            _, utility, point = scored[0]

        if len(self.active) >= self.slots:
            if self.policy == "persistent_fifo":
                victim = min(self.active.values(), key=lambda unit: (unit.last_used, unit.birth))
            elif self.policy == "shuffled_lifecycle":
                victim = rng.choice(list(self.active.values()))
            elif self.policy == "oracle_lifecycle":
                victim = min(
                    self.active.values(),
                    key=lambda unit: oracle_elimination(ranked_candidates, target, unit.point),
                )
            elif self.policy == "credit_lifecycle":
                victim = min(self.active.values(), key=lambda unit: (unit.credit, unit.last_used))
            else:
                victim = min(self.active.values(), key=lambda unit: unit.last_used)
            self.active.pop(victim.point)
            victim.active = False
            self.retirements += 1
            if self.persistent:
                self.archive[victim.point] = victim

        if point in self.archive:
            unit = self.archive.pop(point)
            unit.active = True
            unit.reactivations += 1
            self.reactivations += 1
        else:
            unit = self._birth(point)
        unit.credit = 0.82 * unit.credit + 0.18 * min(1.0, utility / 8.0)
        unit.last_used = self.clock
        unit.uses += 1
        self.active[point] = unit
        self.task_queries += 1
        return point

    def end_task(
        self,
        candidates: list[Expr],
        target: Expr,
        success: bool,
        observed_points: list[tuple[int, int, int, int]],
    ) -> dict:
        top = candidates[: min(72, len(candidates))]
        points = list(dict.fromkeys(observed_points))
        target_values = {point: eval_poly(target.poly, point) for point in points}
        marginals = {}
        for point in points:
            # Credit is assigned after feedback: how many surviving hypotheses
            # this observation rejects. Selection never sees the target value
            # before the point is queried.
            marginal = mean(
                float(eval_poly(candidate.poly, point) != target_values[point])
                for candidate in top
            ) if top else 0.0
            marginals[point] = marginal
            unit = self.active.get(point) or self.archive.get(point)
            if unit is None:
                continue
            observed = marginal + 0.15 * float(success)
            if self.policy == "shuffled_lifecycle":
                observed = 0.5 * observed + 0.5 * random.Random(self.clock * 7919 + sum(point)).random()
            unit.credit = 0.75 * unit.credit + 0.25 * observed
            unit.marginal_sum += marginal

        for unit in self.archive.values():
            if unit.point not in marginals:
                unit.credit *= 0.85
        stale = [
            point
            for point, unit in self.archive.items()
            if unit.credit < 0.12 and self.clock - unit.last_used >= 3
        ]
        for point in stale:
            self.archive.pop(point)
            self.eliminations += 1

        return {
            "probe_queries": self.task_queries,
            "probe_generated": self.task_generated,
            "active_probe_count": len(self.active),
            "archived_probe_count": len(self.archive),
            "mean_active_credit": mean(unit.credit for unit in self.active.values()),
            "mean_active_age": mean(self.clock - unit.birth + 1 for unit in self.active.values()),
            "mean_marginal_utility": mean(marginals.values()) if marginals else 0.0,
            "active_probe_points": [list(point) for point in sorted(self.active)],
            "active_probe_credits": [
                self.active[point].credit for point in sorted(self.active)
            ],
            "cumulative_births": self.births,
            "cumulative_retirements": self.retirements,
            "cumulative_reactivations": self.reactivations,
            "cumulative_eliminations": self.eliminations,
        }


def revisit_tasks() -> list[Task]:
    original = benchmark_tasks()["recursive_reuse"]
    return [
        Task(
            name=f"revisit_d{task.level}",
            family="recursive_revisit",
            level=task.level,
            target=add_expr(task.target, W),
            source_macros=frozenset((f"R{task.level}",)),
        )
        for task in original
    ]


def task_stream() -> list[tuple[str, Task]]:
    tasks = benchmark_tasks()
    return (
        [("formation", task) for task in tasks["recursive_reuse"]]
        + [("shift", task) for task in tasks["low_reuse_control"]]
        + [("revisit", task) for task in revisit_tasks()]
    )


def update_macro_archive(
    archive: dict[str, Macro], phase: str, task: Task, result: dict
) -> None:
    used = set()
    for identifier in result["selected_macros"]:
        if f"[{identifier}]" in result["solution"] and identifier in archive:
            archive[identifier].credit += 1.0
            archive[identifier].uses += 1
            used.add(identifier)
    for identifier, macro in archive.items():
        if identifier not in used:
            macro.credit *= 0.985
    if not result["success"]:
        return
    prefix = {"formation": "R", "shift": "N", "revisit": "V"}[phase]
    archive[f"{prefix}{task.level}"] = Macro(f"{prefix}{task.level}", task.target, credit=1.2)


def search_with_probe_lifecycle(
    task: Task,
    archive: dict[str, Macro],
    manager: ProbeManager,
    candidate_budget: int,
    shared_initial: list[tuple[int, int, int, int]],
    rng: random.Random,
    macro_slots: int = 4,
    beam_width: int = 90,
    max_rounds: int = 8,
    max_search_cost: int = 19,
) -> dict:
    started = time.perf_counter()
    active_probes = manager.start_task(shared_initial)
    observed_probes = list(active_probes)
    used_this_task = set(observed_probes)
    target_values = [eval_poly(task.target.poly, point) for point in observed_probes]
    selected, retrieval_evaluations = retrieve_macros(
        archive, task, observed_probes, target_values, "phi_recursive", macro_slots, rng
    )
    selected_ids = {macro.identifier for macro in selected}
    atoms = list(PRIMITIVES) + [macro.as_atom() for macro in selected]
    best_by_poly = {atom.poly: atom for atom in atoms}
    evaluated = len(best_by_poly)
    solution = next((atom for atom in best_by_poly.values() if atom.poly == task.target.poly), None)
    beam = list(best_by_poly.values())

    for _ in range(max_rounds):
        if solution is not None or evaluated >= candidate_budget:
            break
        beam.sort(key=lambda expr: candidate_score(expr, target_values, observed_probes), reverse=True)
        beam = beam[:beam_width]
        partners = atoms + beam[:24]
        proposals = {}
        for left in beam:
            for right in partners:
                for constructor in (add_expr, mul_expr):
                    candidate = constructor(left, right)
                    if candidate.search_cost > max_search_cost or candidate.poly in best_by_poly:
                        continue
                    current = proposals.get(candidate.poly)
                    if current is None or candidate.search_cost < current.search_cost:
                        proposals[candidate.poly] = candidate
                if evaluated + len(proposals) >= candidate_budget:
                    break
            if evaluated + len(proposals) >= candidate_budget:
                break
        remaining = candidate_budget - evaluated
        proposals = dict(list(proposals.items())[:remaining])
        if not proposals:
            break
        for polynomial, candidate in proposals.items():
            best_by_poly[polynomial] = candidate
            if polynomial == task.target.poly:
                solution = candidate
                break
        evaluated += len(proposals)
        if solution is not None:
            break

        ranked = sorted(
            best_by_poly.values(),
            key=lambda expr: candidate_score(expr, target_values, observed_probes),
            reverse=True,
        )
        new_point = manager.replace(ranked, task.target, used_this_task, rng)
        if new_point is not None:
            used_this_task.add(new_point)
            observed_probes.append(new_point)
            target_values = [eval_poly(task.target.poly, point) for point in observed_probes]
        beam = ranked[:beam_width]

    ranked_final = sorted(
        best_by_poly.values(),
        key=lambda expr: candidate_score(expr, target_values, observed_probes),
        reverse=True,
    )
    lifecycle = manager.end_task(
        ranked_final, task.target, solution is not None, observed_probes
    )
    solution_text = solution.text if solution is not None else ""
    retrieval_recall = (
        len(selected_ids & task.source_macros) / len(task.source_macros)
        if task.source_macros
        else 0.0
    )
    return {
        "success": solution is not None,
        "candidates_evaluated": evaluated,
        "retrieval_candidates_evaluated": retrieval_evaluations,
        "selected_macros": sorted(selected_ids),
        "retrieval_recall": retrieval_recall,
        "solution": solution_text,
        "solution_search_cost": solution.search_cost if solution is not None else 0,
        "seconds": time.perf_counter() - started,
        **lifecycle,
    }


def aggregate(records: list[dict]) -> list[dict]:
    groups = {}
    for row in records:
        key = (row["policy"], row["phase"], row["level"])
        groups.setdefault(key, []).append(row)
    output = []
    for (policy, phase, level), rows in sorted(groups.items()):
        values = [float(row["success"]) for row in rows]
        output.append(
            {
                "policy": policy,
                "phase": phase,
                "level": level,
                "degree": rows[0]["degree"],
                "runs": len(rows),
                "success_rate": mean(values),
                "success_se": pstdev(values) / math.sqrt(len(values)),
                "probe_queries_mean": mean(row["probe_queries"] for row in rows),
                "probe_generated_mean": mean(row["probe_generated"] for row in rows),
                "active_probe_count_mean": mean(row["active_probe_count"] for row in rows),
                "archived_probe_count_mean": mean(row["archived_probe_count"] for row in rows),
                "mean_active_credit": mean(row["mean_active_credit"] for row in rows),
                "mean_marginal_utility": mean(row["mean_marginal_utility"] for row in rows),
                "retrieval_recall_mean": mean(row["retrieval_recall"] for row in rows),
                "candidates_mean": mean(row["candidates_evaluated"] for row in rows),
            }
        )
    return output


def run(seeds: int, candidate_budget: int, output_name: str) -> Path:
    records = []
    stream = task_stream()
    for seed in range(seeds):
        for policy in POLICIES:
            archive = foundation_macros()
            manager = ProbeManager(policy, slots=4, query_budget=8)
            for stream_index, (phase, task) in enumerate(stream):
                shared_rng = random.Random(seed * 100003 + stream_index * 193 + 17)
                shared_initial = shared_rng.sample(list(DOMAIN), 4)
                rng = random.Random(seed * 100003 + stream_index * 193 + POLICIES.index(policy) * 1009 + 29)
                result = search_with_probe_lifecycle(
                    task, archive, manager, candidate_budget, shared_initial, rng
                )
                records.append(
                    {
                        "seed": seed,
                        "policy": policy,
                        "stream_index": stream_index,
                        "phase": phase,
                        "task": task.name,
                        "level": task.level,
                        "degree": degree(task.target.poly),
                        "expanded_size": task.target.expanded_size,
                        "macro_archive_size_before": len(archive),
                        **result,
                    }
                )
                update_macro_archive(archive, phase, task, result)

    summary = aggregate(records)
    lifecycle_totals = []
    for policy in POLICIES:
        final_rows = [
            row
            for row in records
            if row["policy"] == policy and row["stream_index"] == len(stream) - 1
        ]
        lifecycle_totals.append(
            {
                "policy": policy,
                "births_mean": mean(row["cumulative_births"] for row in final_rows),
                "retirements_mean": mean(row["cumulative_retirements"] for row in final_rows),
                "reactivations_mean": mean(row["cumulative_reactivations"] for row in final_rows),
                "eliminations_mean": mean(row["cumulative_eliminations"] for row in final_rows),
                "final_archive_mean": mean(row["archived_probe_count"] for row in final_rows),
            }
        )

    output = {
        "experiment": "EXP245",
        "claim_scope": "probe lifecycle in bounded polynomial structure discovery",
        "seeds": seeds,
        "candidate_budget": candidate_budget,
        "active_probe_slots": 4,
        "probe_query_budget_per_task": 8,
        "policies": list(POLICIES),
        "stream": [
            {
                "index": index,
                "phase": phase,
                "task": task.name,
                "degree": degree(task.target.poly),
                "expanded_size": task.target.expanded_size,
            }
            for index, (phase, task) in enumerate(stream)
        ],
        "controls": {
            "same_macro_organization": "phi_recursive",
            "same_candidate_budget": True,
            "same_active_probe_slots": True,
            "same_probe_query_budget": True,
            "same_first_task_initial_probes": True,
            "task_observations_persist_within_task": True,
            "learned_lifecycle_retained_slots": 2,
            "exact_polynomial_verifier": True,
        },
        "summary": summary,
        "lifecycle_totals": lifecycle_totals,
        "records": records,
    }
    requested_path = Path(output_name)
    path = requested_path if requested_path.parent != Path(".") else RESULTS / requested_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(path)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=12)
    parser.add_argument("--candidate-budget", type=int, default=1800)
    parser.add_argument("--output", default="EXP245_probe_lifecycle.json")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.seeds, args.candidate_budget, args.output)
