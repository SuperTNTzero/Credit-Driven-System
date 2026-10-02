"""EXP243-long: one uninterrupted organization-formation event stream.

Unlike EXP243, this experiment does not synthesize earlier JSON artifacts.
Each run owns one persistent state Sigma=(W, Phi, C, U, B, H) and advances it
through candidate generation, active probing, attributed credit, exact
verification, lifecycle selection, consolidation, damage, and recovery.

The domain is bounded symbolic arithmetic (integer polynomials). Exact
canonical polynomial equality acts as the trusted validity kernel. This makes
the causal loop auditable; it is not a claim of open-world theorem discovery.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean, pstdev
from typing import Iterable

from exp244_complexity_scaling import (
    DOMAIN,
    ONE,
    PRIMITIVES,
    W as VAR_W,
    X,
    Y,
    Z,
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
)


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
METHODS = (
    "full_loop",
    "shuffled_credit",
    "no_phi",
    "no_compile",
    "program_memory",
    "random_verify",
    "phase_reset",
)
FOUNDATION_IDS = frozenset("ABCDEF")


@dataclass
class CandidateState:
    expression: Expr
    credit: float
    born: int
    last_seen: int
    observations: int = 0
    verifier_calls: int = 0
    valid_hits: int = 0
    active: bool = True


@dataclass
class UnitState:
    identifier: str
    expression: Expr
    credit: float
    born: int
    dependencies: set[str] = field(default_factory=set)
    uses: int = 0
    active: bool = True
    damaged: bool = False
    recoveries: int = 0

    def as_macro(self) -> Macro:
        return Macro(self.identifier, self.expression, self.credit, self.uses)


@dataclass
class ProgramMemory:
    signature: tuple[int, ...]
    expression: Expr
    born: int
    uses: int = 0


@dataclass
class ClosedLoopState:
    method: str
    clock: int = 0
    resource: float = 130.0
    phi: dict[str, float] = field(default_factory=dict)
    candidates: dict[tuple, CandidateState] = field(default_factory=dict)
    units: dict[str, UnitState] = field(default_factory=dict)
    program_memory: list[ProgramMemory] = field(default_factory=list)
    history: list[dict] = field(default_factory=list)
    births: int = 0
    promotions: int = 0
    eliminations: int = 0
    retirements: int = 0
    reactivations: int = 0
    damage_events: int = 0
    recovery_events: int = 0
    wrong_consolidations: int = 0
    total_candidate_evaluations: int = 0
    total_queries: int = 0
    total_verifier_calls: int = 0
    resource_exhaustions: int = 0

    def __post_init__(self) -> None:
        if self.units:
            return
        for identifier, macro in foundation_macros().items():
            self.units[identifier] = UnitState(
                identifier=identifier,
                expression=macro.expression,
                credit=1.0,
                born=0,
            )
            self.phi[identifier] = 1.0

    @property
    def has_phi(self) -> bool:
        return self.method not in ("no_phi", "program_memory")

    @property
    def can_compile(self) -> bool:
        return self.method not in ("no_compile", "program_memory")

    def event(self, kind: str, **payload) -> None:
        self.history.append({"clock": self.clock, "kind": kind, **payload})


@dataclass(frozen=True)
class StreamTask:
    name: str
    phase: str
    target: Expr
    source_units: frozenset[str]
    unit_id: str
    is_revisit: bool = False


def build_stream() -> list[StreamTask]:
    reuse = benchmark_tasks()["recursive_reuse"]
    r1, r2, r3, r4, r5 = (task.target for task in reuse)
    a = foundation_macros()["A"].expression
    b = foundation_macros()["B"].expression
    c = foundation_macros()["C"].expression
    d = foundation_macros()["D"].expression
    e = foundation_macros()["E"].expression
    f = foundation_macros()["F"].expression

    stream = [
        StreamTask("form_r1", "formation", r1, frozenset(("D", "C")), "R1"),
        StreamTask("form_r2", "formation", r2, frozenset(("R1", "A", "E")), "R2"),
        StreamTask("form_r3", "formation", r3, frozenset(("R2", "B", "C")), "R3"),
        StreamTask("form_r4", "formation", r4, frozenset(("R3", "C", "D")), "R4"),
    ]
    recursive_units = (("R1", r1), ("R2", r2), ("R3", r3), ("R4", r4))
    offsets = (X, Y, Z, VAR_W)
    for cycle in range(4):
        for index, (identifier, expression) in enumerate(recursive_units):
            offset = offsets[(cycle + index) % len(offsets)]
            stream.append(
                StreamTask(
                    f"reuse_c{cycle + 1}_{identifier.lower()}",
                    "transfer",
                    add_expr(expression, offset),
                    frozenset((identifier,)),
                    f"P{cycle + 1}{index + 1}",
                )
            )
    stream.extend(
        [
        StreamTask("transfer_r2_f", "transfer", add_expr(r2, f), frozenset(("R2", "F")), "T1"),
        StreamTask(
            "transfer_r3_ad",
            "transfer",
            add_expr(r3, mul_expr(a, d)),
            frozenset(("R3", "A", "D")),
            "T2",
        ),
        StreamTask(
            "transfer_r4_be",
            "transfer",
            add_expr(r4, mul_expr(b, e)),
            frozenset(("R4", "B", "E")),
            "T3",
        ),
        StreamTask("damage_probe_r3", "damage", r3, frozenset(("R3",)), "R3", True),
        StreamTask("recover_r3", "recovery", r3, frozenset(("R2", "B", "C")), "R3", True),
        StreamTask("postrepair_r3_w", "recovery", add_expr(r3, VAR_W), frozenset(("R3",)), "P1"),
        StreamTask("late_r5", "late", r5, frozenset(("R4", "A", "E")), "R5"),
        StreamTask("revisit_r1", "revisit", r1, frozenset(("R1",)), "R1", True),
        StreamTask(
            "novel_r4_ac",
            "revisit",
            add_expr(r4, mul_expr(a, c)),
            frozenset(("R4", "A", "C")),
            "N1",
        ),
        ]
    )
    return stream


SIGNATURE_POINTS = (
    (-2, -1, 0, 1),
    (-1, 2, 1, 0),
    (0, -2, 2, 1),
    (1, 0, -1, 2),
    (2, 1, 0, -1),
)


def behavior_signature(expression: Expr) -> tuple[int, ...]:
    return tuple(eval_poly(expression.poly, point) for point in SIGNATURE_POINTS)


def active_units(state: ClosedLoopState) -> list[UnitState]:
    return [unit for unit in state.units.values() if unit.active and not unit.damaged]


def transitive_damage(state: ClosedLoopState, identifier: str) -> set[str]:
    damaged = {identifier}
    changed = True
    while changed:
        changed = False
        for unit in state.units.values():
            if unit.identifier in damaged:
                continue
            if unit.dependencies & damaged:
                damaged.add(unit.identifier)
                changed = True
    return damaged


def damage_critical_unit(state: ClosedLoopState, identifier: str = "R3") -> dict:
    affected = transitive_damage(state, identifier) if identifier in state.units else set()
    for item in affected:
        state.units[item].damaged = True
    if affected:
        state.damage_events += 1
    state.event("damage", target=identifier, affected=sorted(affected))
    return {"damage_target": identifier, "damaged_units": sorted(affected)}


def reset_phase_state(state: ClosedLoopState) -> None:
    if state.method != "phase_reset":
        return
    state.phi = {identifier: 1.0 for identifier in FOUNDATION_IDS}
    state.candidates.clear()
    state.event("phase_reset")


def unit_retrieval_score(
    unit: UnitState,
    peers: list[UnitState],
    probes: list[tuple[int, int, int, int]],
    target_values: list[int],
    phi_value: float,
) -> float:
    atom = unit.as_macro().as_atom()
    transformed = [atom]
    for primitive in (ONE, X, Y, Z, VAR_W):
        transformed.append(add_expr(atom, primitive))
        transformed.append(mul_expr(atom, primitive))
    # Responsibility is relational: a reusable unit can look weak alone but
    # become decisive when composed with one other available unit.
    for peer in peers:
        if peer.identifier == unit.identifier:
            continue
        other = peer.as_macro().as_atom()
        pair = (add_expr(atom, other), mul_expr(atom, other))
        transformed.extend(pair)
        for partial in pair:
            for second_peer in peers:
                if second_peer.identifier in (unit.identifier, peer.identifier):
                    continue
                second = second_peer.as_macro().as_atom()
                transformed.append(add_expr(partial, second))
                transformed.append(mul_expr(partial, second))
            for primitive in (ONE, X, Y, Z, VAR_W):
                transformed.append(add_expr(partial, primitive))
                transformed.append(mul_expr(partial, primitive))
    fit = max(candidate_score(expr, target_values, probes) for expr in transformed)
    return fit + 0.08 * phi_value + 0.008 * math.log1p(unit.uses)


def retrieve_units(
    state: ClosedLoopState,
    probes: list[tuple[int, int, int, int]],
    target_values: list[int],
    rng: random.Random,
    slots: int | None = None,
) -> tuple[list[UnitState], int]:
    available = active_units(state)
    if slots is None:
        # Formation receives broad exploration bandwidth. Deployment is
        # capacity constrained so persistent organization must select among
        # more available units than can be called.
        slots = 8 if state.clock <= 4 else 5
    if state.method == "program_memory":
        return [], 0

    cue_probes = probes[:1]
    cue_values = target_values[:1]
    scored = [
        (
            unit_retrieval_score(
                unit,
                available,
                cue_probes,
                cue_values,
                state.phi.get(unit.identifier, 0.0),
            ),
            unit.identifier,
            unit,
        )
        for unit in available
    ]
    scored.sort(key=lambda item: (-item[0], item[1]))
    cue_scored = [
        (
            unit_retrieval_score(unit, available, cue_probes, cue_values, 0.0),
            unit.identifier,
            unit,
        )
        for unit in available
    ]
    cue_scored.sort(key=lambda item: (-item[0], item[1]))
    if state.method == "no_phi":
        selected = [item[2] for item in cue_scored[:slots]]
    else:
        cue_slots = min(3, slots)
        selected = [item[2] for item in cue_scored[:cue_slots]]
        selected_ids = {unit.identifier for unit in selected}
        selected.extend(
            item[2]
            for item in scored
            if item[1] not in selected_ids
            and len(selected) < slots
        )
    n = len(available)
    retrieval_cost = sum(
        11 + 22 * max(0, n - 1) + 4 * max(0, n - 1) * max(0, n - 2)
        for _ in available
    )
    return selected, retrieval_cost


def choose_probe(
    ranked: list[Expr],
    used: set[tuple[int, int, int, int]],
    rng: random.Random,
    random_choice: bool,
) -> tuple[int, int, int, int]:
    available = [point for point in DOMAIN if point not in used]
    sample = rng.sample(available, min(72, len(available)))
    if random_choice or len(ranked) < 2:
        return rng.choice(sample)
    top = ranked[: min(32, len(ranked))]
    best = None
    for point in sample:
        values = [eval_poly(candidate.poly, point) for candidate in top]
        center = mean(values)
        variance = mean((value - center) ** 2 for value in values)
        partition = len(set(values)) / len(values)
        score = math.log1p(variance) + partition
        item = (score, point)
        if best is None or item > best:
            best = item
    assert best is not None
    return best[1]


def register_candidates(
    state: ClosedLoopState,
    candidates: Iterable[Expr],
    probes: list[tuple[int, int, int, int]],
    target_values: list[int],
) -> None:
    for expression in candidates:
        record = state.candidates.get(expression.poly)
        fit = candidate_score(expression, target_values, probes)
        if record is None:
            state.candidates[expression.poly] = CandidateState(
                expression=expression,
                credit=fit,
                born=state.clock,
                last_seen=state.clock,
                observations=len(probes),
            )
            state.births += 1
        else:
            if expression.search_cost < record.expression.search_cost:
                record.expression = expression
            record.last_seen = state.clock
            record.observations += 1
            record.credit = 0.82 * record.credit + 0.18 * fit


def apply_observation_credit(
    state: ClosedLoopState,
    ranked: list[Expr],
    point: tuple[int, int, int, int],
    target_value: int,
    rng: random.Random,
) -> None:
    records = [state.candidates[expr.poly] for expr in ranked[: min(90, len(ranked))] if expr.poly in state.candidates]
    deltas = [0.18 if eval_poly(record.expression.poly, point) == target_value else -0.12 for record in records]
    if state.method == "shuffled_credit":
        rng.shuffle(deltas)
    if state.method in ("no_phi", "program_memory"):
        deltas = [0.0] * len(deltas)
    for record, delta in zip(records, deltas):
        record.credit = 0.94 * record.credit + delta


def candidate_rank(
    state: ClosedLoopState,
    expression: Expr,
    probes: list[tuple[int, int, int, int]],
    target_values: list[int],
) -> float:
    base = candidate_score(expression, target_values, probes)
    if not state.has_phi:
        return base
    record = state.candidates.get(expression.poly)
    persistent = record.credit if record else 0.0
    macro_credit = sum(state.phi.get(identifier, 0.0) for identifier in expression.macro_ids)
    return base + 0.01 * persistent + 0.005 * macro_credit


def verify_candidates(
    state: ClosedLoopState,
    ranked: list[Expr],
    target: Expr,
    remaining_calls: int,
    rng: random.Random,
) -> tuple[Expr | None, int]:
    if remaining_calls <= 0 or not ranked:
        return None, 0
    count = min(2, remaining_calls, len(ranked))
    if state.method == "random_verify":
        selected = rng.sample(ranked[: min(80, len(ranked))], count)
    else:
        selected = ranked[:count]
    for expression in selected:
        record = state.candidates.get(expression.poly)
        if record:
            record.verifier_calls += 1
        state.total_verifier_calls += 1
        state.resource -= 2.5
        valid = expression.poly == target.poly
        state.event("verify", candidate=expression.text, valid=valid)
        if valid:
            if record:
                record.valid_hits += 1
            return expression, len(selected)
    return None, len(selected)


def update_unit_credit(
    state: ClosedLoopState,
    selected_ids: list[str],
    solution: Expr | None,
    rng: random.Random,
) -> None:
    if not state.has_phi:
        return
    responsible = [identifier for identifier in selected_ids if solution and identifier in solution.macro_ids]
    deltas = {identifier: (-0.015 if identifier not in responsible else 0.30) for identifier in selected_ids}
    if state.method == "shuffled_credit" and deltas:
        values = list(deltas.values())
        rng.shuffle(values)
        deltas = dict(zip(deltas, values))
    for identifier, delta in deltas.items():
        old = state.phi.get(identifier, 0.0)
        state.phi[identifier] = max(-2.0, min(4.0, 0.985 * old + delta))
        if identifier in state.units and identifier in responsible:
            state.units[identifier].credit += max(0.0, delta)
            state.units[identifier].uses += 1
    state.event("credit", responsible=sorted(responsible), selected=selected_ids)


def consolidate(
    state: ClosedLoopState,
    task: StreamTask,
    solution: Expr,
) -> None:
    if state.method == "program_memory":
        signature = behavior_signature(task.target)
        existing = next((item for item in state.program_memory if item.signature == signature), None)
        if existing is None:
            state.program_memory.append(ProgramMemory(signature, solution, state.clock))
            state.event("memory_write", task=task.name)
        return
    if not state.can_compile:
        state.event("promotion_blocked", task=task.name)
        return

    dependencies = set(solution.macro_ids)
    existing = state.units.get(task.unit_id)
    was_damaged = bool(existing and existing.damaged)
    state.units[task.unit_id] = UnitState(
        identifier=task.unit_id,
        expression=task.target,
        credit=max(1.2, state.phi.get(task.unit_id, 0.0)),
        born=state.clock,
        dependencies=dependencies,
        recoveries=(existing.recoveries + 1 if was_damaged else (existing.recoveries if existing else 0)),
    )
    state.phi[task.unit_id] = max(1.2, state.phi.get(task.unit_id, 0.0))
    state.promotions += 1
    if was_damaged:
        state.recovery_events += 1
    state.event(
        "promote",
        unit=task.unit_id,
        dependencies=sorted(dependencies),
        recovery=was_damaged,
    )


def lifecycle_update(state: ClosedLoopState, archive_limit: int = 260) -> dict:
    for record in state.candidates.values():
        if record.last_seen < state.clock:
            record.credit *= 0.94
    stale = [
        poly
        for poly, record in state.candidates.items()
        if state.clock - record.last_seen >= 3 and record.credit < -0.28
    ]
    for poly in stale:
        state.candidates.pop(poly)
        state.eliminations += 1
    if len(state.candidates) > archive_limit:
        ordered = sorted(
            state.candidates.items(),
            key=lambda item: (item[1].credit, item[1].last_seen, item[1].born),
        )
        for poly, _ in ordered[: len(state.candidates) - archive_limit]:
            state.candidates.pop(poly)
            state.eliminations += 1

    nonfoundation = [unit for unit in state.units.values() if unit.identifier not in FOUNDATION_IDS]
    for unit in nonfoundation:
        if unit.damaged:
            continue
        maintenance = 0.025 + 0.003 * math.log1p(unit.expression.expanded_size)
        state.resource -= maintenance
        if unit.credit < 0.15 and state.clock - unit.born >= 4:
            unit.active = False
            state.retirements += 1
    if len(nonfoundation) > 12:
        active = [unit for unit in nonfoundation if unit.active and not unit.damaged]
        active.sort(key=lambda unit: (state.phi.get(unit.identifier, 0.0), unit.uses, unit.born))
        for unit in active[: max(0, len(active) - 12)]:
            unit.active = False
            state.retirements += 1

    state.resource = min(180.0, state.resource + 30.0)
    if state.resource <= 0:
        state.resource_exhaustions += 1
        state.resource = 0.0
    return {
        "candidate_archive": len(state.candidates),
        "active_units": len(active_units(state)),
        "resource": state.resource,
    }


def memory_retrieve(
    state: ClosedLoopState,
    task: StreamTask,
    initial_probes: list[tuple[int, int, int, int]],
) -> Expr | None:
    if state.method != "program_memory" or not state.program_memory:
        return None
    observed = tuple(eval_poly(task.target.poly, point) for point in initial_probes)
    matches = []
    for item in state.program_memory:
        predicted = tuple(eval_poly(item.expression.poly, point) for point in initial_probes)
        if predicted == observed:
            matches.append(item)
    if len(matches) != 1:
        return None
    matches[0].uses += 1
    state.event("memory_retrieve", task=task.name)
    return matches[0].expression


def search_task(
    state: ClosedLoopState,
    task: StreamTask,
    rng: random.Random,
    intervention_rng: random.Random,
    initial_probes: list[tuple[int, int, int, int]],
    candidate_budget: int,
    verifier_budget: int,
    query_budget: int,
    beam_width: int = 90,
    max_search_cost: int = 19,
) -> dict:
    started = time.perf_counter()
    state.clock += 1
    state.resource -= 1.0
    probes = list(initial_probes)
    used_probes = set(probes)
    target_values = [eval_poly(task.target.poly, point) for point in probes]
    state.total_queries += len(probes)
    state.event("task_start", task=task.name, phase=task.phase)

    memorized = memory_retrieve(state, task, probes)
    if memorized is not None and memorized.poly == task.target.poly:
        state.event("task_success", task=task.name, source="program_memory")
        lifecycle = lifecycle_update(state)
        return {
            "success": True,
            "source": "program_memory",
            "candidate_evaluations": 0,
            "retrieval_evaluations": len(state.program_memory),
            "queries": len(probes),
            "verifier_calls": 0,
            "selected_units": [],
            "source_recall": 0.0,
            "solution": memorized.text,
            "solution_cost": 1,
            "seconds": time.perf_counter() - started,
            **lifecycle,
        }

    selected_units, retrieval_evaluations = retrieve_units(state, probes, target_values, rng)
    selected_ids = [unit.identifier for unit in selected_units]
    source_recall = (
        len(set(selected_ids) & set(task.source_units)) / len(task.source_units)
        if task.source_units
        else 0.0
    )
    atoms = list(PRIMITIVES) + [unit.as_macro().as_atom() for unit in selected_units]
    best = {atom.poly: atom for atom in atoms}
    register_candidates(state, best.values(), probes, target_values)
    evaluated = len(best)
    verifier_calls = 0
    solution = None
    beam = list(best.values())

    for _ in range(8):
        ranked = sorted(
            best.values(),
            key=lambda expr: candidate_rank(state, expr, probes, target_values),
            reverse=True,
        )
        found, calls = verify_candidates(
            state,
            ranked,
            task.target,
            verifier_budget - verifier_calls,
            intervention_rng,
        )
        verifier_calls += calls
        if found is not None:
            solution = found
            break
        if evaluated >= candidate_budget or state.resource <= 2.5:
            break

        beam = ranked[:beam_width]
        partners = atoms + beam[:24]
        proposals: dict[tuple, Expr] = {}
        round_limit = min(candidate_budget - evaluated, max(240, candidate_budget // 6))
        for left in beam:
            for right in partners:
                for constructor in (add_expr, mul_expr):
                    candidate = constructor(left, right)
                    if candidate.search_cost > max_search_cost or candidate.poly in best:
                        continue
                    current = proposals.get(candidate.poly)
                    if current is None or candidate.search_cost < current.search_cost:
                        proposals[candidate.poly] = candidate
                if len(proposals) >= round_limit:
                    break
            if len(proposals) >= round_limit:
                break
        remaining = round_limit
        proposals = dict(list(proposals.items())[:remaining])
        if not proposals:
            break
        best.update(proposals)
        register_candidates(state, proposals.values(), probes, target_values)
        evaluated += len(proposals)
        state.total_candidate_evaluations += len(proposals)
        state.resource -= 0.008 * len(proposals)

        if len(probes) < query_budget:
            ranked = sorted(
                best.values(),
                key=lambda expr: candidate_rank(state, expr, probes, target_values),
                reverse=True,
            )
            point = choose_probe(
                ranked,
                used_probes,
                rng,
                random_choice=state.method in ("random_verify", "program_memory"),
            )
            target_value = eval_poly(task.target.poly, point)
            probes.append(point)
            used_probes.add(point)
            target_values.append(target_value)
            state.total_queries += 1
            state.resource -= 0.75
            apply_observation_credit(
                state, ranked, point, target_value, intervention_rng
            )
            state.event("probe", point=list(point), target_value=target_value)

    update_unit_credit(state, selected_ids, solution, rng)
    if solution is not None:
        consolidate(state, task, solution)
        state.event("task_success", task=task.name, source="verified")
    else:
        state.event("task_failure", task=task.name)
    lifecycle = lifecycle_update(state)
    return {
        "success": solution is not None,
        "source": "verified" if solution is not None else "none",
        "candidate_evaluations": evaluated,
        "retrieval_evaluations": retrieval_evaluations,
        "queries": len(probes),
        "verifier_calls": verifier_calls,
        "selected_units": selected_ids,
        "source_recall": source_recall,
        "solution": solution.text if solution else "",
        "solution_cost": solution.search_cost if solution else 0,
        "seconds": time.perf_counter() - started,
        **lifecycle,
    }


def audit_damaged_behavior(state: ClosedLoopState, task: StreamTask) -> dict:
    """Measure the frozen post-damage call before any recovery update."""
    started = time.perf_counter()
    state.clock += 1
    state.resource -= 1.0
    state.event("task_start", task=task.name, phase=task.phase)
    available = direct_unit_available(state, task.unit_id)
    state.event(
        "damage_behavior_audit",
        task=task.name,
        target_unit=task.unit_id,
        success=available,
    )
    lifecycle = lifecycle_update(state)
    return {
        "success": available,
        "source": "direct_unit" if available else "damaged",
        "candidate_evaluations": 0,
        "retrieval_evaluations": 0,
        "queries": 0,
        "verifier_calls": 0,
        "selected_units": [task.unit_id] if available else [],
        "source_recall": float(available),
        "solution": f"[{task.unit_id}]" if available else "",
        "solution_cost": 1 if available else 0,
        "seconds": time.perf_counter() - started,
        **lifecycle,
    }


def direct_unit_available(state: ClosedLoopState, identifier: str) -> bool:
    unit = state.units.get(identifier)
    return bool(unit and unit.active and not unit.damaged)


def run_one(
    method: str,
    seed: int,
    candidate_budget: int,
    verifier_budget: int,
    query_budget: int,
) -> tuple[list[dict], dict, list[dict]]:
    state = ClosedLoopState(method)
    stream = build_stream()
    rows = []
    previous_phase = stream[0].phase
    damage_audit = {
        "target": "R3",
        "available_before": False,
        "available_after": False,
        "affected": [],
        "recovery_task_index": None,
    }

    for index, task in enumerate(stream):
        if task.phase != previous_phase:
            reset_phase_state(state)
            previous_phase = task.phase
        if task.phase == "damage":
            damage_audit["available_before"] = direct_unit_available(state, "R3")
            info = damage_critical_unit(state, "R3")
            damage_audit["affected"] = info["damaged_units"]
            damage_audit["available_after"] = direct_unit_available(state, "R3")

        probe_rng = random.Random(seed * 100003 + index * 193 + 17)
        initial_probes = probe_rng.sample(list(DOMAIN), 3)
        # Search randomness is paired across methods. Randomized ablations use
        # a separate stream so their interventions cannot alter later probes.
        rng = random.Random(seed * 100003 + index * 193 + 29)
        intervention_rng = random.Random(
            seed * 100003 + index * 193 + METHODS.index(method) * 1009 + 71
        )
        if task.phase == "damage":
            result = audit_damaged_behavior(state, task)
        else:
            result = search_task(
                state,
                task,
                rng,
                intervention_rng,
                initial_probes,
                candidate_budget,
                verifier_budget,
                query_budget,
            )
        if task.phase == "recovery" and task.unit_id == "R3" and direct_unit_available(state, "R3"):
            if damage_audit["recovery_task_index"] is None:
                damage_audit["recovery_task_index"] = index
        rows.append(
            {
                "seed": seed,
                "method": method,
                "stream_index": index,
                "task": task.name,
                "phase": task.phase,
                "degree": degree(task.target.poly),
                "expanded_size": task.target.expanded_size,
                "target_unit": task.unit_id,
                "is_revisit": task.is_revisit,
                "resource_before_flow": state.resource,
                **result,
                "cumulative_births": state.births,
                "cumulative_promotions": state.promotions,
                "cumulative_eliminations": state.eliminations,
                "cumulative_retirements": state.retirements,
                "cumulative_reactivations": state.reactivations,
                "cumulative_damage_events": state.damage_events,
                "cumulative_recovery_events": state.recovery_events,
                "cumulative_wrong_consolidations": state.wrong_consolidations,
                "program_memory_size": len(state.program_memory),
            }
        )

    lifecycle_kinds = {item["kind"] for item in state.history}
    postformation = [
        row for row in rows if row["phase"] not in ("formation", "damage")
    ]
    complex_postformation = [
        row
        for row in postformation
        if row["degree"] >= 2 and not row["is_revisit"]
    ]
    lifecycle_checks = (
        state.births > 0,
        "probe" in lifecycle_kinds,
        "verify" in lifecycle_kinds,
        "credit" in lifecycle_kinds,
        state.promotions > 0,
        sum(unit.uses for unit in state.units.values()) > 0,
        state.damage_events > 0,
        state.recovery_events > 0,
        state.eliminations > 0,
    )
    final = {
        "seed": seed,
        "method": method,
        "tasks": len(stream),
        "success_rate": mean(float(row["success"]) for row in rows),
        "formation_success": mean(float(row["success"]) for row in rows if row["phase"] == "formation"),
        "transfer_success": mean(float(row["success"]) for row in rows if row["phase"] == "transfer"),
        "recovery_success": mean(float(row["success"]) for row in rows if row["phase"] == "recovery"),
        "late_revisit_success": mean(float(row["success"]) for row in rows if row["phase"] in ("late", "revisit")),
        "postformation_success": mean(float(row["success"]) for row in postformation),
        "complex_postformation_success": mean(float(row["success"]) for row in complex_postformation),
        "final_resource": state.resource,
        "resource_exhaustions": state.resource_exhaustions,
        "candidate_births": state.births,
        "candidate_eliminations": state.eliminations,
        "promotions": state.promotions,
        "active_units": len(active_units(state)),
        "nonfoundation_units": sum(identifier not in FOUNDATION_IDS for identifier in state.units),
        "program_memory_size": len(state.program_memory),
        "verifier_calls": state.total_verifier_calls,
        "queries": state.total_queries,
        "candidate_evaluations": state.total_candidate_evaluations,
        "damage_available_before": damage_audit["available_before"],
        "damage_available_after": damage_audit["available_after"],
        "damage_drop": float(damage_audit["available_before"]) - float(damage_audit["available_after"]),
        "damaged_unit_count": len(damage_audit["affected"]),
        "recovery_task_index": damage_audit["recovery_task_index"],
        "recovered": damage_audit["recovery_task_index"] is not None,
        "recovery_events": state.recovery_events,
        "wrong_consolidations": state.wrong_consolidations,
        "lifecycle_event_coverage": sum(lifecycle_checks) / len(lifecycle_checks),
        "history_events": len(state.history),
    }
    return rows, final, state.history


def summarize(final_rows: list[dict]) -> list[dict]:
    fields = (
        "success_rate",
        "formation_success",
        "transfer_success",
        "recovery_success",
        "late_revisit_success",
        "postformation_success",
        "complex_postformation_success",
        "final_resource",
        "resource_exhaustions",
        "candidate_births",
        "candidate_eliminations",
        "promotions",
        "active_units",
        "nonfoundation_units",
        "program_memory_size",
        "verifier_calls",
        "queries",
        "candidate_evaluations",
        "damaged_unit_count",
        "damage_drop",
        "recovered",
        "recovery_events",
        "wrong_consolidations",
        "lifecycle_event_coverage",
    )
    output = []
    for method in METHODS:
        rows = [row for row in final_rows if row["method"] == method]
        item = {"method": method, "seeds": len(rows)}
        for field_name in fields:
            values = [float(row[field_name]) for row in rows]
            item[field_name] = mean(values)
            item[f"{field_name}_sd"] = pstdev(values)
        output.append(item)
    return output


def exact_sign_flip(final_rows: list[dict], method_a: str, method_b: str, field_name: str) -> dict:
    lookup = {(row["method"], row["seed"]): float(row[field_name]) for row in final_rows}
    seeds = sorted({row["seed"] for row in final_rows})
    differences = [lookup[(method_a, seed)] - lookup[(method_b, seed)] for seed in seeds]
    nonzero = [value for value in differences if abs(value) > 1e-12]
    observed = abs(mean(differences))
    if not nonzero:
        p_value = 1.0
    elif len(nonzero) <= 20:
        extreme = 0
        total = 2 ** len(nonzero)
        for mask in range(total):
            signed = [value if mask & (1 << index) else -value for index, value in enumerate(nonzero)]
            if abs(mean(signed)) >= observed - 1e-12:
                extreme += 1
        p_value = extreme / total
    else:
        p_value = float("nan")
    return {
        "method_a": method_a,
        "method_b": method_b,
        "field": field_name,
        "mean_difference": mean(differences),
        "two_sided_exact_p": p_value,
        "paired_seeds": len(differences),
    }


def run(
    seeds: int,
    seed_start: int,
    candidate_budget: int,
    verifier_budget: int,
    query_budget: int,
    output_name: str,
    keep_histories: bool,
    workers: int,
) -> Path:
    started = time.perf_counter()
    task_rows = []
    final_rows = []
    histories = []
    jobs = [(seed, method) for seed in range(seed_start, seed_start + seeds) for method in METHODS]
    completed = []
    if workers <= 1:
        for seed, method in jobs:
            completed.append(
                (
                    seed,
                    method,
                    run_one(method, seed, candidate_budget, verifier_budget, query_budget),
                )
            )
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    run_one,
                    method,
                    seed,
                    candidate_budget,
                    verifier_budget,
                    query_budget,
                ): (seed, method)
                for seed, method in jobs
            }
            for future in as_completed(futures):
                seed, method = futures[future]
                completed.append((seed, method, future.result()))

    for seed, method, (rows, final, history) in sorted(
        completed, key=lambda item: (item[0], METHODS.index(item[1]))
    ):
            task_rows.extend(rows)
            final_rows.append(final)
            if keep_histories or seed == seed_start:
                histories.append({"seed": seed, "method": method, "events": history})

    comparisons = [
        exact_sign_flip(final_rows, "full_loop", method, field_name)
        for method in METHODS
        if method != "full_loop"
        for field_name in ("success_rate", "transfer_success", "late_revisit_success")
    ]
    summary = summarize(final_rows)
    summary_lookup = {row["method"]: row for row in summary}
    gates = {
        "single_state_lineage": all(row["history_events"] > 0 for row in final_rows),
        "all_mechanisms_observed_full": summary_lookup["full_loop"]["lifecycle_event_coverage"] >= 0.99,
        "credit_changes_structure": summary_lookup["full_loop"]["promotions"] > summary_lookup["shuffled_credit"]["promotions"] and summary_lookup["full_loop"]["postformation_success"] > summary_lookup["shuffled_credit"]["postformation_success"],
        "damage_has_specific_effect": summary_lookup["full_loop"]["damage_drop"] >= 0.8,
        "feedback_recovers_damage": summary_lookup["full_loop"]["recovered"] >= 0.8,
        "compile_expands_fixed_budget_reach": summary_lookup["full_loop"]["complex_postformation_success"] > summary_lookup["no_compile"]["complex_postformation_success"],
        "persistent_loop_beats_phase_reset": summary_lookup["full_loop"]["success_rate"] > summary_lookup["phase_reset"]["success_rate"],
        "persistent_loop_beats_program_memory_on_transfer": summary_lookup["full_loop"]["transfer_success"] > summary_lookup["program_memory"]["transfer_success"],
        "no_resource_exhaustion_full": summary_lookup["full_loop"]["resource_exhaustions"] == 0,
        "no_wrong_consolidation_full": summary_lookup["full_loop"]["wrong_consolidations"] == 0,
    }
    output = {
        "experiment": "EXP243-long",
        "kind": "single uninterrupted stateful run; no reuse of EXP227-236 result artifacts",
        "claim_scope": "bounded symbolic arithmetic with exact polynomial validity kernel",
        "seeds": seeds,
        "seed_start": seed_start,
        "methods": list(METHODS),
        "config": {
            "candidate_budget_per_task": candidate_budget,
            "verifier_budget_per_task": verifier_budget,
            "query_budget_per_task": query_budget,
            "initial_resource": 130.0,
            "resource_flow_per_task": 30.0,
            "resource_cap": 180.0,
            "candidate_archive_limit": 260,
            "active_nonfoundation_unit_limit": 12,
            "parallel_workers": workers,
        },
        "controls": {
            "same_task_stream": True,
            "same_primitive_grammar": True,
            "same_initial_probes_by_seed_task": True,
            "same_candidate_budget": True,
            "same_verifier_budget": True,
            "same_query_budget": True,
            "paired_search_random_stream": True,
            "separate_intervention_random_stream": True,
            "no_final_audit_label_during_search": True,
            "exact_validity_kernel": True,
            "state_not_reset_in_full_loop": True,
            "oracle_not_included": True,
        },
        "stream": [
            {
                "index": index,
                "name": task.name,
                "phase": task.phase,
                "degree": degree(task.target.poly),
                "expanded_size": task.target.expanded_size,
                "source_units": sorted(task.source_units),
            }
            for index, task in enumerate(build_stream())
        ],
        "gates": gates,
        "gates_passed": sum(gates.values()),
        "gates_total": len(gates),
        "summary": summary,
        "paired_comparisons": comparisons,
        "final_rows": final_rows,
        "task_rows": task_rows,
        "histories": histories,
        "elapsed_seconds": time.perf_counter() - started,
    }
    requested = Path(output_name)
    path = requested if requested.parent != Path(".") else RESULTS / requested
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(path)
    print(json.dumps({"gates": gates, "elapsed_seconds": output["elapsed_seconds"]}, ensure_ascii=False))
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--candidate-budget", type=int, default=2400)
    parser.add_argument("--verifier-budget", type=int, default=16)
    parser.add_argument("--query-budget", type=int, default=8)
    parser.add_argument("--output", default="EXP243_long_closed_loop.json")
    parser.add_argument("--keep-histories", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        args.seeds,
        args.seed_start,
        args.candidate_budget,
        args.verifier_budget,
        args.query_budget,
        args.output,
        args.keep_histories,
        args.workers,
    )
