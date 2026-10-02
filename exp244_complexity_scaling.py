"""EXP244: complexity scaling in bounded symbolic discovery.

The benchmark increases algebraic degree from linear to fifth-order
polynomials while independently varying whether tasks admit recursive reuse.
All methods share the same primitive grammar, exact polynomial verifier,
candidate budget, and observations. They differ only in active probing and in
whether verified structures can be retrieved and composed as persistent units.
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
from typing import Iterable


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
VARIABLES = ("x", "y", "z", "w")
DOMAIN = tuple(
    (x, y, z, w)
    for x in (-2, -1, 0, 1, 2)
    for y in (-2, -1, 0, 1, 2)
    for z in (-2, -1, 0, 1, 2)
    for w in (-2, -1, 0, 1, 2)
)
METHODS = (
    "flat_random",
    "flat_active",
    "direct_memory",
    "phi_static",
    "phi_shuffled",
    "phi_recursive",
    "oracle_recursive",
)


Monomial = tuple[int, int, int, int]
Polynomial = tuple[tuple[Monomial, int], ...]


def canonical(items: dict[Monomial, int]) -> Polynomial:
    return tuple(sorted((monomial, coefficient) for monomial, coefficient in items.items() if coefficient))


def add_poly(a: Polynomial, b: Polynomial) -> Polynomial:
    result = dict(a)
    for monomial, coefficient in b:
        result[monomial] = result.get(monomial, 0) + coefficient
    return canonical(result)


def mul_poly(a: Polynomial, b: Polynomial) -> Polynomial:
    result: dict[Monomial, int] = {}
    for left, left_coefficient in a:
        for right, right_coefficient in b:
            monomial = tuple(x + y for x, y in zip(left, right))
            result[monomial] = result.get(monomial, 0) + left_coefficient * right_coefficient
    return canonical(result)


def eval_poly(poly: Polynomial, point: tuple[int, int, int, int]) -> int:
    total = 0
    for monomial, coefficient in poly:
        value = coefficient
        for base, exponent in zip(point, monomial):
            value *= base**exponent
        total += value
    return total


def degree(poly: Polynomial) -> int:
    return max((sum(monomial) for monomial, _ in poly), default=0)


@dataclass(frozen=True)
class Expr:
    poly: Polynomial
    text: str
    search_cost: int
    expanded_size: int
    macro_ids: frozenset[str] = frozenset()


def constant(value: int) -> Expr:
    poly = canonical({(0, 0, 0, 0): value})
    return Expr(poly, str(value), 1, 1)


def variable(name: str) -> Expr:
    powers = [0, 0, 0, 0]
    powers[VARIABLES.index(name)] = 1
    return Expr(canonical({tuple(powers): 1}), name, 1, 1)


def add_expr(a: Expr, b: Expr) -> Expr:
    left, right = sorted((a, b), key=lambda item: item.text)
    return Expr(
        add_poly(left.poly, right.poly),
        f"({left.text}+{right.text})",
        left.search_cost + right.search_cost + 1,
        left.expanded_size + right.expanded_size + 1,
        left.macro_ids | right.macro_ids,
    )


def mul_expr(a: Expr, b: Expr) -> Expr:
    left, right = sorted((a, b), key=lambda item: item.text)
    return Expr(
        mul_poly(left.poly, right.poly),
        f"({left.text}*{right.text})",
        left.search_cost + right.search_cost + 1,
        left.expanded_size + right.expanded_size + 1,
        left.macro_ids | right.macro_ids,
    )


def sum_expr(*items: Expr) -> Expr:
    output = items[0]
    for item in items[1:]:
        output = add_expr(output, item)
    return output


def product_expr(*items: Expr) -> Expr:
    output = items[0]
    for item in items[1:]:
        output = mul_expr(output, item)
    return output


ZERO = constant(0)
ONE = constant(1)
X, Y, Z, W = (variable(name) for name in VARIABLES)
PRIMITIVES = (ZERO, ONE, X, Y, Z, W)


@dataclass
class Macro:
    identifier: str
    expression: Expr
    credit: float = 1.0
    uses: int = 0

    def as_atom(self) -> Expr:
        return Expr(
            self.expression.poly,
            f"[{self.identifier}]",
            1,
            self.expression.expanded_size,
            frozenset((self.identifier,)),
        )


@dataclass(frozen=True)
class Task:
    name: str
    family: str
    level: int
    target: Expr
    source_macros: frozenset[str]


def foundation_macros() -> dict[str, Macro]:
    expressions = {
        "A": add_expr(X, ONE),
        "B": add_expr(Y, ONE),
        "C": add_expr(Z, ONE),
        "D": add_expr(X, Y),
        "E": add_expr(Y, Z),
        "F": add_expr(X, W),
    }
    return {name: Macro(name, expression) for name, expression in expressions.items()}


def benchmark_tasks() -> dict[str, list[Task]]:
    foundations = foundation_macros()
    a, b, c, d, e = (foundations[name].expression for name in "ABCDE")

    reusable = []
    r1 = add_expr(d, c)
    reusable.append(Task("reuse_d1", "recursive_reuse", 1, r1, frozenset(("D", "C"))))
    r2 = add_expr(mul_expr(r1, a), e)
    reusable.append(Task("reuse_d2", "recursive_reuse", 2, r2, frozenset(("R1", "A", "E"))))
    r3 = add_expr(mul_expr(r2, b), c)
    reusable.append(Task("reuse_d3", "recursive_reuse", 3, r3, frozenset(("R2", "B", "C"))))
    r4 = add_expr(mul_expr(r3, c), d)
    reusable.append(Task("reuse_d4", "recursive_reuse", 4, r4, frozenset(("R3", "C", "D"))))
    r5 = add_expr(mul_expr(r4, a), e)
    reusable.append(Task("reuse_d5", "recursive_reuse", 5, r5, frozenset(("R4", "A", "E"))))

    # Degree-matched and expanded-size-matched controls. At degree d, the
    # product and linear filler both have size 4d-1, so the target has size
    # 8d-1, exactly matching the recursive-reuse sequence 7,15,23,31,39.
    control_factors = (add_expr(X, Z), add_expr(Y, W), add_expr(Z, W), add_expr(X, X), add_expr(Y, Y))
    filler_leaves = (W, X, Z, Y, X, W, Y, Z, W, X)
    controls = []
    for level in range(1, 6):
        product = product_expr(*control_factors[:level])
        filler = sum_expr(*filler_leaves[: 2 * level])
        target = add_expr(product, filler)
        assert target.expanded_size == reusable[level - 1].target.expanded_size
        assert degree(target.poly) == level
        controls.append(Task(f"control_d{level}", "low_reuse_control", level, target, frozenset()))
    return {"recursive_reuse": reusable, "low_reuse_control": controls}


def normalized_error(expr: Expr, target_values: list[int], probes: list[tuple[int, int, int, int]]) -> float:
    predictions = [eval_poly(expr.poly, point) for point in probes]
    mse = mean((prediction - target) ** 2 for prediction, target in zip(predictions, target_values))
    scale = math.sqrt(mean(target * target for target in target_values)) + 1.0
    return math.sqrt(mse) / scale


def candidate_score(expr: Expr, target_values: list[int], probes: list[tuple[int, int, int, int]]) -> float:
    exact_fraction = mean(
        float(eval_poly(expr.poly, point) == target)
        for point, target in zip(probes, target_values)
    )
    return -normalized_error(expr, target_values, probes) + 0.08 * exact_fraction - 0.0025 * expr.search_cost


def choose_active_probe(
    candidates: list[Expr],
    used: set[tuple[int, int, int, int]],
    rng: random.Random,
) -> tuple[int, int, int, int]:
    available = [point for point in DOMAIN if point not in used]
    sample = rng.sample(available, min(80, len(available)))
    top = candidates[: min(28, len(candidates))]
    if len(top) < 2:
        return rng.choice(sample)
    best_point = sample[0]
    best_score = -1.0
    for point in sample:
        values = [eval_poly(candidate.poly, point) for candidate in top]
        center = mean(values)
        variance = mean((value - center) ** 2 for value in values)
        disagreement = len(set(values)) / len(values)
        score = math.log1p(variance) + disagreement
        if score > best_score:
            best_score = score
            best_point = point
    return best_point


def retrieve_macros(
    archive: dict[str, Macro],
    task: Task,
    probes: list[tuple[int, int, int, int]],
    target_values: list[int],
    method: str,
    slots: int,
    rng: random.Random,
) -> tuple[list[Macro], int]:
    if method in ("flat_random", "flat_active", "direct_memory"):
        return [], 0
    macros = list(archive.values())
    if method == "phi_static":
        macros = [macro for macro in macros if len(macro.identifier) == 1]
    if method == "phi_shuffled":
        rng.shuffle(macros)
        return macros[:slots], 0
    if method == "oracle_recursive" and task.source_macros:
        relevant = [archive[name] for name in task.source_macros if name in archive]
        remaining = [macro for macro in macros if macro.identifier not in task.source_macros]
        remaining.sort(key=lambda macro: (-macro.credit, macro.identifier))
        return (relevant + remaining)[:slots], 0

    scored = []
    retrieval_candidates_evaluated = 0
    all_macro_atoms = {macro.identifier: macro.as_atom() for macro in macros}
    for macro in macros:
        atom = macro.as_atom()
        transformed = [atom]
        for primitive in (ONE, X, Y, Z, W):
            transformed.append(add_expr(atom, primitive))
            transformed.append(mul_expr(atom, primitive))
        # Responsibility can be relational: a unit that looks weak alone may
        # become decisive when paired with another verified unit. This cheap
        # second-order intervention is retrieval-only and does not consume the
        # downstream candidate budget.
        for other_id, other in all_macro_atoms.items():
            if other_id == macro.identifier:
                continue
            transformed.append(add_expr(atom, other))
            transformed.append(mul_expr(atom, other))
        retrieval_candidates_evaluated += len(transformed)
        fit = max(candidate_score(expr, target_values, probes) for expr in transformed)
        score = fit + 0.018 * macro.credit - 0.001 * math.log1p(macro.expression.expanded_size)
        scored.append((score, macro.identifier, macro))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [item[2] for item in scored[:slots]], retrieval_candidates_evaluated


def search_task(
    task: Task,
    method: str,
    archive: dict[str, Macro],
    budget: int,
    rng: random.Random,
    initial_probes: list[tuple[int, int, int, int]],
    macro_slots: int = 4,
    beam_width: int = 90,
    max_rounds: int = 8,
    max_search_cost: int = 19,
) -> dict:
    started = time.perf_counter()
    probes = list(initial_probes)
    target_values = [eval_poly(task.target.poly, point) for point in probes]

    if method == "direct_memory":
        for macro in archive.values():
            if macro.expression.poly == task.target.poly:
                return {
                    "success": True,
                    "candidates_evaluated": 0,
                    "probes": len(probes),
                    "selected_macros": [],
                    "retrieval_recall": 0.0,
                    "solution": macro.expression.text,
                    "solution_search_cost": 1,
                    "solution_expanded_size": macro.expression.expanded_size,
                    "solution_macro_fraction": 0.0,
                    "retrieval_candidates_evaluated": 0,
                    "seconds": time.perf_counter() - started,
                }

    selected, retrieval_candidates_evaluated = retrieve_macros(
        archive, task, probes, target_values, method, macro_slots, rng
    )
    selected_ids = {macro.identifier for macro in selected}
    retrieval_recall = (
        len(selected_ids & task.source_macros) / len(task.source_macros)
        if task.source_macros
        else 0.0
    )
    atoms = list(PRIMITIVES) + [macro.as_atom() for macro in selected]
    best_by_poly = {atom.poly: atom for atom in atoms}
    candidates_evaluated = len(best_by_poly)

    for atom in best_by_poly.values():
        if atom.poly == task.target.poly:
            solution = atom
            break
    else:
        solution = None

    used_probes = set(probes)
    beam = list(best_by_poly.values())
    for _ in range(max_rounds):
        if solution is not None or candidates_evaluated >= budget:
            break
        beam.sort(key=lambda expr: candidate_score(expr, target_values, probes), reverse=True)
        beam = beam[:beam_width]
        partners = atoms + beam[:24]
        proposals: dict[Polynomial, Expr] = {}
        for left in beam:
            for right in partners:
                for constructor in (add_expr, mul_expr):
                    candidate = constructor(left, right)
                    if candidate.search_cost > max_search_cost or candidate.poly in best_by_poly:
                        continue
                    current = proposals.get(candidate.poly)
                    if current is None or candidate.search_cost < current.search_cost:
                        proposals[candidate.poly] = candidate
                if candidates_evaluated + len(proposals) >= budget:
                    break
            if candidates_evaluated + len(proposals) >= budget:
                break
        if not proposals:
            break
        remaining = budget - candidates_evaluated
        if len(proposals) > remaining:
            proposals = dict(list(proposals.items())[:remaining])
        for polynomial, candidate in proposals.items():
            best_by_poly[polynomial] = candidate
            if polynomial == task.target.poly:
                solution = candidate
                break
        candidates_evaluated += len(proposals)
        if solution is not None:
            break

        ranked = sorted(
            best_by_poly.values(),
            key=lambda expr: candidate_score(expr, target_values, probes),
            reverse=True,
        )
        if method == "flat_random":
            point = rng.choice([item for item in DOMAIN if item not in used_probes])
        else:
            point = choose_active_probe(ranked, used_probes, rng)
        probes.append(point)
        used_probes.add(point)
        target_values.append(eval_poly(task.target.poly, point))
        beam = ranked[:beam_width]

    if solution is not None:
        # The exact polynomial canonical form is the independent verifier.
        assert solution.poly == task.target.poly
        macro_fraction = len(solution.macro_ids) / max(1, solution.search_cost)
        solution_text = solution.text
        solution_cost = solution.search_cost
        solution_expanded = solution.expanded_size
    else:
        macro_fraction = 0.0
        solution_text = ""
        solution_cost = 0
        solution_expanded = 0

    return {
        "success": solution is not None,
        "candidates_evaluated": candidates_evaluated,
        "probes": len(probes),
        "selected_macros": sorted(selected_ids),
        "retrieval_recall": retrieval_recall,
        "solution": solution_text,
        "solution_search_cost": solution_cost,
        "solution_expanded_size": solution_expanded,
        "solution_macro_fraction": macro_fraction,
        "retrieval_candidates_evaluated": retrieval_candidates_evaluated,
        "seconds": time.perf_counter() - started,
    }


def update_archive(
    archive: dict[str, Macro],
    method: str,
    task: Task,
    result: dict,
) -> None:
    used = set()
    for identifier in result["selected_macros"]:
        if identifier in result["solution"] and identifier in archive:
            archive[identifier].credit += 1.0
            archive[identifier].uses += 1
            used.add(identifier)
    for identifier, macro in archive.items():
        if identifier not in used:
            macro.credit *= 0.985

    if not result["success"]:
        return
    if method in ("phi_recursive", "phi_shuffled", "oracle_recursive"):
        identifier = f"R{task.level}" if task.family == "recursive_reuse" else f"N{task.level}"
        archive[identifier] = Macro(identifier, task.target, credit=1.2)
    elif method == "direct_memory":
        identifier = f"M_{task.family}_{task.level}"
        archive[identifier] = Macro(identifier, task.target, credit=1.0)


def aggregate(records: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for row in records:
        key = (row["method"], row["budget"], row["family"], row["level"])
        groups.setdefault(key, []).append(row)
    output = []
    for (method, budget, family, level), rows in sorted(groups.items()):
        success = [float(row["success"]) for row in rows]
        evaluated = [row["candidates_evaluated"] for row in rows]
        probes = [row["probes"] for row in rows]
        recalls = [row["retrieval_recall"] for row in rows]
        archive_sizes = [row["archive_size_before"] for row in rows]
        retrieval_evaluations = [row["retrieval_candidates_evaluated"] for row in rows]
        solution_costs = [row["solution_search_cost"] for row in rows if row["success"]]
        macro_fractions = [row["solution_macro_fraction"] for row in rows if row["success"]]
        output.append(
            {
                "method": method,
                "budget": budget,
                "family": family,
                "level": level,
                "degree": rows[0]["degree"],
                "expanded_size": rows[0]["target_expanded_size"],
                "runs": len(rows),
                "success_rate": mean(success),
                "success_std": pstdev(success),
                "success_se": pstdev(success) / math.sqrt(len(success)),
                "candidates_mean": mean(evaluated),
                "probes_mean": mean(probes),
                "retrieval_recall_mean": mean(recalls),
                "archive_size_mean": mean(archive_sizes),
                "retrieval_candidates_mean": mean(retrieval_evaluations),
                "solution_search_cost_mean": mean(solution_costs) if solution_costs else 0.0,
                "solution_macro_fraction_mean": mean(macro_fractions) if macro_fractions else 0.0,
                "seconds_mean": mean(row["seconds"] for row in rows),
            }
        )
    return output


def critical_degrees(summary: list[dict], threshold: float = 0.8) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for row in summary:
        groups.setdefault((row["method"], row["budget"], row["family"]), []).append(row)
    output = []
    for (method, budget, family), rows in sorted(groups.items()):
        critical_degree = 0
        for row in sorted(rows, key=lambda item: item["degree"]):
            if row["degree"] != critical_degree + 1 or row["success_rate"] < threshold:
                break
            critical_degree = row["degree"]
        output.append(
            {
                "method": method,
                "budget": budget,
                "family": family,
                "threshold": threshold,
                "critical_degree": critical_degree,
                "success_auc": mean(row["success_rate"] for row in rows),
            }
        )
    return output


def run(seeds: int, budgets: list[int], output_name: str) -> Path:
    tasks = benchmark_tasks()
    records = []
    for budget in budgets:
        for seed in range(seeds):
            for method in METHODS:
                for family, family_tasks in tasks.items():
                    archive = foundation_macros()
                    for task in family_tasks:
                        rng = random.Random(seed * 100003 + budget * 101 + METHODS.index(method) * 1009 + task.level * 17 + (0 if family == "recursive_reuse" else 7919))
                        probe_rng = random.Random(seed * 100003 + budget * 101 + task.level * 17 + (0 if family == "recursive_reuse" else 7919))
                        initial_probes = probe_rng.sample(list(DOMAIN), 4)
                        result = search_task(task, method, archive, budget, rng, initial_probes)
                        row = {
                            "seed": seed,
                            "method": method,
                            "budget": budget,
                            "family": family,
                            "task": task.name,
                            "level": task.level,
                            "degree": degree(task.target.poly),
                            "target_expanded_size": task.target.expanded_size,
                            "archive_size_before": len(archive),
                            **result,
                        }
                        records.append(row)
                        update_archive(archive, method, task, result)

    summary = aggregate(records)
    critical = critical_degrees(summary)
    output = {
        "experiment": "EXP244",
        "claim_scope": "bounded exact polynomial discovery from degree 1 to 5",
        "seeds": seeds,
        "budgets": budgets,
        "methods": list(METHODS),
        "families": {
            family: [
                {
                    "name": task.name,
                    "level": task.level,
                    "degree": degree(task.target.poly),
                    "expanded_size": task.target.expanded_size,
                    "target": task.target.text,
                    "source_macros": sorted(task.source_macros),
                }
                for task in family_tasks
            ]
            for family, family_tasks in tasks.items()
        },
        "controls": {
            "same_primitives": True,
            "same_exact_verifier": True,
            "same_candidate_budget": True,
            "retrieval_evaluations_accounted_separately": True,
            "same_initial_observations": 4,
            "same_initial_probe_points_within_seed": True,
            "macro_slots": 4,
            "direct_memory_is_not_composable": True,
        },
        "summary": summary,
        "critical_degrees": critical,
        "records": records,
    }
    path = RESULTS / output_name
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(path)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--budgets", type=int, nargs="+", default=[400, 800, 1600])
    parser.add_argument("--output", default="EXP244_complexity_scaling.json")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.seeds, args.budgets, args.output)
