"""
CBGA — the Chu & Beasley genetic algorithm for the MKP (Chu & Beasley 1998,
"A genetic algorithm for the multidimensional knapsack problem",
J. Heuristics 4: 63-86). The canonical MKP metaheuristic and the expected
evolutionary-family competitor.

Steady state: each child is built by binary-tournament selection of two
parents, uniform crossover, flipping `mutation_bits` random bits, and the
paper's REPAIR operator, then replaces the worst population member unless it
duplicates an existing one. Repair makes every individual feasible:

  DROP phase  remove selected items in increasing pseudo-utility order until
              all knapsacks fit
  ADD  phase  insert unselected items in decreasing pseudo-utility order
              whenever they fit

with pseudo-utility u_j = p_j / (lambda . w_j), lambda = the dual values of
the instance's LP relaxation (solved once per problem with scipy/HiGHS —
deterministic). This is the paper's surrogate-weight scheme.

Framework notes:
  - MKP-specific by design (the repair reads profits/weights/capacity from
    problem.extras); it raises on any other problem.
  - Repair uses constraint bookkeeping, not objective calls, exactly as
    published — so a CBGA "function evaluation" is one child evaluation.
    Disclose in the paper that CBGA additionally exploits instance structure
    through repair; that is inherent to the published algorithm.
  - One `step()` produces `children_per_step` children so convergence curves
    stay a sane length; budget experiments should cap `max_fn_calls` anyway.
  - Topology is unused (panmictic population, like the paper's).

Implemented from the paper's published description.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from core.optimizer import Optimizer


def mkp_arrays(problem):
    """The MKP instance data (profits, weights, capacity) an evaluator-free
    repair operator needs, or a clear error for non-MKP problems."""
    extras = problem.extras
    if not all(key in extras for key in ("profits", "weights", "capacity")):
        raise ValueError(
            f"Chu & Beasley repair needs an MKP-style problem exposing profits/"
            f"weights/capacity in extras; {problem.name!r} does not"
        )
    return (
        np.asarray(extras["profits"], dtype=np.float64),
        np.asarray(extras["weights"], dtype=np.float64),  # (m, n)
        np.asarray(extras["capacity"], dtype=np.float64),
    )


def pseudo_utility_orders(profits, weights, capacity):
    """Chu & Beasley's pseudo-utility item orders u_j = p_j / (lambda . w_j)
    with lambda = the dual values of the instance's LP relaxation (solved once
    per problem with scipy/HiGHS — deterministic). Returns (drop_order,
    add_order): least useful first / most useful first.
    """
    from scipy.optimize import linprog

    lp = linprog(
        c=-profits,
        A_ub=weights,
        b_ub=capacity,
        bounds=(0, 1),
        method="highs",
    )
    if not lp.success:
        raise RuntimeError(f"LP relaxation failed: {lp.message}")
    duals = -np.asarray(lp.ineqlin.marginals)  # >= 0 for binding <= constraints
    weighted = duals @ weights
    with np.errstate(divide="ignore"):
        utility = np.where(weighted > 0, profits / weighted, np.inf)
    # ties broken by item index (stable), as any fixed deterministic order
    drop_order = np.argsort(utility, kind="stable")  # least useful first
    return drop_order, drop_order[::-1]


def chu_beasley_repair(x, weights, capacity, drop_order, add_order):
    """The paper's REPAIR, in place: DROP selected items in increasing
    pseudo-utility order until every knapsack fits, then ADD unselected items
    in decreasing order whenever they fit. Constraint bookkeeping only — no
    objective evaluations."""
    load = weights @ x
    if (load > capacity).any():
        for j in drop_order:
            if x[j]:
                x[j] = 0
                load -= weights[:, j]
                if (load <= capacity).all():
                    break
    for j in add_order:
        if not x[j] and (load + weights[:, j] <= capacity).all():
            x[j] = 1
            load += weights[:, j]


class ChuBeasleyGA(Optimizer):

    name = "CBGA"

    def __init__(
        self,
        population_size: int = 100,
        mutation_bits: int = 2,  # the paper's fixed-rate mutation
        children_per_step: int = 25,
        **topology_kwargs,
    ):
        super().__init__(population_size=population_size, **topology_kwargs)
        if mutation_bits < 0:
            raise ValueError(f"mutation_bits must be >= 0, got {mutation_bits}")
        self.mutation_bits = int(mutation_bits)
        self.children_per_step = int(children_per_step)

    # ─────────────────────── problem-specific setup ────────────────────

    def set_problem(self, problem):
        super().set_problem(problem)
        if problem is None:
            return
        try:
            self.profits, self.weights, self.capacity = mkp_arrays(problem)
        except ValueError as error:
            raise ValueError(f"{self.name}: {error}") from None
        # pseudo-utilities from the LP relaxation's dual values (paper §3.2)
        self._drop_order, self._add_order = pseudo_utility_orders(
            self.profits, self.weights, self.capacity
        )

    # ─────────────────────── Optimizer interface ───────────────────────

    def initialize(self, initial_population=None):
        if initial_population is None:
            population = self.random_solutions(self.population_size)
        else:
            population = np.array(initial_population, dtype=np.int8)
        for row in population:
            self._repair(row)
        self.population = population
        self.values, self.violations = self.evaluate(self.population)
        # a multiset: random init on small instances produces duplicate rows,
        # and a plain set would forget the surviving twin when one is replaced
        self._member_keys = Counter(row.tobytes() for row in self.population)

        self.best_solution = None
        self._remember_best()

    def step(self, iteration):
        for _ in range(self.children_per_step):
            child = self._make_child()
            if self._member_keys[child.tobytes()] > 0:
                continue  # duplicate: discard (paper's diversity rule)
            value, violation = self.evaluate(child)
            worst = self.deb_worst_index(self.values, self.violations)
            if self.is_first_better(value, self.values[worst], violation, self.violations[worst]):
                self._member_keys[self.population[worst].tobytes()] -= 1
                self.population[worst] = child
                self.values[worst] = value
                self.violations[worst] = violation
                self._member_keys[child.tobytes()] += 1
        self._remember_best()

    def get_best(self):
        return self.best_solution, self.best_value, self.best_violation

    # ──────────────────────────── operators ────────────────────────────

    def _make_child(self):
        p1 = self._tournament()
        p2 = self._tournament()
        mask = self.rng.random(self.dim) < 0.5  # uniform crossover
        child = np.where(mask, self.population[p1], self.population[p2]).astype(np.int8)
        if self.mutation_bits:
            flip = self.rng.choice(self.dim, size=min(self.mutation_bits, self.dim), replace=False)
            child[flip] ^= 1
        self._repair(child)
        return child

    def _tournament(self):
        a, b = self.rng.integers(self.population_size, size=2)
        return a if self.is_first_better(self.values[a], self.values[b], self.violations[a], self.violations[b]) else b

    def _repair(self, x):
        """In place: DROP until feasible, then ADD what still fits."""
        chu_beasley_repair(x, self.weights, self.capacity, self._drop_order, self._add_order)

    def _remember_best(self):
        idx = self.deb_best_index(self.values, self.violations)
        if self.best_solution is None or self.is_first_better(
            self.values[idx], self.best_value, self.violations[idx], self.best_violation
        ):
            self.best_solution = self.population[idx].copy()
            self.best_value = float(self.values[idx])
            self.best_violation = float(self.violations[idx])
