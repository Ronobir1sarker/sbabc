"""
CETS — Critical Event Tabu Search for the MKP (Glover & Kochenberger 1996,
"Critical event tabu search for multidimensional knapsack problems", in
Osman & Kelly (eds.), Meta-Heuristics: Theory and Applications, 407-427).
The local-search-family competitor.

The method is a strategic oscillation around the feasibility boundary:

  constructive phase  add variables (chosen by surrogate-constraint ratio)
                      while they fit; the last feasible solution is a
                      CRITICAL EVENT; then add `span` more, going infeasible
  destructive phase   drop variables until feasible (the regained-feasibility
                      solution is a critical event), then drop `span` more

  span oscillates 1..p2..1 (p2*span phase-pairs per value while span <= p1,
  p2 pairs above); at each turn-around the first k choices are biased by
  tabu memory: TABU_R = sum of the last t critical solutions, TABU_F = sum
  of all critical solutions. Adds maximize ratio(j) - PEN_R*TABU_R(j) -
  PEN_F*TABU_F(j); drops minimize the same expression (an inducement to
  drop recently-critical variables). k cycles 1..KMAX (period 2t
  turn-arounds). Trial solutions generated at critical events (scans by raw
  objective coefficient that ignore tabu status — the paper's aspiration;
  ~82% of its optima came from these) are recorded as candidates for the
  best-so-far without redirecting the search.

Surrogate weights follow the paper exactly: feasible -> w_i = 1/slack_i;
infeasible constructive -> method (a): 1/slack_i for positive slack, else
2 + |slack_i|; infeasible destructive -> method (b): 0 for non-negative
slack, else 1/(|slack_i| + sum of that row's coefficients over items out of
the knapsack).

Faithfulness notes:
  - The chapter's penalty-definition equation is an image lost to text
    extraction; PEN_R = r* and PEN_F = r*/(C * critical-event count) are
    reconstructed from its prose (r* = the maximum initial-surrogate ratio;
    "PEN_F is scaled by the iteration count so that the long-term influence
    is small", C = 100,000) — PEN_R dominates any ratio, forcing the first
    k adds toward TABU_R(j) = 0 exactly as described, while PEN_F only
    breaks ties.
  - The paper is deterministic (start x = 0, or x = e). Runs under this
    framework need independent repetitions, so initial="random" (default)
    starts from a seeded random vector and enters the phase matching its
    feasibility; initial="zero" reproduces the paper's setup. This is the
    only stochastic element.
  - The paper's six-restart parameter ladder (varying t and p2) is NOT
    replicated; campaigns run one fixed configuration per experiment name.
  - Function evaluations: interior add/drop moves use constraint
    bookkeeping and incremental objective values, exactly as published —
    only critical solutions and trial solutions are scored through the
    counted evaluator (the same disclosure as CBGA's repair: the algorithm
    inherently exploits MKP structure, so a black-box FE count would be
    fiction). Report FE comparisons against it with that caveat.
  - MKP-specific by design (reads profits/weights/capacity extras); raises
    on other problems. Topology and population size are unused
    (single-solution method).

One step() = one constructive + one destructive phase (with their transfer
bookkeeping), so convergence curves have a sane length; stopping is the
framework's (iteration cap, patience, max_fn_calls, stop-at-optimum).
"""

from __future__ import annotations

import numpy as np

from algorithms.cbga import mkp_arrays
from core.optimizer import Optimizer


class CriticalEventTabuSearch(Optimizer):

    name = "CETS"

    def __init__(
        self,
        t: int = 7,  # recency window: last t critical solutions (paper: 3-12)
        p1: int = 3,  # span schedule knee
        p2: int = 7,  # span ceiling
        k_max: int = 4,  # KMAX: ceiling of the tabu-influenced move count
        penalty_constant: float = 100_000.0,  # C in the PEN_F scale
        initial: str = "random",  # "random" | "zero" (paper) | "ones"
        **topology_kwargs,
    ):
        super().__init__(**topology_kwargs)
        if initial not in ("random", "zero", "ones"):
            raise ValueError(f"initial must be 'random', 'zero' or 'ones', got {initial!r}")
        if not 1 <= p1 <= p2:
            raise ValueError(f"need 1 <= p1 <= p2, got {p1}, {p2}")
        self.t = int(t)
        self.p1 = int(p1)
        self.p2 = int(p2)
        self.k_max = int(k_max)
        self.penalty_constant = float(penalty_constant)
        self.initial = initial

    # ─────────────────────── problem-specific setup ────────────────────

    def set_problem(self, problem):
        super().set_problem(problem)
        if problem is None:
            return
        try:
            self.profits, self.weights, self.capacity = mkp_arrays(problem)
        except ValueError as error:
            raise ValueError(f"{self.name}: {error}") from None
        # r*: maximum ratio under the initial surrogate (each constraint
        # scaled by its RHS) — the paper's reference point for the penalties
        with np.errstate(divide="ignore", invalid="ignore"):
            s = (self.weights / self.capacity[:, None]).sum(axis=0)
            ratios = np.where(s > 0, self.profits / s, np.inf)
        finite = ratios[np.isfinite(ratios)]
        self._r_star = float(finite.max()) if len(finite) else 1.0

    # ─────────────────────── Optimizer interface ───────────────────────

    def initialize(self, initial_population=None):
        n = self.dim
        if initial_population is not None:
            seeds = np.atleast_2d(np.array(initial_population, dtype=np.int8))
            self.x = seeds[0].copy()  # single-solution method: first row only
        elif self.initial == "zero":
            self.x = np.zeros(n, dtype=np.int8)
        elif self.initial == "ones":
            self.x = np.ones(n, dtype=np.int8)
        else:
            self.x = self.random_solutions(1)[0]
        self.load = self.weights @ self.x

        self.tabu_recent = []  # the last t critical solutions
        self.tabu_r = np.zeros(n, dtype=np.float64)
        self.tabu_f = np.zeros(n, dtype=np.float64)
        self.n_critical = 0
        self.span = 1
        self.span_increasing = True
        self.pairs_at_span = 0
        self.turnarounds = 0
        self.k = 1

        self.best_solution = None
        self.best_value = -np.inf if self.optimization_type == "max" else np.inf
        self.best_violation = np.inf
        self._consider(self.x)  # the start point itself (x=0 is feasible)

    def step(self, iteration):
        if self._feasible():
            self._constructive_phase()
            self._transfer()
            self._destructive_phase()
            self._transfer()
        else:  # e.g. initial="ones"/random-infeasible: paper's x = e entry
            self._destructive_phase()
            self._transfer()
            self._constructive_phase()
            self._transfer()

    def get_best(self):
        return self.best_solution, self.best_value, self.best_violation

    # ────────────────────────── phase mechanics ────────────────────────

    def _feasible(self) -> bool:
        return bool((self.load <= self.capacity).all())

    def _slack(self):
        return self.capacity - self.load

    def _add(self, j):
        self.x[j] = 1
        self.load += self.weights[:, j]

    def _drop(self, j):
        self.x[j] = 0
        self.load -= self.weights[:, j]

    def _fits(self):
        """Boolean mask over items: adding j keeps every knapsack feasible."""
        return ((self.load[:, None] + self.weights) <= self.capacity[:, None]).all(axis=0)

    def _ratios(self, weights_w):
        """Surrogate ratios c_j / s_j for the given constraint weights."""
        s = weights_w @ self.weights
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(s > 0, self.profits / s, np.inf)

    def _surrogate_weights(self, phase: str):
        slack = self._slack()
        if self._feasible():
            with np.errstate(divide="ignore"):
                return np.where(slack > 0, 1.0 / np.maximum(slack, 1e-12), 1e12)
        if phase == "constructive":  # method (a)
            with np.errstate(divide="ignore"):
                return np.where(slack > 0, 1.0 / slack, 2.0 + np.abs(slack))
        # destructive, method (b): only violated rows carry weight
        out_sum = self.weights @ (1 - self.x)  # row sums over items out of the knapsack
        with np.errstate(divide="ignore"):
            return np.where(slack >= 0, 0.0, 1.0 / (np.abs(slack) + out_sum))

    def _penalized(self, phase: str, count_var: int):
        """value(j) = ratio(j) - penalties, applied to the first k choices
        after a turn-around (paper §2.2)."""
        values = self._ratios(self._surrogate_weights(phase)).astype(np.float64)
        if count_var <= self.k:
            pen_r = self._r_star
            pen_f = self._r_star / (self.penalty_constant * max(1, self.n_critical))
            values = values - pen_r * self.tabu_r - pen_f * self.tabu_f
        return values

    def _record_critical(self, x):
        """Critical event: update the recency window and frequency counts."""
        snapshot = x.copy()
        self.tabu_recent.append(snapshot)
        self.tabu_r += snapshot
        if len(self.tabu_recent) > self.t:
            self.tabu_r -= self.tabu_recent.pop(0)
        self.tabu_f += snapshot
        self.n_critical += 1

    def _consider(self, x):
        """Score a complete candidate through the counted evaluator and keep
        it if it beats the best so far."""
        value, violation = self.evaluate(np.asarray(x, dtype=np.int8))
        if self.best_solution is None or self.is_first_better(
            value, self.best_value, violation, self.best_violation
        ):
            self.best_solution = np.array(x, dtype=np.int8, copy=True)
            self.best_value = float(value)
            self.best_violation = float(violation)

    def _transfer(self):
        """Turn-around bookkeeping: span schedule (paper §2.3, counted in
        constructive+destructive pairs) and the k cycle (paper §2.2)."""
        self.turnarounds += 1
        if self.turnarounds % (2 * self.t) == 0:
            self.k = self.k + 1 if self.k < self.k_max else 1
        if self.turnarounds % 2:
            return  # only completed C+D pairs advance the span schedule
        self.pairs_at_span += 1
        allowed = self.p2 * self.span if self.span <= self.p1 else self.p2
        if self.pairs_at_span < allowed:
            return
        self.pairs_at_span = 0
        if self.span_increasing:
            self.span += 1
            if self.span > self.p2:
                self.span = self.p2
                self.span_increasing = False
        else:
            self.span -= 1
            if self.span < 1:
                self.span = 1
                self.span_increasing = True

    # ───────────────────────── constructive phase ──────────────────────

    def _constructive_phase(self):
        count_var = 0
        # C1: add best-valued fitting variables until none fits
        while True:
            addable = (self.x == 0) & self._fits()
            if not addable.any():
                break
            values = np.where(addable, self._penalized("constructive", count_var + 1), -np.inf)
            self._add(int(np.argmax(values)))
            count_var += 1

        # critical event at the boundary
        self._record_critical(self.x)
        self._consider(self.x)

        # C2: go infeasible by `span` further adds
        count_span = 0
        first_crossing = True
        while True:
            count_span += 1
            candidates = self.x == 0
            if count_span > self.span or not candidates.any():
                break
            values = np.where(candidates, self._penalized("constructive", count_var + 1), -np.inf)
            j = int(np.argmax(values))
            self._add(j)
            count_var += 1
            if first_crossing:
                self._trial_restore_feasibility(exclude=j)
                first_crossing = False

    def _trial_restore_feasibility(self, exclude: int):
        """Trial solution at the constructive critical event (paper §2):
        keep the add that crossed the boundary and look for a different
        variable to drop (scanned by increasing objective coefficient) that
        re-establishes feasibility."""
        selected = np.flatnonzero(self.x == 1)
        selected = selected[selected != exclude]
        for j in selected[np.argsort(self.profits[selected], kind="stable")]:
            if ((self.load - self.weights[:, j]) <= self.capacity).all():
                trial = self.x.copy()
                trial[j] = 0
                self._consider(trial)
                return

    # ───────────────────────── destructive phase ───────────────────────

    def _destructive_phase(self):
        count_var = 0
        # D1: drop worst-valued variables until feasibility is regained
        dropped_last = None
        while not self._feasible():
            candidates = self.x == 1
            if not candidates.any():
                break
            values = np.where(candidates, self._penalized("destructive", count_var + 1), np.inf)
            dropped_last = int(np.argmin(values))
            self._drop(dropped_last)
            count_var += 1

        # critical event on regaining feasibility
        self._record_critical(self.x)
        self._consider(self.x)
        self._trial_fill_up(exclude=dropped_last)

        # D2: continue `span` drops into the feasible region
        count_span = 0
        while True:
            count_span += 1
            candidates = self.x == 1
            if count_span > self.span or not candidates.any():
                break
            values = np.where(candidates, self._penalized("destructive", count_var + 1), np.inf)
            self._drop(int(np.argmin(values)))
            count_var += 1

    def _trial_fill_up(self, exclude):
        """Mirror trial at the destructive critical event: scan the items out
        of the knapsack (by decreasing objective coefficient, skipping the
        one just dropped) for the first that still fits."""
        outside = np.flatnonzero(self.x == 0)
        if exclude is not None:
            outside = outside[outside != exclude]
        order = outside[np.argsort(-self.profits[outside], kind="stable")]
        fits = self._fits()
        for j in order:
            if fits[j]:
                trial = self.x.copy()
                trial[j] = 1
                self._consider(trial)
                return
