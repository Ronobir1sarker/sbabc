"""
BGWO — the binary grey wolf optimizer of Luo & Zhao (2019), "A binary grey
wolf optimizer for the multidimensional knapsack problem", Applied Soft
Computing 83:105645. The second recent competitor, from a non-ABC swarm
family, and the strongest rival in the ABABC paper's own comparison tables.

Per iteration (their Algorithm 3):
  1. prey estimate (eq. 7): x^p = w_a*x_alpha + w_b*x_beta + w_d*x_delta
     + eps, with fitness-proportional leader weights and per-dimension
     Gaussian noise eps ~ N(0, sigma(t)), sigma(t) = exp(-100*t/G) — the
     leaders' estimate sharpens as the hunt progresses;
  2. every wolf builds a trial: y_j = x^p_j - r1*|x^p_j - x^k_j| with
     r1 ~ U[-2,2] (eq. 10; the enhanced GWO of their ref. [44], which fixes
     the original's bias toward the origin), binarized by r2 < phi(y_j)
     (eq. 11) with the V-shaped transform phi(y) = |tanh(y)| (their tuned
     choice; "sshape" gives the sigmoid of the bGWO-o baseline);
  3. the trial is made feasible by the pseudo-utility quick repair (their
     Algorithm 2 = Chu & Beasley's DROP/ADD with an early break, utilities
     from the LP relaxation's dual prices — shared with algorithms/cbga.py);
  4. differentiated position updating (eq. 12): a wolf moves greedily if the
     trial improves it; NON-leader wolves must move even to worse trials
     (diversification), while the three leaders only ever improve — so the
     best three distinct solutions always survive in the population.

The initial population is their Algorithm 1 elite generator: scan items in
descending pseudo-utility, adding each that still fits with probability
1/2 — feasible, high-quality and diverse by construction.

Framework notes:
  - MKP-specific (repair + elite init read the instance data); raises on
    other problems. Topology is unused; population_size = s (paper: 20).
  - One counted evaluation per trial (and per initial wolf), exactly the
    paper's NFE accounting (their budget: NFE = 100,000 -> G = NFE/s
    iterations). Repair is constraint bookkeeping (CBGA precedent).
  - `schedule_iterations` anchors sigma(t)'s decay (the paper's G); set it
    to the planned iteration count (default 5000 = 100k evaluations at
    s = 20).
  - Leaders are re-identified at the start of every iteration as the three
    best pairwise-distinct positions (the paper identifies them once and
    maintains them via an update cascade; re-identification is equivalent
    under eq. 12's invariant and avoids the cascade's archive-loss edge
    case).

Reproduction status (30 runs x 100k evaluations vs their Table 2,
results/bgwo_weing_validation.csv): success rates match or beat the paper
on 7 of 8 Weing instances (weing1/2/4/5/6/8 = 100%; weing3 = 97% vs their
87%). On weing7 this implementation reaches the optimum in ~10-20% of runs
vs the claimed 97% — the same instance where the ABABC reproduction falls
short of ITS paper (which inherited these very competitor numbers from
Luo & Zhao's Table 2, so the claim traces to a single source). A/B tests
of every documented reading (scalar vs per-dimension r1 and eps) and the
paper's "well-stated MKP" preprocessing (weing7 has no removable items)
do not recover it; weing7 is a degenerate plateau instance (71 distinct
weight columns among 105 items). Disclose the residual wherever BGWO
results are reported.
"""

from __future__ import annotations

import numpy as np

from algorithms.cbga import chu_beasley_repair, mkp_arrays, pseudo_utility_orders
from core.optimizer import Optimizer


class BinaryGWO(Optimizer):

    name = "BGWO"

    def __init__(
        self,
        num_wolves: int = 20,  # s in the paper
        transform: str = "vshape",  # "vshape" = |tanh| (tuned); "sshape" = bGWO-o's sigmoid
        schedule_iterations: int = 5000,  # G: anchors sigma(t) = exp(-100 t / G)
        **topology_kwargs,
    ):
        super().__init__(population_size=num_wolves, **topology_kwargs)
        if transform not in ("vshape", "sshape"):
            raise ValueError(f"transform must be 'vshape' or 'sshape', got {transform!r}")
        if schedule_iterations <= 0:
            raise ValueError(f"schedule_iterations must be positive, got {schedule_iterations}")
        self.num_wolves = int(num_wolves)
        self.transform = transform
        self.schedule_iterations = int(schedule_iterations)

    # ─────────────────────── problem-specific setup ────────────────────

    def set_problem(self, problem):
        super().set_problem(problem)
        if problem is None:
            return
        try:
            self.profits, self.weights, self.capacity = mkp_arrays(problem)
        except ValueError as error:
            raise ValueError(f"{self.name}: {error}") from None
        # pseudo-utilities from the LP dual prices (the paper solves the
        # relaxed dual with CPLEX; scipy/HiGHS here, same quantity)
        self._drop_order, self._add_order = pseudo_utility_orders(
            self.profits, self.weights, self.capacity
        )

    # ─────────────────────── Optimizer interface ───────────────────────

    def initialize(self, initial_population=None):
        if initial_population is not None:
            self.wolves = np.array(initial_population, dtype=np.int8)
            for wolf in self.wolves:  # external seeds may be infeasible;
                # -inf values would poison the fitness-weighted prey estimate
                chu_beasley_repair(wolf, self.weights, self.capacity,
                                   self._drop_order, self._add_order)
        else:
            self.wolves = self._elite_population(self.num_wolves)
        self.values, self.violations = self.evaluate(self.wolves)

        self.best_solution = None
        self._remember_best()

    def step(self, iteration):
        progress = min(iteration / self.schedule_iterations, 1.0)
        leaders = self._leader_indices()
        prey = self._estimate_prey(leaders, progress)

        for k in range(self.num_wolves):
            trial = self._trial_position(prey, self.wolves[k])
            chu_beasley_repair(trial, self.weights, self.capacity,
                               self._drop_order, self._add_order)
            value, violation = self.evaluate(trial)
            improves = self.is_first_better(value, self.values[k], violation, self.violations[k])
            if improves or k not in leaders:  # eq. 12: leaders only improve
                self.wolves[k] = trial
                self.values[k] = value
                self.violations[k] = violation
        self._remember_best()

    def get_best(self):
        return self.best_solution, self.best_value, self.best_violation

    # ──────────────────────────── mechanics ────────────────────────────

    def _elite_population(self, count):
        """Algorithm 1: each wolf adds items in descending pseudo-utility
        with probability 1/2 whenever they still fit — feasible and diverse."""
        wolves = np.zeros((count, self.dim), dtype=np.int8)
        for k in range(count):
            load = np.zeros(len(self.capacity))
            for j in self._add_order:  # descending pseudo-utility
                if self.rng.random() < 0.5 and (load + self.weights[:, j] <= self.capacity).all():
                    wolves[k, j] = 1
                    load += self.weights[:, j]
        return wolves

    def _leader_indices(self):
        """The wolves holding the three best pairwise-distinct positions
        (alpha, beta, delta), by Deb-consistent ordering. -inf values sort
        safely here (feasibility ranks first, then value)."""
        order = sorted(
            range(self.num_wolves),
            key=lambda k: (self.violations[k] > 0,
                           -self.values[k] if self.optimization_type == "max" else self.values[k],
                           self.violations[k]),
        )
        leaders, seen = [], set()
        for k in order:
            key = self.wolves[k].tobytes()
            if key not in seen:
                leaders.append(k)
                seen.add(key)
            if len(leaders) == 3:
                break
        return leaders

    def _estimate_prey(self, leaders, progress):
        """Eq. 7: fitness-weighted combination of the leader positions plus
        Gaussian noise whose scale decays as the hunt progresses."""
        positions = self.wolves[leaders].astype(np.float64)
        fitness = np.abs(self.values[leaders].astype(np.float64))
        total = fitness.sum()
        wolf_weights = fitness / total if total > 0 else np.full(len(leaders), 1.0 / len(leaders))
        sigma = np.exp(-100.0 * progress)
        return wolf_weights @ positions + self.rng.normal(0.0, sigma, size=self.dim)

    def _trial_position(self, prey, wolf):
        """Eqs. 10-11: move toward the prey estimate, then binarize through
        the transform function."""
        r1 = self.rng.uniform(-2.0, 2.0, size=self.dim)
        y = prey - r1 * np.abs(prey - wolf)
        if self.transform == "vshape":
            phi = np.abs(np.tanh(y))
        else:  # bGWO-o's S-shaped sigmoid
            phi = 1.0 / (1.0 + np.exp(-10.0 * (y - 0.5)))
        return (self.rng.random(self.dim) < phi).astype(np.int8)

    def _remember_best(self):
        idx = self.deb_best_index(self.values, self.violations)
        if self.best_solution is None or self.is_first_better(
            self.values[idx], self.best_value, self.violations[idx], self.best_violation
        ):
            self.best_solution = self.wolves[idx].copy()
            self.best_value = float(self.values[idx])
            self.best_violation = float(self.violations[idx])
