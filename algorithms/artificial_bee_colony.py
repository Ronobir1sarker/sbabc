"""
Artificial Bee Colony (Karaboga) — the general ABC skeleton every ABC
variant in this repo builds on, with Deb's rule for constraints.

One iteration (`step`) is the paper's cycle:
  employed bees  — each food source is perturbed once; greedy selection
  onlooker bees  — food sources drawn by fitness-proportional roulette, then
                   perturbed; greedy selection
  scout bees     — sources whose trial counter exceeded `limit` are abandoned
                   and re-initialized at random

A variant customizes two orthogonal things:

  distort_operator(source, neighbor)  how a candidate food source is built
                                      (SetBasedABC, BitABC, GeneticABC)
  random_food_sources(n) + decode(x)  what a food source *is*: by default a
                                      binary vector evaluated as-is; AMABC
                                      stores 4 angle-modulation coefficients
                                      and decodes them to a bit string

The default distort operator is a minimal binary adaptation of the paper's
eq. (6): copy one random dimension from a random topological neighbor, or
its complement.
"""

from __future__ import annotations

import numpy as np

from core.optimizer import Optimizer


class ArtificialBeeColony(Optimizer):

    name = "ABC"

    def __init__(
        self,
        num_food_sources: int = 25,
        limit: int = 100,  # the paper's abandonment criterion ("limit")
        fitness_gap: float = 1.0,  # min roulette fitness of a feasible source; keeps it above every infeasible one
        tie_break: str = "candidate",  # who wins a Deb's-rule tie in greedy selection (see _perturb_food_source)
        **topology_kwargs,
    ):
        super().__init__(population_size=num_food_sources, **topology_kwargs)
        if tie_break not in ("candidate", "current"):
            raise ValueError(f"tie_break must be 'candidate' or 'current', got {tie_break!r}")
        self.num_food_sources = num_food_sources
        self.limit = limit
        self.fitness_gap = fitness_gap
        self.tie_break = tie_break

    # ───────────────────── representation hooks ────────────────────────
    #
    # A food source needn't be a solution vector itself: AMABC, for example,
    # evolves 4 angle-modulation coefficients per bee. Variants with such an
    # indirect representation override these two hooks; everything else
    # (phases, trials, roulette, greedy selection) is representation-blind.

    def random_food_sources(self, n: int) -> np.ndarray:
        """`n` random food sources, one per row. Default: random solutions
        (bit vectors, or whatever the problem's init_sampler produces)."""
        return self.random_solutions(n)

    def decode(self, food_source: np.ndarray) -> np.ndarray:
        """Map a food source (or a batch of them) to a solution the problem
        can evaluate. Default: food sources are solutions already.
        """
        return food_source

    def evaluate_food_source(self, food_source):
        """Counted evaluation of a food source (or batch) via `decode`."""
        return self.evaluate(self.decode(food_source))

    # ─────────────────────── Optimizer interface ───────────────────────

    def initialize(self, initial_population=None):
        if initial_population is None:
            self.food_sources = self.random_food_sources(self.num_food_sources)
        else:
            # copy: runs must not share (or mutate the caller's) population
            self.food_sources = np.array(initial_population)

        self.food_values, self.food_violations = self.evaluate_food_source(self.food_sources)
        self.trials = np.zeros(self.num_food_sources, dtype=np.int64)

        self.best_solution = None
        self._remember_best()

    def step(self, iteration):
        self._employed_bee_phase()
        self._onlooker_bee_phase()
        self._scout_bee_phase()
        self._remember_best()

    def get_best(self):
        if self.best_solution is None:
            return None, self.best_value, self.best_violation
        return self.decode(self.best_solution), self.best_value, self.best_violation

    # ──────────────────────────── bee phases ───────────────────────────

    def _employed_bee_phase(self):
        for n in range(self.num_food_sources):
            self._perturb_food_source(n)

    def _onlooker_bee_phase(self):
        N = self.num_food_sources
        chosen = self.rng.choice(N, size=N, p=self._selection_probabilities())
        for n in chosen:
            self._perturb_food_source(n)

    def _scout_bee_phase(self):
        for n in range(self.num_food_sources):
            if self.trials[n] > self.limit:
                self._reinitialize_food_source(n)

    def _reinitialize_food_source(self, n):
        self.food_sources[n, :] = self.random_food_sources(1)[0]
        self.food_values[n], self.food_violations[n] = self.evaluate_food_source(self.food_sources[n, :])
        self.trials[n] = 0

    def _perturb_food_source(self, n):
        """Generate a candidate near source n and apply greedy selection.

        With tie_break="candidate" (default, the setting behind the paper's
        results) ties go to the candidate: a neutral move on a plateau still
        counts as movement, so the trial counter only grows while the source
        is strictly better than everything found near it. "current" is the
        textbook ABC rule — a candidate must be strictly better to replace,
        so plateau steps burn trials and hasten abandonment.
        """
        candidate, value, violation = self.distort_operator(n, self.random_neighbor(n))

        if self.tie_break == "candidate":
            keep_current = self.is_first_better(self.food_values[n], value, self.food_violations[n], violation)
        else:
            keep_current = not self.is_first_better(value, self.food_values[n], violation, self.food_violations[n])

        if keep_current:
            self.trials[n] += 1
        else:
            self.food_sources[n, :] = candidate
            self.food_values[n] = value
            self.food_violations[n] = violation
            self.trials[n] = 0

    def distort_operator(self, source, neighbor):
        """Binary adaptation of the paper's eq. (6): move one random dimension
        of the source toward (or away from) the neighbor. Returns (candidate,
        value, violation). Override this to define a new ABC variant.
        """
        candidate = self.food_sources[source].copy()
        j = self.rng.integers(self.dim)
        neighbor_bit = self.food_sources[neighbor, j]
        candidate[j] = neighbor_bit if self.rng.random() < 0.5 else 1 - neighbor_bit
        value, violation = self.evaluate_food_source(candidate)
        return candidate, value, violation

    # ────────────────────── fitness and best-so-far ────────────────────

    def _selection_probabilities(self):
        """Roulette-wheel probabilities (paper eq. 7-8), adapted for constraints:
        feasible sources score `objective shifted to >= fitness_gap`, infeasible
        sources score 1/(1+violation) ∈ (0, 1], so any feasible source outranks
        every infeasible one when fitness_gap >= 1.
        """
        feasible = self.food_violations <= 0
        scores = np.empty(self.num_food_sources, dtype=np.float64)

        if feasible.any():
            values = self.food_values[feasible]
            if self.optimization_type == "max":
                scores[feasible] = values - values.min() + self.fitness_gap
            else:
                scores[feasible] = values.max() - values + self.fitness_gap
        scores[~feasible] = 1.0 / (1.0 + self.food_violations[~feasible])

        total = scores.sum()
        if total <= 0:
            return np.full(self.num_food_sources, 1.0 / self.num_food_sources)
        return scores / total

    def _best_food_source_index(self):
        return self.deb_best_index(self.food_values, self.food_violations)

    def _remember_best(self):
        """Memorize the best food source seen this run (paper: 'memorize the
        best solution achieved so far'). Copies, so later in-place updates of
        the population can never corrupt the memorized solution.
        """
        idx = self._best_food_source_index()
        if self.best_solution is None or self.is_first_better(
            self.food_values[idx], self.best_value, self.food_violations[idx], self.best_violation
        ):
            self.best_solution = self.food_sources[idx].copy()
            self.best_value = float(self.food_values[idx])
            self.best_violation = float(self.food_violations[idx])
