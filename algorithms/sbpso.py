"""
SBPSO — Set-Based Particle Swarm Optimization (Langeveld & Engelbrecht,
Swarm Intelligence 6:297-342, 2012), on the same vector encoding as
sets/operators.py.

Each particle keeps a position X (a set, as a bit vector), a personal best
Y and a neighborhood best Y_hat (best personal best in its topological
neighborhood, itself included). Velocities carry no momentum — each
iteration builds a fresh velocity from four components (paper's update):

    V = c1*r1 (x) (Y ⊖ X)        cognitive: toward the personal best
      ⊕ c2*r2 (x) (Y_hat ⊖ X)    social: toward the neighborhood best
      ⊕ c3*r3 ⊙+_k A             add elements outside X ∪ Y ∪ Y_hat,
                                 chosen by k-tournament on marginal value
      ⊕ c4*r4 ⊙-  S              remove elements common to X ∩ Y ∩ Y_hat

then X ← X ⊞ V. The k-tournament evaluations are counted, so function-call
budgets are comparable with the ABC family.

c1 and c2 must lie in [0, 1] (the ⊗ operator keeps a fraction of a
velocity); c3 and c4 may exceed 1 (they are elements counts). Defaults are
the paper's tuned values for the small MKP problems.
"""

from __future__ import annotations

import numpy as np

from core.optimizer import Optimizer
from sets import operators as ops


class SetBasedPSO(Optimizer):

    name = "SBPSO"

    def __init__(
        self,
        num_particles: int = 25,
        c1: float = 0.5156,  # cognitive fraction, in [0, 1]
        c2: float = 0.4531,  # social fraction, in [0, 1]
        c3: float = 1.8359,  # how many outside elements to try to add
        c4: float = 2.2578,  # how many common elements to remove
        k: int = 7,  # tournament size for the add operator
        **topology_kwargs,
    ):
        super().__init__(population_size=num_particles, **topology_kwargs)
        for name, value in (("c1", c1), ("c2", c2)):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1] (it scales a velocity), got {value}")
        self.num_particles = num_particles
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.c4 = c4
        self.k = int(k)

    # ─────────────────────── Optimizer interface ───────────────────────

    def initialize(self, initial_population=None):
        N = self.num_particles
        if initial_population is None:
            self.positions = self.random_solutions(N)
        else:
            self.positions = np.array(initial_population, dtype=np.int8)

        self.values, self.violations = self.evaluate(self.positions)

        self.pbest = self.positions.copy()
        self.pbest_values = self.values.copy()
        self.pbest_violations = self.violations.copy()
        # neighborhood bests start at the personal bests and only improve
        self.nbest = self.positions.copy()
        self.nbest_values = self.values.copy()
        self.nbest_violations = self.violations.copy()
        self._update_neighborhood_bests()

        self.best_solution = None
        self._remember_best()

    def step(self, iteration):
        # Move first, then fold the fresh evaluations into the bests. The
        # guides a move sees are identical to updating the bests at the start
        # of the next iteration (they always reflect the previous evaluation),
        # but this order reports the final iteration's evaluations instead of
        # discarding them (they are paid for either way).
        for i in range(self.num_particles):
            self.positions[i] = self._move(i)
        self.values, self.violations = self.evaluate(self.positions)
        self._update_personal_bests()
        self._update_neighborhood_bests()
        self._remember_best()

    def get_best(self):
        return self.best_solution, self.best_value, self.best_violation

    # ─────────────────────────── PSO memory ────────────────────────────

    def _update_personal_bests(self):
        for i in range(self.num_particles):
            if self.is_first_better(
                self.values[i], self.pbest_values[i], self.violations[i], self.pbest_violations[i]
            ):
                self.pbest[i] = self.positions[i].copy()
                self.pbest_values[i] = self.values[i]
                self.pbest_violations[i] = self.violations[i]

    def _update_neighborhood_bests(self):
        for i in range(self.num_particles):
            for l in (i, *self.neighbor_indices[i]):
                if self.is_first_better(
                    self.pbest_values[l], self.nbest_values[i],
                    self.pbest_violations[l], self.nbest_violations[i],
                ):
                    self.nbest[i] = self.pbest[l].copy()
                    self.nbest_values[i] = self.pbest_values[l]
                    self.nbest_violations[i] = self.pbest_violations[l]

    def _remember_best(self):
        for i in range(self.num_particles):
            if self.best_solution is None or self.is_first_better(
                self.pbest_values[i], self.best_value, self.pbest_violations[i], self.best_violation
            ):
                self.best_solution = self.pbest[i].copy()
                self.best_value = float(self.pbest_values[i])
                self.best_violation = float(self.pbest_violations[i])

    # ──────────────────────── velocity/position ────────────────────────

    def _move(self, i):
        x, y, y_hat = self.positions[i], self.pbest[i], self.nbest[i]
        common = x & y & y_hat  # S: elements every guide agrees on
        outside = 1 - (x | y | y_hat)  # A: elements no guide contains

        r1, r2, r3, r4 = self.rng.random(4)
        velocity = ops.add_velocities(
            ops.scalar_mul(self.rng, self.c1 * r1, y - x),
            ops.scalar_mul(self.rng, self.c2 * r2, y_hat - x),
            ops.k_tournament_add(self.rng, self.c3 * r3, outside, x, self.evaluate, self.k, self.optimization_type),
            ops.remove_elements(self.rng, self.c4 * r4, common),
        )
        return ops.apply_velocity(x, velocity)
