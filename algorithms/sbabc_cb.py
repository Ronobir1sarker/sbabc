"""
SBABC-CB — the generic SBABC plus the Chu & Beasley greedy repair.

A demonstration arm, not a new proposed algorithm: it answers "is the gap
between SBABC and the repair-based MKP specialists attributable to domain
knowledge or to the search engine?" by adding exactly one ingredient — the
same repair module CBGA and ABABC use (algorithms/cbga.py) — to the tuned
SBABC configuration, changing nothing else. No separate tuning: holding
every parameter fixed is the point, otherwise the comparison would no
longer isolate knowledge as the single variable.

Where the repair applies (mirroring CBGA, which repairs every child before
its one counted evaluation):

  - every complete candidate the distort operator builds, before its
    counted evaluation;
  - every random food source (initialization and scout re-seeding).

The k-tournament's marginal probes are NOT repaired: they are ranking
devices inside the addition operator, not solutions, and repairing them
would smuggle knowledge into the generic operator itself rather than bolt
it on at the solution boundary.

FE accounting matches CBGA: repair is constraint bookkeeping on
precomputed pseudo-utility orders (no objective calls), so a repaired
candidate still costs exactly one evaluation. MKP-only by construction —
set_problem raises on problems that don't expose profits/weights/capacity.
"""

from __future__ import annotations

import numpy as np

from algorithms.cbga import chu_beasley_repair, mkp_arrays, pseudo_utility_orders
from algorithms.sbabc import SetBasedABC
from sets import operators as ops


class SetBasedABCWithRepair(SetBasedABC):

    name = "SBABC-CB"

    def __init__(self, distort_operator_version: int = 5, **sbabc_kwargs):
        if distort_operator_version != 5:
            raise ValueError(
                "SBABC-CB only implements the spec-driven distort operator "
                f"(version 5), got version {distort_operator_version}"
            )
        super().__init__(distort_operator_version=5, **sbabc_kwargs)

    # ─────────────────────── problem-specific setup ────────────────────

    def set_problem(self, problem):
        super().set_problem(problem)
        if problem is None:
            return
        try:
            self.profits, self.weights, self.capacity = mkp_arrays(problem)
        except ValueError as error:
            raise ValueError(f"{self.name}: {error}") from None
        self._drop_order, self._add_order = pseudo_utility_orders(
            self.profits, self.weights, self.capacity
        )

    def _repair(self, x):
        chu_beasley_repair(x, self.weights, self.capacity, self._drop_order, self._add_order)

    # ─────────────────────── repaired hook points ──────────────────────

    def random_food_sources(self, n: int) -> np.ndarray:
        sources = super().random_food_sources(n)
        for row in sources:
            self._repair(row)
        return sources

    def _distort_from_specs(self, f1, f2):
        """SetBasedABC._distort_from_specs with one insertion: each complete
        candidate is repaired before its counted evaluation (keep the random-
        number consumption order in lockstep with the parent)."""
        common = f1 * f2
        rare = 1 - np.clip(f1 + f2, 0, 1)
        r1, r2, r3 = self.rng.random(3)

        v_toward = (
            ops.scalar_mul(self.rng, r1 * self.beta1, f2 - f1)
            if any("m" in s for s in self.candidate_specs) else None
        )
        v_drop = (
            ops.remove_elements(self.rng, r2 * self.beta2, common)
            if any("d" in s for s in self.candidate_specs) else None
        )
        v_add_shared = (
            ops.k_tournament_add(self.rng, r3 * self.beta3, rare, f1, self.evaluate, self.k, self.optimization_type)
            if self.shared_add_velocity and any("a" in s for s in self.candidate_specs) else None
        )

        candidates = []
        for spec in self.candidate_specs:
            velocities = [v for flag, v in (("m", v_toward), ("d", v_drop)) if flag in spec]
            y = ops.apply_velocity(f1, ops.add_velocities(*velocities)) if velocities else f1
            if "a" in spec:
                v_add = (
                    v_add_shared
                    if v_add_shared is not None
                    else ops.k_tournament_add(
                        self.rng, r3 * self.beta3, rare, y, self.evaluate, self.k, self.optimization_type
                    )
                )
                y = ops.apply_velocity(y, v_add)
            y = np.array(y, dtype=np.int8)  # own the buffer before in-place repair
            self._repair(y)
            candidates.append((y, *self.evaluate(y)))
        return self._better_candidate(*candidates)
