"""
The repaired family — SBPSO-CB, GBABC-CB and BitABC-CB.

Together with SBABC-CB (algorithms/sbabc_cb.py) these complete the
knowledge-isolation design of the paper (its Section 7.4): every problem-independent
engine that admits the construction receives the specialists' Chu & Beasley
repair and NOTHING else — the engine's tuned configuration is unchanged and
deliberately not re-tuned, so problem knowledge is the single manipulated
variable. Four engines under two knowledge conditions, seed-paired with the
main campaign. AMABC admits no repaired variant: its solutions are decoded
from angle-modulation coefficients, and a repaired solution cannot be
written back into them (a structural exclusion, disclosed in the paper).

Where the repair applies, mirroring CBGA (which repairs every child before
its one counted evaluation) and SBABC-CB:

  - every candidate solution, before its counted evaluation — for GBABC-CB
    that is each of the 20 offspring (CBGA's per-child precedent), for
    BitABC-CB the bitwise candidate, for SBPSO-CB the moved position;
  - every random solution (initialization, and scout re-seeding for the
    ABC engines).

SBPSO-CB's k-tournament probes (inside the ⊙+k addition operator) are NOT
repaired, exactly as in SBABC-CB: they are ranking devices inside the
generic operator, not solutions, and repairing them would smuggle knowledge
into the operator itself rather than bolt it on at the solution boundary.

FE accounting matches CBGA: the repair is constraint bookkeeping on
precomputed pseudo-utility orders (no objective calls), so a repaired
candidate still costs exactly one evaluation. The repair itself consumes no
randomness, so every RNG stream stays in lockstep with the unrepaired
engine. MKP-only by construction — set_problem raises on problems that do
not expose profits/weights/capacity.
"""

from __future__ import annotations

import numpy as np

from algorithms.bitabc import BitABC
from algorithms.cbga import chu_beasley_repair, mkp_arrays, pseudo_utility_orders
from algorithms.gbabc import GeneticABC
from algorithms.sbpso import SetBasedPSO


class ChuBeasleyRepairMixin:
    """set_problem precomputation + the in-place repair, shared by the family."""

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


class SetBasedPSOWithRepair(ChuBeasleyRepairMixin, SetBasedPSO):

    name = "SBPSO-CB"

    def random_solutions(self, n: int) -> np.ndarray:
        positions = super().random_solutions(n)
        for row in positions:
            self._repair(row)
        return positions

    def _move(self, i):
        """SBPSO's move unchanged; the resulting position is repaired before
        step() pays its counted evaluation. Tournament probes inside the move
        stay unrepaired (module docstring)."""
        position = np.array(super()._move(i), dtype=np.int8)  # own the buffer
        self._repair(position)
        return position


class GeneticABCWithRepair(ChuBeasleyRepairMixin, GeneticABC):

    name = "GBABC-CB"

    def random_food_sources(self, n: int) -> np.ndarray:
        sources = super().random_food_sources(n)
        for row in sources:
            self._repair(row)
        return sources

    def distort_operator(self, source, neighbor):
        """GeneticABC.distort_operator with one insertion: each of the 20
        offspring is repaired before its counted evaluation (the CBGA
        per-child precedent). Random-number consumption is identical to the
        parent — the repair uses none."""
        if self.best_parent == "gbest" and self.best_solution is not None:
            best = self.best_solution
        else:
            best = self.food_sources[self._best_food_source_index()]

        parents = np.stack(
            [
                self.food_sources[source],
                self.food_sources[neighbor],
                self.food_sources[self.random_neighbor(source)],
                np.zeros(self.dim, dtype=np.int8),
                best,
            ]
        )

        children = []
        for a, b in enumerate(self.rng.permutation(len(parents))):
            children.extend(self._two_point_crossover(parents[a], parents[b]))
        for child in children:
            self._give_ones_if_empty(child)
        grandchildren = [self._swap_one_and_zero(child) for child in children]

        candidates = np.stack(children + grandchildren)
        for row in candidates:
            self._repair(row)
        values, violations = self.evaluate_food_source(candidates)

        best = 0
        for i in range(1, len(candidates)):
            if self.is_first_better(values[i], values[best], violations[i], violations[best]):
                best = i
        return candidates[best], values[best], violations[best]


class BitABCWithRepair(ChuBeasleyRepairMixin, BitABC):

    name = "BitABC-CB"

    def random_food_sources(self, n: int) -> np.ndarray:
        sources = super().random_food_sources(n)
        for row in sources:
            self._repair(row)
        return sources

    def distort_operator(self, source, neighbor):
        """BitABC.distort_operator with one insertion: the candidate is
        repaired before its counted evaluation."""
        x = self.food_sources[source]
        x_k = self.food_sources[neighbor]
        phi = (self.rng.random(self.dim) < self.r).astype(np.int8)

        candidate = x ^ (phi & (x | x_k))
        self._repair(candidate)
        value, violation = self.evaluate_food_source(candidate)
        return candidate, value, violation
