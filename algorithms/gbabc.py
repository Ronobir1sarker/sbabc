"""
GBABC — binary ABC based on genetic operators (Ozturk, Hancer & Karaboga,
2015, Information Sciences 297:154-170).

The paper replaces ABC's perturbation with a "genetic neighbourhood
generator" (their Section 3):

  1. parents = {X_i, two random neighbors, the all-zeros vector, and the
     population-best food source}
  2. match the 5 parents by a random bijection and apply two-point
     crossover to each matched pair  →  10 children
  3. any all-zero child gets `threshold` (the paper's Th) random bits set
     to 1, so no candidate is the empty solution
  4. each child spawns a grandchild by swapping the values of a random
     1-position and a random 0-position  →  10 grandchildren
  5. the candidate is the best of the 20 (here by Deb's rule, matching the
     constrained fitness handling used throughout this framework)

Note the cost: 20 objective evaluations per bee per phase, all counted.
"""

from __future__ import annotations

import numpy as np

from algorithms.artificial_bee_colony import ArtificialBeeColony


class GeneticABC(ArtificialBeeColony):

    name = "GBABC"

    def __init__(
        self,
        threshold: int = 3,  # the paper's Th: ones given to an all-zero child
        best_parent: str = "population",  # which "best" joins the parents (see distort_operator)
        **abc_kwargs,
    ):
        super().__init__(**abc_kwargs)
        if threshold < 1:
            raise ValueError(f"threshold must be >= 1, got {threshold}")
        if best_parent not in ("population", "gbest"):
            raise ValueError(f"best_parent must be 'population' or 'gbest', got {best_parent!r}")
        self.threshold = int(threshold)
        self.best_parent = best_parent

    def distort_operator(self, source, neighbor):
        # The paper's 5th parent is gbest, the best solution found so far;
        # "population" (the best CURRENT food source, which abandonment can
        # roll back) is what the original experiments ran and stays the
        # default so published results remain reproducible.
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
        values, violations = self.evaluate_food_source(candidates)

        best = 0
        for i in range(1, len(candidates)):
            if self.is_first_better(values[i], values[best], violations[i], violations[best]):
                best = i
        return candidates[best], values[best], violations[best]

    def _two_point_crossover(self, p1, p2):
        """Exchange the segment between two random positions (inclusive)."""
        lo, hi = np.sort(self.rng.integers(0, self.dim, size=2))
        c1, c2 = p1.copy(), p2.copy()
        c1[lo : hi + 1] = p2[lo : hi + 1]
        c2[lo : hi + 1] = p1[lo : hi + 1]
        return c1, c2

    def _give_ones_if_empty(self, child):
        if not child.any():
            positions = self.rng.choice(self.dim, size=min(self.threshold, self.dim), replace=False)
            child[positions] = 1

    def _swap_one_and_zero(self, child):
        """Grandchild: a random 1-position and a random 0-position exchange
        values (paper's swap operator). All-ones children have no 0 to swap
        and are returned unchanged.
        """
        ones = np.flatnonzero(child == 1)
        zeros = np.flatnonzero(child == 0)
        grandchild = child.copy()
        if len(ones) and len(zeros):
            grandchild[self.rng.choice(ones)] = 0
            grandchild[self.rng.choice(zeros)] = 1
        return grandchild
