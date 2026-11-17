"""
BitABC — binary ABC using bitwise operations (Jia, Duan & Khan, 2014,
Computers & Industrial Engineering 76:360-365).

The paper replaces the real-arithmetic perturbation of ABC's eq. (3) with a
bitwise update applied to every bit at once (their eq. 5):

    v = x XOR (phi AND (x OR x_k))

where x_k is a random neighbor and each bit of phi is 1 with probability
`r` (their eq. 6; r = 0.5 performed best in the paper's study). Where
phi = 1 the candidate flips set bits off and copies the neighbor's set
bits on; where phi = 0 it keeps x — so small r keeps the candidate close
to x, exactly like a small |phi| in continuous ABC.

(The paper's Fig. 2 worked example contradicts its own eq. (5); the
equation is the normative form and is what is implemented here.)
"""

from __future__ import annotations

import numpy as np

from algorithms.artificial_bee_colony import ArtificialBeeColony


class BitABC(ArtificialBeeColony):

    name = "BitABC"

    def __init__(
        self,
        r: float = 0.5,  # per-bit probability that phi = 1 (paper's eq. 6)
        **abc_kwargs,
    ):
        super().__init__(**abc_kwargs)
        if not 0.0 <= r <= 1.0:
            raise ValueError(f"r must be in [0, 1], got {r}")
        self.r = r

    def distort_operator(self, source, neighbor):
        x = self.food_sources[source]
        x_k = self.food_sources[neighbor]
        phi = (self.rng.random(self.dim) < self.r).astype(np.int8)

        candidate = x ^ (phi & (x | x_k))
        value, violation = self.evaluate_food_source(candidate)
        return candidate, value, violation
