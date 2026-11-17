"""
AMABC — angle-modulated ABC (Pampara & Engelbrecht, 2011).

Instead of evolving bit strings, every food source is a 4-dimensional
continuous coefficient vector theta = (a, b, c, d) of the bit-generating
function

    g(y) = sin(2*pi*(y - a) * b * cos(2*pi*(y - a) * c)) + d

The D-bit solution is obtained by sampling g at D evenly spaced points
y_j = (-D + 1 + 2j) / (D - 1), j = 0..D-1 (spanning [-1, 1]) and setting
bit j to 1 iff g(y_j) > 0. The search itself is the *standard continuous
ABC* on theta: the perturbation is Karaboga's eq. (6) on one random
coefficient, nu_j = theta_j + phi * (theta_j - theta_kj), phi ~ U[-1, 1].

So only the representation hooks differ from the base class: food sources
live in R^4 and `decode` maps them to bit strings for evaluation. The
coefficient initialization ranges follow the SBABC manuscript's parameter
table: a in [-2, 2], b in [0, 10], c in [0, 10], d in [-1, 1]. Coefficients
are unconstrained during the search (the optimization is over all of R^4);
the ranges apply to initial and scout-reinitialized sources only.
"""

from __future__ import annotations

import numpy as np

from algorithms.artificial_bee_colony import ArtificialBeeColony


class AngleModulatedABC(ArtificialBeeColony):

    name = "AMABC"

    def __init__(
        self,
        a_range: tuple[float, float] = (-2.0, 2.0),  # horizontal shift of g
        b_range: tuple[float, float] = (0.0, 10.0),  # sine frequency
        c_range: tuple[float, float] = (0.0, 10.0),  # cosine frequency
        d_range: tuple[float, float] = (-1.0, 1.0),  # vertical shift
        **abc_kwargs,
    ):
        super().__init__(**abc_kwargs)
        self.a_range = tuple(a_range)
        self.b_range = tuple(b_range)
        self.c_range = tuple(c_range)
        self.d_range = tuple(d_range)

    # ───────────────────── representation hooks ────────────────────────

    def random_food_sources(self, n):
        ranges = np.array([self.a_range, self.b_range, self.c_range, self.d_range])
        return self.rng.uniform(ranges[:, 0], ranges[:, 1], size=(n, 4))

    def decode(self, food_source):
        """(..., 4) coefficient vectors → (..., dim) bit strings."""
        theta = np.asarray(food_source, dtype=np.float64)
        a = theta[..., 0, None]
        b = theta[..., 1, None]
        c = theta[..., 2, None]
        d = theta[..., 3, None]

        D = self.dim
        y = np.zeros(1) if D == 1 else (-D + 1 + 2 * np.arange(D)) / (D - 1)
        g = np.sin(2 * np.pi * (y - a) * b * np.cos(2 * np.pi * (y - a) * c)) + d
        return (g > 0).astype(np.int8)

    # ──────────────────────── distort operator ─────────────────────────

    def distort_operator(self, source, neighbor):
        """Standard continuous ABC move (Karaboga eq. 6) on one coefficient."""
        candidate = self.food_sources[source].copy()
        j = self.rng.integers(candidate.shape[-1])
        phi = self.rng.uniform(-1.0, 1.0)
        candidate[j] += phi * (candidate[j] - self.food_sources[neighbor, j])

        value, violation = self.evaluate_food_source(candidate)
        return candidate, value, violation
