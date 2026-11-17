"""
Problem — the single interface every optimization problem exposes.

An algorithm never sees problem internals (weights, datasets, ...); it only
calls ``problem.evaluate(x)`` and reads the metadata fields. To add a new
problem, build a ``Problem`` with an evaluate callable — see
``problems/mkp/problem.py`` for a complete example.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

# evaluate(x) -> (value, violation)
#   x         : binary array of shape (dim,) or a batch (..., dim)
#   value     : objective value(s); by convention -inf for infeasible
#               solutions of maximization problems
#   violation : total constraint violation(s); 0 means feasible
EvaluateFn = Callable[[np.ndarray], tuple]


@dataclass
class Problem:
    name: str
    dim: int
    evaluate: EvaluateFn
    known_optimum: Optional[float] = None
    optimization_type: str = "max"  # "max" or "min"
    extras: dict = field(default_factory=dict)  # anything problem-specific

    def __post_init__(self):
        if self.optimization_type not in ("max", "min"):
            raise ValueError(
                f"optimization_type must be 'max' or 'min', got {self.optimization_type!r}"
            )

    def error(self, value):
        """Gap between a value and the known optimum (non-negative when optimal is known best)."""
        if self.known_optimum is None:
            raise ValueError(f"Problem {self.name!r} has no known optimum")
        if self.optimization_type == "max":
            return self.known_optimum - value
        return value - self.known_optimum

    def error_percentage(self, value):
        """``error`` as a percentage of the known optimum."""
        if self.known_optimum == 0:
            raise ZeroDivisionError(f"Known optimum of {self.name!r} is 0")
        return 100.0 * self.error(value) / abs(self.known_optimum)


def penalized(problem: Problem, factor: float) -> Problem:
    """A penalty-method view of `problem`: violations are folded into the
    objective (subtracted for max, added for min) and reported as 0, so
    Deb's rule degenerates to plain objective comparison. Lets any algorithm
    run as a penalty method without an optimizer switch — the instrument for
    the Deb-vs-penalty constraint-handling comparison.

    Solutions the base problem scores as -inf/+inf when infeasible would
    erase the penalty gradient, so their objective is rebuilt as
    ``penalty from violation only`` measured from 0.
    """
    if factor <= 0:
        raise ValueError(f"penalty factor must be positive, got {factor}")
    sign = -1.0 if problem.optimization_type == "max" else 1.0
    base_evaluate = problem.evaluate

    def evaluate(x):
        value, violation = base_evaluate(x)
        value = np.asarray(value, dtype=np.float64)
        violation = np.asarray(violation)
        penalized_value = np.where(
            np.isfinite(value), value + sign * factor * violation, sign * factor * violation
        )
        zeros = np.zeros_like(violation)
        if penalized_value.ndim == 0:
            return float(penalized_value), 0.0
        return penalized_value, zeros

    return Problem(
        name=f"{problem.name}_penalty{factor:g}",
        dim=problem.dim,
        evaluate=evaluate,
        known_optimum=problem.known_optimum,
        optimization_type=problem.optimization_type,
        extras={**problem.extras, "penalty_factor": factor, "penalized_from": problem.name},
    )
