"""
Set-based operators from Langeveld & Engelbrecht's Set-Based PSO paper
(Swarm Intelligence 6:297-342, 2012), on a fixed-length vector encoding:

  position X : binary vector, X[i] = 1  ⇔ element i is in the set
  velocity V : vector in {-1, 0, 1}, +1 = add element i, -1 = remove element i

All functions are pure: randomness comes in through an explicit `rng`
(a numpy Generator) and objective information through an explicit
`evaluate` callable, so the operators can be unit-tested and reused by any
algorithm (SBPSO, SBABC, set-based GWO, ...).

Convention: `evaluate(x) -> (value, violation)`, feasible ⇔ violation <= 0.
Solutions are compared with Deb's rule, so the operators work for
maximization and minimization alike (pass `optimization_type` where the
objective matters); the MKP benchmarks additionally report value = -inf
when infeasible, which Deb's rule subsumes.
"""

from __future__ import annotations

import math

import numpy as np


def scalar_mul(rng, eta: float, v: np.ndarray) -> np.ndarray:
    """η ⊗ V (paper eq. 13-context): keep ⌊η·|V|⌋ random operations of V, zero the rest."""
    if not 0.0 <= eta <= 1.0:
        raise ValueError(f"eta must be in [0, 1], got {eta}")

    nonzero = np.flatnonzero(v)
    keep = math.floor(eta * len(nonzero))
    result = v.copy()
    drop = rng.choice(nonzero, len(nonzero) - keep, replace=False)
    result[drop] = 0
    return result


def n_fun(rng, beta: float, set_size: int) -> int:
    """N_{β,S} (paper eq. 15): how many elements a β-scaled operation touches.

    At least ⌊β⌋, one more with probability equal to the fractional part of β,
    capped at the set size.
    """
    floor = math.floor(beta)
    extra = 1 if rng.random() < beta - floor else 0
    return min(set_size, floor + extra)


def remove_elements(rng, beta: float, s: np.ndarray) -> np.ndarray:
    """β ⊙- S (paper eq. 14): a velocity removing N_{β,S} random elements of S.

    Removal is deliberately random (no objective information): the paper argues
    these elements are all "good" — they sit in the position, personal best and
    neighborhood best alike — so paying evaluations to pick among them is not
    worth it. This operator exists for exploration.
    """
    members = np.flatnonzero(s)
    n_remove = n_fun(rng, beta, len(members))
    v = np.zeros_like(s)
    v[rng.choice(members, n_remove, replace=False)] = -1
    return v


def k_tournament_add(
    rng, beta: float, available: np.ndarray, x: np.ndarray, evaluate, k: int, optimization_type: str = "max"
) -> np.ndarray:
    """β ⊙+_k A (paper eq. 16, algorithm 2): a velocity adding N_{β,A} elements.

    Each added element wins a k-tournament on its *marginal* objective value
    f(X ∪ {e}) — good additions are rare, so here the paper does spend k extra
    evaluations per element. Tournament entrants are drawn from A with
    replacement, and rounds may pick the same winner twice (adding is idempotent).

    The tournament winner is decided by Deb's rule: any feasible entrant beats
    every infeasible one, feasibles compare by objective (per
    `optimization_type`), infeasibles by violation. Winner selection consumes
    no randomness, so this is bit-identical to the historical
    argmax-with--inf-fallback on maximization problems.
    """
    if optimization_type not in ("max", "min"):
        raise ValueError(f"Invalid optimization_type: {optimization_type}")
    candidates = np.flatnonzero(available)
    n_add = n_fun(rng, beta, len(candidates))
    v = np.zeros_like(x)

    for _ in range(n_add):
        entrants = rng.choice(candidates, size=k, replace=True)
        values = np.empty(k)
        violations = np.empty(k)
        for j, e in enumerate(entrants):
            trial = x.copy()
            trial[e] = 1
            values[j], violations[j] = evaluate(trial)

        feasible = violations <= 0
        if feasible.any():
            if optimization_type == "max":
                winner = np.argmax(np.where(feasible, values, -np.inf))
            else:
                winner = np.argmin(np.where(feasible, values, np.inf))
        else:
            winner = np.argmin(violations)
        v[entrants[winner]] = 1

    return v


def add_velocities(*velocities: np.ndarray) -> np.ndarray:
    """V1 ⊕ V2 ⊕ ... (paper eq. 11): union of operations.

    The paper proves an addition and a removal of the same element can never
    co-occur, so summing and clipping to {-1, 0, 1} is an exact union.
    """
    total = np.sum(velocities, axis=0)
    return np.clip(total, -1, 1).astype(velocities[0].dtype)


def apply_velocity(x: np.ndarray, v: np.ndarray) -> np.ndarray:
    """X ⊞ V (paper eq. 13/18): apply additions and removals to a position."""
    return np.clip(x + v, 0, 1)
