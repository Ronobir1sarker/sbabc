"""
Optimizer — the single base class for all algorithms.

Every algorithm subclasses this and implements three methods:
  - initialize(initial_population) → set up the population for one run
  - step(iteration)                → one iteration of the algorithm
  - get_best()                     → return (solution, value, violation)

The run loop, neighborhood topology, Deb's constraint-handling rule,
evaluation counting, and per-run bookkeeping all live here, so algorithm
files contain only algorithm logic.
"""

from __future__ import annotations

import time
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace

import numpy as np

from problems.base import Problem


@dataclass
class Config:
    """Run-time settings shared by every algorithm (not algorithm hyperparameters)."""

    num_runs: int = 30  # independent repetitions
    max_iteration: int = 5000
    early_stopping_patience: int = 2500  # stop a run after this many non-improving iterations
    max_percentage_error: float = -1.0  # stop when error % <= this; -1 disables
    max_fn_calls: int = -1  # stop once this many evaluations are counted; -1 disables.
    # Checked between iterations, so a run can overshoot by up to one
    # iteration's evaluations — report budgets accordingly.
    seed: int | None = None  # master seed; every run gets its own child seed spawned from it
    verbose: int = 0  # print progress every `verbose` iterations; 0 = silent


class Optimizer(ABC):

    name = "BaseOptimizer"

    def __init__(
        self,
        topology: str = "star",
        neighborhood_radius: int = 1,
        population_size: int = 25,
        optimization_type: str = "max",
    ):
        if optimization_type not in ("max", "min", "auto"):
            raise ValueError(f"Invalid optimization_type: {optimization_type}")

        # "auto" adopts each problem's direction in set_problem(), so one
        # algorithm configuration can serve experiments that mix
        # maximization and minimization domains
        self._auto_optimization_type = optimization_type == "auto"
        self.topology = topology
        self.neighborhood_radius = neighborhood_radius
        self.population_size = population_size
        self.optimization_type = optimization_type

        # Set later via set_problem() / run()
        self.problem: Problem | None = None
        self.dim = None
        self.rng = None
        self.evaluate = None  # counted wrapper around problem.evaluate
        self._fn_calls_current_run = 0

        self.neighbor_indices = self._compute_neighbors()

    # ───────────────────────────── problem ─────────────────────────────

    def set_problem(self, problem: Problem):
        self.problem = problem
        if problem is None:
            self.dim = None
            self.evaluate = None
            return

        self.dim = problem.dim
        if self._auto_optimization_type:
            self.optimization_type = problem.optimization_type
        elif problem.optimization_type != self.optimization_type:
            raise ValueError(
                f"{self.name} is configured for {self.optimization_type!r} but "
                f"problem {problem.name!r} is {problem.optimization_type!r}"
            )

        def counted_evaluate(x):
            # one call per solution vector: a batch (..., dim) counts prod of
            # the leading dimensions, a single (dim,) vector counts 1
            self._fn_calls_current_run += int(np.prod(x.shape[:-1]))
            return problem.evaluate(x)

        self.evaluate = counted_evaluate

    # ──────────────────────────── topology ─────────────────────────────
    #
    # Social topologies from the swarm literature (see the SBPSO paper,
    # Langeveld & Engelbrecht): who each individual may learn from.

    def _compute_neighbors(self) -> np.ndarray:
        calculators = {
            "star": self._star_neighbors,
            "ring": self._ring_neighbors,
            "von_neumann": self._von_neumann_neighbors,
        }
        if self.topology not in calculators:
            raise ValueError(f"Unknown topology: {self.topology!r}; choose from {list(calculators)}")

        raw = calculators[self.topology](self.population_size, self.neighborhood_radius)
        # drop self-loops so "pick a random neighbor" never returns the individual
        # itself, and drop wrap-around duplicates: on a torus/ring whose width is
        # <= 2*radius, distinct offsets reach the same individual, which would
        # silently multiply that neighbor's probability in random_neighbor()
        no_self = [[j for j in row if j != i] for i, row in enumerate(raw)]
        rows = [list(dict.fromkeys(row)) for row in no_self]
        if sum(map(len, rows)) < sum(map(len, no_self)):
            warnings.warn(
                f"{self.topology!r} topology with population_size="
                f"{self.population_size} and radius {self.neighborhood_radius} "
                f"wraps onto itself; duplicate neighbors were removed. Results "
                f"under this configuration differ from code that sampled the "
                f"duplicates with multiplied probability."
            )
        if len({len(row) for row in rows}) != 1:
            # regular topologies always dedup to equal row lengths; anything
            # else means a topology bug, not a user error
            raise AssertionError(f"{self.topology!r} produced ragged neighbor rows")
        return np.array(rows)

    @staticmethod
    def _star_neighbors(N, _radius):
        """Fully connected: everyone is everyone's neighbor."""
        return np.tile(np.arange(N), (N, 1))

    @staticmethod
    def _ring_neighbors(N, radius):
        """Circle: neighbors are the `radius` individuals on each side."""
        offsets = np.arange(-radius, radius + 1)
        return (np.arange(N)[:, None] + offsets) % N

    @staticmethod
    def _von_neumann_neighbors(N, radius):
        """2-D toroidal grid: neighbors within Manhattan distance `radius`."""
        width = int(N**0.5)
        while width > 1 and N % width != 0:
            width -= 1
        height = N // width

        offsets = np.array(
            [
                (dr, dc)
                for dr in range(-radius, radius + 1)
                for dc in range(-radius, radius + 1)
                if abs(dr) + abs(dc) <= radius
            ]
        )
        rows = (np.arange(N) // width)[:, None]
        cols = (np.arange(N) % width)[:, None]
        neigh_rows = (rows + offsets[:, 0]) % height
        neigh_cols = (cols + offsets[:, 1]) % width
        return neigh_rows * width + neigh_cols

    def random_neighbor(self, n: int) -> int:
        """A uniformly random neighbor of individual n (never n itself)."""
        return self.rng.choice(self.neighbor_indices[n])

    def random_solutions(self, n: int) -> np.ndarray:
        """`n` random solutions, one per row — the framework-wide source of
        random (re-)initializations. Default: uniform random bit vectors.

        A problem may ship a domain-appropriate initializer as
        ``extras["init_sampler"]``, a callable ``sampler(rng, n) -> (n, dim)``
        binary array (e.g. randomized covers for SCP, where Bernoulli(0.5)
        starts absurdly far from any good solution). It applies to every
        algorithm identically, so comparisons stay fair, and it must draw all
        randomness from the passed rng to keep runs reproducible.
        """
        sampler = self.problem.extras.get("init_sampler") if self.problem is not None else None
        if sampler is not None:
            return np.asarray(sampler(self.rng, n), dtype=np.int8)
        return self.rng.integers(0, 2, size=(n, self.dim), dtype=np.int8)

    # ─────────────────── Deb's constraint-handling rule ────────────────

    def is_first_better(self, val1, val2, vio1, vio2) -> bool:
        """Deb's rule (Deb, 2000), as used in the constrained ABC paper:
          1. a feasible solution beats an infeasible one
          2. between feasibles, the better objective wins
          3. between infeasibles, the lower violation wins
        Returns True only if solution 1 is *strictly* better.
        """
        feasible1, feasible2 = vio1 <= 0, vio2 <= 0
        if feasible1 != feasible2:
            return feasible1
        if feasible1:  # both feasible
            return val1 > val2 if self.optimization_type == "max" else val1 < val2
        return vio1 < vio2  # both infeasible

    def deb_best_index(self, values, violations) -> int:
        """Index of the population member Deb's rule ranks best: the best
        objective among feasible members, or the lowest violation when
        nothing is feasible. First index wins ties (argmax/argmin order)."""
        values = np.asarray(values, dtype=np.float64)
        violations = np.asarray(violations, dtype=np.float64)
        feasible = violations <= 0
        if feasible.any():
            worst = -np.inf if self.optimization_type == "max" else np.inf
            masked = np.where(feasible, values, worst)
            return int(np.argmax(masked)) if self.optimization_type == "max" else int(np.argmin(masked))
        return int(np.argmin(violations))

    def deb_worst_index(self, values, violations) -> int:
        """Index of the member Deb's rule ranks worst: the highest violation
        if any member is infeasible, else the worst objective."""
        values = np.asarray(values, dtype=np.float64)
        violations = np.asarray(violations, dtype=np.float64)
        if (violations > 0).any():
            return int(np.argmax(violations))
        return int(np.argmin(values)) if self.optimization_type == "max" else int(np.argmax(values))

    # ─────────────────────── algorithm interface ───────────────────────

    @abstractmethod
    def initialize(self, initial_population=None):
        """Set up the population for one run."""

    @abstractmethod
    def step(self, iteration: int):
        """One iteration of the algorithm."""

    @abstractmethod
    def get_best(self):
        """Return (best_solution, best_value, best_violation) found so far this run."""

    # ───────────────────────────── run loop ────────────────────────────

    def run(
        self,
        problem: Problem = None,
        config: Config = None,
        initial_population=None,
        run_seeds=None,
        **overrides,
    ):
        """Execute ``config.num_runs`` independent runs; returns a results dict.

        Keyword overrides are applied on top of `config`, so quick experiments
        can say ``algo.run(problem, max_iteration=100, seed=1)``.

        Each run draws its RNG from its own child seed spawned from
        ``config.seed``, so run r produces the same result no matter which
        process or machine executes it — sequential and parallel execution are
        bit-identical. `run_seeds` lets a parallel executor pass the child
        seeds for the subset of runs this call should compute
        (see experiments/parallel.py).
        """
        if problem is not None:
            self.set_problem(problem)
        if self.problem is None:
            raise ValueError("No problem set. Pass problem= or call set_problem() first.")
        if self.evaluate is None:  # e.g. after unpickling in a worker process
            self.set_problem(self.problem)

        cfg = replace(config or Config(), **overrides)
        if run_seeds is None:
            run_seeds = np.random.SeedSequence(cfg.seed).spawn(cfg.num_runs)
        elif len(run_seeds) != cfg.num_runs:
            raise ValueError(f"Got {len(run_seeds)} run_seeds for {cfg.num_runs} runs")

        self._allocate_run_storage(cfg.num_runs)
        for r in range(cfg.num_runs):
            self.rng = np.random.default_rng(run_seeds[r])
            self._execute_single_run(r, cfg, initial_population)

        return self._build_results()

    # ──────────────────────────── pickling ─────────────────────────────
    #
    # Instances travel to worker processes (experiments/parallel.py). The
    # counted-evaluate closure references self and cannot be pickled; run()
    # rebuilds it from self.problem on the other side.

    def __getstate__(self):
        state = self.__dict__.copy()
        state["evaluate"] = None
        state["rng"] = None
        return state

    def _allocate_run_storage(self, num_runs):
        R = num_runs
        self.best_solution_values = np.zeros(R, dtype=np.float64)
        self.best_solution_violations = np.zeros(R, dtype=np.float64)
        self.best_solution_positions = np.zeros((R, self.dim), dtype=np.int8)
        self.convergence_curves = []  # one array per run; length varies with early stopping
        # cumulative counted evaluations at each convergence point, so curves
        # can be plotted (and truncated) on a function-evaluation axis —
        # per-iteration cost differs enormously between algorithms, so an
        # iteration axis is not budget-fair
        self.fn_call_curves = []
        self.fn_calls_per_run = np.zeros(R, dtype=np.int64)
        self.iterations_per_run = np.zeros(R, dtype=np.int64)
        self.last_improvement_iteration = np.zeros(R, dtype=np.int64)
        self.fn_calls_at_best = np.zeros(R, dtype=np.int64)
        self.wall_time_per_run = np.zeros(R, dtype=np.float64)
        self.cpu_time_per_run = np.zeros(R, dtype=np.float64)

    def _execute_single_run(self, r, cfg, initial_population):
        # deltas, not absolutes: process_time() is cumulative over a worker
        # process that executes many runs back to back
        wall_start, cpu_start = time.perf_counter(), time.process_time()
        self._fn_calls_current_run = 0
        self.initialize(initial_population)

        best_sol, best_val, best_vio = self.get_best()
        curve = [best_val]
        fe_curve = [self._fn_calls_current_run]
        no_improve_count = 0
        last_improve_iter = 0
        fn_calls_at_best = self._fn_calls_current_run
        t = -1  # if max_iteration == 0 we still store consistent results

        for t in range(cfg.max_iteration):
            self.step(t)

            prev_val, prev_vio = best_val, best_vio
            best_sol, best_val, best_vio = self.get_best()
            curve.append(best_val)
            fe_curve.append(self._fn_calls_current_run)

            if self.is_first_better(best_val, prev_val, best_vio, prev_vio):
                no_improve_count = 0
                last_improve_iter = t + 1
                fn_calls_at_best = self._fn_calls_current_run
            else:
                no_improve_count += 1

            self._print_progress(cfg, r, t, best_val)
            if self._should_stop(cfg, no_improve_count, best_val, best_vio):
                break

        if np.isnan(best_val) or np.isnan(best_vio):
            # NaN can never win a Deb comparison and binds as NULL in the
            # sqlite store — always an evaluator bug, so fail loudly here
            raise ValueError(
                f"{self.name} ended run {r} on {self.problem.name!r} with NaN "
                f"(value={best_val}, violation={best_vio}); fix the evaluator"
            )
        self.best_solution_values[r] = best_val
        self.best_solution_violations[r] = best_vio
        self.best_solution_positions[r, :] = best_sol
        self.convergence_curves.append(np.array(curve, dtype=np.float64))
        self.fn_call_curves.append(np.array(fe_curve, dtype=np.int64))
        self.fn_calls_per_run[r] = self._fn_calls_current_run
        self.iterations_per_run[r] = t + 1
        self.last_improvement_iteration[r] = last_improve_iter
        self.fn_calls_at_best[r] = fn_calls_at_best
        self.wall_time_per_run[r] = time.perf_counter() - wall_start
        self.cpu_time_per_run[r] = time.process_time() - cpu_start

    def _print_progress(self, cfg, r, t, best_val):
        if cfg.verbose <= 0 or t % cfg.verbose != 0:
            return
        if self.problem.known_optimum in (None, 0):
            print(f"  [{self.name}] run {r} iter {t}: best = {best_val:.4f}")
        else:
            err = self.problem.error_percentage(best_val)
            print(f"  [{self.name}] run {r} iter {t}: error = {err:.2f}%")

    def _should_stop(self, cfg, no_improve_count, best_val, best_vio) -> bool:
        if no_improve_count >= cfg.early_stopping_patience:
            return True
        if cfg.max_fn_calls >= 0 and self._fn_calls_current_run >= cfg.max_fn_calls:
            return True
        if (
            cfg.max_percentage_error >= 0
            and best_vio <= 0  # evaluators that report raw objectives on
            # infeasible solutions (SCP, MKP "raw", penalized views) can show
            # a value at/above the optimum while violating constraints — an
            # infeasible incumbent must never trigger stop-at-optimum
            and self.problem.known_optimum not in (None, 0)
            # the 1e-9 pad keeps stop-at-optimum (threshold 0.0) from depending
            # on the last ULP of a fractional optimum (mknap1-2's 8706.1):
            # a different numpy/BLAS summation order must not turn an optimal
            # solution into a full-length run
            and self.problem.error_percentage(best_val) <= cfg.max_percentage_error + 1e-9
        ):
            return True
        return False

    def _build_results(self):
        return {
            "algorithm": self.name,
            "problem": self.problem.name,
            "best_values": self.best_solution_values,
            "best_violations": self.best_solution_violations,
            "best_positions": self.best_solution_positions,
            "convergence_curves": self.convergence_curves,
            "fn_call_curves": self.fn_call_curves,
            "fn_calls_per_run": self.fn_calls_per_run,
            "iterations_per_run": self.iterations_per_run,
            "last_improvement_iteration": self.last_improvement_iteration,
            "fn_calls_at_best": self.fn_calls_at_best,
            "wall_time_per_run": self.wall_time_per_run,
            "cpu_time_per_run": self.cpu_time_per_run,
        }
