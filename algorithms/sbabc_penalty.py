"""
SBABC-penalty — the tuned SBABC running as a static penalty method instead
of under Deb's rule (the constraint-handling comparison of the paper's Section 7.10). The
only change is the problem view the search sees: problems.base.penalized
folds factor * violation into the objective and reports violation 0, so
every Deb comparison inside SBABC degenerates to plain objective comparison.
Nothing in the operators changes. A demonstration arm, not a proposed
variant.
Bookkeeping that keeps the stored rows comparable with the Deb arm:
  - the raw (picklable) problem stays in self.problem; the penalized view is
    rebuilt locally in set_problem — its evaluate is a closure and could not
    travel to worker processes anyway;
  - each run's incumbent is re-scored on the raw problem before it is stored,
    so best_value / best_violation are the true profit and the true capacity
    excess (an uncounted evaluation: bookkeeping, like CBGA's repair);
  - stop-at-optimum consults the raw violation, so an infeasible incumbent
    whose penalized value happens to sit at the optimum never ends a run;
  - convergence curves stay in penalized units (not used by the penalty
    table; documented here so nobody plots them against the Deb arm).
Load MKP instances with infeasible_value="raw" for this arm: the default
-inf mask would erase the penalty gradient (penalized() then falls back to
-factor * violation with no profit signal).
"""
import numpy as np

from algorithms.sbabc import SetBasedABC
from problems.base import penalized


class SetBasedABCPenalty(SetBasedABC):
    name = "SBABC-penalty"

    def __init__(self, penalty_factor: float = 100.0, **sbabc_kwargs):
        if penalty_factor <= 0:
            raise ValueError(f"penalty_factor must be positive, got {penalty_factor}")
        self.penalty_factor = float(penalty_factor)
        self._raw_problem = None
        super().__init__(**sbabc_kwargs)

    def set_problem(self, problem):
        if problem is None:
            self._raw_problem = None
            super().set_problem(None)
            return
        if problem.extras.get("penalized_from") is not None:
            raise ValueError(
                f"{self.name} builds its own penalty view; pass the raw problem, "
                f"not {problem.name!r}"
            )
        self._raw_problem = problem
        super().set_problem(penalized(problem, self.penalty_factor))
        # keep the raw, picklable problem; the counted-evaluate closure that
        # super().set_problem built already holds the penalized view
        self.problem = problem

    def _raw_score(self, x) -> tuple[float, float]:
        value, violation = self._raw_problem.evaluate(np.asarray(x))
        return float(value), float(violation)

    def _should_stop(self, cfg, no_improve_count, best_val, best_vio) -> bool:
        if cfg.max_percentage_error >= 0:  # the view reports violation 0 always
            best_sol, _, _ = self.get_best()
            _, best_vio = self._raw_score(best_sol)
        return super()._should_stop(cfg, no_improve_count, best_val, best_vio)

    def _execute_single_run(self, r, cfg, initial_population):
        super()._execute_single_run(r, cfg, initial_population)
        value, violation = self._raw_score(self.best_solution_positions[r])
        self.best_solution_values[r] = value
        self.best_solution_violations[r] = violation
