"""
Sobol-sampled parameter tuning, after the methodology in Langeveld &
Engelbrecht's SBPSO paper: sample parameter sets with a Sobol sequence,
run each candidate on every tuning problem, rank candidates per problem
by mean best value, and pick the candidate with the best average rank.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import math

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from scipy.stats.qmc import Sobol

from core.optimizer import Config
from problems.base import Problem


def sobol_parameter_sets(ranges: dict, sample_size: int) -> pd.DataFrame:
    """Sobol-sample `sample_size` parameter sets (one per row).

    Each entry of `ranges` is either
      - a TUPLE (low, high) of floats → continuous range [low, high)
      - a TUPLE (low, high) of ints   → integer range [low, high], inclusive
      - a LIST  [v1, v2, ...]         → discrete choices (numbers or strings)

    The tuple/list distinction is deliberate: it makes a two-element choice
    list like [1, 3] unambiguous (a list is never read as a range).

    sample_size must be a power of two (a Sobol-sequence requirement).
    """
    if sample_size <= 0 or sample_size & (sample_size - 1) != 0:
        raise ValueError(f"sample_size must be a positive power of 2, got {sample_size}")

    unit_samples = Sobol(d=len(ranges), scramble=False).random_base2(int(math.log2(sample_size)))

    data = {}
    for i, (name, spec) in enumerate(ranges.items()):
        u = unit_samples[:, i]
        if _is_numeric_pair(spec):
            low, high = spec
            if isinstance(low, (int, np.integer)) and isinstance(high, (int, np.integer)):
                if high < low:
                    raise ValueError(f"Invalid integer range for {name!r}: {spec}")
                data[name] = np.minimum((u * (high - low + 1)).astype(int), high - low) + low
            else:
                if not high > low:
                    raise ValueError(f"Invalid float range for {name!r}: {spec}")
                data[name] = u * (high - low) + low
        else:
            choices = list(spec) if isinstance(spec, Sequence) and not isinstance(spec, str) else [spec]
            if not choices:
                raise ValueError(f"Discrete choice list for {name!r} is empty")
            idx = np.minimum((u * len(choices)).astype(int), len(choices) - 1)
            data[name] = np.array(choices, dtype=object)[idx]

    return pd.DataFrame(data, columns=list(ranges))


def _is_numeric_pair(spec) -> bool:
    return (
        isinstance(spec, tuple)
        and len(spec) == 2
        and all(isinstance(x, (int, float, np.integer, np.floating)) for x in spec)
    )


def tune_parameters(
    algorithm_cls,
    problems: list[Problem],
    param_ranges: dict,
    fixed_params: dict = None,
    config: Config = None,
    sample_size: int = 16,
    max_workers: int = None,  # None = sequential; e.g. 6 = process pool with 6 workers
    quiet: bool = False,
    **overrides,
) -> tuple[dict, pd.DataFrame]:
    """Returns (best_params, table); the table is one row per candidate with
    its parameters, per-problem mean error % (when known optima exist),
    and average rank — sorted best first. The error % is derived from the
    Deb-scored mean, so it equals the plain mean when every run was feasible
    and degrades gracefully (instead of turning ±inf) when some were not.

    With `max_workers` set, the candidate × problem grid runs on a process
    pool (one task per pair) and produces identical results to the
    sequential path. On Windows, call this from under
    ``if __name__ == "__main__":`` when parallel.
    """
    from experiments.analysis import deb_scores

    cfg = replace(config or Config(), **overrides)
    param_sets = sobol_parameter_sets(param_ranges, sample_size)
    candidates = [
        algorithm_cls(**params, **(fixed_params or {}))
        for params in param_sets.to_dict("records")
    ]

    # per (problem, candidate): (best_values, best_violations) of every run
    run_outcomes = {}
    if max_workers is not None:
        from experiments.parallel import make_tasks, merge_task_outputs, run_tasks

        tasks = make_tasks(candidates, problems, cfg, runs_per_task=cfg.num_runs)
        outputs = run_tasks(tasks, candidates, problems, cfg, max_workers, quiet)
        merged = merge_task_outputs(outputs, cfg.num_runs)
        for (cid, pid), results in merged.items():
            run_outcomes[pid, cid] = (results["best_values"], results["best_violations"])
    else:
        for pid, problem in enumerate(problems):
            if not quiet:
                print(f"Tuning problem {pid + 1}/{len(problems)}: {problem.name}")
            for cid, candidate in enumerate(candidates):
                candidate.run(problem=problem, config=cfg)
                run_outcomes[pid, cid] = (
                    candidate.best_solution_values.copy(),
                    candidate.best_solution_violations.copy(),
                )

    # Candidate score = mean of Deb-consistent run scores. Runs that ended
    # infeasible score below every feasible run (deeper with more violation)
    # instead of poisoning the mean with ±inf or a misleading raw objective;
    # the candidates of a problem are pooled first so they share one anchor.
    mean_values = np.zeros((len(problems), len(candidates)))
    for pid, problem in enumerate(problems):
        values = np.concatenate([run_outcomes[pid, cid][0] for cid in range(len(candidates))])
        violations = np.concatenate([run_outcomes[pid, cid][1] for cid in range(len(candidates))])
        scores = deb_scores(values, violations, problem.optimization_type)
        mean_values[pid] = scores.reshape(len(candidates), -1).mean(axis=1)

    table = param_sets.copy()
    for pid, problem in enumerate(problems):
        if problem.known_optimum is not None:
            table[f"mean_error_pct[{problem.name}]"] = [
                problem.error_percentage(v) for v in mean_values[pid]
            ]

    # rank candidates per problem (1 = best), averaging ranks on ties
    ranks = np.vstack(
        [
            rankdata(-mean_values[pid] if problem.optimization_type == "max" else mean_values[pid])
            for pid, problem in enumerate(problems)
        ]
    )
    table["avg_rank"] = ranks.mean(axis=0)
    # stable sort: among configurations tied for the best mean rank the earliest
    # sampled is selected (the tie rule stated in the paper, Section 6.2)
    table = table.sort_values("avg_rank", kind="stable").reset_index(drop=True)

    best_params = table.iloc[0][param_sets.columns].to_dict()
    if not quiet:
        print(f"Best parameters (avg rank {table['avg_rank'].iloc[0]:.2f}): {best_params}")
    return best_params, table
