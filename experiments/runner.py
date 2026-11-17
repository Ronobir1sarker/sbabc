"""
Run several algorithms over several problems and tabulate the results.

Typical use::

    from experiments import compare_algorithms
    df = compare_algorithms(
        algorithms=[abc, sbabc],
        problems=load_mkp_problems("mknap1"),
        config=Config(num_runs=15, max_iteration=5000, seed=4),
        save_path="results/comparison.csv",
    )
"""

from __future__ import annotations

from dataclasses import replace

import pandas as pd

from core.optimizer import Config, Optimizer
from experiments.analysis import summarize_runs
from problems.base import Problem


def compare_algorithms(
    algorithms: list[Optimizer],
    problems: list[Problem],
    config: Config = None,
    save_path: str = None,
    quiet: bool = False,
    **overrides,
) -> pd.DataFrame:
    """One row per (problem, algorithm) pair; every algorithm sees the same
    config (and hence the same seed, for paired comparisons).
    """
    cfg = replace(config or Config(), **overrides)

    rows = []
    for problem in problems:
        for algorithm in algorithms:
            results = algorithm.run(problem=problem, config=cfg)
            row = summarize_runs(results, problem)
            rows.append(row)
            if not quiet:
                err = f", mean error {row['mean_error_pct']:.2f}%" if "mean_error_pct" in row else ""
                print(f"[{problem.name}] {algorithm.name}: best {row['best_value']:g}{err}")

    df = pd.DataFrame(rows)
    if save_path:
        _save(df, save_path)
    return df


def _save(df: pd.DataFrame, path: str):
    if path.endswith(".xlsx"):
        df.to_excel(path, index=False)
    else:
        df.to_csv(path, index=False)
    print(f"Saved comparison to {path}")
