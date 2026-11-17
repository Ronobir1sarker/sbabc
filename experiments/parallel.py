"""
Parallel and distributed execution of experiments.

The unit of work is a Task: one algorithm on one problem for a subset of the
independent runs. Because every run draws its RNG from a per-run child seed
(spawned from ``Config.seed`` — see Optimizer.run), a task's result is
bit-identical no matter which process *or machine* executes it, so:

  - local parallelism: ``parallel_compare(...)`` spreads tasks over a process
    pool and returns exactly the same table as ``compare_algorithms(...)``;
  - multiple PCs: build the same task list on each machine, run a different
    shard on each (``shard(tasks, i, n)``), save the raw outputs with
    ``save_task_outputs``, copy the files to one machine, and merge.

``run_experiment.py`` in the repository root wraps all of this in a CLI.

Windows note: anything that creates the process pool must be called from
under ``if __name__ == "__main__":``.
"""

from __future__ import annotations

import glob
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace

import joblib
import numpy as np
import pandas as pd

from core.optimizer import Config, Optimizer
from experiments.analysis import summarize_runs
from problems.base import Problem


@dataclass(frozen=True)
class Task:
    algorithm_index: int
    problem_index: int
    run_indices: tuple  # which of the Config.num_runs repetitions to compute


def make_tasks(algorithms, problems, config: Config, runs_per_task: int = 1) -> list[Task]:
    """The full experiment grid as independent tasks.

    runs_per_task=1 gives the finest grain (best load balancing); raise it if
    individual runs are so short that process overhead dominates.
    """
    return [
        Task(ai, pi, tuple(range(start, min(start + runs_per_task, config.num_runs))))
        for pi in range(len(problems))
        for ai in range(len(algorithms))
        for start in range(0, config.num_runs, runs_per_task)
    ]


def shard(tasks: list[Task], shard_index: int, shard_count: int) -> list[Task]:
    """Deterministic round-robin share of the tasks for machine `shard_index`.

    Every machine must build the identical task list (same experiment
    definition, same runs_per_task) for the shards to partition it correctly.
    """
    if not 0 <= shard_index < shard_count:
        raise ValueError(f"shard_index must be in [0, {shard_count}), got {shard_index}")
    return tasks[shard_index::shard_count]


def restrict_tasks(tasks: list[Task], keep) -> list[Task]:
    """Keep only the run indices for which ``keep(task, run_index)`` is true;
    tasks left with no runs are dropped. Used to give one machine a range of
    run indices and to skip runs already in the results database (resume).
    """
    filtered = []
    for task in tasks:
        kept = tuple(r for r in task.run_indices if keep(task, r))
        if kept:
            filtered.append(Task(task.algorithm_index, task.problem_index, kept))
    return filtered


def _execute_task(task: Task, algorithm: Optimizer, problem: Problem, config: Config):
    """Worker entry point (module-level so it can be pickled to processes)."""
    all_seeds = np.random.SeedSequence(config.seed).spawn(config.num_runs)
    seeds = [all_seeds[i] for i in task.run_indices]
    results = algorithm.run(
        problem=problem,
        config=replace(config, num_runs=len(seeds), verbose=0),
        run_seeds=seeds,
    )
    return task, results


def run_tasks(
    tasks: list[Task],
    algorithms: list[Optimizer],
    problems: list[Problem],
    config: Config,
    max_workers: int = None,
    quiet: bool = False,
    on_result=None,
) -> list[tuple]:
    """Execute tasks on a local process pool; returns [(task, results), ...].

    max_workers=None uses all cores; leave one or two free (e.g. 6 of 7) if
    you want to keep working on the machine. ``on_result(task, results)`` is
    called in this process as each task finishes — e.g. to write the runs to
    the results database immediately, so a crash loses at most one task.

    A task that raises does not abort the campaign: the error is reported,
    every other task still completes (and is saved via ``on_result``), and a
    summary RuntimeError is raised at the end. Resuming the experiment
    recomputes only the failed tasks.
    """
    outputs = []
    failures = []
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                _execute_task, t, algorithms[t.algorithm_index], problems[t.problem_index], config
            ): t
            for t in tasks
        }
        for i, future in enumerate(as_completed(futures), 1):
            task = futures[future]
            label = (
                f"{algorithms[task.algorithm_index].name} on "
                f"{problems[task.problem_index].name} "
                f"runs {task.run_indices[0]}-{task.run_indices[-1]}"
            )
            try:
                task, results = future.result()
            except Exception as error:
                failures.append((label, error))
                print(f"  {i}/{len(tasks)}: {label} FAILED: {error!r}")
                continue
            outputs.append((task, results))
            if on_result is not None:
                on_result(task, results)
            if not quiet:
                print(f"  {i}/{len(tasks)}: {label} done")
    if failures:
        listing = "\n".join(f"  {label}: {error!r}" for label, error in failures)
        raise RuntimeError(
            f"{len(failures)} of {len(tasks)} tasks failed (all completed tasks "
            f"were kept; rerun the experiment to recompute the failures):\n{listing}"
        )
    return outputs


def merge_task_outputs(outputs: list[tuple], num_runs: int) -> dict:
    """Stitch task outputs back into one results dict per (algorithm, problem),
    with runs in their original order. Raises if any run is duplicated or
    missing (e.g. a shard file was forgotten).
    """
    grouped = defaultdict(list)
    for task, results in outputs:
        grouped[(task.algorithm_index, task.problem_index)].append((task, results))

    merged = {}
    for key, chunks in grouped.items():
        first = chunks[0][1]
        combined = {
            "algorithm": first["algorithm"],
            "problem": first["problem"],
            "best_values": np.zeros(num_runs),
            "best_violations": np.zeros(num_runs),
            "best_positions": np.zeros((num_runs, first["best_positions"].shape[1]), dtype=np.int8),
            "convergence_curves": [None] * num_runs,
            "fn_call_curves": [None] * num_runs,
            "fn_calls_per_run": np.zeros(num_runs, dtype=np.int64),
            "iterations_per_run": np.zeros(num_runs, dtype=np.int64),
            "last_improvement_iteration": np.zeros(num_runs, dtype=np.int64),
            "fn_calls_at_best": np.zeros(num_runs, dtype=np.int64),
            "wall_time_per_run": np.zeros(num_runs, dtype=np.float64),
            "cpu_time_per_run": np.zeros(num_runs, dtype=np.float64),
        }
        seen = set()
        for task, results in chunks:
            for offset, r in enumerate(task.run_indices):
                if r in seen:
                    raise ValueError(f"Run {r} of {first['algorithm']}/{first['problem']} appears twice")
                seen.add(r)
                for field in combined:
                    if field in ("algorithm", "problem") or field not in results:
                        # shard files saved by older code lack newer fields;
                        # their slots keep the zero/None fill — treat such
                        # merged aggregates (e.g. timing means) as suspect
                        continue
                    combined[field][r] = results[field][offset]
        missing = set(range(num_runs)) - seen
        if missing:
            raise ValueError(
                f"Runs {sorted(missing)} of {first['algorithm']}/{first['problem']} are missing "
                f"— were all shards merged?"
            )
        merged[key] = combined
    return merged


def results_table(merged: dict, problems: list[Problem]) -> pd.DataFrame:
    """One summary row per (problem, algorithm), in grid order."""
    rows = [
        summarize_runs(results, problems[pi])
        for (ai, pi), results in sorted(merged.items(), key=lambda kv: (kv[0][1], kv[0][0]))
    ]
    return pd.DataFrame(rows)


def parallel_compare(
    algorithms: list[Optimizer],
    problems: list[Problem],
    config: Config = None,
    max_workers: int = None,
    runs_per_task: int = 1,
    save_path: str = None,
    quiet: bool = False,
    **overrides,
) -> pd.DataFrame:
    """Drop-in parallel version of ``compare_algorithms``: same table,
    bit-identical values, spread over a local process pool.
    """
    cfg = replace(config or Config(), **overrides)
    tasks = make_tasks(algorithms, problems, cfg, runs_per_task)
    outputs = run_tasks(tasks, algorithms, problems, cfg, max_workers, quiet)
    df = results_table(merge_task_outputs(outputs, cfg.num_runs), problems)
    if save_path:
        from experiments.runner import _save

        _save(df, save_path)
    return df


def save_task_outputs(path: str, outputs: list[tuple]):
    joblib.dump(outputs, path)
    print(f"Saved {len(outputs)} task results to {path}")


def load_task_outputs(pattern: str) -> list[tuple]:
    """Load and concatenate task outputs from a glob pattern or single path."""
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No shard files match {pattern!r}")
    outputs = []
    for path in paths:
        chunk = joblib.load(path)
        print(f"Loaded {len(chunk)} task results from {path}")
        outputs.extend(chunk)
    return outputs
