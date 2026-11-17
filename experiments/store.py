"""
SQLite results database: one row per completed run.

Why a database instead of result files? Three properties that matter for a
long-running research project:

  - crash resilience: rows are written as each run finishes, so an
    interrupted experiment resumes where it stopped (completed runs are
    skipped, nothing is recomputed);
  - multi-PC merging: each machine writes its own local .sqlite file for its
    share of the run indices; ``merge_from`` combines them, and the primary
    key (experiment, algorithm, problem, run_index) makes duplicated or
    conflicting runs an error instead of a silent mistake;
  - recall: the experiment's config, every algorithm's exact parameters, and
    the git commit are stored next to the results, so any number in the
    final table traces back to what produced it.

Schema
    experiments            name → master seed, run budget, git commit, date
    experiment_algorithms  (experiment, algorithm) → parameters as JSON
    runs                   (experiment, algorithm, problem, run_index) →
                           metrics + best position and convergence curve
                           (NumPy arrays stored as .npy blobs)

Because every run's RNG comes from a per-run child seed spawned from the
experiment's master seed (see Optimizer.run), the merged database is
bit-identical to what a single machine would have produced.
"""

from __future__ import annotations

import inspect
import io
import json
import sqlite3
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from core.optimizer import Config, Optimizer
from experiments.analysis import summarize_runs
from problems.base import Problem

_SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    name                    TEXT PRIMARY KEY,
    master_seed             INTEGER NOT NULL,
    num_runs                INTEGER NOT NULL,
    max_iteration           INTEGER NOT NULL,
    early_stopping_patience INTEGER NOT NULL,
    max_percentage_error    REAL    NOT NULL,
    git_commit              TEXT,
    created_utc             TEXT    NOT NULL,
    max_fn_calls            INTEGER
);

CREATE TABLE IF NOT EXISTS experiment_algorithms (
    experiment  TEXT NOT NULL REFERENCES experiments(name),
    algorithm   TEXT NOT NULL,
    class_name  TEXT NOT NULL,
    params_json TEXT NOT NULL,
    PRIMARY KEY (experiment, algorithm)
);

CREATE TABLE IF NOT EXISTS runs (
    experiment                 TEXT    NOT NULL REFERENCES experiments(name),
    algorithm                  TEXT    NOT NULL,
    problem                    TEXT    NOT NULL,
    run_index                  INTEGER NOT NULL,
    best_value                 REAL    NOT NULL,
    best_violation             REAL    NOT NULL,
    fn_calls                   INTEGER NOT NULL,
    iterations                 INTEGER NOT NULL,
    last_improvement_iteration INTEGER NOT NULL,
    fn_calls_at_best           INTEGER NOT NULL,
    best_position              BLOB    NOT NULL,
    convergence                BLOB    NOT NULL,
    wall_time                  REAL,
    cpu_time                   REAL,
    fn_call_curve              BLOB,
    PRIMARY KEY (experiment, algorithm, problem, run_index)
);

-- which code and library versions produced runs for each experiment, one row
-- per distinct environment seen. Per-run seeding only guarantees bit-identical
-- results when every machine runs the same code on the same numpy, and the
-- documented multi-PC workflow gives machines disjoint run indices — so the
-- runs-level conflict check can never catch drift. This table lets merge_from
-- detect it instead. git_commit is "" (not NULL) when git is unavailable, so
-- the primary key deduplicates correctly.
CREATE TABLE IF NOT EXISTS environments (
    experiment     TEXT NOT NULL,
    git_commit     TEXT NOT NULL,
    python_version TEXT NOT NULL,
    numpy_version  TEXT NOT NULL,
    scipy_version  TEXT NOT NULL,
    recorded_utc   TEXT NOT NULL,
    PRIMARY KEY (experiment, git_commit, python_version, numpy_version, scipy_version)
);
"""

# Columns added after the original schema; databases created before them are
# migrated in place on open (or on attach, for merging). Appended (never
# inserted) so that positional `INSERT ... SELECT *` between old and new
# databases keeps lining up — _SCHEMA above must list them last too.
_MIGRATIONS = {
    "runs": (("wall_time", "REAL"), ("cpu_time", "REAL"), ("fn_call_curve", "BLOB")),
    "experiments": (("max_fn_calls", "INTEGER"),),
}

_ENVIRONMENTS_TABLE = """
CREATE TABLE IF NOT EXISTS {schema}.environments (
    experiment     TEXT NOT NULL,
    git_commit     TEXT NOT NULL,
    python_version TEXT NOT NULL,
    numpy_version  TEXT NOT NULL,
    scipy_version  TEXT NOT NULL,
    recorded_utc   TEXT NOT NULL,
    PRIMARY KEY (experiment, git_commit, python_version, numpy_version, scipy_version)
)
"""


def _migrate(con, schema: str = "main"):
    for table, columns in _MIGRATIONS.items():
        existing = {row[1] for row in con.execute(f"PRAGMA {schema}.table_info({table})")}
        for column, sql_type in columns:
            if column not in existing:
                con.execute(f"ALTER TABLE {schema}.{table} ADD COLUMN {column} {sql_type}")
    # tables added after the original schema (executescript can't target an
    # attached database, so table creation must go through the migration too)
    con.execute(_ENVIRONMENTS_TABLE.format(schema=schema))


# ─────────────────────────── small helpers ───────────────────────────


def _to_blob(array) -> bytes:
    """NumPy array → .npy bytes (self-describing: dtype and shape included)."""
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(array), allow_pickle=False)
    return buffer.getvalue()


def _from_blob(blob: bytes) -> np.ndarray:
    return np.load(io.BytesIO(blob), allow_pickle=False)


def _json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    raise TypeError(f"Algorithm parameter of type {type(value).__name__} is not JSON-storable")


def algorithm_params(algorithm: Optimizer) -> dict:
    """The algorithm's constructor parameters, read back off the instance.

    Walks the __init__ signatures of the whole class hierarchy and collects
    every named parameter that exists as an attribute, so the database
    records what actually ran (not what you remember configuring).
    """
    params = {}
    for cls in type(algorithm).__mro__:
        init = cls.__dict__.get("__init__")
        if init is None:
            continue
        for p in inspect.signature(init).parameters.values():
            if p.name == "self" or p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
                continue
            if p.name not in params and hasattr(algorithm, p.name):
                params[p.name] = getattr(algorithm, p.name)
    return params


def current_environment() -> dict:
    """The code and library versions that determine run results, for the
    environments provenance table."""
    import platform

    import scipy

    return {
        "git_commit": current_git_commit() or "",
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
    }


def current_git_commit() -> str | None:
    """Short commit hash of the working directory, "+dirty" if there are
    uncommitted changes; None when git is unavailable.
    """
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        ).stdout.strip()
        return commit + ("+dirty" if status else "")
    except (OSError, subprocess.CalledProcessError):
        return None


# ───────────────────────────── the store ─────────────────────────────


class RunStore:
    """Per-run results database. Open one per machine; merge later.

    Usage:
        store = RunStore("results/results.sqlite")
        store.register_experiment("mknap1_final", algorithms, config)
        ... after each completed task:
        store.save_task_results("mknap1_final", task.run_indices, results)
    """

    def __init__(self, path: str):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.con = sqlite3.connect(self.path)
        self.con.executescript(_SCHEMA)
        _migrate(self.con)
        self.con.commit()

    def close(self):
        self.con.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ──────────────────────── experiment metadata ────────────────────────

    def register_experiment(
        self,
        name: str,
        algorithms: list[Optimizer],
        config: Config,
        git_commit: str | None = None,
    ):
        """Record what the experiment name means: config, per-algorithm
        parameters, code version. Registering the same name twice is fine if
        nothing changed; changed config or parameters raise, because that
        would silently mix incomparable results under one name.
        """
        if config.seed is None:
            raise ValueError(
                "Experiments stored in the database need a fixed Config.seed — "
                "it is what makes runs resumable and mergeable across machines."
            )
        names = [a.name for a in algorithms]
        if len(set(names)) != len(names):
            raise ValueError(f"Algorithm names must be unique within an experiment, got {names}")

        settings = {
            "master_seed": config.seed,
            "num_runs": config.num_runs,
            "max_iteration": config.max_iteration,
            "early_stopping_patience": config.early_stopping_patience,
            "max_percentage_error": config.max_percentage_error,
            "max_fn_calls": config.max_fn_calls,
        }
        existing = self.con.execute(
            "SELECT master_seed, num_runs, max_iteration, early_stopping_patience, "
            "max_percentage_error, max_fn_calls FROM experiments WHERE name = ?",
            (name,),
        ).fetchone()
        if existing is not None:
            # NULL max_fn_calls (row written before the column existed) means "disabled"
            existing = tuple(-1 if v is None else v for v in existing)

        if existing is None:
            self.con.execute(
                "INSERT INTO experiments (name, master_seed, num_runs, max_iteration, "
                "early_stopping_patience, max_percentage_error, max_fn_calls, git_commit, "
                "created_utc) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (name, *settings.values(),
                 git_commit or current_git_commit(),
                 datetime.now(timezone.utc).isoformat(timespec="seconds")),
            )
        elif existing != tuple(settings.values()):
            raise ValueError(
                f"Experiment {name!r} already exists with different settings "
                f"{dict(zip(settings, existing))} vs now {settings}. "
                f"Use a new experiment name when the setup changes."
            )

        for algorithm in algorithms:
            params = json.dumps(
                algorithm_params(algorithm), sort_keys=True, default=_json_default
            )
            stored = self.con.execute(
                "SELECT params_json FROM experiment_algorithms WHERE experiment = ? AND algorithm = ?",
                (name, algorithm.name),
            ).fetchone()
            if stored is None:
                self.con.execute(
                    "INSERT INTO experiment_algorithms VALUES (?, ?, ?, ?)",
                    (name, algorithm.name, type(algorithm).__name__, params),
                )
            elif stored[0] != params:
                raise ValueError(
                    f"Algorithm {algorithm.name!r} in experiment {name!r} was registered "
                    f"with different parameters:\n  stored: {stored[0]}\n  now:    {params}\n"
                    f"Rename the algorithm or use a new experiment name."
                )
        self.con.commit()

    def record_environment(self, experiment: str, git_commit: str | None = None):
        """Record THIS machine's code/library versions for the experiment.

        Call it when runs are actually about to be computed (not on every
        registration): a zero-run invocation from a dirty tree must not
        plant an environment row that later makes merge_from cry drift for
        runs that environment never produced. The experiments row only keeps
        the first machine's commit, which is exactly what merge_from cannot
        verify for the other machines of a multi-PC campaign — this table
        is what it checks instead.
        """
        environment = current_environment()
        if git_commit is not None:
            environment["git_commit"] = git_commit
        self.con.execute(
            "INSERT OR IGNORE INTO environments VALUES (?, ?, ?, ?, ?, ?)",
            (experiment, *environment.values(),
             datetime.now(timezone.utc).isoformat(timespec="seconds")),
        )
        self.con.commit()

    def experiment_config(self, experiment: str) -> Config:
        row = self.con.execute(
            "SELECT master_seed, num_runs, max_iteration, early_stopping_patience, "
            "max_percentage_error, max_fn_calls FROM experiments WHERE name = ?",
            (experiment,),
        ).fetchone()
        if row is None:
            raise KeyError(f"No experiment named {experiment!r} in {self.path}")
        return Config(
            seed=row[0], num_runs=row[1], max_iteration=row[2],
            early_stopping_patience=row[3], max_percentage_error=row[4],
            max_fn_calls=-1 if row[5] is None else row[5],
        )

    def algorithm_parameters(self, experiment: str) -> pd.DataFrame:
        """Recall exactly which parameters each algorithm ran with."""
        df = pd.read_sql(
            "SELECT algorithm, class_name, params_json FROM experiment_algorithms "
            "WHERE experiment = ?",
            self.con,
            params=(experiment,),
        )
        df["params"] = df.pop("params_json").map(json.loads)
        return df

    # ─────────────────────────── writing runs ───────────────────────────

    def completed_runs(self, experiment: str) -> set[tuple]:
        """{(algorithm, problem, run_index), ...} already in the database."""
        rows = self.con.execute(
            "SELECT algorithm, problem, run_index FROM runs WHERE experiment = ?", (experiment,)
        ).fetchall()
        return set(rows)

    def save_task_results(self, experiment: str, run_indices, results: dict):
        """Store one task's output (the dict Optimizer.run returns) as one row
        per run. Called as each task finishes, so a crash loses at most the
        task in flight.
        """
        rows = [
            (
                experiment,
                results["algorithm"],
                results["problem"],
                int(run_index),
                float(results["best_values"][offset]),
                float(results["best_violations"][offset]),
                int(results["fn_calls_per_run"][offset]),
                int(results["iterations_per_run"][offset]),
                int(results["last_improvement_iteration"][offset]),
                int(results["fn_calls_at_best"][offset]),
                _to_blob(results["best_positions"][offset]),
                _to_blob(results["convergence_curves"][offset]),
                float(results["wall_time_per_run"][offset]),
                float(results["cpu_time_per_run"][offset]),
                _to_blob(results["fn_call_curves"][offset]),
            )
            for offset, run_index in enumerate(run_indices)
        ]
        self.con.executemany(
            "INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows
        )
        self.con.commit()

    # ─────────────────────────── reading runs ───────────────────────────

    def load_runs(self, experiment: str) -> pd.DataFrame:
        """All scalar per-run metrics as a flat table (blobs excluded)."""
        return pd.read_sql(
            "SELECT algorithm, problem, run_index, best_value, best_violation, fn_calls, "
            "iterations, last_improvement_iteration, fn_calls_at_best, wall_time, cpu_time "
            "FROM runs WHERE experiment = ? ORDER BY algorithm, problem, run_index",
            self.con,
            params=(experiment,),
        )

    def load_results(self, experiment: str, algorithm: str, problem: str) -> dict:
        """Rebuild the results dict Optimizer.run would have returned for this
        (algorithm, problem) pair — including positions and convergence curves.
        """
        rows = self.con.execute(
            "SELECT run_index, best_value, best_violation, fn_calls, iterations, "
            "last_improvement_iteration, fn_calls_at_best, best_position, convergence, "
            "wall_time, cpu_time, fn_call_curve "
            "FROM runs WHERE experiment = ? AND algorithm = ? AND problem = ? "
            "ORDER BY run_index",
            (experiment, algorithm, problem),
        ).fetchall()
        if not rows:
            raise KeyError(f"No runs for {algorithm!r} on {problem!r} in experiment {experiment!r}")
        columns = list(zip(*rows))
        return {
            "algorithm": algorithm,
            "problem": problem,
            "run_indices": np.array(columns[0], dtype=np.int64),
            "best_values": np.array(columns[1], dtype=np.float64),
            "best_violations": np.array(columns[2], dtype=np.float64),
            "fn_calls_per_run": np.array(columns[3], dtype=np.int64),
            "iterations_per_run": np.array(columns[4], dtype=np.int64),
            "last_improvement_iteration": np.array(columns[5], dtype=np.int64),
            "fn_calls_at_best": np.array(columns[6], dtype=np.int64),
            "best_positions": np.stack([_from_blob(b) for b in columns[7]]),
            "convergence_curves": [_from_blob(b) for b in columns[8]],
            # NULL (pre-migration rows) becomes NaN
            "wall_time_per_run": np.array(columns[9], dtype=np.float64),
            "cpu_time_per_run": np.array(columns[10], dtype=np.float64),
            # None for rows stored before the FE-axis existed
            "fn_call_curves": [None if b is None else _from_blob(b) for b in columns[11]],
        }

    def run_counts(self, experiment: str) -> pd.DataFrame:
        """How many of the experiment's runs each (algorithm, problem) pair has."""
        num_runs = self.experiment_config(experiment).num_runs
        df = pd.read_sql(
            "SELECT algorithm, problem, COUNT(*) AS runs_done FROM runs "
            "WHERE experiment = ? GROUP BY algorithm, problem",
            self.con,
            params=(experiment,),
        )
        df["runs_expected"] = num_runs
        return df

    def summary_table(self, experiment: str, problems: list[Problem]) -> pd.DataFrame:
        """One row of summary statistics per (problem, algorithm) — the same
        metrics as compare_algorithms. Requires the experiment to be complete;
        raises listing what is missing otherwise (merge or run the rest first).
        """
        num_runs = self.experiment_config(experiment).num_runs
        algorithms = [
            row[0]
            for row in self.con.execute(
                "SELECT algorithm FROM experiment_algorithms WHERE experiment = ?", (experiment,)
            )
        ]
        by_name = {p.name: p for p in problems}
        done = self.completed_runs(experiment)

        missing = {
            (algorithm, problem.name): [
                r for r in range(num_runs) if (algorithm, problem.name, r) not in done
            ]
            for problem in problems
            for algorithm in algorithms
        }
        missing = {pair: runs for pair, runs in missing.items() if runs}
        if missing:
            listing = "\n".join(f"  {a} on {p}: runs {r}" for (a, p), r in missing.items())
            raise ValueError(
                f"Experiment {experiment!r} is incomplete:\n{listing}\n"
                f"Run the missing runs or merge the other machine's database first."
            )

        rows = [
            summarize_runs(self.load_results(experiment, algorithm, problem.name), by_name[problem.name])
            for problem in problems
            for algorithm in algorithms
        ]
        return pd.DataFrame(rows)

    # ───────────────────────────── merging ──────────────────────────────

    def merge_from(self, other_path: str, allow_env_mismatch: bool = False) -> int:
        """Copy runs from another machine's database file into this one.

        Experiment metadata must agree (same name → same settings and
        parameters). A run present in both files with identical results is
        skipped; the same run with *different* results is an error — it means
        the two machines did not execute the same experiment definition.
        Returns the number of newly inserted runs.

        Because multi-PC shards have *disjoint* run indices, the runs-level
        conflict check can never catch two machines running different code or
        numpy versions — that would silently poison the merged results. The
        environments table records (git commit, python, numpy, scipy) per
        machine at register time, and merging raises when they disagree for a
        shared experiment. Pass ``allow_env_mismatch=True`` only after
        convincing yourself the difference cannot affect results (databases
        written before this check carry no environment rows and merge
        unchecked).
        """
        if not Path(other_path).exists():
            raise FileNotFoundError(other_path)
        con = self.con
        con.execute("ATTACH DATABASE ? AS other", (str(other_path),))
        try:
            # bring a database written by older code up to the current schema,
            # so the positional INSERT ... SELECT * below lines up
            _migrate(con, "other")
            # metadata must agree; whole-row comparison done in pandas
            for table, keys in (
                ("experiments", ["name"]),
                ("experiment_algorithms", ["experiment", "algorithm"]),
            ):
                mine = pd.read_sql(f"SELECT * FROM main.{table}", con)
                theirs = pd.read_sql(f"SELECT * FROM other.{table}", con)
                if table == "experiments":
                    # rows written before the column existed carry NULL = "disabled"
                    for df in (mine, theirs):
                        df["max_fn_calls"] = df["max_fn_calls"].astype("float64").fillna(-1)
                compare_cols = [c for c in mine.columns if c not in ("git_commit", "created_utc")]
                joined = mine[compare_cols].merge(
                    theirs[compare_cols], on=keys, suffixes=("_mine", "_theirs")
                )
                for column in compare_cols:
                    if column in keys:
                        continue
                    diff = joined[joined[f"{column}_mine"] != joined[f"{column}_theirs"]]
                    if len(diff):
                        raise ValueError(
                            f"{other_path} disagrees with this database on {table}.{column}:\n"
                            f"{diff[keys + [f'{column}_mine', f'{column}_theirs']].to_string(index=False)}"
                        )
                con.execute(f"INSERT OR IGNORE INTO main.{table} SELECT * FROM other.{table}")

            # environment drift: every experiment with runs in both files must
            # have been produced by the same code and library versions
            env_cols = "git_commit, python_version, numpy_version, scipy_version"
            drift = con.execute(
                f"SELECT m.experiment, m.{env_cols.replace(', ', ', m.')}, "
                f"o.{env_cols.replace(', ', ', o.')} "
                f"FROM main.environments m JOIN other.environments o "
                f"USING (experiment) "
                f"WHERE m.experiment IN (SELECT DISTINCT experiment FROM other.runs) "
                f"AND (m.git_commit != o.git_commit "
                f"     OR m.python_version != o.python_version "
                f"     OR m.numpy_version != o.numpy_version "
                f"     OR m.scipy_version != o.scipy_version)"
            ).fetchall()
            if drift and not allow_env_mismatch:
                lines = "\n".join(
                    f"  {e}: here ({g1}, py {p1}, numpy {n1}, scipy {s1}) "
                    f"vs there ({g2}, py {p2}, numpy {n2}, scipy {s2})"
                    for e, g1, p1, n1, s1, g2, p2, n2, s2 in drift[:10]
                )
                raise ValueError(
                    f"{other_path} was produced by a different code/library "
                    f"environment — its shard may not be bit-comparable with "
                    f"this database:\n{lines}\n"
                    f"Rerun on matching versions, or pass "
                    f"allow_env_mismatch=True if the difference is known to "
                    f"be harmless."
                )
            con.execute("INSERT OR IGNORE INTO main.environments SELECT * FROM other.environments")

            conflicting = con.execute(
                "SELECT m.experiment, m.algorithm, m.problem, m.run_index, m.best_value, o.best_value "
                "FROM main.runs m JOIN other.runs o "
                "USING (experiment, algorithm, problem, run_index) "
                "WHERE m.best_value != o.best_value OR m.fn_calls != o.fn_calls "
                "OR m.best_violation != o.best_violation "
                "OR m.iterations != o.iterations "
                "OR m.last_improvement_iteration != o.last_improvement_iteration "
                "OR m.fn_calls_at_best != o.fn_calls_at_best"
            ).fetchall()
            if conflicting:
                lines = "\n".join(
                    f"  {e}/{a}/{p} run {r}: {v1} here vs {v2} there"
                    for e, a, p, r, v1, v2 in conflicting[:10]
                )
                raise ValueError(
                    f"{len(conflicting)} runs in {other_path} conflict with this database "
                    f"(same key, different results) — the machines did not run the same "
                    f"experiment definition:\n{lines}"
                )

            before = con.execute("SELECT COUNT(*) FROM main.runs").fetchone()[0]
            con.execute("INSERT OR IGNORE INTO main.runs SELECT * FROM other.runs")
            inserted = con.execute("SELECT COUNT(*) FROM main.runs").fetchone()[0] - before
            con.commit()
        except BaseException:
            con.rollback()
            raise
        finally:
            con.execute("DETACH DATABASE other")
        return inserted
