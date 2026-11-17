"""Summary and comparison statistics for the multi-run results returned by
Optimizer.run().

Besides the per-(algorithm, problem) summary row, this module provides the
pairwise statistics the paper's tables need next to every comparison:
Wilcoxon tests, Cliff's delta effect sizes, and bootstrap confidence
intervals (all deterministic — the bootstrap is seeded).
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from problems.base import Problem

# How the paired Wilcoxon signed-rank test treats pairs of runs with equal
# values. "wilcox" discards them, as in Wilcoxon's original procedure and as
# statistical software does by default. This is the rule the paper states
# (Section 6.3), and every significance mark of its tables follows it.
WILCOXON_ZERO_METHOD = "wilcox"


def summarize_runs(results: dict, problem: Problem) -> dict:
    """Flatten one algorithm-on-one-problem result into a row of metrics.

    Value/error statistics are computed over *feasible* runs only — never
    trust the value channel of an infeasible run (the MKP convention masks it
    to -inf, but SCP-style evaluators report the raw objective there, which
    would otherwise masquerade as a best). Infeasible runs still show up in
    num_runs vs feasible_runs. With zero feasible runs the value metrics are
    NaN.

    Error metrics are only present when the problem's optimum is known;
    a "success" is a feasible run whose error percentage is <= 1e-9 — a
    relative tolerance rather than float equality, so instances with a
    fractional optimum (mknap1-2's 8706.1) cannot flip to zero successes on
    a numpy/BLAS whose dot-product summation order differs by one ULP.
    When known_optimum is a best-known value rather than a proven optimum
    (extras["optimum_proven"] is False for mknapcb 30.250/30.500 and the gk
    set), "success" therefore means "reached or beat the best known" and
    mean_error_pct can be negative — word paper tables accordingly.

    Standard deviations are the *sample* std (ddof=1), the convention journal
    tables assume for n independent repetitions; NaN when fewer than 2
    feasible runs exist.
    """
    values = results["best_values"]
    feasible = results["best_violations"] <= 0
    feasible_values = values[feasible]

    if len(feasible_values):
        best = np.max(feasible_values) if problem.optimization_type == "max" else np.min(feasible_values)
    else:
        best = np.nan

    row = {
        "algorithm": results["algorithm"],
        "problem": problem.name,
        "best_value": best,
        "mean_value": np.mean(feasible_values) if len(feasible_values) else np.nan,
        "std_value": np.std(feasible_values, ddof=1) if len(feasible_values) > 1 else np.nan,
        "feasible_runs": int(feasible.sum()),
        "num_runs": len(values),
        "mean_fn_calls": np.mean(results["fn_calls_per_run"]),
        "mean_fn_calls_at_best": np.mean(results["fn_calls_at_best"]),
        "mean_iterations": np.mean(results["iterations_per_run"]),
    }
    if "cpu_time_per_run" in results:  # absent in results stored before timing existed
        row["mean_cpu_time"] = np.mean(results["cpu_time_per_run"])
        row["mean_wall_time"] = np.mean(results["wall_time_per_run"])

    if problem.known_optimum is not None:
        errors = np.array([problem.error_percentage(v) for v in feasible_values])
        row.update(
            best_error_pct=problem.error_percentage(best) if len(feasible_values) else np.nan,
            mean_error_pct=np.mean(errors) if len(errors) else np.nan,
            std_error_pct=np.std(errors, ddof=1) if len(errors) > 1 else np.nan,
            success_runs=int(np.sum(errors <= 1e-9)),
        )
    return row


# ───────────────────── pairwise comparison statistics ─────────────────────


def cliffs_delta(a, b) -> float:
    """Cliff's delta in [-1, 1]: P(a > b) - P(a < b) over all cross pairs.

    Purely ordinal, so it tolerates ties, non-normality and ±inf values —
    which is why it is the recommended effect size for metaheuristic run
    results. Positive means sample `a` tends to be larger.
    """
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    greater = (a[:, None] > b[None, :]).sum()
    less = (a[:, None] < b[None, :]).sum()
    return float(greater - less) / (len(a) * len(b))


def delta_magnitude(delta: float) -> str:
    """Romano et al. (2006) verbal labels for |Cliff's delta|."""
    d = abs(delta)
    if d < 0.147:
        return "negligible"
    if d < 0.33:
        return "small"
    if d < 0.474:
        return "medium"
    return "large"


def bootstrap_ci(
    a, b, statistic=np.mean, n_boot: int = 10_000, confidence: float = 0.95, seed: int = 0
) -> tuple[float, float]:
    """Seeded percentile-bootstrap CI for statistic(a) - statistic(b).

    Resamples the two groups independently (the runs are independent
    repetitions). Returns (low, high); contains 0 ⇔ no reliable difference.
    """
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    rng = np.random.default_rng(seed)
    diffs = np.array(
        [
            statistic(rng.choice(a, len(a), replace=True))
            - statistic(rng.choice(b, len(b), replace=True))
            for _ in range(n_boot)
        ]
    )
    tail = 100 * (1 - confidence) / 2
    return float(np.percentile(diffs, tail)), float(np.percentile(diffs, 100 - tail))


def deb_scores(values, violations, optimization_type: str = "max") -> np.ndarray:
    """Finite per-run scores ordered by Deb's rule, for rank-based statistics.

    Feasible runs keep their objective value. Infeasible runs score strictly
    worse than every feasible one, ordered among themselves by violation —
    never trust the value channel of an infeasible run (raw for SCP-style
    evaluators, ±inf for the MKP convention; both are handled here). The
    infeasible magnitudes are artificial, so tests on scores containing them
    are ordinally Deb-consistent but their signed-rank magnitudes for those
    pairs are not meaningful — feasibility rates should be reported alongside.
    """
    values = np.asarray(values, dtype=np.float64)
    violations = np.asarray(violations, dtype=np.float64)
    feasible = violations <= 0
    scores = values.copy()

    finite_feasible = values[feasible & np.isfinite(values)]
    worst = 0.0
    if len(finite_feasible):
        worst = finite_feasible.min() if optimization_type == "max" else finite_feasible.max()
    step = 1.0 + violations[~feasible]
    scores[~feasible] = worst - step if optimization_type == "max" else worst + step
    return scores


def pairwise_stats(
    a,
    b,
    optimization_type: str = "max",
    paired: bool = True,
    n_boot: int = 10_000,
    seed: int = 0,
    violations_a=None,
    violations_b=None,
) -> dict:
    """Everything a results table needs for "algorithm A vs algorithm B" on
    one problem, from their per-run best values.

    Pass `violations_a`/`violations_b` whenever available: infeasible runs are
    then mapped through `deb_scores`, which keeps the tests Deb-consistent for
    evaluators that report raw objectives on infeasible solutions and keeps
    ±inf out of the test statistics. Without them, non-finite values are
    treated as infeasible with unit violation.

    `paired=True` uses the Wilcoxon signed-rank test — correct here because
    run r of every algorithm starts from the same child seed (common random
    numbers); pairs of runs with equal values are discarded
    (`WILCOXON_ZERO_METHOD`). `paired=False` falls back to the rank-sum test
    for unequal run counts. Effect direction is normalized so positive delta / "A" winner
    always means A performed *better*, regardless of min/max.
    """
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if violations_a is None:
        violations_a = (~np.isfinite(a)).astype(np.float64)
    if violations_b is None:
        violations_b = (~np.isfinite(b)).astype(np.float64)
    # one shared "worst feasible" anchor: score the pooled samples together
    pooled = deb_scores(
        np.concatenate([a, b]), np.concatenate([violations_a, violations_b]), optimization_type
    )
    a, b = pooled[: len(a)], pooled[len(a) :]
    all_feasible = (np.asarray(violations_a) <= 0).all() and (np.asarray(violations_b) <= 0).all()
    sign = 1.0 if optimization_type == "max" else -1.0

    if paired and len(a) == len(b):
        if np.array_equal(a, b):
            p_value = 1.0  # identical samples: signed-rank is undefined, difference is surely absent
        else:
            p_value = float(stats.wilcoxon(a, b, zero_method=WILCOXON_ZERO_METHOD).pvalue)
        test = "wilcoxon_signed_rank"
    else:
        p_value = float(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue)
        test = "wilcoxon_rank_sum"

    delta = cliffs_delta(sign * a, sign * b)
    result = {
        "test": test,
        "p_value": p_value,
        "cliffs_delta": delta,  # > 0 ⇔ A better
        "delta_magnitude": delta_magnitude(delta),
        "winner": "A" if delta > 0 else ("B" if delta < 0 else "tie"),
    }
    # A mean-difference CI is only meaningful over true objective values,
    # not the artificial magnitudes deb_scores assigns infeasible runs
    if all_feasible:
        low, high = bootstrap_ci(sign * a, sign * b, n_boot=n_boot, seed=seed)
        result.update(mean_diff_ci_low=low, mean_diff_ci_high=high, ci_excludes_zero=low > 0 or high < 0)
    return result


def holm_correction(p_values) -> list[float]:
    """Holm's step-down adjustment for one family of hypotheses; returns
    adjusted p-values in the input order (monotone, capped at 1).

    Controls the family-wise error rate with no independence assumption —
    the declared multiple-comparison stance for the paper's tables: on one
    problem, testing the reference against m competitors is m simultaneous
    hypotheses, so significance stars belong on these adjusted values.
    """
    p = np.asarray(p_values, dtype=np.float64)
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, idx in enumerate(np.argsort(p, kind="stable")):
        running = max(running, (len(p) - rank) * p[idx])
        adjusted[idx] = min(1.0, running)
    return [float(x) for x in adjusted]


def significance_table(store, experiment: str, problems: list[Problem], reference: str):
    """One row per (problem, competitor): the reference algorithm compared
    against every other algorithm of the experiment, with Wilcoxon p-value
    (raw, plus `p_holm` adjusted over the problem's competitor family),
    Cliff's delta (+ magnitude) and a bootstrap CI on the mean difference.

    Every comparison requires both algorithms to cover the *same* run
    indices: the paired statistics rely on run r sharing one child seed
    (common random numbers), and pairing two equal-length but different run
    sets — possible after mid-campaign task failures — would silently break
    that. Complete the experiment's missing runs first.

    `store` is a RunStore; passed in (not imported) to keep this module
    dependency-free of the persistence layer.
    """
    import pandas as pd

    algorithms = list(store.algorithm_parameters(experiment)["algorithm"])
    if reference not in algorithms:
        raise ValueError(f"Reference {reference!r} not in experiment {experiment!r}: {algorithms}")

    rows = []
    for problem in problems:
        ref = store.load_results(experiment, reference, problem.name)
        problem_rows = []
        for algorithm in algorithms:
            if algorithm == reference:
                continue
            other = store.load_results(experiment, algorithm, problem.name)
            if not np.array_equal(ref["run_indices"], other["run_indices"]):
                raise ValueError(
                    f"{reference!r} and {algorithm!r} on {problem.name!r} cover "
                    f"different run indices ({ref['run_indices'].tolist()} vs "
                    f"{other['run_indices'].tolist()}) — positional pairing would "
                    f"silently mismatch the common-random-numbers runs. Complete "
                    f"the experiment's missing runs first."
                )
            row = {"problem": problem.name, "reference": reference, "algorithm": algorithm}
            row.update(
                pairwise_stats(
                    ref["best_values"],
                    other["best_values"],
                    problem.optimization_type,
                    violations_a=ref["best_violations"],
                    violations_b=other["best_violations"],
                )
            )
            problem_rows.append(row)
        for row, adjusted in zip(problem_rows, holm_correction([r["p_value"] for r in problem_rows])):
            row["p_holm"] = adjusted
        rows.extend(problem_rows)
    return pd.DataFrame(rows)
