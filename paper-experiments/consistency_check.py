"""Reproduce Table tab:consistency (Section 7.8, consistency across benchmarks).

Reads results/results.sqlite read-only. For each benchmark the algorithms are
ranked on every instance by the mean best objective value over feasible runs
(ties receive their mid-rank, as in Section 6.3) and the average rank over the
instances is reported together with the standing it implies. Also prints the
two summaries (mean rank over the four benchmarks, worst standing), the
variant with the Glover-Kochenberger instances as a fifth benchmark, and the
six-algorithm standings with ABABC included, all quoted in the text.

Usage: .venv/bin/python paper-experiments/consistency_check.py
"""
import os
import sqlite3

import pandas as pd
from scipy.stats import rankdata

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(ROOT, "results", "results.sqlite")

FIVE = ["SBABC", "SBPSO", "GBABC", "BitABC", "AMABC"]
SIX = FIVE + ["ABABC"]
BENCHMARKS = [
    ("MKP (64)", "mkp_competitors_rev2"),
    ("MKP large (90)", "mkp_large90_rev2"),
    ("MDP (15)", "mdp_main_rev1"),
    ("FS (7)", "fs_main_rev1"),
]
FIFTH = ("GK (11)", "mkgk_rev2")


def average_ranks(con, experiment, algorithms):
    """Average rank of each algorithm over the instances of one experiment."""
    df = pd.read_sql(
        "select algorithm, problem, best_value, best_violation from runs "
        "where experiment=?", con, params=(experiment,))
    df = df[df.algorithm.isin(algorithms)]
    infeasible = df[df.best_violation > 0]
    if len(infeasible):
        print(f"  note: {experiment} has {len(infeasible)} infeasible final runs "
              f"{infeasible.groupby('algorithm').size().to_dict()} (excluded)")
    feasible = df[df.best_violation == 0]
    means = feasible.groupby(["problem", "algorithm"]).best_value.mean().unstack()[algorithms]
    if means.isna().any().any():
        raise RuntimeError(f"{experiment}: an algorithm has no feasible run on some instance")
    # all four domains maximise, so a larger mean value is the better rank
    ranks = means.apply(lambda row: pd.Series(rankdata(-row.values), index=row.index), axis=1)
    return ranks.mean(), len(means)


def standings(avg):
    return pd.Series(rankdata(avg.values, method="min"), index=avg.index).astype(int)


def summarise(con, algorithms, benchmarks, title):
    print(f"\n== {title}")
    avgs, stands = {}, {}
    for label, exp in benchmarks:
        avg, n = average_ranks(con, exp, algorithms)
        avgs[label], stands[label] = avg, standings(avg)
        assert n == int(label.split("(")[1].rstrip(")")), (label, n)
    table = pd.DataFrame({
        label: [f"{stands[label][a]} ({avgs[label][a]:.2f})" for a in algorithms]
        for label, _ in benchmarks}, index=algorithms)
    table["Mean rank"] = [f"{sum(avgs[l][a] for l, _ in benchmarks) / len(benchmarks):.2f}"
                          for a in algorithms]
    table["Worst standing"] = [max(stands[l][a] for l, _ in benchmarks) for a in algorithms]
    print(table.to_string())


def main():
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    summarise(con, FIVE, BENCHMARKS, "Table tab:consistency: five algorithms, four benchmarks")
    summarise(con, FIVE, BENCHMARKS + [FIFTH], "Variant quoted in the text: GK as a fifth benchmark")
    summarise(con, SIX, BENCHMARKS, "Six-algorithm standings quoted in the text (ABABC included)")


if __name__ == "__main__":
    main()
