"""Reproduce Table tab:mkp-large9 (Section 7.3) and the nine-instance means
quoted in the conclusion.

Reads results/results.sqlite read-only. For the nine index-00 instances of the
main comparison (mkp_competitors_rev2) it prints each algorithm's mean error
(%) relative to the best-known value over its feasible runs, the table's two
average-rank rows (over all 64 instances on the raw mean objective, ties at
their mid-rank; once among all nine algorithms, once within the
problem-independent tier), and, for the conclusion, every algorithm's mean and
largest error over the nine together with the same figures for SBABC-CB from
mkp_cb_rev1.

Usage: .venv/bin/python paper-experiments/large9_check.py
"""
import os
import sqlite3
import sys

import numpy as np
from scipy.stats import rankdata

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from problems.mkp import load_mkp_problems  # noqa: E402

DB = os.path.join(ROOT, "results", "results.sqlite")
MAIN = "mkp_competitors_rev2"
CB = "mkp_cb_rev1"
ORDER = ["SBABC", "SBPSO", "GBABC", "BitABC", "AMABC", "ABABC", "CBGA", "CETS", "BGWO"]
TIER = ORDER[:5]


def mean_best(con, experiment, algorithm, problem):
    rows = con.execute(
        "SELECT best_value FROM runs WHERE experiment=? AND algorithm=? AND problem=? AND best_violation=0",
        (experiment, algorithm, problem),
    ).fetchall()
    return float(np.mean([r[0] for r in rows])) if rows else float("nan")


def main():
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    problems = [r[0] for r in con.execute("SELECT DISTINCT problem FROM runs WHERE experiment=?", (MAIN,))]
    large = sorted(p for p in problems if p.endswith("-00"))
    optimum = {}
    for suite in [f"mknapcb{i}" for i in range(1, 10)]:
        for pr in load_mkp_problems(suite):
            if pr.name in large:
                optimum[pr.name] = pr.known_optimum
    assert len(large) == 9 and set(optimum) == set(large), large

    def err(experiment, alg, p):
        return (optimum[p] - mean_best(con, experiment, alg, p)) / optimum[p] * 100

    print("Mean error (%) on the nine large instances, main comparison")
    print(f"{'Instance':10s}" + "".join(f"{a:>8s}" for a in ORDER))
    table = {a: [err(MAIN, a, p) for p in large] for a in ORDER}
    for i, p in enumerate(large):
        print(f"{p:10s}" + "".join(f"{table[a][i]:8.2f}" for a in ORDER))

    # Average-rank rows over all 64 instances, raw mean objective, mid-rank ties.
    def avg_rank(algs):
        ranks = []
        for p in problems:
            means = np.array([mean_best(con, MAIN, a, p) for a in algs])
            ranks.append(rankdata(-means, method="average"))
        return np.mean(ranks, axis=0)
    all9 = avg_rank(ORDER)
    tier = avg_rank(TIER)
    print(f"{'Avg. rank (all 64)':20s}" + "".join(f"{r:8.2f}" for r in all9))
    print(f"{'Avg. rank (tier)':20s}" + "".join(f"{r:8.2f}" for r in tier))

    print("\nOver the nine: mean and largest error (%), as quoted in Section 9")
    for a in ORDER:
        print(f"{a:10s} mean {np.mean(table[a]):6.2f}   max {max(table[a]):6.2f}")
    cb = [err(CB, "SBABC-CB", p) for p in large]
    print(f"{'SBABC-CB':10s} mean {np.mean(cb):6.2f}   max {max(cb):6.2f}   (per instance: "
          + ", ".join(f"{e:.2f}" for e in cb) + ")")


if __name__ == "__main__":
    main()
