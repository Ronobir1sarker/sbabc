"""One-at-a-time check of the tuned SBABC on the knapsack (experiment
mkp_sensitivity_rev1), quoted in the Limitations of the paper and tabulated
in the supplementary material. The database is opened read-only.

    python paper-experiments/sensitivity_oat_check.py

The experiment runs the tuned configuration and 20 others that differ from
it in one parameter (four further levels each of beta1, beta2, beta3, k and
the abandonment limit) on the 19 instances of the ablation, 30 runs each at
10^5 evaluations. For every setting the script prints the mean error over
the ten small and the nine large instances and the number of instances on
which the setting is significantly better than, worse than, or
indistinguishable from the tuned configuration: paired Wilcoxon per instance
(experiments.analysis.pairwise_stats), Holm over the 19 instances of the
setting, alpha 0.05. The last line gives the count quoted in the paper: the
settings of beta3, k and the limit that are better than the tuned one on at
least one instance.
"""
import os
import sqlite3
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import run_experiment  # noqa: E402
from experiments.analysis import holm_correction, pairwise_stats  # noqa: E402

DB = os.path.join(ROOT, "results", "results.sqlite")
EXPERIMENT = "mkp_sensitivity_rev1"
REFERENCE = "SBABC_tuned"
ALPHA = 0.05


def main():
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    runs = pd.read_sql(
        "select algorithm, problem, run_index, best_value, best_violation from runs "
        "where experiment=? order by algorithm, problem, run_index", con, params=(EXPERIMENT,))
    problems = run_experiment._ablation_instances()
    best = {p.name: p.known_optimum for p in problems}
    small, large = [p.name for p in problems[:10]], [p.name for p in problems[10:]]
    assert sorted(best) == sorted(runs.problem.unique()), "the experiment must cover the 19 ablation instances"

    def side(algorithm, problem):
        one = runs[(runs.algorithm == algorithm) & (runs.problem == problem)]
        return one.best_value.values, one.best_violation.values

    def error(algorithm, names):
        mean = runs[(runs.algorithm == algorithm) & (runs.best_violation <= 0)].groupby("problem").best_value.mean()
        return float(np.mean([100 * (best[p] - mean[p]) / best[p] for p in names]))

    settings = sorted((a for a in runs.algorithm.unique() if a != REFERENCE),
                      key=lambda s: (s.split("=")[0], float(s.split("=")[1])))
    print(f"{'setting':20s} {'small err %':>11s} {'large err %':>11s}   better/worse/no difference (19 instances)")
    print(f"{'tuned configuration':20s} {error(REFERENCE, small):11.3f} {error(REFERENCE, large):11.2f}   reference")
    better_somewhere = []
    for setting in settings:
        stats = []
        for problem in best:
            a, va = side(setting, problem)
            b, vb = side(REFERENCE, problem)
            stats.append(pairwise_stats(a, b, "max", paired=True, violations_a=va, violations_b=vb))
        adjusted = holm_correction([s["p_value"] for s in stats])
        won = sum(1 for s, p in zip(stats, adjusted) if p < ALPHA and s["winner"] == "A")
        lost = sum(1 for s, p in zip(stats, adjusted) if p < ALPHA and s["winner"] == "B")
        label = setting.replace("SBABC_", "")
        print(f"{label:20s} {error(setting, small):11.3f} {error(setting, large):11.2f}   {won}/{lost}/{len(stats) - won - lost}")
        if won and label.split("=")[0] in ("beta3", "k", "limit"):
            better_somewhere.append(label)
    quoted = [s for s in settings if s.replace("SBABC_", "").split("=")[0] in ("beta3", "k", "limit")]
    print(f"\nsettings of beta3, k and the limit: {len(quoted)}; significantly better than the tuned "
          f"configuration on at least one instance: {len(better_somewhere)} {better_somewhere}")


if __name__ == "__main__":
    main()
