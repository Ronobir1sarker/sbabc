"""Reproduce the tie-rule comparison on the single-candidate layout quoted in
Sections 5.3, 5.4 and 7.3.

Reads results/results.sqlite read-only. The strict-improvement rule of the
original ABC (a candidate replaces its source only if it is strictly better)
is compared with the tie-accepting rule used in the experiments, run for run,
on the 19 ablation instances, in two settings:

  ablation setting   mkp_tierule_rev1 / SBABC_mda_strict
                     against mkp_ablation_rev2 / SBABC_mda_only   (seed 9)
  tuned setting      mkp_tierule_main_rev1 / SBABC_strict
                     against mkp_competitors_rev2 / SBABC          (seed 4)

Scoring is that of Table tab:ablation (ablation_rev2_check.py): paired
Wilcoxon per instance over the 30 runs, Holm over the 19 instances at 0.05;
mean error over the ten small and the nine large instances; evaluations per
iteration.

Usage: .venv/bin/python paper-experiments/tie_rule_check.py
"""
import os
import sqlite3
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import run_experiment as rx  # noqa: E402

DB = os.path.join(ROOT, "results", "results.sqlite")
SETTINGS = [
    ("ablation setting (star topology, seed 9)",
     ("mkp_tierule_rev1", "SBABC_mda_strict"), ("mkp_ablation_rev2", "SBABC_mda_only")),
    ("tuned setting (ring topology, seed 4)",
     ("mkp_tierule_main_rev1", "SBABC_strict"), ("mkp_competitors_rev2", "SBABC")),
]
SENT = -1e18  # an infeasible run (-inf) ranks below every feasible value


def load(con, experiment, algorithm, names):
    df = pd.read_sql(
        "select problem, run_index, best_value, fn_calls, iterations from runs "
        "where experiment=? and algorithm=?", con, params=(experiment, algorithm))
    return df[df.problem.isin(names)]


def main():
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    probs = rx._ablation_instances()
    opt = {p.name: p.known_optimum for p in probs}
    dim = {p.name: p.dim for p in probs}
    names = [p.name for p in probs]
    small, large = names[:10], names[10:]
    for label, strict_key, ties_key in SETTINGS:
        s, t = load(con, *strict_key, names), load(con, *ties_key, names)
        print(f"\n== {label}: {strict_key[0]}/{strict_key[1]} against {ties_key[0]}/{ties_key[1]}")
        if len(s) != 570 or len(t) != 570:
            print(f"   incomplete: {len(s)} strict runs, {len(t)} stored runs (570 expected)")
            continue
        S = s.pivot(index="run_index", columns="problem", values="best_value").replace(-np.inf, SENT)
        T = t.pivot(index="run_index", columns="problem", values="best_value").replace(-np.inf, SENT)
        pvals, signs = [], []
        for n in names:
            d = S[n].values - T[n].values
            pvals.append(1.0 if np.all(d == 0) else wilcoxon(S[n].values, T[n].values).pvalue)
            signs.append(np.sign(d.mean()))
        order = np.argsort(pvals)
        m = len(pvals)
        sig = [False] * m
        for rank, i in enumerate(order):  # Holm over the 19 instances
            if pvals[i] <= 0.05 / (m - rank):
                sig[i] = True
            else:
                break
        err = lambda df, n: 100 * (opt[n] - df[df.problem == n].best_value.mean()) / opt[n]  # noqa: E731
        print(f"   {'instance':10s} {'n':>4s} {'ties':>8s} {'strict':>8s} {'p':>8s}")
        for n, p, sg, sgn in zip(names, pvals, sig, signs):
            mark = "strict better" if sg and sgn > 0 else ("strict worse" if sg and sgn < 0 else "")
            print(f"   {n:10s} {dim[n]:4d} {err(t, n):8.3f} {err(s, n):8.3f} {p:8.4f}  {mark}")
        w = sum(1 for sg, sgn in zip(sig, signs) if sg and sgn > 0)
        l = sum(1 for sg, sgn in zip(sig, signs) if sg and sgn < 0)  # noqa: E741
        mean = lambda df, ns: np.mean([err(df, n) for n in ns])  # noqa: E731
        epi = lambda df: df.fn_calls.sum() / df.iterations.sum()  # noqa: E731
        print(f"   strict against ties-accepted (better/worse/no difference): {w}/{l}/{m - w - l}")
        print(f"   mean error, ten small instances: ties {mean(t, small):.3f}  strict {mean(s, small):.3f}")
        print(f"   mean error, nine large instances: ties {mean(t, large):.2f}  strict {mean(s, large):.2f}")
        print(f"   evaluations per iteration: ties {epi(t):.1f}  strict {epi(s):.1f}")


if __name__ == "__main__":
    main()
