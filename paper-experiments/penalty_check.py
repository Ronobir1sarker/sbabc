"""Rebuild the constraint-handling table of §7.10 (tab:penalty) from the
results database: the tuned SBABC under Deb's rule (the main comparison,
mkp_competitors_rev2) against the static-penalty variants of a penalty
experiment on the nine alpha=0.25 "-00" instances. Per instance and factor:
mean error (%) over feasible runs, the feasible-run count, and the paired
Wilcoxon test against the Deb runs (same seeds), Holm-corrected within the
instance across its factors, as Section 6.3 states. Infeasible runs enter
the test through deb_scores (below every feasible value), never through
their raw profit.

    .venv/bin/python paper-experiments/penalty_check.py [experiment]

Default experiment: mkp_penalty_rev2 (the paper's). mkp_penalty_rev1 is the
frozen, superseded design (see run_experiment.build_mkp_penalty_rev2)."""
import os, sys, sqlite3, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from experiments.analysis import pairwise_stats, holm_correction
from problems.mkp import load_mkp_problems

EXP = sys.argv[1] if len(sys.argv) > 1 else "mkp_penalty_rev2"
DEB_EXP = "mkp_competitors_rev2"
con = sqlite3.connect(f"file:{os.path.join(ROOT, 'results', 'results.sqlite')}?mode=ro", uri=True)
q = "select experiment, algorithm, problem, run_index, best_value, best_violation from runs where experiment=?"
pen = pd.read_sql(q, con, params=(EXP,))
deb = pd.read_sql(q + " and algorithm='SBABC'", con, params=(DEB_EXP,))
probs = [load_mkp_problems(f"mknapcb{i}")[0] for i in range(1, 10)]
names = [p.name for p in probs]
opt = {p.name: p.known_optimum for p in probs}
deb = deb[deb.problem.isin(names)]
factors = sorted(pen.algorithm.unique(), key=lambda a: float(a[len("SBABC_pen"):]))
assert len(deb) == 30 * 9, len(deb)
assert len(pen) == 30 * 9 * len(factors), (len(pen), factors)

def err(values, name):
    return 100 * (opt[name] - np.mean(values)) / opt[name]

header = ["Instance", "Deb's rule"] + [f"penalty x{a[len('SBABC_pen'):]}" for a in factors]
print(" & ".join(header) + r" \\")
n_sig = 0; feas_total = {a: 0 for a in factors}; raw_sig = []
for name in names:
    d = deb[deb.problem == name].sort_values("run_index")
    assert (d.best_violation <= 0).all()
    cells = [name, f"{err(d.best_value.values, name):.2f}"]
    pvals, stats = [], []
    for a in factors:
        g = pen[(pen.algorithm == a) & (pen.problem == name)].sort_values("run_index")
        assert list(g.run_index) == list(d.run_index)
        feas = g.best_violation.values <= 0
        feas_total[a] += int(feas.sum())
        st = pairwise_stats(g.best_value.values, d.best_value.values, "max",
                            violations_a=g.best_violation.values, violations_b=d.best_violation.values)
        pvals.append(st["p_value"]); stats.append(st)
        mean_err = err(g.best_value.values[feas], name) if feas.any() else np.nan
        cells.append((f"{mean_err:.2f}" if feas.all() else
                      (f"{mean_err:.2f} ({int(feas.sum())}/30)" if feas.any() else "-- (0/30)")))
    adj = holm_correction(pvals)
    for a, st, p_raw, p_adj in zip(factors, stats, pvals, adj):
        raw_sig.append(p_raw < 0.05)
        if p_adj < 0.05:
            n_sig += 1
            print(f"  SIGNIFICANT {name} {a}: winner={st['winner']} p={p_raw:.4f} holm={p_adj:.4f} delta={st['cliffs_delta']:.2f}")
    print(" & ".join(cells) + r" \\" + "   % raw p: " + ", ".join(f"{p:.3f}" for p in pvals))
print(f"\nfeasible runs per factor: {feas_total}  (of 270 each)")
print(f"significant after Holm within instance: {n_sig} of {9 * len(factors)}; raw p<0.05: {sum(raw_sig)}")
