"""Score mkp_ablation_rev2 exactly as Table 4's rows are scored (evals/iter,
small/large mean error, W/L/ns vs the four-candidate reference with paired
Wilcoxon per instance + Holm over 19), after (a) verifying that the mda
control reproduces rev1 digit for digit and (b) verifying that the scoring
reproduces every published rev1 row."""
import os, sys, io, sqlite3, warnings
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import run_experiment as rx

con = sqlite3.connect(f"file:{os.path.join(ROOT, 'results', 'results.sqlite')}?mode=ro", uri=True)
d = pd.read_sql("select experiment, algorithm, problem, run_index, best_value, best_violation, "
                "fn_calls, iterations, best_position from runs where experiment in "
                "('mkp_ablation_rev1','mkp_ablation_rev2')", con)
probs = rx._ablation_instances()
opt = {p.name: p.known_optimum for p in probs}
small = [p.name for p in probs[:10]]; large = [p.name for p in probs[10:]]
assert all(o is not None for o in opt.values())

# (a) reproduction control
r1 = d[(d.experiment == "mkp_ablation_rev1") & (d.algorithm == "SBABC_mda_only")].set_index(["problem", "run_index"])
r2 = d[(d.experiment == "mkp_ablation_rev2") & (d.algorithm == "SBABC_mda_only")].set_index(["problem", "run_index"])
r1, r2 = r1.align(r2, join="inner")
same = (r1.best_value == r2.best_value) & (r1.fn_calls == r2.fn_calls) & (r1.iterations == r2.iterations)
print(f"mda control: {int(same.sum())}/{len(same)} runs identical to rev1 (value, fn_calls, iterations)")
d = d[~((d.experiment == "mkp_ablation_rev2") & (d.algorithm == "SBABC_mda_only"))]  # keep rev1's row

SENT = -1e18  # infeasible (-inf) ranks below every feasible value in the paired test
def pivot(alg):
    g = d[d.algorithm == alg]
    return g.pivot(index="run_index", columns="problem", values="best_value").replace(-np.inf, SENT)

def wlns(alg, ref):
    A, R = pivot(alg), pivot(ref)
    pvals, signs = [], []
    for p in opt:
        a, r = A[p].values, R[p].values
        diff = a - r
        if np.all(diff == 0):
            pvals.append(1.0); signs.append(0); continue
        pvals.append(wilcoxon(a, r).pvalue); signs.append(np.sign(diff.mean()))
    # Holm over the 19 tests
    order = np.argsort(pvals); m = len(pvals); sig = [False] * m
    for rank, i in enumerate(order):
        if pvals[i] <= 0.05 / (m - rank): sig[i] = True
        else: break
    w = sum(1 for s, g in zip(sig, signs) if s and g > 0)
    l = sum(1 for s, g in zip(sig, signs) if s and g < 0)
    return f"{w}/{l}/{m - w - l}"

def row(alg):
    g = d[d.algorithm == alg]
    epi = g.fn_calls.sum() / g.iterations.sum()
    means = g.groupby("problem").best_value.mean()
    err = {p: 100 * (opt[p] - means[p]) / opt[p] for p in opt}
    infeas = [p for p in opt if not np.isfinite(err[p])]
    f = lambda ps: "--" if any(p in infeas for p in ps) else f"{np.mean([err[p] for p in ps]):.3f}"
    return epi, f(small), f(large), len(infeas)

print("\n(b) scoring check: rev1 rows (published: full ref, textbook 0/0/19, sharedV3 0/2/17, mda 9/0/10,"
      " no_add 0/11/8, no_drop 0/17/2, move 0/18/1, drop 0/19/0, add 0/19/0)")
for alg in ["SBABC_full", "SBABC_textbook_ties", "SBABC_sharedV3", "SBABC_mda_only", "SBABC_no_add",
            "SBABC_no_drop", "SBABC_move_only", "SBABC_drop_only", "SBABC_add_only"]:
    epi, s, l, ninf = row(alg)
    print(f"  {alg:22s} {epi:6.1f}  small {s:>7}  large {l:>6}  vs full {wlns(alg, 'SBABC_full') if alg != 'SBABC_full' else 'ref.':8s} infeasible-instances {ninf}")

print("\n(c) NEW single-candidate arms")
for alg in ["SBABC_mda_only", "SBABC_md_only", "SBABC_ma_only", "SBABC_da_only", "SBABC_move_only", "SBABC_drop_only", "SBABC_add_only"]:
    epi, s, l, ninf = row(alg)
    print(f"  {alg:22s} {epi:6.1f}  small {s:>7}  large {l:>6}  vs full {wlns(alg, 'SBABC_full'):8s} vs mda {wlns(alg, 'SBABC_mda_only'):8s} infeasible-instances {ninf}")

print("\n(d) per-instance mean error %, new arms vs mda and full")
tab = {}
for alg in ["SBABC_full", "SBABC_mda_only", "SBABC_md_only", "SBABC_ma_only", "SBABC_da_only"]:
    means = d[d.algorithm == alg].groupby("problem").best_value.mean()
    tab[alg.replace("SBABC_", "").replace("_only", "")] = pd.Series({p: 100 * (opt[p] - means[p]) / opt[p] for p in opt})
print(pd.DataFrame(tab).loc[list(opt)].round(3).to_string())

print("\n(e) mean number of selected items in final solutions (large instances)")
from experiments.store import _from_blob
g = d[d.problem.isin(large) & d.algorithm.isin(["SBABC_full", "SBABC_mda_only", "SBABC_md_only", "SBABC_ma_only", "SBABC_da_only"])].copy()
g["size"] = g.best_position.map(lambda b: int(_from_blob(b).sum()))
print(g.groupby("algorithm")["size"].mean().round(1).to_string())
