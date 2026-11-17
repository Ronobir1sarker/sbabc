"""Figure 1 of the manuscript: mean best-so-far error vs consumed function
evaluations on the smallest (5.100-00) and largest (30.500-00) of the nine
large instances of the main comparison — problem-independent tier in the
left column, problem-specific methods on a finer error scale in the right.
Curves average 30 runs and begin once the first run attains feasibility
(infeasible best-so-far values are -inf under the MKP convention).

    .venv/bin/python paper-experiments/fig_convergence.py
"""
import sys, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.store import RunStore
from problems.mkp import load_mkp_problems

EXPERIMENT = "mkp_competitors_rev2"
INDEPENDENT = ["SBABC", "SBPSO", "GBABC", "BitABC", "AMABC"]
SPECIFIC = ["ABABC", "CBGA", "CETS", "BGWO"]
INSTANCES = ["5.100-00", "30.500-00"]
BUDGET = 100_000
GRID = np.linspace(0, BUDGET, 401)

optima = {}
for set_name in ("mknapcb1", "mknapcb9"):
    for p in load_mkp_problems(set_name)[:1]:
        optima[p.name] = p.known_optimum

store = RunStore(str(ROOT / "results" / "results.sqlite"))


def mean_error_curve(algorithm, instance):
    """Mean over runs of the best-so-far error, sampled on the FE grid;
    NaN where any run is still infeasible (the curve starts afterwards)."""
    results = store.load_results(EXPERIMENT, algorithm, instance)
    opt = optima[instance]
    sampled = []
    for values, fes in zip(results["convergence_curves"], results["fn_call_curves"]):
        err = 100.0 * (opt - np.asarray(values, dtype=np.float64)) / opt  # inf while infeasible
        idx = np.clip(np.searchsorted(fes, GRID, side="right") - 1, 0, len(fes) - 1)
        curve = err[idx]
        curve[GRID < fes[0]] = np.inf
        sampled.append(curve)
    mean = np.mean(sampled, axis=0)
    return np.where(np.isfinite(mean), mean, np.nan)


fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.4))
for row, instance in enumerate(INSTANCES):
    for col, (algorithms, title) in enumerate(
        [(INDEPENDENT, "problem-independent"), (SPECIFIC, "problem-specific (finer scale)")]
    ):
        ax = axes[row][col]
        for name in algorithms:
            curve = mean_error_curve(name, instance)
            bold = name == "SBABC"
            ax.plot(GRID / 1e3, curve, lw=2.2 if bold else 1.2,
                    color="black" if bold else None, label=name)
        ax.set_title(f"{instance} — {title}", fontsize=9)
        ax.set_xlabel(r"function evaluations ($\times 10^3$)", fontsize=8)
        if col == 0:
            ax.set_ylabel("mean error (%)", fontsize=8)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=7, frameon=False)
        ax.grid(alpha=0.25, lw=0.4)
fig.tight_layout()
manuscript = ROOT / "set based abc latex files" / "Figures"
out = (manuscript if manuscript.is_dir() else ROOT / "results" / "figures") / "fig_convergence.pdf"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, bbox_inches="tight")
print("wrote", out)
