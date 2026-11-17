"""Figure 2 of the manuscript: mean wall-clock time per run vs instance size
on four Glover-Kochenberger instances at the common budget (data: the
mkgk_rev2 campaign, all algorithms on one machine).

    .venv/bin/python paper-experiments/fig_walltime.py
"""
import sys, pathlib, sqlite3

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ALGORITHMS = ["SBABC", "SBPSO", "GBABC", "BitABC", "AMABC", "ABABC", "CBGA", "CETS", "BGWO"]
SIZES = {"gk01": 100, "gk06": 200, "gk09": 1500, "gk11": 2500}

con = sqlite3.connect(f"file:{ROOT / 'results' / 'results.sqlite'}?mode=ro", uri=True)
d = pd.read_sql(
    "select problem, algorithm, avg(wall_time) w from runs "
    "where experiment='mkgk_rev2' and problem in ('gk01','gk06','gk09','gk11') "
    "group by problem, algorithm", con)
d["n"] = d.problem.map(SIZES)

fig, ax = plt.subplots(figsize=(6.6, 4.4))
for name in ALGORITHMS:
    sub = d[d.algorithm == name].sort_values("n")
    bold = name == "SBABC"
    ax.plot(sub.n, sub.w, marker="o", ms=3.5, lw=2.2 if bold else 1.2,
            color="black" if bold else None, label=name)
ax.set_xlabel("number of items $n$", fontsize=9)
ax.set_ylabel("mean wall-clock time per run (s)", fontsize=9)
ax.tick_params(labelsize=8)
ax.legend(fontsize=7, frameon=False, ncol=2)
ax.grid(alpha=0.25, lw=0.4)
fig.tight_layout()
manuscript = ROOT / "set based abc latex files" / "Figures"
out = (manuscript if manuscript.is_dir() else ROOT / "results" / "figures") / "fig_walltime.pdf"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, bbox_inches="tight")
print("wrote", out)
