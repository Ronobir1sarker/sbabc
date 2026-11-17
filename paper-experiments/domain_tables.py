"""Reproduces the maximum diversity table (tab:mdp) and the feature selection
table (tab:fs) of the paper from results/results.sqlite (opened read-only).

    python paper-experiments/domain_tables.py --domain mdp
    python paper-experiments/domain_tables.py --domain fs

Each row prints the table's cells (mean error % over feasible runs on MDP;
mean accuracy % / mean subset size on FS) followed by the significance mark
of the paper's protocol (Section 6.3): paired Wilcoxon over the 30
common-seed runs, Holm-corrected within each instance over the five
competitors, alpha 0.05; '+' = SBABC significantly better, '-' = worse. The
last lines give the average instance rank per algorithm and SBABC's tally of
wins / losses / non-significant comparisons against each rival.
"""
import argparse, io, os, sqlite3, sys
import numpy as np
from scipy.stats import rankdata

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from experiments.analysis import pairwise_stats, holm_correction  # noqa: E402

ALGS = ["SBABC", "SBPSO", "GBABC", "BitABC", "AMABC", "ABABC"]
MDP = ["SOM-b_2_n100_m20", "SOM-b_6_n200_m40", "SOM-b_9_n300_m30", "SOM-b_13_n400_m40",
       "SOM-b_17_n500_m50", "GKD-c_1_n500_m50", "GKD-c_2_n500_m50", "GKD-c_3_n500_m50",
       "GKD-c_4_n500_m50", "MDG-a_1_n500_m50", "MDG-a_2_n500_m50", "MDG-a_3_n500_m50",
       "MDG-b_1_n500_m50", "MDG-b_2_n500_m50", "MDG-b_3_n500_m50"]
FS = ["fs_wine", "fs_wdbc", "fs_ionosphere", "fs_sonar", "fs_segment", "fs_german_credit",
      "fs_hill_valley"]


def connect():
    return sqlite3.connect(f"file:{os.path.join(ROOT, 'results', 'results.sqlite')}?mode=ro", uri=True)


def load(con, experiment, algorithm, problem, positions=False):
    rows = con.execute(
        "select best_value, best_violation, best_position from runs "
        "where experiment=? and algorithm=? and problem=? order by run_index",
        (experiment, algorithm, problem)).fetchall()
    values = np.array([r[0] for r in rows], float)
    violations = np.array([r[1] for r in rows], float)
    sizes = np.array([int(np.load(io.BytesIO(r[2])).sum()) for r in rows]) if positions else None
    return values, violations, sizes


def marks_for(sb, rivals, optimization_type):
    """Significance marks of the rivals against SBABC on one instance."""
    stats = {a: pairwise_stats(sb[0], v[0], optimization_type=optimization_type, paired=True,
                               violations_a=sb[1], violations_b=v[1]) for a, v in rivals.items()}
    adjusted = holm_correction([stats[a]["p_value"] for a in rivals])
    out = {}
    for a, p in zip(rivals, adjusted):
        out[a] = "" if p >= 0.05 else ("+" if stats[a]["winner"] == "A" else "-")
    return out


def table(domain):
    con = connect()
    if domain == "mdp":
        from problems.mdp import load_mdp_problems
        experiment, order, opt_type = "mdp_main_rev1", MDP, "max"
        problems = {p.name: p for p in load_mdp_problems(order)}
    else:
        from problems.feature_selection import load_fs_problems
        experiment, order, opt_type = "fs_main_rev1", FS, "max"
        problems = {p.name: p for p in load_fs_problems([o[3:] for o in order])}
    ranks = {a: [] for a in ALGS}
    tally = {a: [0, 0, 0] for a in ALGS[1:]}
    for name in order:
        vals = {a: load(con, experiment, a, name, positions=(domain == "fs")) for a in ALGS}
        marks = marks_for(vals["SBABC"], {a: vals[a] for a in ALGS[1:]}, opt_type)
        for a, m in marks.items():
            tally[a][0 if m == "+" else 1 if m == "-" else 2] += 1
        means = np.array([vals[a][0][vals[a][1] <= 0].mean() for a in ALGS])
        for a, r in zip(ALGS, rankdata(-means, method="average")):
            ranks[a].append(r)
        cells = []
        for a in ALGS:
            v, vio, sz = vals[a]
            if domain == "mdp":
                best = problems[name].known_optimum
                cells.append(f"{100 * (best - v[vio <= 0].mean()) / best:6.2f}{marks.get(a, ''):1s}")
            else:
                n = problems[name].dim
                acc = (v - 0.1 * (1 - sz / n)) / 0.9  # eq. (fs) with alpha = 0.9
                cells.append(f"{100 * acc.mean():5.1f}/{sz.mean():5.1f}{marks.get(a, ''):1s}")
        label = name if domain == "mdp" else f"{name} (n={problems[name].dim})"
        print(f"{label:28s} " + "  ".join(cells))
    print("avg rank " + " ".join(f"{a} {np.mean(ranks[a]):.2f}" for a in ALGS))
    print("SBABC wins/losses/n.s. vs each rival:", tally)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=("mdp", "fs"), required=True)
    table(ap.parse_args().domain)
