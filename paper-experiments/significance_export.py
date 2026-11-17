"""Exports the statistics behind every significance mark of the paper to CSV
files in results/significance/ (the database is opened read-only).

    python paper-experiments/significance_export.py

One file per family of comparisons:

    mkp_main.csv        main comparison, 64 instances, SBABC against 8     (tab:mkp-large9)
    mkp_classes.csv     90 instances, nine class summaries, SBABC against 8 (tab:mkp-classes)
    mkp_gk.csv          Glover-Kochenberger, 11 instances, SBABC against 8  (tab:gk)
    mkp_ablation.csv    operator ablation, 19 instances per configuration   (tab:ablation)
    mkp_repair.csv      repair-equipped family, 64 instances, SBABC-CB against 8 (tab:cb)
    mdp.csv             maximum diversity, 15 instances, SBABC against 5    (tab:mdp)
    fs.csv              feature selection, 7 datasets, SBABC against 5      (tab:fs)
    mkp_penalty.csv     Deb's rule against four penalty weights, 9 instances (tab:penalty)
    mkp_one_at_a_time.csv  tuned SBABC against 20 settings that change one
                        parameter, 19 instances each (Limitations; supplementary material)

Each row is one comparison of a reference with one other algorithm:

    p_value       paired Wilcoxon signed-rank test (experiments.analysis.pairwise_stats)
    p_holm        Holm adjustment within the family named in column `family`
    significant   1 if p_holm < 0.05
    better        which side has the better mean: reference, other or tie
    cliffs_delta  Cliff's delta, positive when the reference is better, with its
                  magnitude class (negligible, small, medium, large)
    ci_low/high   bootstrap 95 % confidence interval of the mean difference,
                  positive when the reference is better; empty when a run of
                  either side is infeasible, because the difference of objective
                  values is then not meaningful

A per-instance row compares the paired runs (30, or 10 on the
Glover-Kochenberger instances) and its family is the set of algorithms
compared on that instance. A class row of mkp_classes.csv compares the ten
per-instance mean errors of a class. In mkp_ablation.csv the reference is
the reference configuration of the table's block, the other side is the
configuration tested, and the family of a configuration is its 19 instances,
as in the paper's table; the last 19 rows test the strict-improvement rule
under the tuned configuration against the main comparison's SBABC. In
mkp_one_at_a_time.csv the reference is the tuned configuration and the family
of a setting is its 19 instances.
"""
import os
import sqlite3
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from experiments.analysis import holm_correction, pairwise_stats  # noqa: E402
from problems.mkp import load_mkp_problems  # noqa: E402

DB = os.path.join(ROOT, "results", "results.sqlite")
OUT = os.path.join(ROOT, "results", "significance")
ALPHA = 0.05
MKP = ["SBPSO", "GBABC", "BitABC", "AMABC", "ABABC", "CBGA", "CETS", "BGWO"]
GENERIC = ["SBPSO", "GBABC", "BitABC", "AMABC", "ABABC"]
REPAIR = ["SBABC", "SBPSO-CB", "GBABC-CB", "BitABC-CB", "ABABC", "CBGA", "CETS", "BGWO"]
ABLATION = [  # (configuration, reference) pairs of tab:ablation
    ("SBABC_full", "SBABC_mda_only"), ("SBABC_md_only", "SBABC_mda_only"), ("SBABC_ma_only", "SBABC_mda_only"),
    ("SBABC_da_only", "SBABC_mda_only"), ("SBABC_move_only", "SBABC_mda_only"), ("SBABC_drop_only", "SBABC_mda_only"),
    ("SBABC_add_only", "SBABC_mda_only"), ("SBABC_textbook_ties", "SBABC_full"), ("SBABC_sharedV3", "SBABC_full"),
    ("SBABC_mda_strict", "SBABC_mda_only"),
]


def runs(con, *experiments):
    marks = ",".join("?" * len(experiments))
    return pd.read_sql(
        f"select experiment, algorithm, problem, run_index, best_value, best_violation from runs "
        f"where experiment in ({marks}) order by algorithm, problem, run_index", con, params=experiments)


def compare(reference, other, optimization_type="max"):
    """One comparison from two aligned (values, violations) pairs."""
    stats = pairwise_stats(reference[0], other[0], optimization_type, paired=True,
                           violations_a=reference[1], violations_b=other[1])
    feasible_ref = reference[0][reference[1] <= 0]
    feasible_other = other[0][other[1] <= 0]
    return {
        "n": len(reference[0]),
        "mean_reference": float(feasible_ref.mean()) if len(feasible_ref) else float("nan"),
        "mean_other": float(feasible_other.mean()) if len(feasible_other) else float("nan"),
        "feasible_reference": int(len(feasible_ref)),
        "feasible_other": int(len(feasible_other)),
        "p_value": stats["p_value"],
        "better": {"A": "reference", "B": "other", "tie": "tie"}[stats["winner"]],
        "cliffs_delta": stats["cliffs_delta"],
        "delta_magnitude": stats["delta_magnitude"],
        "ci_low": stats.get("mean_diff_ci_low", float("nan")),
        "ci_high": stats.get("mean_diff_ci_high", float("nan")),
    }


def close_family(rows):
    """Holm-adjust the p-values of one family in place."""
    for row, adjusted in zip(rows, holm_correction([r["p_value"] for r in rows])):
        row["p_holm"] = adjusted
        row["significant"] = int(adjusted < ALPHA)
    return rows


def per_instance(frame, reference, others, family_label, optimization_type="max", instances=None):
    out = []
    for problem in (instances if instances is not None else sorted(frame.problem.unique())):
        one = frame[frame.problem == problem]
        value = one.pivot(index="run_index", columns="algorithm", values="best_value")
        violation = one.pivot(index="run_index", columns="algorithm", values="best_violation")
        rows = []
        for o in others:
            row = {"instance": problem, "reference": reference, "other": o, "family": family_label}
            row.update(compare((value[reference].values, violation[reference].values),
                               (value[o].values, violation[o].values), optimization_type))
            rows.append(row)
        out.extend(close_family(rows))
    return out


def write(name, rows):
    columns = ["instance", "reference", "other", "family", "n", "feasible_reference", "feasible_other",
               "mean_reference", "mean_other", "p_value", "p_holm", "significant", "better",
               "cliffs_delta", "delta_magnitude", "ci_low", "ci_high"]
    frame = pd.DataFrame(rows)[columns]
    path = os.path.join(OUT, name)
    frame.to_csv(path, index=False, float_format="%.6g")
    won = int(((frame.significant == 1) & (frame.better == "reference")).sum())
    lost = int(((frame.significant == 1) & (frame.better == "other")).sum())
    print(f"{name:20s} {len(frame):4d} comparisons   reference significantly better {won:3d}, worse {lost:3d}")


def main():
    os.makedirs(OUT, exist_ok=True)
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)

    main_runs = runs(con, "mkp_competitors_rev2")
    write("mkp_main.csv", per_instance(main_runs, "SBABC", MKP, "the 8 other algorithms on the instance"))
    write("mkp_gk.csv", per_instance(runs(con, "mkgk_rev2"), "SBABC", MKP, "the 8 other algorithms on the instance"))

    family = pd.concat([main_runs[main_runs.algorithm.isin(["SBABC", "ABABC", "CBGA", "CETS", "BGWO"])],
                        runs(con, "mkp_cb_rev1")])
    write("mkp_repair.csv", per_instance(family, "SBABC-CB", REPAIR, "the 8 other columns on the instance"))

    write("mdp.csv", per_instance(runs(con, "mdp_main_rev1"), "SBABC", GENERIC, "the 5 other algorithms on the instance"))
    write("fs.csv", per_instance(runs(con, "fs_main_rev1"), "SBABC", GENERIC, "the 5 other algorithms on the dataset"))

    penalty = runs(con, "mkp_penalty_rev2")
    deb = main_runs[(main_runs.algorithm == "SBABC") & main_runs.problem.isin(penalty.problem.unique())]
    write("mkp_penalty.csv", per_instance(pd.concat([deb, penalty]), "SBABC",
                                          ["SBABC_pen1", "SBABC_pen10", "SBABC_pen100", "SBABC_pen1000"],
                                          "the 4 penalty weights on the instance"))

    # class summaries: ten per-instance mean errors per class, Holm within the class
    best = {}
    for suite in [f"mknapcb{i}" for i in range(1, 10)]:
        for problem in load_mkp_problems(suite):
            best[problem.name] = problem.known_optimum
    scale = runs(con, "mkp_large90_rev2")
    means = scale[scale.best_violation <= 0].groupby(["problem", "algorithm"]).best_value.mean().unstack()
    error = pd.DataFrame({a: [100 * (best[p] - means.loc[p, a]) / best[p] for p in means.index]
                          for a in means.columns}, index=means.index)
    rows = []
    for cls in sorted({p.split("-")[0] for p in error.index}, key=lambda c: [int(x) for x in c.split(".")]):
        members = [p for p in error.index if p.split("-")[0] == cls]
        zeros = np.zeros(len(members))
        family_rows = []
        for o in MKP:
            row = {"instance": f"class {cls} ({len(members)} instances, mean error %)", "reference": "SBABC",
                   "other": o, "family": "the 8 other algorithms on the class"}
            row.update(compare((error.loc[members, "SBABC"].values, zeros), (error.loc[members, o].values, zeros), "min"))
            family_rows.append(row)
        rows.extend(close_family(family_rows))
    write("mkp_classes.csv", rows)

    # ablation: one family per configuration, over its 19 instances
    ablation = runs(con, "mkp_ablation_rev1", "mkp_ablation_rev2", "mkp_tierule_rev1")
    ablation = ablation[~((ablation.experiment == "mkp_ablation_rev2") & (ablation.algorithm == "SBABC_mda_only"))]
    rows = []
    for configuration, reference in ABLATION:
        family_rows = []
        for problem in sorted(ablation.problem.unique()):
            def side(name):
                one = ablation[(ablation.algorithm == name) & (ablation.problem == problem)].sort_values("run_index")
                return one.best_value.values, one.best_violation.values
            row = {"instance": problem, "reference": reference, "other": configuration,
                   "family": f"the 19 instances of {configuration}"}
            row.update(compare(side(reference), side(configuration)))
            family_rows.append(row)
        rows.extend(close_family(family_rows))
    # the tie rule under the tuned configuration, against the main comparison's SBABC
    tuned = runs(con, "mkp_tierule_main_rev1")
    family_rows = []
    for problem in sorted(tuned.problem.unique()):
        a = tuned[tuned.problem == problem].sort_values("run_index")
        b = main_runs[(main_runs.algorithm == "SBABC") & (main_runs.problem == problem)].sort_values("run_index")
        row = {"instance": problem, "reference": "SBABC (tuned configuration)", "other": "SBABC_strict",
               "family": "the 19 instances of SBABC_strict"}
        row.update(compare((b.best_value.values, b.best_violation.values), (a.best_value.values, a.best_violation.values)))
        family_rows.append(row)
    rows.extend(close_family(family_rows))
    write("mkp_ablation.csv", rows)

    # one parameter changed at a time: one family per setting, over its 19 instances
    oat = runs(con, "mkp_sensitivity_rev1")
    settings = sorted((a for a in oat.algorithm.unique() if a != "SBABC_tuned"),
                      key=lambda s: (s.split("=")[0], float(s.split("=")[1])))
    rows = []
    for setting in settings:
        family_rows = []
        for problem in sorted(oat.problem.unique()):
            def one(name):
                part = oat[(oat.algorithm == name) & (oat.problem == problem)]
                return part.best_value.values, part.best_violation.values
            row = {"instance": problem, "reference": "SBABC_tuned", "other": setting,
                   "family": f"the 19 instances of {setting}"}
            row.update(compare(one("SBABC_tuned"), one(setting)))
            family_rows.append(row)
        rows.extend(close_family(family_rows))
    write("mkp_one_at_a_time.csv", rows)
    print(f"\nwritten to {OUT}")


if __name__ == "__main__":
    main()
