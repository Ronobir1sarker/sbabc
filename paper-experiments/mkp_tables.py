"""Reproduces the six knapsack tables of the paper from results/results.sqlite
(opened read-only):

    tab:mkp-small    small suites: mean error and instances solved
    tab:mkp-large9   nine large instances of the main comparison
    tab:mkp-classes  90 Chu-Beasley instances, class means
    tab:gk           Glover-Kochenberger instances
    tab:walltime     wall-clock time per run on four of them
    tab:cb           knowledge isolation: the repair-equipped family

    python paper-experiments/mkp_tables.py            # print every table
    python paper-experiments/mkp_tables.py --check    # and compare each cell
                                                      # with the manuscript

Conventions (Section 6.3 of the paper). Mean error is 100 (v* - mean) / v*
over the feasible runs of an instance. A per-instance row tests the paired
runs (30, or 10 on the Glover-Kochenberger instances); a class row tests the
ten per-instance mean errors. Every test is the framework's paired Wilcoxon
(experiments.analysis.pairwise_stats) with Holm's correction within the row;
'+' marks the reference as significantly better than the column, '-' as
worse. Average ranks rank the algorithms on each instance by mean objective
value, ties at their mid-rank.

With --check the script reads the tables of
"set based abc latex files/sections/07_results.tex", when that file is
present, and lists every cell that differs from the value computed here.
"""
import argparse
import os
import re
import sqlite3
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from experiments.analysis import holm_correction, pairwise_stats  # noqa: E402
from problems.mkp import load_mkp_problems  # noqa: E402

DB = os.path.join(ROOT, "results", "results.sqlite")
TEX = os.path.join(ROOT, "set based abc latex files", "sections", "07_results.tex")
ALGS = ["SBABC", "SBPSO", "GBABC", "BitABC", "AMABC", "ABABC", "CBGA", "CETS", "BGWO"]
TIER = ALGS[:5]
CB_COLUMNS = ["SBABC", "SBABC-CB", "SBPSO-CB", "GBABC-CB", "BitABC-CB", "ABABC", "CBGA", "CETS", "BGWO"]
WALLTIME_ROWS = ["gk01", "gk06", "gk09", "gk11"]
ALPHA = 0.05


def reference_values():
    best, size = {}, {}
    for suite in ["mknap1", "mknap2", "gk"] + [f"mknapcb{i}" for i in range(1, 10)]:
        for problem in load_mkp_problems(suite):
            best[problem.name], size[problem.name] = problem.known_optimum, problem.dim
    return best, size


def runs(con, experiment):
    return pd.read_sql(
        "select algorithm, problem, run_index, best_value, best_violation, wall_time "
        "from runs where experiment=? order by algorithm, problem, run_index",
        con, params=(experiment,))


def mean_feasible(frame, columns):
    feasible = frame[frame.best_violation <= 0]
    return feasible.groupby(["problem", "algorithm"]).best_value.mean().unstack()[columns]


def errors(means, best):
    return pd.DataFrame({a: [100 * (best[p] - means.loc[p, a]) / best[p] for p in means.index]
                         for a in means.columns}, index=means.index)


def average_rank(means, columns):
    ranks = np.array([rankdata(-means.loc[p, columns].values, method="average") for p in means.index])
    return dict(zip(columns, ranks.mean(axis=0)))


def run_marks(frame, problem, reference, others):
    """Marks of the reference against each other algorithm on one instance (paired runs)."""
    one = frame[frame.problem == problem]
    value = one.pivot(index="run_index", columns="algorithm", values="best_value")
    violation = one.pivot(index="run_index", columns="algorithm", values="best_violation")
    stats = [pairwise_stats(value[reference].values, value[o].values, "max", paired=True,
                            violations_a=violation[reference].values, violations_b=violation[o].values)
             for o in others]
    adjusted = holm_correction([s["p_value"] for s in stats])
    return {o: ("" if p >= ALPHA else "+" if s["winner"] == "A" else "-")
            for o, p, s in zip(others, adjusted, stats)}


def class_marks(error_frame, instances, reference, others):
    """Marks of the reference against each other algorithm over a class (per-instance mean errors)."""
    stats = [pairwise_stats(error_frame.loc[instances, reference].values,
                            error_frame.loc[instances, o].values, "min", paired=True) for o in others]
    adjusted = holm_correction([s["p_value"] for s in stats])
    return {o: ("" if p >= ALPHA else "+" if s["winner"] == "A" else "-")
            for o, p, s in zip(others, adjusted, stats)}


def solved(frame, best):
    hit = [abs(v - best[p]) <= 1e-9 * max(1.0, abs(best[p])) and c <= 0
           for v, p, c in zip(frame.best_value, frame.problem, frame.best_violation)]
    return frame.assign(hit=hit).groupby(["problem", "algorithm"]).hit.any().unstack()


# --------------------------------------------------------------------------- tables
def compute(con):
    """Every cell of the six tables: {table: {(row, column): (number or None, mark)}}."""
    best, size = reference_values()
    out = {}

    main = runs(con, "mkp_competitors_rev2")
    means = mean_feasible(main, ALGS)
    err = errors(means, best)
    small1 = [p for p in means.index if p.startswith("mknap1-")]
    small2 = [p for p in means.index if p.startswith("mknap2-")]
    large = sorted((p for p in means.index if re.fullmatch(r"\d+\.\d+-00", p)),
                   key=lambda p: [int(x) for x in p.split("-")[0].split(".")])
    hit = solved(main, best)

    cells = {}
    for a in ALGS:
        cells[(a, "mknap1 error")] = (err.loc[small1, a].mean(), "")
        cells[(a, "mknap1 solved")] = (float(hit.loc[small1, a].sum()), "")
        cells[(a, "mknap2 error")] = (err.loc[small2, a].mean(), "")
        cells[(a, "mknap2 solved")] = (float(hit.loc[small2, a].sum()), "")
    out["tab:mkp-small"] = cells

    cells = {}
    for p in large:
        marks = run_marks(main, p, "SBABC", ALGS[1:])
        for a in ALGS:
            cells[(p, a)] = (err.loc[p, a], marks.get(a, ""))
    for label, columns in (("Avg. rank (all 64)", ALGS), ("Avg. rank (tier)", TIER)):
        for a, r in average_rank(means, columns).items():
            cells[(label, a)] = (r, "")
    out["tab:mkp-large9"] = cells

    scale = runs(con, "mkp_large90_rev2")
    scale_means = mean_feasible(scale, ALGS)
    scale_err = errors(scale_means, best)
    classes = sorted({p.split("-")[0] for p in scale_means.index},
                     key=lambda c: [int(x) for x in c.split(".")])
    cells = {}
    for c in classes:
        members = [p for p in scale_means.index if p.split("-")[0] == c]
        marks = class_marks(scale_err, members, "SBABC", ALGS[1:])
        for a in ALGS:
            cells[(f"{c} ({len(members)})", a)] = (scale_err.loc[members, a].mean(), marks.get(a, ""))
    for label, columns in (("Avg. rank (all 90)", ALGS), ("Avg. rank (tier)", TIER)):
        for a, r in average_rank(scale_means, columns).items():
            cells[(label, a)] = (r, "")
    out["tab:mkp-classes"] = cells

    gk = runs(con, "mkgk_rev2")
    gk_means = mean_feasible(gk, ALGS)
    gk_err = errors(gk_means, best)
    order = sorted(gk_means.index, key=lambda p: int(p[2:]))
    cells = {}
    for p in order:
        marks = run_marks(gk, p, "SBABC", ALGS[1:])
        cells[(p, "n")] = (float(size[p]), "")
        for a in ALGS:
            cells[(p, a)] = (gk_err.loc[p, a], marks.get(a, ""))
    for label, columns in (("Avg. rank (all 11)", ALGS), ("Avg. rank (tier)", TIER)):
        for a, r in average_rank(gk_means, columns).items():
            cells[(label, a)] = (r, "")
    out["tab:gk"] = cells

    wall = gk.groupby(["problem", "algorithm"]).wall_time.mean().unstack()[ALGS]
    cells = {}
    for p in WALLTIME_ROWS:
        cells[(p, "n")] = (float(size[p]), "")
        for a in ALGS:
            cells[(p, a)] = (wall.loc[p, a], "")
    out["tab:walltime"] = cells

    repaired = runs(con, "mkp_cb_rev1")
    family = pd.concat([main[main.algorithm.isin(["SBABC", "ABABC", "CBGA", "CETS", "BGWO"])], repaired])
    family_means = mean_feasible(family, CB_COLUMNS)
    family_err = errors(family_means, best)
    others = [c for c in CB_COLUMNS if c != "SBABC-CB"]
    tally = {o: [0, 0] for o in others}
    marks = {p: run_marks(family, p, "SBABC-CB", others) for p in family_means.index}
    for p in family_means.index:
        for o in others:
            tally[o][0] += marks[p][o] == "+"
            tally[o][1] += marks[p][o] == "-"
    cells = {}
    for p in large:
        for a in CB_COLUMNS:
            cells[(p, a)] = (family_err.loc[p, a], marks[p].get(a, ""))
    for a, r in average_rank(family_means, CB_COLUMNS).items():
        cells[("Avg. rank (all 64)", a)] = (r, "")
    for o in others:
        cells[("W--L vs SBABC-CB (64)", o)] = (None, f"{tally[o][0]}--{tally[o][1]}")
    out["tab:cb"] = cells
    return out


DECIMALS = {"tab:mkp-small": 3, "tab:walltime": 1}
LAYOUT = {  # column keys of each table, in the order of the LaTeX columns after the row label
    "tab:mkp-small": ["mknap1 error", "mknap1 solved", "mknap2 error", "mknap2 solved"],
    "tab:mkp-large9": ALGS,
    "tab:mkp-classes": ALGS,
    "tab:gk": ["n"] + ALGS,
    "tab:walltime": ["n"] + ALGS,
    "tab:cb": CB_COLUMNS,
}


def show(tables):
    for label, cells in tables.items():
        columns = LAYOUT[label]
        rows = list(dict.fromkeys(r for r, _ in cells))
        print(f"\n== {label}")
        print(f"{'':24s}" + "".join(f"{c[:12]:>13s}" for c in columns))
        for r in rows:
            line = f"{r:24s}"
            for c in columns:
                value, mark = cells.get((r, c), (None, ""))
                if value is None:
                    text = mark or "--"
                elif c in ("n",) or c.endswith("solved"):
                    text = f"{int(value)}"
                else:
                    text = f"{value:.{DECIMALS.get(label, 2)}f}{mark}"
                line += f"{text:>13s}"
            print(line)


# --------------------------------------------------------------------------- check against the manuscript
def latex_rows(tex, label):
    start = tex.find("\\label{" + label + "}")
    a = tex.find("\\begin{tabular}", start)
    b = tex.find("\\end{tabular}", a)
    rows = []
    for line in tex[a:b].split("\\\\"):
        line = re.sub(r"\\(toprule|midrule|bottomrule)|\\cmidrule\([lr]+\)\{[^}]*\}", "", " ".join(line.split()))
        if "&" in line and "multicolumn" not in line:
            rows.append([c.strip() for c in line.split("&")])
    return rows


def check(tables):
    if not os.path.exists(TEX):
        print(f"\n--check: {TEX} is not present, nothing to compare")
        return 0
    tex = open(TEX).read()
    differences, compared = [], 0
    for label, cells in tables.items():
        columns = LAYOUT[label]
        known_rows = {r for r, _ in cells}
        for row in latex_rows(tex, label):
            name = row[0].replace("\\ ", " ").replace("\\,", " ")
            name = " ".join(name.split())
            if name not in known_rows:
                continue
            for column, cell in zip(columns, row[1:]):
                if (name, column) not in cells:
                    continue
                value, mark = cells[(name, column)]
                compared += 1
                if value is None:                       # a wins--losses tally
                    if cell != mark:
                        differences.append((label, name, column, cell, mark))
                    continue
                printed = re.search(r"-?\d+(?:\.\d+)?", cell)
                printed_mark = "+" if "^{+}" in cell else "-" if "^{-}" in cell else ""
                decimals = 0 if (column == "n" or column.endswith("solved")) else DECIMALS.get(label, 2)
                if printed is None or abs(float(printed.group(0)) - value) > 0.5 * 10 ** -decimals + 1e-9:
                    differences.append((label, name, column, cell, f"{value:.{decimals + 2}f}"))
                elif printed_mark != mark:
                    differences.append((label, name, column, cell, f"mark '{mark}'"))
    print(f"\n--check: {compared} cells compared with the manuscript, {len(differences)} differ")
    for d in differences:
        print("   ", d)
    return len(differences)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="compare every cell with the manuscript's tables")
    args = parser.parse_args()
    connection = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    computed = compute(connection)
    show(computed)
    if args.check:
        sys.exit(1 if check(computed) else 0)
