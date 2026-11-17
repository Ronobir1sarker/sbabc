"""Builds the LaTeX source of the paper's supplementary material from the
results database (opened read-only) and the tuning files.

    python paper-experiments/supplement.py

Writes supplement.tex next to the manuscript ("set based abc latex files/
supplement/") when that folder exists, and into results/supplement/
otherwise. Compile it with any LaTeX engine (two passes for the contents).

Sections: S1 instances and reference values; S2 the tuned configuration of
every algorithm in every domain; S3 all tuning tables; S4 per-instance
results of the knapsack experiments; S5 components removed from the
four-candidate layout; S6 the one-at-a-time check; S7 the reproduction of
ABABC and BGWO on the Weing instances. Significance marks follow Section 6.3
of the paper: paired Wilcoxon over the runs of an instance, pairs with equal
values discarded, Holm's correction within the row.
"""
import csv
import importlib.util
import json
import os
import sqlite3
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import run_experiment  # noqa: E402
import run_tuning  # noqa: E402
from experiments.analysis import holm_correction, pairwise_stats  # noqa: E402
from problems.feature_selection import load_fs_problems  # noqa: E402
from problems.mdp import load_mdp_problems  # noqa: E402
from problems.mkp import load_mkp_problems  # noqa: E402
from problems.scp import load_scp_problems  # noqa: E402

_spec = importlib.util.spec_from_file_location("mkp_tables", os.path.join(ROOT, "paper-experiments", "mkp_tables.py"))
mkp_tables = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mkp_tables)

DB = os.path.join(ROOT, "results", "results.sqlite")
TUNING = os.path.join(ROOT, "results", "tuning")
MANUSCRIPT = os.path.join(ROOT, "set based abc latex files")
OUT_DIR = os.path.join(MANUSCRIPT, "supplement") if os.path.isdir(MANUSCRIPT) else os.path.join(ROOT, "results", "supplement")
ALGS = mkp_tables.ALGS
GENERIC = ["SBABC", "SBPSO", "GBABC", "BitABC", "AMABC", "ABABC"]
DOMAINS = [("mkp", "Multidimensional knapsack"), ("mdp", "Maximum diversity"),
           ("fs", "Feature selection"), ("scp", "Set covering")]
PARAM_NAMES = {
    "beta1": r"$\beta_1$", "beta2": r"$\beta_2$", "beta3": r"$\beta_3$", "k": r"$k$", "limit": "limit",
    "candidate_set": "layout", "scout_bee_phase_version": "scout", "topology": "topology",
    "neighborhood_radius": "radius", "c1": r"$c_1$", "c2": r"$c_2$", "c3": r"$c_3$", "c4": r"$c_4$",
    "r": r"$r$", "threshold": "Th", "credit_alpha": r"$\alpha$", "pursuit_beta": r"$\beta$",
    "p_min": r"$p_{\min}$", "reward_window": r"$W$", "reward_type": "reward", "num_wolves": "wolves",
    "transform": "transfer", "population_size": "population", "mutation_bits": "mutation bits",
    "t": r"$t$", "p2": r"$p_2$", "k_max": r"$k_{\max}$", "avg_rank": "mean rank",
}


def tex(s):
    return str(s).replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")


def fmt(v):
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    try:
        x = float(v)
    except (TypeError, ValueError):
        return tex(str(v).replace("von_neumann", "von Neumann"))
    if x == int(x) and abs(x) < 1e6:
        return str(int(x))
    return f"{x:.4g}"


def longtable(columns, align, rows, caption, label=None, size=r"\footnotesize"):
    head = " & ".join(columns) + r" \\"
    out = [r"{" + size, r"\begin{longtable}{@{}" + align + r"@{}}",
           r"\caption{" + caption + "}" + (r"\label{" + label + "}" if label else "") + r"\\",
           r"\toprule", head, r"\midrule", r"\endfirsthead",
           r"\multicolumn{" + str(len(columns)) + r"}{@{}l}{\emph{(continued)}}\\", r"\toprule", head, r"\midrule", r"\endhead",
           r"\bottomrule", r"\endfoot"]
    out += [" & ".join(r) + r" \\" for r in rows]
    out += [r"\end{longtable}", "}"]
    return "\n".join(out)


def mark(m):
    return {"+": r"$^{+}$", "-": r"$^{-}$"}.get(m, "")


# --------------------------------------------------------------------------- S1
def section_instances():
    parts = [r"\section{Instances and reference values}\label{s:instances}",
             "Errors are measured against the proven optimum where one is known and against the best-known value "
             "of the benchmark library otherwise (Section~6.1 of the paper). The tables give, for every instance, "
             "its size, its reference value and which of the two the value is."]
    small = load_mkp_problems("mknap1") + load_mkp_problems("mknap2")
    rows = [[tex(p.name), str(p.dim), str(p.extras["num_of_constraints"]), fmt(p.known_optimum), "proven"] for p in small]
    parts.append(longtable(["Instance", "$n$", "$m$", "Reference value", "Status"], "lrrrl", rows,
                           "Small knapsack suites mknap1 (7 instances) and mknap2 (48 instances) of the OR-Library."))
    scale = run_experiment.build_mkp_large90_rev2()[1]
    rows = [[tex(p.name), str(p.dim), str(p.extras["num_of_constraints"]), fmt(p.extras.get("tightness", "")), fmt(p.known_optimum),
             "proven" if p.extras.get("optimum_proven") else "best known"] for p in scale]
    parts.append(longtable(["Instance", "$n$", "$m$", r"$\alpha$", "Reference value", "Status"], "lrrrrl", rows,
                           "Chu--Beasley instances of the scale study (ten per class; the nine with index 00 are also "
                           "the large instances of the main comparison)."))
    gk = load_mkp_problems("gk")
    rows = [[tex(p.name), str(p.dim), str(p.extras["num_of_constraints"]), fmt(p.known_optimum), "best known"] for p in gk]
    parts.append(longtable(["Instance", "$n$", "$m$", "Reference value", "Status"], "lrrrl", rows,
                           "Glover--Kochenberger instances."))
    mdp_test = run_experiment.build_mdp_main_rev1()[1]
    rows = [[tex(p.name), str(p.dim), str(p.extras["m"]), fmt(p.known_optimum), "best known"] for p in mdp_test]
    parts.append(longtable(["Instance", "$n$", "$m$", "Reference value", "Status"], "lrrrl", rows,
                           "Maximum diversity instances of MDPLIB used in the test experiment."))
    scp = load_scp_problems(["scp44", "scp45", "scp54", "scp55", "scp64", "scpb1"])
    rows = [[tex(p.name), str(p.extras["num_rows"]), str(p.dim), fmt(p.known_optimum), "proven"] for p in scp]
    parts.append(longtable(["Instance", "rows", "columns", "Optimum", "Status"], "lrrrl", rows,
                           "Set covering instances of the OR-Library (also the tuning instances of this domain)."))
    fs = load_fs_problems(["wine", "wdbc", "ionosphere", "sonar", "segment", "german_credit", "hill_valley", "glass", "vehicle"])
    rows = [[tex(p.name.replace("fs_", "")), str(p.dim), str(p.extras["n_samples"]), str(p.extras["n_classes"]),
             "tuning" if p.name in ("fs_glass", "fs_vehicle") else "test"] for p in fs]
    parts.append(longtable(["Dataset", "features", "samples", "classes", "Use"], "lrrrl", rows,
                           "Feature-selection datasets (no reference values: the comparison ranks the fitness)."))
    rows = []
    for domain, title in DOMAINS:
        for p in run_tuning.tuning_problems(domain):
            ref = "" if domain == "fs" else fmt(p.known_optimum)
            rows.append([title, tex(p.name.replace("fs_", "")), str(p.dim), ref])
    parts.append(longtable(["Domain", "Tuning instance", "$n$", "Reference value"], "llrr", rows,
                           "Tuning instances. Where a domain has a test experiment, they are disjoint from its test instances."))
    return "\n\n".join(parts)


# --------------------------------------------------------------------------- S2, S3
def tuning_rows(domain, algorithm):
    path = os.path.join(TUNING, f"{domain}_{algorithm}.csv")
    with open(path) as handle:
        rows = list(csv.DictReader(handle))
    params = [c for c in rows[0] if not c.startswith(("mean_error_pct", "fitness", "mean_fitness")) and c != "avg_rank"]
    return rows, params


def section_tuned():
    parts = [r"\section{Tuned configurations}\label{s:tuned}",
             "The configuration selected for every algorithm in every domain by the tuning protocol of Section~6.2 "
             "of the paper: the configuration with the best mean rank over the tuning instances, ties going to the "
             "earliest sampled. Colony sizes are fixed (25 food sources or particles; 40 for ABABC), and only the "
             "population of CBGA and the pack of BGWO are tuned in size. Table~3 of the paper repeats the "
             "configuration of SBABC."]
    for domain, title in DOMAINS:
        best = json.load(open(os.path.join(TUNING, f"{domain}_best_configs.json")))
        rows = []
        for algorithm in [a for a in ALGS if a in best]:
            rows_csv, params = tuning_rows(domain, algorithm)
            chosen = best[algorithm]
            cells = [f"{PARAM_NAMES.get(p, tex(p))} = {fmt(chosen.get(p, rows_csv[0][p]))}" for p in params]
            rows.append([algorithm, ", ".join(cells)])
        parts.append(longtable(["Algorithm", "Selected configuration"], r"lp{0.78\linewidth}", rows,
                               f"{title}: selected configurations."))
    return "\n\n".join(parts)


def section_tuning_tables():
    parts = [r"\section{Tuning tables}\label{s:tuning}",
             "Each table lists the 32 Sobol-sampled configurations of one algorithm in one domain, best first, "
             "with their mean rank over the tuning instances (rank 1 is best). The first row is the selected "
             "configuration. Where several configurations share the best mean rank, the caption gives their "
             "number; the earliest sampled of them is selected. The tuning errors of every configuration on every "
             "tuning instance are in the files results/tuning/\\textless domain\\textgreater\\_\\textless "
             "algorithm\\textgreater.csv of the repository."]
    for domain, title in DOMAINS:
        algorithms = [a for a in ALGS if os.path.exists(os.path.join(TUNING, f"{domain}_{a}.csv"))]
        for algorithm in algorithms:
            rows_csv, params = tuning_rows(domain, algorithm)
            ranks = [float(r["avg_rank"]) for r in rows_csv]
            ties = sum(1 for x in ranks if abs(x - ranks[0]) < 1e-12)
            columns = [PARAM_NAMES.get(p, tex(p)) for p in params] + ["mean rank"]
            rows = [[fmt(r[p]) for p in params] + [f"{float(r['avg_rank']):.2f}"] for r in rows_csv]
            caption = f"{title}, {algorithm}: the 32 tuning configurations."
            if ties > 1:
                caption += f" {ties} configurations share the best mean rank."
            parts.append(longtable(columns, "l" * len(params) + "r", rows, caption, size=r"\scriptsize"))
    return "\n\n".join(parts)


# --------------------------------------------------------------------------- S4
def per_instance_block(frame, columns, reference, title, label_note):
    best, _ = mkp_tables.reference_values()
    means = mkp_tables.mean_feasible(frame, columns)
    errors = mkp_tables.errors(means, best)
    others = [c for c in columns if c != reference]
    order = list(means.index)
    small = sorted((p for p in order if p.startswith("mknap")),
                   key=lambda p: (int(p.split("-")[0][5:]), int(p.split("-")[1])))
    cb = sorted((p for p in order if not p.startswith("mknap")),
                key=lambda p: [int(x) for x in p.split("-")[0].split(".")] + [int(p.split("-")[1])])
    rows = []
    for p in small + cb:
        marks = mkp_tables.run_marks(frame, p, reference, others)
        rows.append([tex(p)] + [f"{errors.loc[p, c]:.3f}{mark(marks.get(c, ''))}" for c in columns])
    return longtable(["Instance"] + [tex(c) for c in columns], "l" + "r" * len(columns), rows,
                     f"{title} Mean error (\\%) over the feasible runs. {label_note}", size=r"\scriptsize")


def section_per_instance(con):
    parts = [r"\section{Per-instance results of the knapsack experiments}\label{s:perinstance}",
             "Mean error (\\%) of every algorithm on every instance, relative to the reference value of "
             "Section~\\ref{s:instances}. Superscripts mark the reference algorithm as significantly better ($+$) "
             "or worse ($-$) than the column (paired Wilcoxon over the runs, pairs with equal values discarded, Holm "
             "within the row, $\\alpha = 0.05$). The Glover--Kochenberger, maximum diversity, feature-selection and "
             "set covering results are given per instance in the paper itself (Tables~7, 11, 12 and 13)."]
    main = mkp_tables.runs(con, "mkp_competitors_rev2")
    parts.append(per_instance_block(main, ALGS, "SBABC", "Main comparison, 64 instances, 30 runs.",
                                    "Superscripts compare SBABC with the column (Tables~4 and~5 of the paper)."))
    scale = mkp_tables.runs(con, "mkp_large90_rev2")
    parts.append(per_instance_block(scale, ALGS, "SBABC", "Scale study, 90 Chu--Beasley instances, 30 runs.",
                                    "Superscripts compare SBABC with the column on each instance; Table~6 of the paper "
                                    "tests the class means."))
    repaired = mkp_tables.runs(con, "mkp_cb_rev1")
    family = pd.concat([main[main.algorithm.isin(["SBABC", "ABABC", "CBGA", "CETS", "BGWO"])], repaired])
    parts.append(per_instance_block(family, mkp_tables.CB_COLUMNS, "SBABC-CB",
                                    "Repair-equipped family, 64 instances, 30 runs.",
                                    "Superscripts compare SBABC-CB with the column (Table~10 of the paper)."))
    return "\n\n".join(parts)


# --------------------------------------------------------------------------- S5, S6
def family_tally(runs, configuration, reference, problems):
    stats = []
    for problem in problems:
        a = runs[(runs.algorithm == configuration) & (runs.problem == problem)].sort_values("run_index")
        b = runs[(runs.algorithm == reference) & (runs.problem == problem)].sort_values("run_index")
        stats.append(pairwise_stats(a.best_value.values, b.best_value.values, "max", paired=True,
                                    violations_a=a.best_violation.values, violations_b=b.best_violation.values))
    adjusted = holm_correction([s["p_value"] for s in stats])
    won = sum(1 for s, p in zip(stats, adjusted) if p < 0.05 and s["winner"] == "A")
    lost = sum(1 for s, p in zip(stats, adjusted) if p < 0.05 and s["winner"] == "B")
    return f"{won}/{lost}/{len(stats) - won - lost}"


def summary(runs, configuration, best, small, large):
    part = runs[runs.algorithm == configuration]
    per_iteration = part.fn_calls.sum() / part.iterations.sum()
    means = part[part.best_violation <= 0].groupby("problem").best_value.mean()
    error = lambda names: np.mean([100 * (best[p] - means[p]) / best[p] for p in names])
    return per_iteration, error(small), error(large)


def section_ablation(con):
    problems = run_experiment._ablation_instances()
    best = {p.name: p.known_optimum for p in problems}
    small, large = [p.name for p in problems[:10]], [p.name for p in problems[10:]]
    runs = pd.read_sql("select experiment, algorithm, problem, run_index, best_value, best_violation, fn_calls, iterations "
                       "from runs where experiment='mkp_ablation_rev1'", con)
    rows = []
    for configuration, label in [("SBABC_full", "four candidates (m, md, ma, mda)"),
                                 ("SBABC_no_add", "without add (m, md)"), ("SBABC_no_drop", "without drop (m, ma)")]:
        per_iteration, e_small, e_large = summary(runs, configuration, best, small, large)
        tally = "ref." if configuration == "SBABC_full" else family_tally(runs, configuration, "SBABC_full", list(best))
        rows.append([label, f"{per_iteration:.1f}", f"{e_small:.3f}", f"{e_large:.2f}", tally])
    text = ("Section~7.3 of the paper removes each component from the single composite. Removing a component from "
            "the four-candidate layout instead leaves the candidates that do not apply it, and gives the same "
            "verdict: both reduced layouts are significantly worse than the full one on most instances and better "
            "on none. Same 19 instances, budget and seed as Table~9 of the paper.")
    return "\n\n".join([r"\section{Components removed from the four-candidate layout}\label{s:ablation}", text,
                        longtable(["Layout", "Evals./iter.", "Small err.\\,\\%", "Large err.\\,\\%", "W/L/ns"], "lrrrl", rows,
                                  "Components removed from the four-candidate layout (W/L/ns against the full layout; "
                                  "paired Wilcoxon per instance, Holm over the 19 instances).")])


def section_one_at_a_time(con):
    problems = run_experiment._ablation_instances()
    best = {p.name: p.known_optimum for p in problems}
    small, large = [p.name for p in problems[:10]], [p.name for p in problems[10:]]
    runs = pd.read_sql("select algorithm, problem, run_index, best_value, best_violation, fn_calls, iterations "
                       "from runs where experiment='mkp_sensitivity_rev1'", con)
    settings = sorted((a for a in runs.algorithm.unique() if a != "SBABC_tuned"),
                      key=lambda s: (s.split("=")[0], float(s.split("=")[1])))
    _, e_small, e_large = summary(runs, "SBABC_tuned", best, small, large)
    rows = [["tuned configuration", f"{e_small:.3f}", f"{e_large:.2f}", "ref."]]
    for setting in settings:
        _, e_small, e_large = summary(runs, setting, best, small, large)
        name, value = setting.replace("SBABC_", "").split("=")
        rows.append([f"{PARAM_NAMES.get(name, tex(name))} = {value}", f"{e_small:.3f}", f"{e_large:.2f}",
                     family_tally(runs, setting, "SBABC_tuned", list(best))])
    text = ("Section~8.5 of the paper cites this check. The tuned SBABC of the knapsack (Table~3 of the paper) runs "
            "on the 19 instances of the ablation together with 20 settings that change one parameter each, 30 runs "
            "per instance at $10^5$ evaluations. None of the twelve settings of $\\beta_3$, $k$ and the abandonment "
            "limit is significantly better than the tuned configuration on any instance. A larger move weight "
            "($\\beta_1 = 1$) is better on three instances, in line with the sensitivity analysis of Section~7.9.")
    return "\n\n".join([r"\section{One-at-a-time check of the tuned configuration}\label{s:oat}", text,
                        longtable(["Setting", "Small err.\\,\\%", "Large err.\\,\\%", "W/L/ns"], "lrrl", rows,
                                  "One parameter changed at a time (W/L/ns: setting significantly better than, worse "
                                  "than or indistinguishable from the tuned configuration; paired Wilcoxon per "
                                  "instance, Holm over the 19 instances).")])


# --------------------------------------------------------------------------- S7
def section_weing():
    names = {f"mknap2-{i}": f"weing{i - 2}" for i in range(3, 11)}
    ababc = {r["problem"]: r for r in csv.DictReader(open(os.path.join(ROOT, "results", "ababc_weing_validation.csv")))}
    bgwo = {r["problem"]: r for r in csv.DictReader(open(os.path.join(ROOT, "results", "bgwo_weing_validation.csv")))}
    rows = []
    for key, label in names.items():
        a, b = ababc[key], bgwo[key]
        rows.append([label, f"{float(a['success_pct']):.0f}", f"{float(a['paper_pct']):.0f}",
                     f"{float(b['success_pct']):.0f}", f"{float(b['paper_pct']):.0f}"])
    text = (
        "Both recent comparators were checked against the Weing instances (mknap2-3 to mknap2-10) of their own "
        "publications, with 30 runs at the evaluation budget of the experiments. The table gives the percentage of "
        "runs that reach the optimum, measured and published. Each implementation matches or exceeds the published "
        "rate on seven of the eight instances.\n\n"
        "The exception is in both cases weing7 ($n = 105$ items, $m = 2$ constraints, optimum 1\\,095\\,445). "
        "The re-implemented BGWO reaches the optimum in 20\\,\\% of the runs against a published 97\\,\\%, and ABABC "
        "in none against a published 40\\,\\%. Its best run ends at 1\\,095\\,382, the level that the ABABC publication "
        "reports for its other competitors on this instance. Several readings of each method were tested without "
        "recovering the published rates. For BGWO these were the scalar and the per-dimension interpretation of its "
        "random factors and the preprocessing of the instance, which removes no item from weing7. For ABABC they "
        "were six variants of its bit-count rule, onlooker selection, utility weights, the lower bound of its "
        "scaling factor, the update rule of the authors' public code and tournament selection of neighbours. The two publications' "
        "tables share their weing7 values, so the published rates for this instance trace to a single source. The "
        "instance is a plateau case: of its 105 items only 71 have distinct weight columns, so many near-optimal "
        "solutions tie.\n\n"
        "None of this affects the comparisons of the paper, which use new runs of every algorithm under identical "
        "budgets and seeds. The gap concerns only the success rates published for this one instance.")
    return "\n\n".join([r"\section{Reproduction of ABABC and BGWO on the Weing instances}\label{s:weing}", text,
                        longtable(["Instance", "ABABC", "published", "BGWO", "published"], "lrrrr", rows,
                                  "Percentage of 30 runs reaching the optimum, measured here and published.")])


def main():
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    preamble = r"""\documentclass[10pt,a4paper]{article}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage[margin=2cm]{geometry}
\usepackage{amsmath,booktabs,longtable,array}
\usepackage[hidelinks]{hyperref}
\renewcommand{\thesection}{S\arabic{section}}
\renewcommand{\thetable}{S\arabic{table}}
\setlength{\LTcapwidth}{\linewidth}
\begin{document}
\begin{center}
{\large\bfseries Supplementary material}\\[0.4em]
A Generic Set Based Artificial Bee Colony Algorithm for Combinatorial Optimization\\[0.2em]
Ronobir Chandra Sarker and Jagdish Chand Bansal
\end{center}

This supplement is generated from the database of all runs by the script
\texttt{paper-experiments/supplement.py} of the repository. Section, table and figure numbers without the
prefix S refer to the paper.
\begin{itemize}\setlength{\itemsep}{0pt}
\item Repository: \url{https://github.com/Ronobir1sarker/sbabc} (archived as doi:10.5281/zenodo.24326043)
\item Database: \url{https://doi.org/10.5281/zenodo.24326069}
\end{itemize}

\tableofcontents
"""
    body = [section_instances(), section_tuned(), section_tuning_tables(), section_per_instance(con),
            section_ablation(con), section_one_at_a_time(con), section_weing()]
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "supplement.tex")
    with open(path, "w") as handle:
        handle.write(preamble + "\n\n".join(body) + "\n\\end{document}\n")
    print("written", path)


if __name__ == "__main__":
    main()
