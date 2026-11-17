"""Reproduces the parameter-sensitivity table of the paper (tab:sensitivity)
from the tuning tables results/tuning/<domain>_SBABC.csv.

    python paper-experiments/sensitivity_table.py            # print the table
    python paper-experiments/sensitivity_table.py --check    # and compare it
                                                             # with the manuscript

Each tuning table holds the 32 Sobol-sampled configurations of SBABC in one
domain with their mean tuning rank (rank 1 is best). For a continuous
parameter the script prints the Spearman rank correlation between the
parameter's value and that rank, starred where p < 0.05, so a negative value
means that larger settings help. For a categorical parameter it prints the
mean tuning rank of each level, best first.

With --check every entry is looked up in the table of
"set based abc latex files/sections/07_results.tex", when that file is
present.
"""
import argparse
import csv
import os
import sys

import numpy as np
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TUNING = os.path.join(ROOT, "results", "tuning")
TEX = os.path.join(ROOT, "set based abc latex files", "sections", "07_results.tex")
DOMAINS = ["mkp", "mdp", "fs", "scp"]
CONTINUOUS = ["beta1", "beta2", "beta3", "k", "limit"]
CATEGORICAL = {"candidate_set": {"four": "four", "single": "single"},
               "topology": {"von_neumann": "vN", "ring": "ring", "star": "star"},
               "scout_bee_phase_version": {"1": "v1", "2": "v2"}}


def configurations(domain):
    with open(os.path.join(TUNING, f"{domain}_SBABC.csv")) as handle:
        return list(csv.DictReader(handle))


def compute():
    table = {}
    for domain in DOMAINS:
        rows = configurations(domain)
        assert len(rows) == 32, (domain, len(rows))
        rank = np.array([float(r["avg_rank"]) for r in rows])
        for parameter in CONTINUOUS:
            rho, p = spearmanr([float(r[parameter]) for r in rows], rank)
            table[(parameter, domain)] = (float(rho), bool(p < 0.05))
        for parameter, names in CATEGORICAL.items():
            if parameter not in rows[0]:
                table[(parameter, domain)] = None            # not tunable in this domain
                continue
            levels = {}
            for r in rows:
                levels.setdefault(names[r[parameter]], []).append(float(r["avg_rank"]))
            table[(parameter, domain)] = sorted(((np.mean(v), k) for k, v in levels.items()))
    return table


def show(table):
    print(f"{'parameter':26s}" + "".join(f"{d.upper():>34s}" for d in DOMAINS))
    for parameter in CONTINUOUS:
        cells = [f"{rho:+.2f}{'*' if star else ''}" for rho, star in (table[(parameter, d)] for d in DOMAINS)]
        print(f"{parameter:26s}" + "".join(f"{c:>34s}" for c in cells))
    for parameter in CATEGORICAL:
        cells = []
        for d in DOMAINS:
            levels = table[(parameter, d)]
            cells.append("--" if levels is None else " < ".join(f"{name} ({mean:.1f})" for mean, name in levels))
        print(f"{parameter:26s}" + "".join(f"{c:>34s}" for c in cells))


def check(table):
    if not os.path.exists(TEX):
        print(f"\n--check: {TEX} is not present, nothing to compare")
        return 0
    tex = open(TEX).read()
    start = tex.find("\\label{tab:sensitivity}")
    block = tex[start:tex.find("\\end{tabular}", start)]
    data_rows = [line for line in block.split("\n") if "&" in line]
    row_of = {"beta1": "beta_1", "beta2": "beta_2", "beta3": "beta_3", "k": "$k$", "limit": "limit",
              "candidate_set": "candidate layout", "topology": "topology", "scout_bee_phase_version": "scout variant"}
    missing, compared = [], 0
    for parameter in list(CONTINUOUS) + list(CATEGORICAL):
        line = next(l for l in data_rows if l.lstrip().startswith(("$\\" + row_of[parameter], row_of[parameter])))
        cells = [c.strip() for c in line.rstrip("\\ ").split("&")][1:]
        for d, cell in zip(DOMAINS, cells):
            entry = table[(parameter, d)]
            compared += 1
            if parameter in CONTINUOUS:
                rho, star = entry
                expected = f"${'+' if rho >= 0 else '-'}{abs(rho):.2f}" + ("^{*}$" if star else "$")
                if cell != expected:
                    missing.append((parameter, d, cell, expected))
            elif entry is None:
                if cell != "--":
                    missing.append((parameter, d, cell, "--"))
            else:
                expected = [f"{name} ({mean:.1f})" for mean, name in entry]
                found = [x.strip() for x in cell.replace("\\makecell[tl]{", "").rstrip("}").split("\\\\")]
                if found != expected:
                    missing.append((parameter, d, cell, expected))
    print(f"\n--check: {compared} cells compared with the manuscript, {len(missing)} differ")
    for m in missing:
        print("   ", m)
    return len(missing)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="compare every entry with the manuscript's table")
    args = parser.parse_args()
    computed = compute()
    show(computed)
    if args.check:
        sys.exit(1 if check(computed) else 0)
