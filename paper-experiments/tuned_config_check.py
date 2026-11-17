"""Rebuild the rows of the paper's tuned-configuration table (tab:tuned, Section 6.2)
from results/tuning/<domain>_best_configs.json and verify the tie facts stated there:
on feature selection nine SBABC configurations tie for the best mean rank (ABABC all 32),
the configuration selected on MKP/MDP/SCP is among them, and in every one of the 27
tuning tables the selected row is the earliest sampled among those tied for best.
Run from the repository root:  .venv/bin/python paper-experiments/tuned_config_check.py
"""
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from run_tuning import _domain_spaces  # noqa: E402
from experiments.tuning import sobol_parameter_sets  # noqa: E402

DOMAINS = ("mkp", "mdp", "fs", "scp")  # column order of tab:tuned
TUNING = "results/tuning"


def table_rows():
    best = {d: json.load(open(f"{TUNING}/{d}_best_configs.json"))["SBABC"] for d in DOMAINS}
    assert all(best[d]["num_food_sources"] == 25 for d in DOMAINS)

    def topo(b):
        return "star" if b["topology"] == "star" else f"{b['topology']} ({b['neighborhood_radius']})"

    rows = [
        ("$\\beta_1$ (move weight)", [f"{best[d]['beta1']:.2f}" for d in DOMAINS]),
        ("$\\beta_2$ (drop weight)", [f"{best[d]['beta2']:.2f}" for d in DOMAINS]),
        ("$\\beta_3$ (add rate)", [f"{best[d]['beta3']:.2f}" for d in DOMAINS]),
        ("$k$ (tournament size)", [str(best[d]["k"]) for d in DOMAINS]),
        ("limit (abandonment)", [str(best[d]["limit"]) for d in DOMAINS]),
        ("candidate layout", [best[d]["candidate_set"] for d in DOMAINS]),
        ("topology (radius)", [topo(best[d]) for d in DOMAINS]),
        ("scout variant", ["v1" if d == "mkp" else "--" for d in DOMAINS]),
    ]
    assert best["mkp"]["scout_bee_phase_version"] == 1
    return best, rows


def tie_facts():
    """Return {(domain, algorithm): (tied_at_top, pick_is_earliest_sampled, top_rows)}."""
    out = {}
    for domain in DOMAINS:
        for name, (_, ranges, _) in _domain_spaces(domain).items():
            table = pd.read_csv(f"{TUNING}/{domain}_{name}.csv")
            gen = sobol_parameter_sets(ranges, 32)
            keys = list(ranges)
            num = [k for k in keys if gen[k].dtype.kind in "fi"]
            cat = [k for k in keys if k not in num]
            order = []
            for _, r in table.iterrows():
                hit = (gen[num].sub(r[num].astype(float)).abs() < 1e-6).all(axis=1)
                hit &= (gen[cat].astype(str) == r[cat].astype(str)).all(axis=1)
                # a small space (BGWO) repeats a configuration within the 32 samples;
                # the earliest sampled index is the one that matters for the tie rule
                assert hit.any(), (domain, name, dict(r[keys]))
                order.append(int(gen.index[hit][0]))
            table["gen_order"] = order
            top = table[table.avg_rank == table.avg_rank.min()]
            out[(domain, name)] = (len(top), int(table.gen_order.iloc[0]) == int(top.gen_order.min()), top)
    return out


def main():
    best, rows = table_rows()
    print("Parameter & " + " & ".join(d.upper() for d in DOMAINS) + r" \\")
    for label, cells in rows:
        print(f"{label} & " + " & ".join(cells) + r" \\")

    facts = tie_facts()
    assert all(ok for _, ok, _ in facts.values()), "a selected row is not the earliest sampled among ties"
    assert facts[("fs", "SBABC")][0] == 9 and facts[("fs", "ABABC")][0] == 32
    assert all(facts[(d, "SBABC")][0] == 1 for d in ("mkp", "mdp", "scp"))
    shared = {k: best["mdp"][k] for k in ("beta1", "beta2", "beta3", "k", "limit", "candidate_set", "topology")}
    assert shared == {k: best["scp"][k] for k in shared}
    assert all(abs(best["mkp"][k] - shared[k]) < 1e-9 for k in ("beta1", "beta2", "beta3")) and \
        best["mkp"]["k"] == shared["k"] and best["mkp"]["limit"] == shared["limit"] and \
        best["mkp"]["candidate_set"] == shared["candidate_set"] == "single" and best["mkp"]["topology"] == "ring"
    fs_top = facts[("fs", "SBABC")][2]
    in_fs_tie = ((fs_top.beta1 - shared["beta1"]).abs() < 1e-6) & (fs_top.candidate_set == "single") & (fs_top.topology == "ring")
    assert in_fs_tie.any(), "the constrained-domain configuration is not among the FS ties"
    print("\nties at top per table:", {f"{d}/{n}": t for (d, n), (t, _, _) in facts.items() if t > 1})
    print("all 27 selected rows are the earliest sampled among their ties: OK")
    print("constrained-domain configuration is among the nine FS ties: OK")


if __name__ == "__main__":
    main()
