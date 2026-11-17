"""Unified parameter tuning for the experiments: within a domain, every algorithm is tuned with the SAME
protocol — 32 Sobol-sampled configurations x the same runs x the same
tuning instances at the same evaluation budget, best configuration by mean
rank across instances. State the protocol verbatim in the paper.

Domains (--domain, default mkp):
  mkp   9 algorithms; tuning instances = Chu & Beasley -15/-25 instances
        (mid/high tightness, 100-500 items), DISJOINT from every test
        instance (tests use mknap1/mknap2 + the -00 instances); 5 runs at
        the campaign's 100k-evaluation budget.
  scp   the 6 domain-generic algorithms (CBGA/CETS/BGWO are MKP-only),
        minimization; tuning instances scp44/45, scp54/55, scp64, scpb1 —
        disjoint from the scp_main_rev1 test set; 5 runs x 100k.
  fs    the same 6 algorithms; tuning datasets glass + vehicle, which are
        thereby EXCLUDED from the fs_main_rev1 table; 3 runs x 25k
        evaluations (wrapper evaluations are ~10^3 x MKP's cost).
  mdp   the same 6 algorithms on the maximum diversity problem (MDPLIB,
        maximization, cardinality constraint); tuning instances
        problems.mdp.TUNING_INSTANCES — disjoint from mdp_main_rev1's
        DEFAULT_INSTANCES; 5 runs x 100k.

Run it (leave a core free with --workers 6):
    python run_tuning.py --domain mkp --workers 6

One CSV per algorithm lands in results/tuning/ as it finishes; rerunning
skips algorithms whose CSV already exists (delete a CSV to redo one). When
a domain completes, the best configurations are saved to
results/tuning/<domain>_best_configs.json — run_experiment.py reads that
file automatically when building the domain's campaigns, so no manual
copying is needed.
"""

import argparse
import json
import os

from algorithms import (
    AdaptiveBinaryABC,
    AngleModulatedABC,
    BinaryGWO,
    BitABC,
    ChuBeasleyGA,
    CriticalEventTabuSearch,
    GeneticABC,
    SetBasedABC,
    SetBasedPSO,
)
from core import Config
from experiments.store import _json_default
from experiments.tuning import tune_parameters

OUT_DIR = os.path.join("results", "tuning")
SAMPLE_SIZE = 32  # Sobol configurations per algorithm (power of 2)

# tune at the budget you will test at (mkp/scp: 100k FE; fs: 25k — wrapper
# evaluations are vastly more expensive, mirrored by fs_main_rev1)
CONFIGS = {
    "mkp": Config(num_runs=5, max_iteration=5000, early_stopping_patience=5000,
                  max_percentage_error=0.0, max_fn_calls=100_000, seed=11),
    "scp": Config(num_runs=5, max_iteration=5000, early_stopping_patience=5000,
                  max_percentage_error=0.0, max_fn_calls=100_000, seed=11),
    "mdp": Config(num_runs=5, max_iteration=5000, early_stopping_patience=5000,
                  max_percentage_error=0.0, max_fn_calls=100_000, seed=11),
    "fs": Config(num_runs=3, max_iteration=2000, early_stopping_patience=2000,
                 max_percentage_error=-1.0,  # no known optima for FS fitness
                 max_fn_calls=25_000, seed=11),
}

# one topology sub-space shared by the swarm/bee algorithms (star ignores
# the radius; the tuner samples it anyway, which is harmless)
_TOPOLOGY = {
    "topology": ["star", "ring", "von_neumann"],
    "neighborhood_radius": (1, 2),
}

# name -> (class, sobol ranges, fixed constructor params)
# ranges follow each paper's stated/experimented intervals; populations stay
# at the published sizes so the spaces tune behavior, not colony scale
MKP_SPACES = {
    "SBABC": (
        SetBasedABC,
        {
            "beta1": (0.01, 1.0),
            "beta2": (0.0, 3.0),
            "beta3": (0.0, 6.0),
            "k": (3, 15),
            "limit": (30, 240),
            # operator layout as a categorical, like BGWO's transform or the
            # topology: "four" (the published operator) spends 4 evaluations
            # per bee where "single" spends 1, and the rev1 ablation showed
            # the extra iterations dominate on large instances — let the
            # unified protocol make that call, not a post-hoc choice
            "candidate_set": ["single", "four"],
            # never exercised by rev1 (every campaign ran the v1 default);
            # v2 re-seeds infeasible sources by shrinking instead of at random
            "scout_bee_phase_version": [1, 2],
            **_TOPOLOGY,
        },
        # v5 with candidate_set="four" is bit-identical to the published v4
        # (proven to the PCG64 state by the rev1 audit), so fixing v5 here
        # changes nothing except making candidate_set tunable
        {"num_food_sources": 25, "distort_operator_version": 5},
    ),
    "SBPSO": (
        SetBasedPSO,
        {
            "c1": (0.0, 1.0),  # velocity-probability scales: [0, 1] by definition
            "c2": (0.0, 1.0),
            "c3": (0.0, 3.0),
            "c4": (0.0, 3.0),
            "k": (3, 15),
            **_TOPOLOGY,
        },
        {"num_particles": 25},
    ),
    "GBABC": (
        GeneticABC,
        {"limit": (30, 240), "threshold": (1, 10), **_TOPOLOGY},
        {"num_food_sources": 25, "best_parent": "gbest"},
    ),
    "BitABC": (
        BitABC,
        {"limit": (30, 240), "r": (0.1, 0.9), **_TOPOLOGY},
        {"num_food_sources": 25},
    ),
    "AMABC": (
        AngleModulatedABC,
        {"limit": (30, 240), **_TOPOLOGY},
        {"num_food_sources": 25},
    ),
    "CBGA": (
        ChuBeasleyGA,
        {"population_size": (50, 200), "mutation_bits": (1, 4)},
        {},
    ),
    "ABABC": (
        AdaptiveBinaryABC,
        {
            "credit_alpha": (0.1, 0.9),
            "pursuit_beta": (0.1, 0.9),
            "p_min": (0.01, 0.2),
            "reward_window": (5, 50),
            "reward_type": ["instant", "average", "extreme"],
        },
        {"num_food_sources": 40, "schedule_iterations": 100_000 // (2 * 40)},
    ),
    "CETS": (
        CriticalEventTabuSearch,
        {"t": (3, 12), "p2": (5, 9), "k_max": (2, 6)},
        {"p1": 3},
    ),
    "BGWO": (
        BinaryGWO,
        # BGWO spends exactly num_wolves evaluations per iteration, so below
        # s=20 the max_iteration=5000 cap binds before the 100k budget and the
        # candidate would be tuned (and later run) FE-starved — the lower
        # bound stays at 20 to keep every configuration budget-equal
        {"num_wolves": (20, 40), "transform": ["vshape", "sshape"]},
        # sigma(t) decays within the first ~5% of iterations, so anchoring
        # the schedule at s=20's iteration count is safe for the whole range
        {"schedule_iterations": 100_000 // 20},
    ),
}


# The 6 domain-generic algorithms for scp/fs: the MKP spaces minus the
# MKP-only competitors, with domain-appropriate fixed parameters layered on.
_GENERIC = ("SBABC", "SBPSO", "GBABC", "BitABC", "AMABC", "ABABC")


def _domain_spaces(domain: str) -> dict:
    if domain == "mkp":
        return MKP_SPACES
    extra = {"optimization_type": "min"} if domain == "scp" else {}
    spaces = {}
    for name in _GENERIC:
        cls, ranges, fixed = MKP_SPACES[name]
        fixed = {**fixed, **extra}
        if name == "SBABC":
            # scout v2's shrink premise is knapsack-only: SCP inverts it
            # (dropping columns UNcovers rows — the constructor refuses),
            # MDP's equality constraint breaks it (dropping below m adds
            # violation), and FS has no constraints so v2 degenerates to
            # v1; pin v1 rather than spend a Sobol dimension on a dead or
            # invalid choice
            ranges = {key: spec for key, spec in ranges.items() if key != "scout_bee_phase_version"}
            fixed["scout_bee_phase_version"] = 1
        if name == "ABABC" and domain == "fs":
            # schedule anchored to the FS budget: 25k FE / (2 * SN)
            fixed["schedule_iterations"] = 25_000 // (2 * 40)
        spaces[name] = (cls, ranges, fixed)
    return spaces


def tuning_problems(domain: str):
    if domain == "mkp":
        from problems.mkp import load_mkp_problems

        picks = {  # set -> instance indices (disjoint from the -00 test instances)
            "mknapcb1": (15, 25),  # 5.100-15, 5.100-25
            "mknapcb4": (15,),  # 10.100-15
            "mknapcb2": (15,),  # 5.250-15
            "mknapcb5": (15,),  # 10.250-15
            "mknapcb3": (15,),  # 5.500-15
        }
        problems = []
        for set_name, indices in picks.items():
            instances = load_mkp_problems(set_name)
            problems += [instances[i] for i in indices]
        return problems
    if domain == "scp":
        from problems.scp import load_scp_problems

        # disjoint from scp_main_rev1's test set (the DEFAULT_INSTANCES)
        return load_scp_problems(["scp44", "scp45", "scp54", "scp55", "scp64", "scpb1"])
    if domain == "mdp":
        from problems.mdp import TUNING_INSTANCES, load_mdp_problems

        # disjoint from mdp_main_rev1's test set (the DEFAULT_INSTANCES)
        return load_mdp_problems(TUNING_INSTANCES)
    from problems.feature_selection import load_fs_problems

    # tuning-only datasets, excluded from the fs_main_rev1 table
    return load_fs_problems(["glass", "vehicle"])


def _best_from_csv(path: str, ranges: dict, fixed: dict) -> dict:
    """Recover an algorithm's best configuration from its finished tuning CSV
    (tune_parameters sorts it best-first), so a rerun after an interruption
    rebuilds <domain>_best_configs.json without hours of re-tuning."""
    import pandas as pd

    table = pd.read_csv(path)
    missing = [key for key in ranges if key not in table.columns]
    if missing or not len(table):
        raise SystemExit(
            f"{path} exists but is unusable "
            f"({'empty' if not len(table) else f'missing columns {missing}'}); "
            f"delete it to re-tune"
        )
    best = table.iloc[0][list(ranges)].to_dict()
    return {**_coerce(ranges, best), **fixed}


def _coerce(ranges: dict, best: dict) -> dict:
    """Pandas upcasts a mixed row to float; restore int-typed parameters so
    the JSON round-trips into constructor-ready values."""
    out = {}
    for key, value in best.items():
        spec = ranges.get(key)
        if isinstance(spec, tuple) and all(isinstance(x, int) for x in spec):
            out[key] = int(value)
        elif isinstance(spec, tuple):
            out[key] = float(value)
        else:
            out[key] = value
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", choices=("mkp", "scp", "fs", "mdp"), default="mkp")
    parser.add_argument("--workers", type=int, default=None, help="process-pool size")
    parser.add_argument("--only", metavar="NAME", help="tune a single algorithm")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    spaces = _domain_spaces(args.domain)
    if args.only and args.only not in spaces:
        parser.error(
            f"--only {args.only!r} is not a {args.domain} algorithm; "
            f"choose from: {', '.join(spaces)}"
        )
    config = CONFIGS[args.domain]
    problems = tuning_problems(args.domain)
    print(f"[{args.domain}] tuning instances: {[p.name for p in problems]}")

    best_configs = {}
    for name, (cls, ranges, fixed) in spaces.items():
        if args.only and name != args.only:
            continue
        out_path = os.path.join(OUT_DIR, f"{args.domain}_{name}.csv")
        if os.path.exists(out_path):
            # an interruption can leave finished CSVs with no JSON entry, so
            # recover the best row instead of dropping the algorithm
            best_configs[name] = _best_from_csv(out_path, ranges, fixed)
            print(f"{name}: {out_path} exists — skipping (best configuration "
                  f"recovered from it; delete the CSV to re-tune)")
            continue
        print(f"\n=== Tuning {name}: {SAMPLE_SIZE} configs x {config.num_runs} runs "
              f"x {len(problems)} instances ===")
        best, table = tune_parameters(
            cls, problems, ranges, fixed_params=fixed, config=config,
            sample_size=SAMPLE_SIZE, max_workers=args.workers,
        )
        table.to_csv(out_path, index=False)
        best_configs[name] = {**_coerce(ranges, best), **fixed}
        print(f"{name}: saved {out_path}\n  best: {best_configs[name]}")

    done = {}
    json_path = os.path.join(OUT_DIR, f"{args.domain}_best_configs.json")
    if os.path.exists(json_path):
        done = json.load(open(json_path))
    done.update(best_configs)
    with open(json_path, "w") as f:
        json.dump(done, f, indent=2, default=_json_default)
    print(f"\nBest configurations so far ({len(done)}/{len(spaces)}) -> {json_path}")
    for name, params in done.items():
        print(f"  {name}: {params}")


if __name__ == "__main__":
    main()
