"""Experiment driver: multi-core execution, multi-PC run splitting, and a
SQLite results database with per-run rows.

The experiments are predefined — pick one with --experiment (see
CAMPAIGNS; default mkp_competitors_rev2). Every campaign automatically uses
the tuned configurations from results/tuning/<domain>_best_configs.json
when that file exists (produced by run_tuning.py) and falls back to the
published parameters with a loud warning otherwise. Every completed run is
written to the database immediately, so an interrupted experiment resumes
where it stopped — rerunning the same command never recomputes runs that
are already stored.

All runs on this machine (leave a core free with --workers 6):
    python run_experiment.py --experiment mkp_competitors_rev2 --workers 6

Split the 30 runs across two PCs (same repo, same code + numpy versions —
the merge refuses mismatched environments; per-run seeding makes the merged
result bit-identical to a single-machine run):
    python run_experiment.py --runs 0-14      # PC 1
    python run_experiment.py --runs 15-29     # PC 2
    # copy PC 2's results/results.sqlite here as results/results_pc2.sqlite:
    python run_experiment.py --merge results/results_pc2.sqlite

When all runs are in the database the summary table is printed and saved as
CSV (--save to change the path). To just re-print it later:
    python run_experiment.py --experiment <name> --summary

Recall what an experiment ran with (config, parameters, git commit):
    python run_experiment.py --experiment <name> --params
"""

import argparse
import os
import re

from algorithms import (
    AdaptiveBinaryABC,
    AngleModulatedABC,
    BinaryGWO,
    BitABC,
    BitABCWithRepair,
    GeneticABCWithRepair,
    SetBasedPSOWithRepair,
    ChuBeasleyGA,
    CriticalEventTabuSearch,
    GeneticABC,
    SetBasedABC,
    SetBasedABCPenalty,
    SetBasedABCWithRepair,
    SetBasedPSO,
)
from core import Config
from experiments.parallel import make_tasks, restrict_tasks, run_tasks
from experiments.store import RunStore

# Every run gets this evaluation budget; FE-equal comparison is the fairness
# instrument for every experiment (per-iteration FE costs differ up to 20x
# between these algorithms). 100k matches the recent-competitor literature
# (Durgut & Aydın 2021). CETS is the disclosed exception: its interior moves
# are constraint bookkeeping, so the iteration cap is what actually bounds it.
MAX_FN_CALLS = 100_000
FS_MAX_FN_CALLS = 25_000  # wrapper evaluations are ~10^3 x an MKP evaluation


def _tuned_configs(domain: str) -> dict:
    """The per-algorithm tuned parameters run_tuning.py wrote, or {} with a
    loud warning — the R4-fair campaigns must run on tuned configurations."""
    path = os.path.join("results", "tuning", f"{domain}_best_configs.json")
    if os.path.exists(path):
        import json

        print(f"Using tuned configurations from {path}")
        return json.load(open(path))
    print(
        f"WARNING: {path} not found — building with PUBLISHED/fallback "
        f"parameters. For the paper campaigns run `python run_tuning.py "
        f"--domain {domain}` to completion first (shared tuning protocol)."
    )
    return {}


def _tuned_params(tuned, key, fallback):
    """The tuned parameters for `key`, or `fallback` when no tuning JSON
    exists at all. A JSON that exists but lacks `key` is an error: silently
    mixing tuned and published parameters within one campaign is exactly the
    unfairness the shared tuning protocol exists to prevent."""
    if not tuned:
        return fallback
    if key not in tuned:
        raise SystemExit(
            f"The tuned-configuration JSON lacks {key!r} (it has: "
            f"{', '.join(sorted(tuned))}). Finish the tuning first "
            f"(python run_tuning.py --only {key}) or delete/rename the JSON "
            f"to run the whole campaign on published parameters."
        )
    return tuned[key]


def _make(cls, key, tuned, display=None, **fallback):
    """One algorithm: the tuned configuration when available, else fallback."""
    algorithm = cls(**_tuned_params(tuned, key, fallback))
    if display:
        algorithm.name = display
    return algorithm


def _mkp_algorithms(tuned):
    """The 9-algorithm MKP comparison set."""
    shared = dict(topology="von_neumann", neighborhood_radius=2)
    return [
        _make(
            SetBasedABC,
            "SBABC",
            tuned,
            num_food_sources=25,
            limit=120,
            beta1=0.5,
            beta2=1.5,
            beta3=3,
            k=5,
            distort_operator_version=4,
            **shared,
        ),
        _make(
            SetBasedPSO,
            "SBPSO",
            tuned,
            num_particles=25,
            c1=0.5156,
            c2=0.4531,
            c3=1.8359,
            c4=2.2578,
            k=7,
            **shared,
        ),
        _make(
            GeneticABC,
            "GBABC",
            tuned,
            num_food_sources=25,
            limit=100,
            threshold=3,
            best_parent="gbest",
            **shared,
        ),
        _make(BitABC, "BitABC", tuned, num_food_sources=25, limit=100, r=0.5, **shared),
        _make(
            AngleModulatedABC, "AMABC", tuned, num_food_sources=25, limit=100, **shared
        ),
        _make(ChuBeasleyGA, "CBGA", tuned, population_size=100),
        _make(
            AdaptiveBinaryABC,
            "ABABC",
            tuned,
            schedule_iterations=MAX_FN_CALLS // (2 * 40),
        ),
        _make(CriticalEventTabuSearch, "CETS", tuned),  # random start: run variance
        _make(BinaryGWO, "BGWO", tuned, schedule_iterations=MAX_FN_CALLS // 20),
    ]


def _generic_algorithms(tuned, domain: str):
    """The 6 domain-generic algorithms for SCP/FS (CBGA/CETS/BGWO are
    MKP-only: their repair/choice rules read knapsack instance structure)."""
    direction = "min" if domain == "scp" else "max"
    budget = FS_MAX_FN_CALLS if domain == "fs" else MAX_FN_CALLS
    shared = dict(
        topology="von_neumann", neighborhood_radius=2, optimization_type=direction
    )
    # scout v2's shrink premise is knapsack-only: SCP inverts it and MDP's
    # equality constraint breaks it (FS is unconstrained: v2 == v1 there)
    sbabc_extra = {"scout_bee_phase_version": 1} if domain in ("scp", "mdp") else {}
    return [
        _make(
            SetBasedABC,
            "SBABC",
            tuned,
            num_food_sources=25,
            limit=120,
            beta1=0.5,
            beta2=1.5,
            beta3=3,
            k=5,
            distort_operator_version=4,
            **sbabc_extra,
            **shared,
        ),
        _make(
            SetBasedPSO,
            "SBPSO",
            tuned,
            num_particles=25,
            c1=0.5156,
            c2=0.4531,
            c3=1.8359,
            c4=2.2578,
            k=7,
            **shared,
        ),
        _make(
            GeneticABC,
            "GBABC",
            tuned,
            num_food_sources=25,
            limit=100,
            threshold=3,
            best_parent="gbest",
            **shared,
        ),
        _make(BitABC, "BitABC", tuned, num_food_sources=25, limit=100, r=0.5, **shared),
        _make(
            AngleModulatedABC, "AMABC", tuned, num_food_sources=25, limit=100, **shared
        ),
        _make(
            AdaptiveBinaryABC,
            "ABABC",
            tuned,
            schedule_iterations=budget // (2 * 40),
            optimization_type=direction,
        ),
    ]


def build_mkp_competitors_rev1():
    """SBABC vs every comparator on the small-MKP testbed (all of
    mknap1 + mknap2) plus one instance per Chu & Beasley class, 30 runs,
    FE-equal at 100k. A run ends at the budget, the iteration cap, or the
    (best-)known value.

    FROZEN record of the four-candidate SBABC configuration (the ablation
    then showed the single composite candidate dominates at equal FE, and
    the retune on the widened space confirmed it). Superseded by
    mkp_competitors_rev2; register_experiment now refuses to compute under
    this name because the tuned JSON changed — use --summary/--params."""
    from problems.mkp import load_mkp_problems

    problems = load_mkp_problems("mknap1") + load_mkp_problems("mknap2")
    for set_name in (f"mknapcb{i}" for i in range(1, 10)):
        problems += load_mkp_problems(set_name)[:1]  # the m.n-00 instance
    config = Config(
        num_runs=30,
        max_iteration=5000,
        early_stopping_patience=5000,
        max_percentage_error=0.0,
        max_fn_calls=MAX_FN_CALLS,
        seed=4,
    )
    return _mkp_algorithms(_tuned_configs("mkp")), problems, config


def build_mkp_competitors_rev2():
    """The main comparison on the RETUNED SBABC: the widened space made the candidate
    layout and scout variant tunable, and the tuner picked the single
    composite (mda) candidate with scout v1 / ring decisively (avg rank
    1.17 vs 3.50 for the best four-candidate sample; 4.30% vs 7.35% at
    n=500). Identical protocol to rev1 — same instances, budget, and seed,
    so every non-SBABC row reproduces rev1 bit-for-bit as a cross-check."""
    return build_mkp_competitors_rev1()


def build_mkp_large90_rev2():
    """Scale study: 10 instances per Chu & Beasley class (90 total; indices
    0-3/10-12/20-22 span all three tightness blocks), answering the
    '27 instances is unfair' complaint. 30 runs, FE-equal at 100k.
    Named rev2 from the start: it was held until the retune settled the
    candidate-set question and never ran under the rev1 configuration."""
    from problems.mkp import load_mkp_problems

    indices = (0, 1, 2, 3, 10, 11, 12, 20, 21, 22)
    problems = []
    for set_name in (f"mknapcb{i}" for i in range(1, 10)):
        instances = load_mkp_problems(set_name)
        problems += [instances[i] for i in indices]
    config = Config(
        num_runs=30,
        max_iteration=5000,
        early_stopping_patience=5000,
        max_percentage_error=0.0,
        max_fn_calls=MAX_FN_CALLS,
        seed=5,
    )
    return _mkp_algorithms(_tuned_configs("mkp")), problems, config


def build_mkgk_rev2():
    """The 11 Glover-Kochenberger instances up to n=2500, m=100 —
    the scalability evidence (quality + the recorded runtimes). 10 runs.
    Named rev2 from the start, like mkp_large90_rev2."""
    from problems.mkp import load_mkp_problems

    problems = load_mkp_problems("gk")
    config = Config(
        num_runs=10,
        max_iteration=5000,
        early_stopping_patience=5000,
        max_percentage_error=0.0,
        max_fn_calls=MAX_FN_CALLS,
        seed=6,
    )
    return _mkp_algorithms(_tuned_configs("mkp")), problems, config


def build_mkp_cb_rev1():
    """The knowledge-isolation FAMILY (Section 7.4 of the paper): every
    problem-independent engine that admits the construction — SBABC, SBPSO,
    GBABC, BitABC — runs its tuned configuration UNCHANGED plus the
    Chu & Beasley repair (the same module CBGA and ABABC use), applied to
    every candidate and every random solution before its counted
    evaluation. AMABC is structurally excluded: its solutions are decoded
    from angle-modulation coefficients, and a repaired solution cannot be
    written back. Deliberately not tuned separately — holding every
    parameter fixed isolates domain knowledge as the single manipulated
    variable. Same 64 instances, budget, and seed as the competitors
    campaigns: cross-experiment comparisons stay paired by run seed."""
    from problems.mkp import load_mkp_problems

    problems = load_mkp_problems("mknap1") + load_mkp_problems("mknap2")
    for set_name in (f"mknapcb{i}" for i in range(1, 10)):
        problems += load_mkp_problems(set_name)[:1]  # the m.n-00 instance

    tuned = _tuned_configs("mkp")
    shared = dict(topology="von_neumann", neighborhood_radius=2)
    sbabc = _tuned_params(tuned, "SBABC", dict(
        num_food_sources=25, limit=120, beta1=0.5, beta2=1.5, beta3=3, k=5, **shared,
    ))
    algorithms = [
        SetBasedABCWithRepair(**{**sbabc, "distort_operator_version": 5}),
        _make(
            SetBasedPSOWithRepair, "SBPSO", tuned, num_particles=25,
            c1=0.5156, c2=0.4531, c3=1.8359, c4=2.2578, k=7, **shared,
        ),
        _make(
            GeneticABCWithRepair, "GBABC", tuned, num_food_sources=25,
            limit=100, threshold=3, best_parent="gbest", **shared,
        ),
        _make(BitABCWithRepair, "BitABC", tuned, num_food_sources=25, limit=100, r=0.5, **shared),
    ]
    config = Config(
        num_runs=30,
        max_iteration=5000,
        early_stopping_patience=5000,
        max_percentage_error=0.0,
        max_fn_calls=MAX_FN_CALLS,
        seed=4,
    )
    return algorithms, problems, config


def build_mdp_main_rev1():
    """The Maximum Diversity domain (maximization, cardinality
    equality constraint, quadratic objective) on 15 MDPLIB instances
    (SOM-b / GKD-c / MDG-a / MDG-b, n = 100-500; best-known values, not
    proven optima — disjoint from the tuning instances), 30 runs, FE-equal
    at 100k. Replaces set covering as the second non-knapsack domain: the
    SCP diagnostic showed every knowledge-free algorithm stuck at 250-2000%
    above optimum where a trivial greedy scores ~3%."""
    from problems.mdp import load_mdp_problems

    problems = load_mdp_problems()  # DEFAULT_INSTANCES
    config = Config(
        num_runs=30,
        max_iteration=5000,
        early_stopping_patience=5000,
        max_percentage_error=0.0,
        max_fn_calls=MAX_FN_CALLS,
        seed=10,
    )
    return _generic_algorithms(_tuned_configs("mdp"), "mdp"), problems, config


def build_scp_main_rev1():
    """The Set Covering domain (minimization) on the 15 default
    OR-Library instances (proven optima; disjoint from the tuning set),
    30 runs, FE-equal at 100k."""
    from problems.scp import load_scp_problems

    problems = load_scp_problems()  # DEFAULT_INSTANCES
    config = Config(
        num_runs=30,
        max_iteration=5000,
        early_stopping_patience=5000,
        max_percentage_error=0.0,
        max_fn_calls=MAX_FN_CALLS,
        seed=7,
    )
    return _generic_algorithms(_tuned_configs("scp"), "scp"), problems, config


def build_fs_main_rev1():
    """Feature selection (k-NN wrapper, alpha=0.9) on 7 benchmark
    datasets (13-100 features) — glass and vehicle are tuning-only, and
    madelon (500 features) is excluded on measured wrapper cost (one run
    at the 25k budget was measured in the tens of hours; the exclusion
    predates any comparative results — Section 6.1 of
    the paper). 30 runs at the 25k wrapper-evaluation budget; no known
    optima, so runs end only at the budget."""
    from problems.feature_selection import DATASETS, load_fs_problems

    names = [n for n in DATASETS if n not in ("glass", "vehicle", "madelon")]
    problems = load_fs_problems(names)
    config = Config(
        num_runs=30,
        max_iteration=2000,
        early_stopping_patience=2000,
        max_percentage_error=-1.0,
        max_fn_calls=FS_MAX_FN_CALLS,
        seed=8,
    )
    return _generic_algorithms(_tuned_configs("fs"), "fs"), problems, config


def _ablation_instances():
    """The ablation's 19 MKP instances: ten small ones with proven optima
    (mknap1-2..7, four mknap2) and the tightest (index-00) instance of each
    Chu-Beasley class."""
    from problems.mkp import load_mkp_problems

    small = load_mkp_problems("mknap1") + load_mkp_problems("mknap2")
    by_name = {p.name: p for p in small}
    problems = [by_name[f"mknap1-{i}"] for i in range(2, 8)]  # 10-50 items
    problems += [by_name[n] for n in ("mknap2-1", "mknap2-2", "mknap2-9", "mknap2-10")]
    for set_name in (f"mknapcb{i}" for i in range(1, 10)):
        problems += load_mkp_problems(set_name)[:1]
    return problems


def build_mkp_ablation_rev1():
    """The SBABC component ablation — nine arms around the TUNED base configuration: the full
    4-candidate operator, the manuscript's shared-V3 semantics, reduced
    candidate sets down to single components, and the textbook tie rule.
    10 small + 9 large instances, 30 runs, FE-equal at 100k.

    COMPLETE and frozen around the pre-retune tuned base (the arms override
    candidate_specs explicitly, so its within-campaign comparisons stand);
    the retune has since changed the JSON, so computing under this name is
    refused — use --summary/--params. The three single-candidate layouts it
    lacks (md, ma, da) are build_mkp_ablation_rev2, on the same base."""
    problems = _ablation_instances()

    tuned = _tuned_configs("mkp")
    base = _tuned_params(
        tuned,
        "SBABC",
        dict(
            num_food_sources=25,
            limit=120,
            beta1=0.5,
            beta2=1.5,
            beta3=3,
            k=5,
            distort_operator_version=4,
            topology="von_neumann",
            neighborhood_radius=2,
        ),
    )
    arms = {  # name -> constructor overrides on the tuned base
        "SBABC_full": dict(candidate_specs=("m", "md", "ma", "mda")),
        "SBABC_sharedV3": dict(
            candidate_specs=("m", "md", "ma", "mda"), shared_add_velocity=True
        ),
        "SBABC_mda_only": dict(candidate_specs=("mda",)),
        "SBABC_no_add": dict(candidate_specs=("m", "md")),
        "SBABC_no_drop": dict(candidate_specs=("m", "ma")),
        "SBABC_move_only": dict(candidate_specs=("m",)),
        "SBABC_drop_only": dict(candidate_specs=("d",)),
        "SBABC_add_only": dict(candidate_specs=("a",)),
        "SBABC_textbook_ties": dict(
            candidate_specs=("m", "md", "ma", "mda"), tie_break="current"
        ),
    }
    algorithms = []
    for name, overrides in arms.items():
        params = {**base, "distort_operator_version": 5, **overrides}
        algorithm = SetBasedABC(**params)
        algorithm.name = name
        algorithms.append(algorithm)

    config = Config(
        num_runs=30,
        max_iteration=5000,
        early_stopping_patience=5000,
        max_percentage_error=0.0,
        max_fn_calls=MAX_FN_CALLS,
        seed=9,
    )
    return algorithms, problems, config


_ABLATION_REV1_BASE = dict(  # the pre-retune tuned SBABC that mkp_ablation_rev1 ran on
    # Verbatim from the database (experiment_algorithms, SBABC_full minus the
    # layout keys); NOT results/tuning/mkp_best_configs.json, which the retune
    # has since changed (topology ring). test_smoke cross-checks it.
    num_food_sources=25,
    limit=102,
    beta1=0.6596875,
    beta2=1.96875,
    beta3=0.1875,
    k=7,
    topology="star",
    neighborhood_radius=1,
    scout_bee_phase_version=1,
    tie_break="candidate",
)


def build_mkp_ablation_rev2():
    """Ablation, completion: the three single-candidate layouts
    missing from rev1, so that every non-empty subset of {m, d, a} has been
    run as a single candidate — the component question is then answered on
    the operator SBABC actually uses (the single composite), not only by
    deletions from the four-candidate layout. mda is re-run as a
    reproduction control: it must match rev1 digit for digit. Same frozen
    base, instances, budget and seed as rev1, so rows pair run for run with
    rev1's."""
    problems = _ablation_instances()
    arms = {
        "SBABC_md_only": dict(candidate_specs=("md",)),
        "SBABC_ma_only": dict(candidate_specs=("ma",)),
        "SBABC_da_only": dict(candidate_specs=("da",)),
        "SBABC_mda_only": dict(candidate_specs=("mda",)),  # control = rev1 row
    }
    algorithms = []
    for name, overrides in arms.items():
        algorithm = SetBasedABC(**_ABLATION_REV1_BASE, distort_operator_version=5, **overrides)
        algorithm.name = name
        algorithms.append(algorithm)

    config = Config(
        num_runs=30,
        max_iteration=5000,
        early_stopping_patience=5000,
        max_percentage_error=0.0,
        max_fn_calls=MAX_FN_CALLS,
        seed=9,  # = rev1: identical per-run seeds, hence identical initial colonies
    )
    return algorithms, problems, config


_SBABC_SINGLE_FALLBACK = dict(  # only used when no tuning JSON exists at all
    num_food_sources=25,
    limit=120,
    beta1=0.5,
    beta2=1.5,
    beta3=3,
    k=5,
    distort_operator_version=5,
    candidate_set="single",
    topology="von_neumann",
    neighborhood_radius=2,
)

SENSITIVITY_LEVELS = {  # off-centre levels; the tuned value is the shared centre arm
    "beta1": (0.05, 0.3, 0.85, 1.0),
    "beta2": (0.0, 0.5, 1.0, 3.0),
    "beta3": (0.0, 1.0, 3.0, 6.0),
    "k": (1, 3, 11, 15),
    "limit": (30, 60, 160, 240),
}


def build_mkp_sensitivity_rev1():
    """One-at-a-time sensitivity of the TUNED
    SBABC around its tuned point. Each of beta1, beta2, beta3, k and limit
    takes four off-centre levels spanning its tuning range while every other
    parameter stays at the tuned value; the centre arm (SBABC_tuned) is
    shared by the five sweeps, so each sweep reads as five arms. Same 19
    instances as the ablation (10 small + 9 large alpha=0.25), 30 runs,
    FE-equal at 100k. A level equal to the tuned value collapses into the
    centre arm rather than duplicating it."""
    from problems.mkp import load_mkp_problems

    small = load_mkp_problems("mknap1") + load_mkp_problems("mknap2")
    by_name = {p.name: p for p in small}
    problems = [by_name[f"mknap1-{i}"] for i in range(2, 8)]  # 10-50 items
    problems += [by_name[n] for n in ("mknap2-1", "mknap2-2", "mknap2-9", "mknap2-10")]
    for set_name in (f"mknapcb{i}" for i in range(1, 10)):
        problems += load_mkp_problems(set_name)[:1]

    base = _tuned_params(_tuned_configs("mkp"), "SBABC", _SBABC_SINGLE_FALLBACK)
    arms = {"SBABC_tuned": {}}
    for param, levels in SENSITIVITY_LEVELS.items():
        for level in levels:
            if abs(float(level) - float(base[param])) < 1e-9:
                continue  # already the centre arm
            arms[f"SBABC_{param}={level:g}"] = {param: level}

    algorithms = []
    for name, overrides in arms.items():
        algorithm = SetBasedABC(**{**base, **overrides})
        algorithm.name = name
        algorithms.append(algorithm)

    config = Config(
        num_runs=30,
        max_iteration=5000,
        early_stopping_patience=5000,
        max_percentage_error=0.0,
        max_fn_calls=MAX_FN_CALLS,
        seed=12,
    )
    return algorithms, problems, config


PENALTY_FACTORS = (1.0, 10.0, 100.0)


def build_mkp_penalty_rev1():
    """FROZEN, superseded by mkp_penalty_rev2 (see its docstring for why).
    The first design of the penalty comparison: the tuned SBABC
    under static penalties 1 / 10 / 100 on the nine alpha=0.25 "-00"
    Chu & Beasley instances, 30 runs, FE-equal at 100k, SEED 4 — the
    competitors campaigns' seed, so every run pairs with the main
    campaign's SBABC (Deb's rule) row for the paired comparison; the Deb
    column of the paper's table is taken from mkp_competitors_rev2, not
    re-run here. Instances keep the DEFAULT infeasible_value="minus_inf":
    penalized() rebuilds a masked solution's objective as
    -factor * violation, so any feasible profit outranks every infeasible
    score and every run ends feasible — the property the draft reports.
    (Penalizing RAW profits instead was tried and rejected: a mild factor
    is negligible against profit magnitudes, most runs end infeasible, and
    the draft's Table 15 is not reproduced. Verified 2026-10-09: the
    masked design matches the draft to its printed precision.)"""
    from problems.mkp import load_mkp_problems

    problems = []
    for set_name in (f"mknapcb{i}" for i in range(1, 10)):
        problems += load_mkp_problems(set_name)[:1]

    base = _tuned_params(_tuned_configs("mkp"), "SBABC", _SBABC_SINGLE_FALLBACK)
    algorithms = []
    for factor in PENALTY_FACTORS:
        arm = SetBasedABCPenalty(penalty_factor=factor, **base)
        arm.name = f"SBABC_pen{factor:g}"
        algorithms.append(arm)

    config = Config(
        num_runs=30,
        max_iteration=5000,
        early_stopping_patience=5000,
        max_percentage_error=0.0,
        max_fn_calls=MAX_FN_CALLS,
        seed=4,
    )
    return algorithms, problems, config


PENALTY_FACTORS_REV2 = (1.0, 10.0, 100.0, 1000.0)


def build_mkp_penalty_rev2():
    """The penalty comparison of the paper's Section 7.10: the tuned SBABC as a static
    penalty method, profit - factor * capacity excess, at factors 1 / 10 /
    100 / 1000 on the nine alpha=0.25 "-00" Chu & Beasley instances, 30
    runs, 100k evaluations, SEED 4 = the competitors campaigns' seed, so
    every run pairs with mkp_competitors_rev2's SBABC (Deb's rule) row,
    which is the Deb column of the paper's table (not re-run here).
    Instances are loaded with infeasible_value="raw" so the penalty sees
    true profits: this is the classical static penalty, under which a
    factor that is too small (1) leaves most runs infeasible, which the
    table reports (statistics over feasible runs, plus the feasible count).
    Supersedes mkp_penalty_rev1 (frozen): there the default -inf masking
    made penalized() score an infeasible solution as -factor * violation,
    an ordering that coincides with Deb's rule in every comparison, so rev1
    differed from the Deb runs only through the onlooker roulette (verified
    2026-11-03: with the roulette neutralised, rev1's three factors and
    Deb's rule are bit-identical)."""
    from problems.mkp import load_mkp_problems

    problems = []
    for set_name in (f"mknapcb{i}" for i in range(1, 10)):
        problems += load_mkp_problems(set_name, infeasible_value="raw")[:1]

    base = _tuned_params(_tuned_configs("mkp"), "SBABC", _SBABC_SINGLE_FALLBACK)
    algorithms = []
    for factor in PENALTY_FACTORS_REV2:
        arm = SetBasedABCPenalty(penalty_factor=factor, **base)
        arm.name = f"SBABC_pen{factor:g}"
        algorithms.append(arm)

    config = Config(
        num_runs=30,
        max_iteration=5000,
        early_stopping_patience=5000,
        max_percentage_error=0.0,
        max_fn_calls=MAX_FN_CALLS,
        seed=4,
    )
    return algorithms, problems, config


def build_mkp_tierule_rev1():
    """Tie rule on the single-candidate layout: rev1
    tested the strict-improvement rule of the original ABC on the
    four-candidate layout only (SBABC_textbook_ties), while the tuned
    knapsack configuration uses the single composite candidate. This arm
    is SBABC_mda_only of mkp_ablation_rev2 with tie_break="current" and
    nothing else changed: same frozen base, instances, budget and seed, so
    its rows pair run for run with that stored arm. The strict rule is the
    one the convergence theorem assumes."""
    problems = _ablation_instances()
    algorithm = SetBasedABC(
        **{**_ABLATION_REV1_BASE, "tie_break": "current"},
        distort_operator_version=5,
        candidate_specs=("mda",),
    )
    algorithm.name = "SBABC_mda_strict"
    config = Config(
        num_runs=30,
        max_iteration=5000,
        early_stopping_patience=5000,
        max_percentage_error=0.0,
        max_fn_calls=MAX_FN_CALLS,
        seed=9,  # = mkp_ablation_rev1/rev2: identical per-run seeds
    )
    return [algorithm], problems, config


def build_mkp_tierule_main_rev1():
    """The same question under the configuration the knapsack comparison
    runs: the tuned SBABC of mkp_competitors_rev2 with
    tie_break="current" and nothing else changed, on the 19 ablation
    instances (all of them part of that comparison), same budget and seed,
    so its rows pair run for run with the stored SBABC runs."""
    problems = _ablation_instances()
    base = _tuned_params(_tuned_configs("mkp"), "SBABC", _SBABC_SINGLE_FALLBACK)
    algorithm = SetBasedABC(**{**base, "tie_break": "current"})
    algorithm.name = "SBABC_strict"
    config = Config(
        num_runs=30,
        max_iteration=5000,
        early_stopping_patience=5000,
        max_percentage_error=0.0,
        max_fn_calls=MAX_FN_CALLS,
        seed=4,  # = mkp_competitors_rev2: identical per-run seeds
    )
    return [algorithm], problems, config


CAMPAIGNS = {
    "mkp_sensitivity_rev1": build_mkp_sensitivity_rev1,
    "mkp_penalty_rev1": build_mkp_penalty_rev1,  # frozen: --summary only (superseded by rev2)
    "mkp_penalty_rev2": build_mkp_penalty_rev2,
    "mkp_competitors_rev1": build_mkp_competitors_rev1,  # frozen: --summary only
    "mkp_competitors_rev2": build_mkp_competitors_rev2,
    "mkp_large90_rev2": build_mkp_large90_rev2,
    "mkgk_rev2": build_mkgk_rev2,
    "mkp_cb_rev1": build_mkp_cb_rev1,
    "mdp_main_rev1": build_mdp_main_rev1,
    "scp_main_rev1": build_scp_main_rev1,
    "fs_main_rev1": build_fs_main_rev1,
    "mkp_ablation_rev1": build_mkp_ablation_rev1,  # frozen: --summary only
    "mkp_ablation_rev2": build_mkp_ablation_rev2,
    "mkp_tierule_rev1": build_mkp_tierule_rev1,
    "mkp_tierule_main_rev1": build_mkp_tierule_main_rev1,
}


def build_experiment(name: str = "mkp_competitors_rev2"):
    """(experiment name, algorithms, problems, config) for a campaign. Must
    be identical on every machine that runs a share of it; the experiment
    name is the comparability unit, and register_experiment refuses to reuse
    a name whose settings or parameters changed (e.g. tuned configs landing
    after untuned runs started — never mix those)."""
    algorithms, problems, config = CAMPAIGNS[name]()
    return name, algorithms, problems, config


def parse_run_range(text: str, num_runs: int) -> set[int]:
    """'0-14' → {0, ..., 14}; '7' → {7}. Bounds-checked against num_runs."""
    match = re.fullmatch(r"(\d+)(?:-(\d+))?", text.strip())
    if match is None:
        raise ValueError(f"--runs {text!r} must look like 'A' or 'A-B'")
    start = int(match.group(1))
    stop = int(match.group(2)) if match.group(2) else start
    if not 0 <= start <= stop < num_runs:
        raise ValueError(
            f"--runs {text!r} is outside this experiment's 0-{num_runs - 1}"
        )
    return set(range(start, stop + 1))


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--experiment",
        choices=sorted(CAMPAIGNS),
        default="mkp_competitors_rev2",
        help="which predefined campaign to run (default: %(default)s)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="process-pool size (default: all cores)",
    )
    parser.add_argument(
        "--runs",
        metavar="A-B",
        help="only run indices A..B inclusive (for multi-PC), e.g. 0-14",
    )
    parser.add_argument(
        "--db",
        default="results/results.sqlite",
        help="results database (default: %(default)s)",
    )
    parser.add_argument(
        "--merge",
        metavar="FILE",
        help="merge another machine's database file into --db",
    )
    parser.add_argument(
        "--allow-env-mismatch",
        action="store_true",
        help="merge even when the other database was produced by a different "
        "code/library environment (see RunStore.merge_from)",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="only print/save the table from the database",
    )
    parser.add_argument(
        "--params",
        action="store_true",
        help="show the stored config and algorithm parameters",
    )
    parser.add_argument(
        "--save", default="results/experiment.csv", help="path for the final CSV table"
    )
    return parser.parse_args()


def print_params(store, experiment):
    """Show the stored config and per-algorithm parameters (--params)."""
    try:
        config = store.experiment_config(experiment)
    except KeyError as missing:
        print(f"\n{missing.args[0]}")
        return
    print(f"Experiment {experiment!r}: {config}")
    for _, row in store.algorithm_parameters(experiment).iterrows():
        print(f"\n{row['algorithm']} ({row['class_name']}):")
        for key, value in sorted(row["params"].items()):
            print(f"  {key} = {value}")


def compute_missing_runs(args, store, experiment, algorithms, problems, config):
    """Run every task not yet in the database (optionally restricted by --runs)."""
    store.register_experiment(experiment, algorithms, config)
    tasks = make_tasks(algorithms, problems, config, runs_per_task=1)
    total = sum(len(t.run_indices) for t in tasks)

    if args.runs:
        selected = parse_run_range(args.runs, config.num_runs)
        tasks = restrict_tasks(tasks, lambda t, r: r in selected)

    done = store.completed_runs(experiment)
    tasks = restrict_tasks(
        tasks,
        lambda t, r: (
            algorithms[t.algorithm_index].name,
            problems[t.problem_index].name,
            r,
        )
        not in done,
    )
    todo = sum(len(t.run_indices) for t in tasks)
    print(
        f"Experiment {experiment!r}: {total} runs total, {len(done)} already in "
        f"{args.db}, {todo} to compute now"
    )
    if tasks:
        # only run-producing invocations record this machine's environment,
        # so merge_from's drift check reflects code that made actual rows
        store.record_environment(experiment)
        run_tasks(
            tasks,
            algorithms,
            problems,
            config,
            max_workers=args.workers,
            on_result=lambda task, results: store.save_task_results(
                experiment, task.run_indices, results
            ),
        )


def print_summary(args, store, experiment, problems):
    """Print the final summary table and save it as CSV; on an incomplete
    experiment, show how many runs are in the database instead."""
    try:
        df = store.summary_table(experiment, problems)
    except KeyError as missing:
        print(f"\n{missing.args[0]}")
        return
    except ValueError as incomplete:
        print(f"\n{incomplete}")
        print("\nRuns in the database so far:")
        print(store.run_counts(experiment).to_string(index=False))
        return

    save_dir = os.path.dirname(args.save)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    df.to_csv(args.save, index=False)
    print(f"\nExperiment complete — saved final table to {args.save}\n")
    columns = [
        c
        for c in (
            "problem",
            "algorithm",
            "best_value",
            "mean_error_pct",
            "success_runs",
            "num_runs",
            "mean_fn_calls",
        )
        if c in df.columns
    ]
    print(df[columns].to_string(index=False))


def main():
    args = parse_args()
    experiment, algorithms, problems, config = build_experiment(args.experiment)
    store = RunStore(args.db)

    if args.params:
        print_params(store, experiment)
        return

    if args.merge:
        inserted = store.merge_from(
            args.merge, allow_env_mismatch=args.allow_env_mismatch
        )
        print(f"Merged {inserted} new runs from {args.merge} into {args.db}")
    elif not args.summary:
        try:
            compute_missing_runs(args, store, experiment, algorithms, problems, config)
        except RuntimeError as failures:
            # completed tasks are already saved; show progress, then fail
            print(f"\n{failures}")
            print_summary(args, store, experiment, problems)
            raise SystemExit(1)

    print_summary(args, store, experiment, problems)


if __name__ == "__main__":
    main()
