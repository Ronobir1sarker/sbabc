"""Fast end-to-end checks of the framework.

Run from the repository root with either
    python tests/test_smoke.py
    python -m pytest tests
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from algorithms import (
    AdaptiveBinaryABC,
    AngleModulatedABC,
    ArtificialBeeColony,
    BinaryGWO,
    BitABC,
    CriticalEventTabuSearch,
    GeneticABC,
    SetBasedABC,
    SetBasedPSO,
)
from core import Config
from experiments import RunStore, compare_algorithms, parallel_compare, sobol_parameter_sets
from experiments.parallel import (
    load_task_outputs,
    make_tasks,
    merge_task_outputs,
    restrict_tasks,
    results_table,
    run_tasks,
    save_task_outputs,
    shard,
)
from problems.mkp import load_mkp_problems

PROBLEM = load_mkp_problems("mknap1")[0]  # 6 items, known optimum 3800
CFG = Config(num_runs=3, max_iteration=200, early_stopping_patience=150, seed=1)


def run_and_check(algorithm, max_error_pct=2.0):
    results = algorithm.run(problem=PROBLEM, config=CFG)

    # the reported positions must reproduce the reported values
    # (guards against the best-solution aliasing bug of the old code)
    for pos, val, vio in zip(
        results["best_positions"], results["best_values"], results["best_violations"]
    ):
        value, violation = PROBLEM.evaluate(pos.astype(np.int8))
        assert value == val, f"{algorithm.name}: position gives {value}, reported {val}"
        assert violation == vio

    assert (results["fn_calls_per_run"] > 0).all()
    assert (results["wall_time_per_run"] > 0).all()
    assert (results["cpu_time_per_run"] >= 0).all()  # clock resolution can floor tiny runs to 0
    assert len(results["convergence_curves"]) == CFG.num_runs

    # the FE-axis curve must align point-for-point with the value curve, so
    # convergence can be plotted/truncated on a budget-fair evaluation axis
    assert len(results["fn_call_curves"]) == CFG.num_runs
    for fe, curve, total in zip(
        results["fn_call_curves"], results["convergence_curves"], results["fn_calls_per_run"]
    ):
        assert len(fe) == len(curve)
        assert (np.diff(fe) >= 0).all()
        assert fe[-1] == total

    best = results["best_values"].max()
    error = PROBLEM.error_percentage(best)
    assert error <= max_error_pct, f"{algorithm.name} error too high: {best} ({error:.2f}%)"
    return results


def test_abc():
    run_and_check(ArtificialBeeColony(num_food_sources=15, limit=50))


def test_baseline_abc_variants():
    """The paper's binary ABC baselines share the ABC skeleton and only
    change the representation and/or the distort operator."""
    run_and_check(BitABC(num_food_sources=15, limit=50, topology="von_neumann"))
    run_and_check(GeneticABC(num_food_sources=15, limit=50, topology="von_neumann"))
    run_and_check(AngleModulatedABC(num_food_sources=15, limit=50, topology="von_neumann"))


def test_sbpso():
    run_and_check(SetBasedPSO(num_particles=15, topology="von_neumann", neighborhood_radius=2))


def test_amabc_decode_matches_positions():
    """AMABC evolves 4 coefficients but must report *binary* solutions."""
    algo = AngleModulatedABC(num_food_sources=10, limit=30)
    results = algo.run(problem=PROBLEM, config=CFG)
    positions = results["best_positions"]
    assert positions.shape == (CFG.num_runs, PROBLEM.dim)
    assert set(np.unique(positions)) <= {0, 1}


def test_sbabc_all_versions():
    for version in (1, 3, 4):
        for scout in (1, 2):
            algo = SetBasedABC(
                num_food_sources=15,
                limit=50,
                distort_operator_version=version,
                scout_bee_phase_version=scout,
                topology="von_neumann",
            )
            run_and_check(algo)


def test_sbabc_spec_distort_reproduces_v4():
    """Version 5 with the default specs must be bit-identical to version 4
    (same RNG consumption order), or ablation results won't be comparable
    with the paper's main results."""
    kwargs = dict(num_food_sources=15, limit=50, topology="von_neumann")
    a = SetBasedABC(distort_operator_version=4, **kwargs).run(problem=PROBLEM, config=CFG)
    b = SetBasedABC(distort_operator_version=5, **kwargs).run(problem=PROBLEM, config=CFG)
    assert np.array_equal(a["best_values"], b["best_values"])
    assert np.array_equal(a["fn_calls_per_run"], b["fn_calls_per_run"])
    assert all(np.array_equal(x, y) for x, y in zip(a["convergence_curves"], b["convergence_curves"]))


def test_sbabc_ablation_variants_run():
    """Component subsets (incl. delete-without-move, add-without-move) and the
    shared-V3 manuscript variant must all execute."""
    cases = [
        (("mda",), False),
        (("d",), False),
        (("a",), False),
        (("m", "md"), False),
        (("m", "md", "ma", "mda"), True),
    ]
    cfg = Config(num_runs=2, max_iteration=60, early_stopping_patience=60, seed=3)
    for specs, shared in cases:
        algo = SetBasedABC(
            num_food_sources=10, limit=30, distort_operator_version=5,
            candidate_specs=specs, shared_add_velocity=shared, topology="von_neumann",
        )
        results = algo.run(problem=PROBLEM, config=cfg)
        assert (results["fn_calls_per_run"] > 0).all(), (specs, shared)


def test_ababc():
    """Durgut & Aydın's adaptive binary ABC (the recent 2021 competitor):
    Chu & Beasley repair keeps every result feasible, the adaptive-pursuit
    probabilities stay a valid distribution, ablation arms with a reduced
    operator pool run, and seeded runs reproduce."""
    algo = AdaptiveBinaryABC(num_food_sources=15, schedule_iterations=200)
    results = run_and_check(algo)
    assert (results["best_violations"] == 0).all()  # repair guarantees feasibility
    assert algo.limit is None  # the recorded parameter never drifts...
    assert algo._limit == 15 * PROBLEM.dim // 2  # ...the SN*D/2 rule resolves here
    assert (algo._op_probs > 0).all()

    for pool in (("binABC",), ("disABC",), ("ibinABC",), ("binABC", "ibinABC")):
        arm = AdaptiveBinaryABC(
            num_food_sources=10, operators=pool, schedule_iterations=60
        )
        r = arm.run(problem=PROBLEM, config=Config(num_runs=2, max_iteration=60,
                                                   early_stopping_patience=60, seed=3))
        assert (r["fn_calls_per_run"] > 0).all(), pool

    a = AdaptiveBinaryABC(num_food_sources=10, schedule_iterations=200).run(problem=PROBLEM, config=CFG)
    b = AdaptiveBinaryABC(num_food_sources=10, schedule_iterations=200).run(problem=PROBLEM, config=CFG)
    assert np.array_equal(a["best_values"], b["best_values"])
    assert np.array_equal(a["fn_calls_per_run"], b["fn_calls_per_run"])


def test_tie_break_current():
    run_and_check(ArtificialBeeColony(num_food_sources=15, limit=50, tie_break="current"), max_error_pct=5.0)


def test_k_tournament_add_minimization():
    """The tournament must follow Deb's rule in both directions: the set-based
    operators are supposed to be objective-agnostic (needed for the Set
    Covering minimization domain)."""
    from sets import operators as ops

    costs = np.array([10.0, 1.0, 5.0, 3.0])

    def eval_min(x):  # minimize cost; element 0 is infeasible
        return float((costs * x).sum()), float(2.0 if x[0] else 0.0)

    x = np.zeros(4, dtype=np.int8)
    # only one candidate and it is infeasible: least-violation fallback must still add it
    v = ops.k_tournament_add(
        np.random.default_rng(1), 1.0, np.array([1, 0, 0, 0], np.int8), x, eval_min, 4, "min"
    )
    assert v[0] == 1

    # among feasible entrants the LOWEST marginal cost must win every tournament
    # it enters; with k=32 over 3 candidates the cheapest (element 1) is drawn
    # into essentially every tournament
    rng = np.random.default_rng(2)
    for _ in range(20):
        v = ops.k_tournament_add(
            rng, 1.0, np.array([0, 1, 1, 1], np.int8), x, eval_min, 32, "min"
        )
        assert v[1] == 1, "min-mode tournament did not pick the cheapest feasible element"

    try:
        ops.k_tournament_add(rng, 1.0, np.ones(4, np.int8), x, eval_min, 4, "best")
        assert False, "invalid optimization_type must raise"
    except ValueError:
        pass


def test_scp_domain():
    """Set covering (minimization): parser, random-cover initializer, and an
    end-to-end SBABC/SBPSO descent. Skips when the OR-Library file has not
    been downloaded (problems.scp.download_scp_instances needs network)."""
    from problems.scp import load_scp_problems

    try:
        problem = load_scp_problems(["scp41"])[0]
    except FileNotFoundError:
        raise unittest.SkipTest("problems/scp/data/scp41.txt not downloaded")
    assert problem.optimization_type == "min" and problem.dim == 1000

    inits = problem.extras["init_sampler"](np.random.default_rng(0), 5)
    values, violations = problem.evaluate(inits)
    assert (violations == 0).all()  # random covers are feasible by construction

    cfg = Config(num_runs=2, max_iteration=30, early_stopping_patience=30, seed=13)
    for algo in (
        SetBasedABC(num_food_sources=10, limit=30, optimization_type="min",
                    scout_bee_phase_version=1, topology="von_neumann"),
        SetBasedPSO(num_particles=10, optimization_type="min", topology="von_neumann"),
    ):
        results = algo.run(problem=problem, config=cfg)
        assert (results["best_violations"] == 0).all()
        for curve in results["convergence_curves"]:
            assert curve[-1] <= curve[0]  # descending on a minimization problem

    a = SetBasedABC(num_food_sources=8, limit=20, optimization_type="min").run(problem=problem, config=cfg)
    b = SetBasedABC(num_food_sources=8, limit=20, optimization_type="min").run(problem=problem, config=cfg)
    assert np.array_equal(a["best_values"], b["best_values"])


def test_feature_selection_domain():
    """FS wrapper problem: deterministic evaluation, Deb-compatible
    infeasibility for the empty mask, and end-to-end SBABC run."""
    from problems.feature_selection import load_fs_problems

    problem = load_fs_problems(["wine"], n_folds=5)[0]  # bundled dataset, no network
    assert problem.dim == 13

    all_feats = np.ones(13, dtype=np.int8)
    v1, vio1 = problem.evaluate(all_feats)
    v2, _ = problem.evaluate(all_feats)
    assert v1 == v2 and vio1 == 0.0
    assert 0.8 <= v1 <= 1.0  # alpha-weighted fitness of a strong classifier
    assert problem.evaluate(np.zeros(13, dtype=np.int8))[1] == 1.0  # empty set infeasible

    cfg = Config(num_runs=2, max_iteration=5, early_stopping_patience=5, seed=11)
    algo = SetBasedABC(num_food_sources=8, limit=20)
    results = algo.run(problem=problem, config=cfg)
    assert (results["best_violations"] == 0).all()
    assert (results["best_values"] >= v1).all()  # never worse than all-features

    repeat = SetBasedABC(num_food_sources=8, limit=20).run(problem=problem, config=cfg)
    assert np.array_equal(results["best_values"], repeat["best_values"])


def test_penalized_problem_wrapper():
    """The penalty-method view (the penalty comparison): violations fold into the
    objective, the reported violation is 0, and algorithms run unchanged."""
    from problems import penalized
    from problems.mkp import load_mkp_problems

    raw = load_mkp_problems("mknap1", infeasible_value="raw")[0]
    pen = penalized(raw, factor=100.0)

    x_feasible = np.zeros(raw.dim, dtype=np.int8)
    x_all = np.ones(raw.dim, dtype=np.int8)  # overloads every knapsack
    raw_val, raw_vio = raw.evaluate(x_all)
    assert raw_vio > 0 and np.isfinite(raw_val)  # raw mode reports the true profit
    pen_val, pen_vio = pen.evaluate(x_all)
    assert pen_vio == 0
    assert pen_val == raw_val - 100.0 * raw_vio
    assert pen.evaluate(x_feasible)[0] == raw.evaluate(x_feasible)[0]  # feasible unchanged

    # -inf convention: the penalty must still produce a finite gradient
    masked = load_mkp_problems("mknap1")[0]
    pen_masked = penalized(masked, factor=100.0)
    val, vio = pen_masked.evaluate(x_all)
    assert np.isfinite(val) and vio == 0

    # end to end: SBABC on the penalized problem still finds the optimum region
    algo = SetBasedABC(num_food_sources=15, limit=50, topology="von_neumann")
    results = algo.run(problem=pen, config=CFG)
    best = results["best_values"].max()
    assert raw.error_percentage(best) <= 2.0  # optimum is feasible, so penalty-optimum = optimum


def test_max_fn_calls_budget():
    algo = SetBasedABC(num_food_sources=10, limit=30)
    results = algo.run(
        problem=PROBLEM,
        config=Config(num_runs=2, max_iteration=500, early_stopping_patience=500, max_fn_calls=3000, seed=2),
    )
    assert (results["iterations_per_run"] < 500).all()  # the budget stopped every run early
    assert (results["fn_calls_per_run"] >= 3000).all()  # after crossing the budget


def test_seed_reproducibility():
    a = SetBasedABC(num_food_sources=15).run(problem=PROBLEM, config=CFG)
    b = SetBasedABC(num_food_sources=15).run(problem=PROBLEM, config=CFG)
    assert np.array_equal(a["best_values"], b["best_values"])
    assert np.array_equal(a["fn_calls_per_run"], b["fn_calls_per_run"])


def test_compare_returns_table():
    df = compare_algorithms(
        algorithms=[ArtificialBeeColony(num_food_sources=10, limit=30)],
        problems=load_mkp_problems("mknap1")[:2],
        config=Config(num_runs=2, max_iteration=50, early_stopping_patience=50, seed=0),
        quiet=True,
    )
    assert len(df) == 2
    assert {"algorithm", "problem", "best_value", "mean_error_pct"} <= set(df.columns)


def _tiny_experiment():
    algorithms = [
        ArtificialBeeColony(num_food_sources=10, limit=30),
        SetBasedABC(num_food_sources=10, limit=30),
    ]
    problems = load_mkp_problems("mknap1")[:2]
    config = Config(num_runs=4, max_iteration=30, early_stopping_patience=30, seed=7)
    return algorithms, problems, config


def test_parallel_matches_sequential():
    algorithms, problems, config = _tiny_experiment()
    sequential = compare_algorithms(algorithms, problems, config, quiet=True)
    parallel = parallel_compare(algorithms, problems, config, max_workers=2, quiet=True)

    key = ["problem", "algorithm"]
    sequential = sequential.sort_values(key).reset_index(drop=True)
    parallel = parallel.sort_values(key).reset_index(drop=True)
    for column in ("best_value", "mean_value", "mean_fn_calls", "mean_error_pct"):
        assert (sequential[column] == parallel[column]).all(), column


def test_shard_and_merge_reconstruct_full_experiment(tmp_path=None):
    import tempfile
    from pathlib import Path

    algorithms, problems, config = _tiny_experiment()
    tasks = make_tasks(algorithms, problems, config, runs_per_task=2)
    shards = [shard(tasks, i, 2) for i in range(2)]
    assert sorted(shards[0] + shards[1], key=str) == sorted(tasks, key=str)

    with tempfile.TemporaryDirectory() as tmp:
        for i, subset in enumerate(shards):  # as if each ran on a different PC
            outputs = run_tasks(subset, algorithms, problems, config, max_workers=2, quiet=True)
            save_task_outputs(str(Path(tmp) / f"shard{i}.joblib"), outputs)
        combined = load_task_outputs(str(Path(tmp) / "shard*.joblib"))

    df = results_table(merge_task_outputs(combined, config.num_runs), problems)
    reference = compare_algorithms(algorithms, problems, config, quiet=True)
    key = ["problem", "algorithm"]
    df = df.sort_values(key).reset_index(drop=True)
    reference = reference.sort_values(key).reset_index(drop=True)
    assert (df["mean_value"] == reference["mean_value"]).all()


def test_sqlite_store_two_pc_workflow():
    """Each 'PC' computes a range of run indices into its own database; the
    merged database resumes correctly, matches the sequential reference,
    and recalls the algorithm parameters."""
    import tempfile
    from pathlib import Path

    algorithms, problems, config = _tiny_experiment()
    tasks = make_tasks(algorithms, problems, config, runs_per_task=1)
    shares = ({0, 1}, {2, 3})  # config.num_runs == 4, split like --runs 0-1 / 2-3

    def names(task, r):
        return (algorithms[task.algorithm_index].name, problems[task.problem_index].name, r)

    with tempfile.TemporaryDirectory() as tmp:
        stores = [RunStore(str(Path(tmp) / f"pc{i}.sqlite")) for i in range(2)]
        for store, share in zip(stores, shares):
            store.register_experiment("tiny", algorithms, config)
            store.record_environment("tiny")  # as the driver does before computing runs
            todo = restrict_tasks(tasks, lambda t, r: r in share)
            todo = restrict_tasks(todo, lambda t, r: names(t, r) not in store.completed_runs("tiny"))
            run_tasks(
                todo, algorithms, problems, config, max_workers=2, quiet=True,
                on_result=lambda t, res: store.save_task_results("tiny", t.run_indices, res),
            )
            # resume: everything this PC was assigned is now stored, nothing left
            done = store.completed_runs("tiny")
            assert {r for (_, _, r) in done} == share
            assert restrict_tasks(todo, lambda t, r: names(t, r) not in done) == []

        # summary must refuse while runs are missing
        try:
            stores[0].summary_table("tiny", problems)
            assert False, "expected incomplete-experiment error"
        except ValueError as e:
            assert "incomplete" in str(e)

        merged = stores[0].merge_from(stores[1].path)
        assert merged == len(algorithms) * len(problems) * len(shares[1])
        assert stores[0].merge_from(stores[1].path) == 0  # re-merge is a no-op

        df = stores[0].summary_table("tiny", problems)
        reference = compare_algorithms(algorithms, problems, config, quiet=True)
        key = ["problem", "algorithm"]
        df = df.sort_values(key).reset_index(drop=True)
        reference = reference.sort_values(key).reset_index(drop=True)
        for column in ("best_value", "mean_value", "mean_fn_calls", "mean_error_pct"):
            assert (df[column] == reference[column]).all(), column

        # stored positions must reproduce stored values
        results = stores[0].load_results("tiny", algorithms[1].name, problems[0].name)
        for pos, val in zip(results["best_positions"], results["best_values"]):
            assert problems[0].evaluate(pos.astype(np.int8))[0] == val

        # runtimes survive the store round-trip
        assert (results["wall_time_per_run"] > 0).all()
        assert np.isfinite(results["cpu_time_per_run"]).all()

        # FE-axis curves survive the round-trip, aligned with the value curves
        for fe, curve in zip(results["fn_call_curves"], results["convergence_curves"]):
            assert fe is not None and len(fe) == len(curve)
            assert (np.diff(fe) >= 0).all()

        # each machine recorded its environment, and it survived the merge
        envs = stores[0].con.execute(
            "SELECT COUNT(*) FROM environments WHERE experiment = 'tiny'"
        ).fetchone()[0]
        assert envs >= 1

        # pairwise statistics work straight off the store
        from experiments.analysis import significance_table

        sig = significance_table(stores[0], "tiny", problems, reference=algorithms[1].name)
        assert len(sig) == len(problems) * (len(algorithms) - 1)
        assert ((sig["p_value"] >= 0) & (sig["p_value"] <= 1)).all()
        assert sig["cliffs_delta"].between(-1, 1).all()

        # parameter recall
        params = stores[0].algorithm_parameters("tiny")
        sbabc_params = params[params["algorithm"] == "SBABC"]["params"].iloc[0]
        assert sbabc_params["limit"] == 30 and sbabc_params["distort_operator_version"] == 4
        assert stores[0].experiment_config("tiny").seed == config.seed

        # changed settings under the same name must be rejected
        try:
            stores[0].register_experiment("tiny", algorithms, CFG)
            assert False, "expected settings-conflict error"
        except ValueError as e:
            assert "different settings" in str(e)

        # a shard produced under a different code/library environment must be
        # refused (disjoint run indices make the runs-level check blind to it)
        stores[1].con.execute(
            "INSERT OR IGNORE INTO environments VALUES "
            "('tiny', 'deadbee+dirty', '0.0.0', '0.0.0', '0.0.0', '2000-01-01T00:00:00+00:00')"
        )
        stores[1].con.commit()
        try:
            stores[0].merge_from(stores[1].path)
            assert False, "expected environment-mismatch error"
        except ValueError as e:
            assert "environment" in str(e)
        assert stores[0].merge_from(stores[1].path, allow_env_mismatch=True) == 0

        for store in stores:
            store.close()


def _exploding_evaluate(x):
    raise RuntimeError("boom")


def test_run_tasks_survives_worker_failure():
    """One failing task must not abort the campaign (audit: a transient fault
    in a week-long run would discard every result after it): the other tasks'
    results are still delivered to on_result, and a summary error is raised at
    the end so the resume machinery recomputes only the failures."""
    from problems.base import Problem

    algorithms = [ArtificialBeeColony(num_food_sources=10, limit=30)]
    problems = [
        load_mkp_problems("mknap1")[0],
        Problem(name="boom", dim=6, evaluate=_exploding_evaluate),
    ]
    config = Config(num_runs=2, max_iteration=20, early_stopping_patience=20, seed=5)
    tasks = make_tasks(algorithms, problems, config, runs_per_task=1)

    saved = []
    try:
        run_tasks(
            tasks, algorithms, problems, config, max_workers=2, quiet=True,
            on_result=lambda t, res: saved.append(t),
        )
        assert False, "expected the end-of-campaign failure summary"
    except RuntimeError as e:
        assert "2 of 4 tasks failed" in str(e)
    good = [t for t in tasks if t.problem_index == 0]
    assert sorted(saved, key=str) == sorted(good, key=str)


def test_bgwo():
    """Luo & Zhao's binary grey wolf optimizer (the second recent
    competitor): elite initialization and repair keep everything feasible,
    both transform functions run, and seeded runs reproduce."""
    results = run_and_check(BinaryGWO(schedule_iterations=200))
    assert (results["best_violations"] == 0).all()

    cfg = Config(num_runs=2, max_iteration=60, early_stopping_patience=60, seed=17)
    s = BinaryGWO(transform="sshape", schedule_iterations=60).run(problem=PROBLEM, config=cfg)
    assert (s["best_violations"] == 0).all()

    a = BinaryGWO(schedule_iterations=60).run(problem=PROBLEM, config=cfg)
    b = BinaryGWO(schedule_iterations=60).run(problem=PROBLEM, config=cfg)
    assert np.array_equal(a["best_values"], b["best_values"])
    assert np.array_equal(a["fn_calls_per_run"], b["fn_calls_per_run"])


def test_cets():
    """Glover & Kochenberger's critical event tabu search (the local-search
    competitor): strategic oscillation keeps the best solution feasible,
    the paper's deterministic x=0 start reproduces exactly, and the random
    start stays seeded-reproducible."""
    results = run_and_check(CriticalEventTabuSearch())
    assert (results["best_violations"] == 0).all()

    cfg = Config(num_runs=2, max_iteration=100, early_stopping_patience=100, seed=9)
    a = CriticalEventTabuSearch(initial="zero").run(problem=PROBLEM, config=cfg)
    assert a["best_values"][0] == a["best_values"][1]  # deterministic from x=0
    b = CriticalEventTabuSearch(initial="zero").run(problem=PROBLEM, config=cfg)
    assert np.array_equal(a["best_values"], b["best_values"])

    c = CriticalEventTabuSearch().run(problem=PROBLEM, config=cfg)
    d = CriticalEventTabuSearch().run(problem=PROBLEM, config=cfg)
    assert np.array_equal(c["best_values"], d["best_values"])  # seeded random start


def test_large_mkp_loaders():
    """Chu & Beasley mknapcb sets and the Glover-Kochenberger gk set parse
    into Problems with the published dimensions and best-known values.
    Skips when the OR-Library downloads are absent."""
    try:
        cb = load_mkp_problems("mknapcb1")
        cb9 = load_mkp_problems("mknapcb9")
        cb5 = load_mkp_problems("mknapcb5")
        gk = load_mkp_problems("gk")
    except FileNotFoundError as missing:
        raise unittest.SkipTest(f"OR-Library data not downloaded: {missing}")
    assert len(cb) == 30 and cb[0].name == "5.100-00"
    assert cb[0].dim == 100 and cb[0].extras["num_of_constraints"] == 5
    assert cb[0].known_optimum == 24381 and cb[0].extras["optimum_proven"]
    assert {p.extras["tightness"] for p in cb} == {0.25, 0.5, 0.75}
    assert all(p.known_optimum and p.known_optimum > 0 for p in cb)

    # the curated 2024-era table supersedes the stale 1998 mkcbres floors,
    # and the still-open classes are flagged as best-known-only
    assert cb9[0].name == "30.500-00" and cb9[0].known_optimum == 116056
    assert not cb9[0].extras["optimum_proven"]
    assert all(p.extras["optimum_proven"] for p in cb5)  # 10.250: fully proven
    zero = np.zeros(cb[0].dim, dtype=np.int8)
    value, violation = cb[0].evaluate(zero)
    assert value == 0 and violation == 0
    ones = np.ones(cb[0].dim, dtype=np.int8)
    assert cb[0].evaluate(ones)[1] > 0  # selecting everything must overload

    assert [p.name for p in gk[:3]] == ["gk01", "gk02", "gk03"]
    assert len(gk) == 11 and gk[10].dim == 2500
    assert gk[0].known_optimum == 3766  # embedded in the .dat header
    assert gk[3].known_optimum == 5767  # gk04 header carries 0: GK_BEST_KNOWN path
    assert all(p.known_optimum and p.known_optimum > 0 for p in gk)


def test_sobol_parameter_sets():
    sets = sobol_parameter_sets(
        {"beta1": (0.0, 1.0), "k": (3, 11), "topology": ["star", "ring"]}, 8
    )
    assert len(sets) == 8
    assert sets["beta1"].between(0, 1).all()
    assert sets["k"].between(3, 11).all()
    assert set(sets["topology"]) <= {"star", "ring"}


def test_sbabc_rejects_invalid_parameters():
    """beta1 > 1 used to construct fine and crash stochastically mid-run when
    r1*beta1 > 1 was first drawn (scalar_mul's domain is [0, 1]); the
    constructor must catch it — and beta2/beta3/k nonsense — upfront, before
    a multi-day campaign shard is wasted on it."""
    for kwargs in (
        dict(beta1=1.5), dict(beta1=-0.1), dict(beta2=-1.0),
        dict(beta3=-0.5), dict(k=0),
    ):
        try:
            SetBasedABC(num_food_sources=5, limit=10, **kwargs)
            assert False, f"expected ValueError for {kwargs}"
        except ValueError:
            pass
    # the boundary values are all legal
    SetBasedABC(num_food_sources=5, limit=10, beta1=1.0, beta2=0.0, beta3=0.0, k=1)


def test_scout_v2_rejects_minimization():
    """Scout v2 shrinks infeasible sources — a knapsack-only premise that is
    inverted on covering-style problems (dropping columns UNcovers rows), so
    pairing it with minimization must fail loudly at construction (explicit
    "min") and at set_problem (when "auto" resolves to min), not silently
    drive every source toward the empty set."""
    from problems.base import Problem

    try:
        SetBasedABC(num_food_sources=5, limit=10, scout_bee_phase_version=2,
                    optimization_type="min")
        assert False, "expected ValueError at construction"
    except ValueError as e:
        assert "scout_bee_phase_version" in str(e)

    algo = SetBasedABC(num_food_sources=5, limit=10, scout_bee_phase_version=2,
                       optimization_type="auto")
    min_problem = Problem(name="tiny_min", dim=4,
                          evaluate=lambda x: (float(np.sum(x)), 0.0),
                          optimization_type="min")
    try:
        algo.set_problem(min_problem)
        assert False, "expected ValueError when auto resolves to min"
    except ValueError as e:
        assert "scout_bee_phase_version" in str(e)


def test_holm_correction():
    """Holm step-down: hand-checked adjustments, capping at 1, and the
    running-max monotonicity that prevents a smaller raw p from ending up
    with a larger adjusted p."""
    from experiments.analysis import holm_correction

    np.testing.assert_allclose(holm_correction([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    np.testing.assert_allclose(
        holm_correction([0.001, 0.02, 0.008, 0.9]), [0.004, 0.04, 0.024, 0.9]
    )
    assert holm_correction([0.5]) == [0.5]
    assert holm_correction([0.6, 0.7]) == [1.0, 1.0]


def _fake_results(algorithm: str, problem: str, values) -> dict:
    """A minimal, feasible results dict with everything save_task_results
    stores — for tests that need runs in a store without running anything."""
    n = len(values)
    return {
        "algorithm": algorithm,
        "problem": problem,
        "best_values": np.asarray(values, dtype=np.float64),
        "best_violations": np.zeros(n),
        "fn_calls_per_run": np.full(n, 10, dtype=np.int64),
        "iterations_per_run": np.full(n, 5, dtype=np.int64),
        "last_improvement_iteration": np.full(n, 3, dtype=np.int64),
        "fn_calls_at_best": np.full(n, 8, dtype=np.int64),
        "best_positions": np.zeros((n, 4), dtype=np.int8),
        "convergence_curves": [np.zeros(3) for _ in range(n)],
        "wall_time_per_run": np.full(n, 0.1),
        "cpu_time_per_run": np.full(n, 0.1),
        "fn_call_curves": [np.arange(3, dtype=np.int64) for _ in range(n)],
    }


def test_significance_table_requires_aligned_runs():
    """Equal-length but *different* run sets (possible after mid-campaign
    task failures on both sides) must be refused: positional pairing would
    silently mismatch the common-random-numbers runs the Wilcoxon
    signed-rank test relies on. Matching run sets must produce the p_holm
    column next to the raw p-values."""
    import tempfile
    from pathlib import Path

    from experiments.analysis import significance_table
    from problems.base import Problem

    algorithms = [
        ArtificialBeeColony(num_food_sources=5, limit=10),
        SetBasedABC(num_food_sources=5, limit=10),
    ]
    a, b = algorithms[0].name, algorithms[1].name
    problem = Problem(name="tiny", dim=4, evaluate=lambda x: (float(np.sum(x)), 0.0))
    config = Config(num_runs=4, max_iteration=5, early_stopping_patience=5, seed=3)

    with tempfile.TemporaryDirectory() as tmp:
        store = RunStore(str(Path(tmp) / "aligned.sqlite"))
        store.register_experiment("misaligned", algorithms, config)
        store.save_task_results("misaligned", [0, 1, 2], _fake_results(a, "tiny", [1.0, 2.0, 3.0]))
        store.save_task_results("misaligned", [0, 1, 3], _fake_results(b, "tiny", [1.5, 2.5, 3.5]))
        try:
            significance_table(store, "misaligned", [problem], reference=a)
            assert False, "expected run-index alignment error"
        except ValueError as e:
            assert "run indices" in str(e)

        store.register_experiment("aligned", algorithms, config)
        store.save_task_results("aligned", [0, 1, 2], _fake_results(a, "tiny", [1.0, 2.0, 3.0]))
        store.save_task_results("aligned", [0, 1, 2], _fake_results(b, "tiny", [1.5, 2.5, 3.5]))
        sig = significance_table(store, "aligned", [problem], reference=a)
        assert "p_holm" in sig.columns
        assert (sig["p_holm"] >= sig["p_value"] - 1e-15).all()
        assert sig["p_holm"].between(0, 1).all()
        store.close()


def test_tuned_config_guard():
    """A tuned-configuration JSON that exists but lacks an algorithm must
    stop the campaign build (SystemExit): silently mixing tuned and published
    parameters within one campaign is the R4 unfairness the shared tuning
    protocol exists to prevent."""
    import run_experiment

    fallback = dict(num_food_sources=5, limit=10)
    assert run_experiment._make(SetBasedABC, "SBABC", {}, **fallback).limit == 10
    tuned = {"SBABC": dict(num_food_sources=5, limit=99)}
    assert run_experiment._make(SetBasedABC, "SBABC", tuned, **fallback).limit == 99
    try:
        run_experiment._make(SetBasedABC, "SBABC", {"SBPSO": {}}, **fallback)
        assert False, "expected SystemExit on a partial tuned JSON"
    except SystemExit as e:
        assert "SBABC" in str(e)


def test_best_from_csv_recovery():
    """run_tuning must rebuild an algorithm's best configuration from its
    finished CSV on resume — otherwise an interrupted tuning campaign writes
    the best-configs JSON without the algorithms that already finished, and
    the only recovery would be hours of re-tuning."""
    import tempfile

    import pandas as pd

    import run_tuning

    ranges = {"beta1": (0.01, 1.0), "k": (3, 15), "topology": ["star", "ring"]}
    fixed = {"num_food_sources": 25}
    table = pd.DataFrame(
        {"beta1": [0.5, 0.9], "k": [5, 7], "topology": ["ring", "star"],
         "avg_rank": [1.0, 2.0]}
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "mkp_SBABC.csv")
        table.to_csv(path, index=False)
        best = run_tuning._best_from_csv(path, ranges, fixed)
        assert best == {"beta1": 0.5, "k": 5, "topology": "ring", "num_food_sources": 25}
        assert isinstance(best["k"], int)

        pd.DataFrame(columns=table.columns).to_csv(path, index=False)
        try:
            run_tuning._best_from_csv(path, ranges, fixed)
            assert False, "expected SystemExit on an empty tuning CSV"
        except SystemExit:
            pass


def test_scp_loader_rejects_empty_file():
    """A 0-byte instance file is a failed download: the loader must refuse it
    with a clear message (the raw parser would die on a bare StopIteration),
    and the downloader treats it as absent so it gets re-fetched."""
    import tempfile

    from problems.scp.problem import load_scp_problems

    with tempfile.TemporaryDirectory() as tmp:
        open(os.path.join(tmp, "scp41.txt"), "w").close()
        try:
            load_scp_problems(["scp41"], data_dir=tmp)
            assert False, "expected FileNotFoundError on an empty instance file"
        except FileNotFoundError as e:
            assert "empty" in str(e)


def test_candidate_set_alias():
    """candidate_set is the tuner-facing alias for the two operator layouts
    the experiments compare (tuples cannot ride through the Sobol choice-list
    machinery); an explicit candidate_specs must always win — the ablation
    arms pass specs over a tuned base that may carry candidate_set."""
    assert SetBasedABC(num_food_sources=5, limit=10).candidate_specs == ("m", "md", "ma", "mda")
    assert SetBasedABC(num_food_sources=5, limit=10, candidate_set="single").candidate_specs == ("mda",)

    explicit = SetBasedABC(num_food_sources=5, limit=10,
                           candidate_set="single", candidate_specs=("m", "md"))
    assert explicit.candidate_specs == ("m", "md")

    try:
        SetBasedABC(num_food_sources=5, limit=10, candidate_set="both")
        assert False, "expected ValueError for an unknown candidate_set"
    except ValueError as e:
        assert "candidate_set" in str(e)


def test_sbabc_tuning_space_layout():
    """The mkp space tunes the operator layout and scout version; scp/fs must
    NOT tune the scout (v2's shrink premise is knapsack-only — the constructor
    refuses it under minimization, which would crash candidate construction
    mid-tuning) and must pin it to v1 exactly once, in fixed."""
    import run_tuning

    _, ranges, fixed = run_tuning._domain_spaces("mkp")["SBABC"]
    assert ranges["candidate_set"] == ["single", "four"]
    assert ranges["scout_bee_phase_version"] == [1, 2]
    assert fixed["distort_operator_version"] == 5

    for domain in ("scp", "fs"):
        _, ranges, fixed = run_tuning._domain_spaces(domain)["SBABC"]
        assert "scout_bee_phase_version" not in ranges, domain
        assert fixed["scout_bee_phase_version"] == 1, domain
        assert ranges["candidate_set"] == ["single", "four"], domain

    # every mkp SBABC Sobol sample must construct (32 = the campaign size)
    _, ranges, fixed = run_tuning._domain_spaces("mkp")["SBABC"]
    for params in sobol_parameter_sets(ranges, 32).to_dict("records"):
        SetBasedABC(**params, **fixed)


def test_sbabc_cb_repairs_to_feasibility():
    """SBABC-CB = tuned SBABC + Chu-Beasley repair at the solution boundary:
    every food source and every candidate must be feasible from the first
    evaluation on (repair is bookkeeping, so FE accounting matches CBGA's
    one-evaluation-per-candidate rule)."""
    from algorithms import SetBasedABCWithRepair

    algorithm = SetBasedABCWithRepair(num_food_sources=5, limit=10, candidate_set="single")
    algorithm.set_problem(PROBLEM)
    algorithm.rng = np.random.default_rng(11)  # what run() does per child seed
    algorithm.initialize()
    assert (algorithm.food_violations == 0).all(), "repair must make init feasible"
    for iteration in range(5):
        algorithm.step(iteration)
        assert (algorithm.food_violations == 0).all(), "candidates must stay feasible"
    _, best_value, best_violation = algorithm.get_best()
    assert best_violation == 0
    assert best_value == 3800, "6-item mknap1-1 must be solved with repair"

    try:
        SetBasedABCWithRepair(num_food_sources=5, limit=10, distort_operator_version=4)
        assert False, "expected ValueError: SBABC-CB is spec-driven (v5) only"
    except ValueError as e:
        assert "version" in str(e)


def test_mkp_cb_campaign_isolates_knowledge():
    """mkp_cb_rev1 = the knowledge-isolation FAMILY (Section 7.4 of the paper):
    each repairable engine's TUNED configuration with repair as the only
    change (no separate tuning) on the competitors testbed with the same
    seed, so cross-experiment comparisons stay paired. AMABC is structurally
    excluded (solutions decode from angle-modulation coefficients)."""
    import run_experiment

    algorithms, problems, config = run_experiment.build_mkp_cb_rev1()
    assert [a.name for a in algorithms] == ["SBABC-CB", "SBPSO-CB", "GBABC-CB", "BitABC-CB"]
    assert len(problems) == 64 and config.seed == 4

    base, _, _ = run_experiment.build_mkp_competitors_rev2()
    engines = {a.name: a for a in base}
    identity = {
        "SBABC-CB": ("SBABC", ("beta1", "beta2", "beta3", "k", "limit", "candidate_specs",
                               "scout_bee_phase_version", "topology", "num_food_sources")),
        "SBPSO-CB": ("SBPSO", ("c1", "c2", "c3", "c4", "k", "num_particles", "topology")),
        "GBABC-CB": ("GBABC", ("threshold", "best_parent", "limit", "num_food_sources", "topology")),
        "BitABC-CB": ("BitABC", ("r", "limit", "num_food_sources", "topology")),
    }
    for arm in algorithms:
        engine_name, attrs = identity[arm.name]
        for attr in attrs:
            assert getattr(arm, attr) == getattr(engines[engine_name], attr), (arm.name, attr)


def test_cb_family_repairs_to_feasibility():
    """SBPSO-CB / GBABC-CB / BitABC-CB: every random solution and every
    candidate must be feasible from its first counted evaluation on (repair
    at the solution boundary, FE accounting unchanged)."""
    from algorithms import BitABCWithRepair, GeneticABCWithRepair, SetBasedPSOWithRepair

    pso = SetBasedPSOWithRepair(num_particles=5)
    pso.set_problem(PROBLEM)
    pso.rng = np.random.default_rng(7)
    pso.initialize()
    assert (pso.violations == 0).all(), "repair must make the initial swarm feasible"
    for t in range(5):
        pso.step(t)
        assert (pso.violations == 0).all() and (pso.pbest_violations == 0).all()
    assert pso.get_best()[2] == 0

    for cls in (GeneticABCWithRepair, BitABCWithRepair):
        algorithm = cls(num_food_sources=5, limit=10)
        algorithm.set_problem(PROBLEM)
        algorithm.rng = np.random.default_rng(7)
        algorithm.initialize()
        assert (algorithm.food_violations == 0).all(), cls.name
        for t in range(5):
            algorithm.step(t)
            assert (algorithm.food_violations == 0).all(), cls.name
        assert algorithm.get_best()[2] == 0, cls.name


def test_mdp_domain():
    """Maximum diversity problem: shipped MDPLIB instances load with their
    best-known values, the evaluator matches a brute-force pair sum and
    reports |#selected - m| as violation, the initializer yields feasible
    m-subsets, and the tuning/test instance sets are disjoint."""
    from problems.mdp import (
        DEFAULT_INSTANCES, KNOWN_BEST, TUNING_INSTANCES, MDPEvaluator,
        RandomSubsetSampler, load_mdp_problems, parse_mdp_text,
    )

    assert not set(DEFAULT_INSTANCES) & set(TUNING_INSTANCES)
    assert all(name in KNOWN_BEST for name in DEFAULT_INSTANCES + TUNING_INSTANCES)

    (p,) = load_mdp_problems(["SOM-b_2_n100_m20"])
    assert p.dim == 100 and p.extras["m"] == 20 and p.known_optimum == 1195
    assert p.optimization_type == "max" and p.extras["optimum_proven"] is False
    rng = np.random.default_rng(3)
    x = p.extras["init_sampler"](rng, 4)
    values, violations = p.evaluate(x)
    assert (violations == 0).all() and (x.sum(axis=1) == 20).all()
    D = p.evaluate.distances
    idx = np.flatnonzero(x[0])
    brute = sum(D[i, j] for a, i in enumerate(idx) for j in idx[a + 1:])
    assert abs(values[0] - brute) < 1e-9
    x[0, np.flatnonzero(x[0] == 0)[:3]] = 1  # 23 selected -> violation 3
    assert p.evaluate(x[0])[1] == 3

    # parser round-trip on a tiny hand-written instance
    text = "3 2\n0 1 1.5\n0 2 2.0\n1 2 4.0\n"
    distances, m = parse_mdp_text(text)
    assert m == 2 and distances[1, 2] == distances[2, 1] == 4.0 and distances[0, 0] == 0
    value, violation = MDPEvaluator(distances, m)(np.array([0, 1, 1]))
    assert value == 4.0 and violation == 0
    assert RandomSubsetSampler(3, 2)(rng, 5).sum(axis=1).tolist() == [2] * 5


def test_mdp_tuning_and_campaign_layout():
    """The mdp tuning space pins SBABC's scout to v1 (v2's shrink premise
    fails under an equality constraint), tunes on TUNING_INSTANCES, and the
    mdp_main_rev1 campaign runs the 6 generic algorithms on the 15 test
    instances with the full 100k budget (not the FS budget)."""
    import run_experiment
    import run_tuning

    _, ranges, fixed = run_tuning._domain_spaces("mdp")["SBABC"]
    assert "scout_bee_phase_version" not in ranges and fixed["scout_bee_phase_version"] == 1
    assert "optimization_type" not in fixed  # maximization is the default
    assert [p.name for p in run_tuning.tuning_problems("mdp")] == list(
        __import__("problems.mdp", fromlist=["TUNING_INSTANCES"]).TUNING_INSTANCES
    )

    algorithms, problems, config = run_experiment.build_mdp_main_rev1()
    assert len(algorithms) == 6 and len(problems) == 15
    assert config.max_fn_calls == run_experiment.MAX_FN_CALLS and config.seed == 10
    ababc = next(a for a in algorithms if a.name == "ABABC")
    assert ababc.schedule_iterations == run_experiment.MAX_FN_CALLS // (2 * 40)
    sbabc = next(a for a in algorithms if a.name == "SBABC")
    assert sbabc.scout_bee_phase_version == 1 and sbabc.optimization_type == "max"


def test_mkp_sensitivity_and_penalty_layout():
    """Sensitivity: 21 arms (shared tuned centre + 4 off-centre levels x 5
    parameters) on the ablation's 19 instances; the centre must equal the
    competitors campaign's SBABC and each arm must differ from it in exactly
    its named parameter. Penalty: the tuned SBABC under three static penalties
    (1/10/100) on the -inf-masked index-00 instances (rev1, frozen) and
    under 1/10/100/1000 on the raw-valued ones (rev2, the paper's table),
    seed 4 = the main campaign's, so rows pair run-for-run with its SBABC
    (Deb) rows; a penalty run must store the incumbent's RAW
    profit/violation and keep the raw problem picklable."""
    import run_experiment
    from algorithms import SetBasedABCPenalty
    from problems.base import penalized

    algorithms, problems, config = run_experiment.build_mkp_sensitivity_rev1()
    assert len(algorithms) == 21 and len(problems) == 19 and config.seed == 12
    assert len({a.name for a in algorithms}) == 21
    centre = next(a for a in algorithms if a.name == "SBABC_tuned")
    base, _, _ = run_experiment.build_mkp_competitors_rev2()
    sbabc = next(a for a in base if a.name == "SBABC")
    swept = tuple(run_experiment.SENSITIVITY_LEVELS)
    for attr in swept + ("candidate_specs", "scout_bee_phase_version", "topology"):
        assert getattr(centre, attr) == getattr(sbabc, attr), attr
    for arm in algorithms:
        if arm is centre:
            continue
        param, level = arm.name[len("SBABC_"):].split("=")
        assert float(getattr(arm, param)) == float(level), arm.name
        assert all(getattr(arm, p) == getattr(centre, p) for p in swept if p != param), arm.name

    algorithms, problems, config = run_experiment.build_mkp_penalty_rev1()
    assert [a.name for a in algorithms] == ["SBABC_pen1", "SBABC_pen10", "SBABC_pen100"]
    # seed 4 = the competitors campaigns' seed: rows pair run-for-run with
    # the main campaign's SBABC (Deb) rows, which serve as the Deb column
    assert len(problems) == 9 and config.seed == 4 and config.num_runs == 30
    assert all(p.name.endswith("-00") for p in problems)
    # default -inf masking: penalized() then scores infeasible solutions as
    # -factor*violation, so every run ends feasible (the draft's property)
    assert all(p.evaluate.infeasible_value == "minus_inf" for p in problems)
    for arm in algorithms:
        for attr in swept + ("candidate_specs",):
            assert getattr(arm, attr) == getattr(centre, attr), (arm.name, attr)

    algorithms, problems, config = run_experiment.build_mkp_penalty_rev2()
    assert [a.name for a in algorithms] == [
        "SBABC_pen1", "SBABC_pen10", "SBABC_pen100", "SBABC_pen1000"
    ]
    assert len(problems) == 9 and config.seed == 4 and config.num_runs == 30
    assert all(p.name.endswith("-00") for p in problems)
    # raw profits: the penalty is profit - factor * excess, the classical
    # static penalty (rev1's -inf masking made it order-equivalent to Deb)
    assert all(p.evaluate.infeasible_value == "raw" for p in problems)
    for arm in algorithms:
        for attr in swept + ("candidate_specs",):
            assert getattr(arm, attr) == getattr(centre, attr), (arm.name, attr)

    algorithm = SetBasedABCPenalty(
        penalty_factor=100.0, num_food_sources=5, limit=10, candidate_set="single"
    )
    algorithm.run(PROBLEM, Config(num_runs=2, max_iteration=30, early_stopping_patience=30, seed=5))
    for r in range(2):
        value, violation = PROBLEM.evaluate(algorithm.best_solution_positions[r])
        assert algorithm.best_solution_values[r] == value
        assert algorithm.best_solution_violations[r] == violation
    assert algorithm.problem is PROBLEM, "raw problem must stay attached (picklable)"
    try:
        algorithm.set_problem(penalized(PROBLEM, 100.0))
        assert False, "double wrapping must be refused"
    except ValueError:
        pass


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
            except unittest.SkipTest as reason:
                print(f"SKIP {name} ({reason})")
            else:
                print(f"PASS {name}")
    print("All smoke tests passed.")

def test_mkp_ablation_rev2_layout():
    """rev2 completes the single-candidate subsets (md, ma, da) plus the mda
    reproduction control, on rev1's exact base, instances and seed: its base
    must equal what the database recorded for rev1's SBABC_full (not the
    retuned JSON), and run r must draw rev1's run-r seed."""
    import json
    import os
    import sqlite3

    import run_experiment

    algorithms, problems, config = run_experiment.build_mkp_ablation_rev2()
    assert [a.name for a in algorithms] == [
        "SBABC_md_only", "SBABC_ma_only", "SBABC_da_only", "SBABC_mda_only"
    ]
    assert [a.candidate_specs for a in algorithms] == [("md",), ("ma",), ("da",), ("mda",)]
    assert all(a.distort_operator_version == 5 and not a.shared_add_velocity for a in algorithms)
    assert len(problems) == 19 and config.seed == 9 and config.num_runs == 30
    assert config.max_fn_calls == run_experiment.MAX_FN_CALLS
    assert [p.name for p in problems] == [p.name for p in run_experiment._ablation_instances()]

    db = os.path.join(os.path.dirname(run_experiment.__file__), "results", "results.sqlite")
    if not os.path.exists(db):
        return  # clean checkout: the cross-check needs the stored rev1 record
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    row = con.execute(
        "SELECT params_json FROM experiment_algorithms "
        "WHERE experiment = 'mkp_ablation_rev1' AND algorithm = 'SBABC_full'"
    ).fetchone()
    if row is None:
        return
    stored = json.loads(row[0])
    for key, value in run_experiment._ABLATION_REV1_BASE.items():
        assert stored[key] == value, key
    settings = con.execute(
        "SELECT master_seed, num_runs, max_fn_calls FROM experiments WHERE name = 'mkp_ablation_rev1'"
    ).fetchone()
    assert settings == (config.seed, config.num_runs, config.max_fn_calls)
