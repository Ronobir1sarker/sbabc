"""Reproduces the set covering boundary study of the paper (tab:scp and the
constraint-handling check of that section).

Without flags: prints the table rows -- random-cover and greedy anchors
(computed here) and every algorithm's tuned result from results/tuning/scp_*.csv
(the selected configuration = the first row of each CSV, which the tuning sorts
best first, earliest sampled among ties).

With --penalty: re-runs the six tuned problem-independent algorithms under
Deb's rule and as static penalty methods (factors 100 and 1000) with 3 runs
each at the tuning budget and seed, re-scores every incumbent on the raw
problem, and prints mean error over feasible runs with the feasible count.
Results are also written to results/scp_penalty_check.json. About two hours
on 14 workers.

With --probe INSTANCE [INSTANCE ...]: one tuned SBABC run per instance
(tuning budget and seed, run index 0) instrumented to record how each
candidate fares in greedy selection -- infeasible (uncovers a row), feasible
but worse, equal, better -- the mean number of columns a candidate adds and
removes, and the number of scout resets. Prints one JSON line per instance;
the paper quotes scp44 and scpb1. About half a minute and two minutes.
"""
import argparse, csv, glob, json, os, sys, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from problems.scp import load_scp_problems, KNOWN_OPTIMA  # noqa: E402

INSTANCES = ["scp44", "scp45", "scp54", "scp55", "scp64", "scpb1"]
ALGS = ["SBABC", "SBPSO", "GBABC", "BitABC", "AMABC", "ABABC"]
FACTORS = [None, 100.0, 1000.0]
TUNING_DIR = os.path.join(ROOT, "results", "tuning")


def column_costs(problem):
    return problem.evaluate(np.eye(problem.dim, dtype=np.int8))[0]


def greedy_cover(problem):
    """Chvatal's greedy (minimum cost per newly covered row), then redundant
    columns dropped most expensive first."""
    cover = problem.extras["init_sampler"].cover_by_column
    costs = column_costs(problem)
    m, n = cover.shape
    covered = np.zeros(m, dtype=bool)
    x = np.zeros(n, dtype=np.int8)
    while not covered.all():
        new = cover[~covered].sum(axis=0)
        ratio = np.where(new > 0, costs / np.maximum(new, 1), np.inf)
        j = int(np.argmin(ratio))
        x[j] = 1
        covered |= cover[:, j]
    counts = cover @ x.astype(np.int64)
    for j in np.argsort(-costs):
        if x[j] and (counts[cover[:, j]] >= 2).all():
            x[j] = 0
            counts[cover[:, j]] -= 1
    value, violation = problem.evaluate(x)
    assert violation == 0
    return float(value)


def tuned_rows():
    rows = {}
    for name in ALGS:
        with open(os.path.join(TUNING_DIR, f"scp_{name}.csv")) as fh:
            table = list(csv.DictReader(fh))
        top = table[0]
        assert float(top["avg_rank"]) == min(float(r["avg_rank"]) for r in table)
        rows[name] = [float(top[f"mean_error_pct[{i}]"]) for i in INSTANCES]
    return rows


def anchors(n_samples=1000, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for p in load_scp_problems(INSTANCES):
        opt = KNOWN_OPTIMA[p.name]
        values, violations = p.evaluate(p.extras["init_sampler"](rng, n_samples))
        assert (violations == 0).all()
        out[p.name] = (100 * (values.mean() - opt) / opt, 100 * (greedy_cover(p) - opt) / opt)
    return out


def print_table():
    anc, tuned = anchors(), tuned_rows()
    print("Instance & Random cover & " + " & ".join(ALGS) + r" & Greedy \\")
    for j, inst in enumerate(INSTANCES):
        cells = " & ".join(f"{tuned[a][j]:.0f}" for a in ALGS)
        print(f"{inst} & {anc[inst][0]:.0f} & {cells} & {anc[inst][1]:.1f} \\\\")
    best = [min(tuned[a][j] for a in ALGS) for j in range(len(INSTANCES))]
    print("best of the six per instance:", [f"{b:.0f}" for b in best],
          [min(ALGS, key=lambda a: tuned[a][j]) for j in range(len(INSTANCES))])


# ---------------------------------------------------------------- penalty
def _task(args):
    name, factor, inst = args
    from core import Config
    from problems.base import penalized
    from run_tuning import MKP_SPACES
    from algorithms.sbabc_penalty import SetBasedABCPenalty
    best = json.load(open(os.path.join(TUNING_DIR, "scp_best_configs.json")))
    problem = load_scp_problems([inst])[0]
    opt = KNOWN_OPTIMA[inst]
    params = dict(best[name])
    params.setdefault("optimization_type", "min")
    cfg = Config(num_runs=3, max_iteration=5000, early_stopping_patience=5000,
                 max_percentage_error=0.0, max_fn_calls=100_000, seed=11)
    t0 = time.time()
    if factor is None:
        alg = MKP_SPACES[name][0](**params)
        alg.run(problem=problem, config=cfg)
        values, violations = alg.best_solution_values, alg.best_solution_violations
    elif name == "SBABC":
        alg = SetBasedABCPenalty(penalty_factor=factor, **params)
        alg.run(problem=problem, config=cfg)
        values, violations = alg.best_solution_values, alg.best_solution_violations
    else:
        alg = MKP_SPACES[name][0](**params)
        alg.run(problem=penalized(problem, factor), config=cfg)
        values, violations = problem.evaluate(alg.best_solution_positions)
    values, violations = np.asarray(values, float), np.asarray(violations, float)
    feasible = violations <= 0
    err = [100 * (v - opt) / opt for v in values[feasible]]
    return {"algorithm": name, "factor": factor, "instance": inst,
            "mean_error_pct": float(np.mean(err)) if err else None,
            "feasible_runs": int(feasible.sum()), "errors": err,
            "seconds": round(time.time() - t0, 1)}


def run_penalty(workers):
    import multiprocessing as mp
    tasks = [(a, f, i) for a in ALGS for f in FACTORS for i in INSTANCES]
    results = []
    with mp.get_context("fork").Pool(workers) as pool:
        for r in pool.imap_unordered(_task, tasks):
            results.append(r)
            print(f"{len(results):3d}/{len(tasks)} {r['algorithm']:6s} factor={r['factor']} {r['instance']}: "
                  f"{r['mean_error_pct'] if r['mean_error_pct'] is None else round(r['mean_error_pct'], 1)}% "
                  f"feasible {r['feasible_runs']}/3 ({r['seconds']}s)", flush=True)
    with open(os.path.join(ROOT, "results", "scp_penalty_check.json"), "w") as fh:
        json.dump(results, fh, indent=1)
    print_penalty(results)


def print_penalty(results):
    table = {(r["algorithm"], r["factor"], r["instance"]): r for r in results}
    print("\nmean error % over feasible runs (feasible/3) -- Deb | penalty 100 | penalty 1000")
    for a in ALGS:
        for inst in INSTANCES:
            cells = []
            for f in FACTORS:
                r = table[(a, f, inst)]
                cells.append(f"{r['mean_error_pct']:7.1f} ({r['feasible_runs']})" if r["mean_error_pct"] is not None else "   inf  (0)")
            print(f"{a:6s} {inst:6s} " + " | ".join(cells))


# ------------------------------------------------------------------ probe
def probe(inst):
    from core import Config
    from algorithms.sbabc import SetBasedABC
    params = json.load(open(os.path.join(TUNING_DIR, "scp_best_configs.json")))["SBABC"]
    problem = load_scp_problems([inst])[0]
    opt = KNOWN_OPTIMA[inst]

    class Probe(SetBasedABC):
        def initialize(self, *a, **k):
            super().initialize(*a, **k)
            self.stats = dict(candidates=0, infeasible=0, worse=0, equal=0, better=0,
                              scouts=0, added=0, removed=0, iterations=0)

        def step(self, it):
            super().step(it)
            self.stats["iterations"] += 1

        def _perturb_food_source(self, n):  # the skeleton's selection, with counters
            src = self.food_sources[n].copy()
            candidate, value, violation = self.distort_operator(n, self.random_neighbor(n))
            cur_v, cur_vio = self.food_values[n], self.food_violations[n]
            st = self.stats
            st["candidates"] += 1
            st["added"] += int(((candidate == 1) & (src == 0)).sum())
            st["removed"] += int(((candidate == 0) & (src == 1)).sum())
            if violation > 0 and cur_vio <= 0:
                st["infeasible"] += 1
            elif self.is_first_better(cur_v, value, cur_vio, violation):
                st["worse"] += 1
            elif value == cur_v and violation == cur_vio:
                st["equal"] += 1
            else:
                st["better"] += 1
            if self.is_first_better(cur_v, value, cur_vio, violation):
                self.trials[n] += 1
            else:
                self.food_sources[n, :] = candidate
                self.food_values[n] = value
                self.food_violations[n] = violation
                self.trials[n] = 0

        def _reinitialize_food_source(self, n):
            self.stats["scouts"] += 1
            super()._reinitialize_food_source(n)

    alg = Probe(**params)
    cfg = Config(num_runs=1, max_iteration=5000, early_stopping_patience=5000,
                 max_percentage_error=0.0, max_fn_calls=100_000, seed=11)
    t0 = time.time()
    alg.run(problem=problem, config=cfg)
    st = alg.stats
    n = st["candidates"]
    print(json.dumps(dict(
        instance=inst, error_pct=round(100 * (alg.best_solution_values[0] - opt) / opt, 1),
        seconds=round(time.time() - t0, 1), **st,
        infeasible_pct=round(100 * st["infeasible"] / n, 1), worse_pct=round(100 * st["worse"] / n, 2),
        equal_pct=round(100 * st["equal"] / n, 2), better_pct=round(100 * st["better"] / n, 3),
        mean_added=round(st["added"] / n, 1), mean_removed=round(st["removed"] / n, 1),
        scouts_per_source=round(st["scouts"] / params["num_food_sources"], 1))), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--penalty", action="store_true")
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--print-json", action="store_true", help="print the stored penalty results")
    ap.add_argument("--probe", nargs="+", metavar="INSTANCE", help="instrumented SBABC run(s)")
    a = ap.parse_args()
    if a.probe:
        for inst in a.probe:
            probe(inst)
    elif a.print_json:
        print_penalty(json.load(open(os.path.join(ROOT, "results", "scp_penalty_check.json"))))
    elif a.penalty:
        run_penalty(a.workers)
    else:
        print_table()
