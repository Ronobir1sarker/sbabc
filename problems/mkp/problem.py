"""
Multidimensional Knapsack Problem (MKP).

    maximize   profits . x
    subject to weights @ x <= capacity   (m constraints)
               x in {0, 1}^n

Benchmark sets under ``data/``:
  mknap1, mknap2      small OR-Library instances, stored as joblib files
                      (``*_data_recover.py`` rebuilds them from the .txt)
  mknapcb1..mknapcb9  the 270 large Chu & Beasley instances (OR-Library
                      .txt token streams; 30 instances per file, named by the
                      literature convention "m.n-XX", e.g. "5.100-00").
                      The files carry optimum 0; values come from
                      data/mknapcb_best_known.txt, the cross-verified
                      2024-era compilation (proven optima for all classes
                      except 30.250/30.500 — extras["optimum_proven"] says
                      which; see the file header for sources), falling back
                      to mkcbres.txt (Chu & Beasley's stale 1998 GA floors)
                      if the curated file is missing.
  gk                  the 11 Glover-Kochenberger instances gk01-gk11
                      (data/gk/*.dat, header "n m best"; best is 0 from
                      gk04 on, filled from GK_BEST_KNOWN below).
"""

from __future__ import annotations

import os
import re

import joblib
import numpy as np

from problems.base import Problem

_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

BENCHMARK_SETS = ("mknap1", "mknap2") + tuple(f"mknapcb{i}" for i in range(1, 10)) + ("gk",)

# Best-known values (NOT proven optima) for gk04-gk11; gk01-gk03 embed
# theirs in the .dat header. Verified 2026-07-11 against two independent
# tables: arXiv:2003.06617 (ICAwICA, Table 4 = Drake's compilation) and the
# FEPSS study (Applied Soft Computing 2024; arXiv:2210.03918, Table 6),
# whose new lower bounds for the largest instances are included here.
# NB: several papers (e.g. Durgut & Aydın 2021) tabulate older values
# (gk05: 7560, gk06: 7677, gk07: 19220, gk08: 18806, gk09: 58087,
# gk10: 57295, gk11: 95237) — use those numbers, not these, when comparing
# against such a paper's reported gaps.
GK_BEST_KNOWN = {
    "gk01": 3766, "gk02": 3958, "gk03": 5656, "gk04": 5767,
    "gk05": 7561, "gk06": 7680, "gk07": 19221, "gk08": 18808,
    "gk09": 58091, "gk10": 57296, "gk11": 95239,
}


class MKPEvaluator:
    """Evaluate MKP solutions; works on a single vector (dim,) or a batch (..., dim).

    Returns (value, violation): value is the total profit and violation the
    summed capacity excess (0 = feasible). With the default
    infeasible_value="minus_inf" the profit of an infeasible solution is
    masked to -inf (the convention of the published results); "raw" reports
    the true profit instead, which penalty-method experiments need — the
    framework compares solutions with Deb's rule either way.
    """

    def __init__(self, profits, weights, capacity, infeasible_value: str = "minus_inf"):
        if infeasible_value not in ("minus_inf", "raw"):
            raise ValueError(f"infeasible_value must be 'minus_inf' or 'raw', got {infeasible_value!r}")
        self.profits = np.asarray(profits)
        self.weights = np.asarray(weights)  # shape (m, n)
        self.capacity = np.asarray(capacity)  # shape (m,)
        self.infeasible_value = infeasible_value

    def __call__(self, x):
        total_weights = np.tensordot(x, self.weights, axes=([x.ndim - 1], [1]))
        capacity = self.capacity[(np.newaxis,) * (x.ndim - 1) + (...,)]
        violation = np.sum(np.maximum(0, total_weights - capacity), axis=-1)
        profit = x @ self.profits
        if self.infeasible_value == "raw":
            return profit.astype(np.float64), violation
        return np.where(violation == 0, profit, -np.inf), violation


def _make_problem(name, profits, weights, capacity, known_optimum, infeasible_value, **extra):
    profits = np.asarray(profits, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    capacity = np.asarray(capacity, dtype=np.float64)
    return Problem(
        name=name,
        dim=len(profits),
        evaluate=MKPEvaluator(profits, weights, capacity, infeasible_value),
        known_optimum=known_optimum,
        optimization_type="max",
        extras={
            "num_of_constraints": weights.shape[0],
            # instance data for algorithms with problem-specific repair
            # operators (e.g. the Chu & Beasley GA, ABABC)
            "profits": profits,
            "weights": weights,
            "capacity": capacity,
            **extra,
        },
    )


def _chu_beasley_best_known(data_dir) -> dict:
    """Instance name -> (value, proven) for the mknapcb instances.

    Preferred source: data/mknapcb_best_known.txt, the curated compilation
    of 2024-era values (see its header for provenance) — 150 of the 1998 values
    were stale, and it records which classes are proven optimal (tags
    opt/opt-cb) vs open best-knowns (30.250, 30.500). Falls back to
    OR-Library's mkcbres.txt (Chu & Beasley's 1998 GA results, first table
    only) with proven=False everywhere when the curated file is absent.
    """
    curated = os.path.join(data_dir, "mknapcb_best_known.txt")
    if os.path.exists(curated):
        table = {}
        for line in open(curated, encoding="utf-8-sig"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            name, value, tag = line.split()
            table[name] = (float(value), tag in ("opt", "opt-cb"))
        return table
    text = open(os.path.join(data_dir, "mkcbres.txt")).read()
    text = text[: text.index("LP optimal")]
    return {
        name: (float(value), False)
        for name, value in re.findall(r"(\d+\.\d+-\d+)\s+(\d+)", text)
    }


def _load_mknapcb(set_name, data_dir, infeasible_value) -> list[Problem]:
    """One OR-Library mknapcb file: whitespace token stream of K instances,
    each 'n m opt(=0), profits, m x n coefficients, m capacities'. Instances
    within a file cycle through tightness 0.25 / 0.50 / 0.75 in blocks of 10."""
    tokens = np.array(open(os.path.join(data_dir, set_name + ".txt")).read().split(), dtype=np.float64)
    best_known = _chu_beasley_best_known(data_dir)

    problems = []
    pos = 1
    for k in range(int(tokens[0])):
        n, m = int(tokens[pos]), int(tokens[pos + 1])
        pos += 3  # skip the always-zero optimum field
        profits = tokens[pos:pos + n]; pos += n
        weights = tokens[pos:pos + m * n].reshape(m, n); pos += m * n
        capacity = tokens[pos:pos + m]; pos += m
        name = f"{m}.{n}-{k:02d}"
        value, proven = best_known.get(name, (None, False))
        problems.append(
            _make_problem(name, profits, weights, capacity, value, infeasible_value,
                          tightness=(0.25, 0.5, 0.75)[k // 10],
                          # paper tables must say "best known", not "optimum",
                          # for the still-open classes (30.250, 30.500)
                          optimum_proven=proven)
        )
    if pos != len(tokens):
        raise ValueError(f"{set_name}.txt: {len(tokens) - pos} unparsed trailing tokens")
    return problems


def _load_gk(data_dir, infeasible_value) -> list[Problem]:
    """The Glover-Kochenberger instances: one file per instance under
    data/gk/, header 'n m best', then profits, m x n coefficients, capacities."""
    gk_dir = os.path.join(data_dir, "gk")
    problems = []
    for filename in sorted(os.listdir(gk_dir)):
        if not filename.endswith(".dat"):
            continue
        name = filename[:-4]
        tokens = np.array(open(os.path.join(gk_dir, filename)).read().split(), dtype=np.float64)
        n, m, best = int(tokens[0]), int(tokens[1]), float(tokens[2])
        pos = 3
        profits = tokens[pos:pos + n]; pos += n
        weights = tokens[pos:pos + m * n].reshape(m, n); pos += m * n
        capacity = tokens[pos:pos + m]; pos += m
        if pos != len(tokens):
            raise ValueError(f"{filename}: {len(tokens) - pos} unparsed trailing tokens")
        known = best if best > 0 else GK_BEST_KNOWN.get(name)
        problems.append(_make_problem(name, profits, weights, capacity, known, infeasible_value))
    return problems


def load_mkp_problems(
    set_name: str = "mknap1", data_dir: str = _DATA_DIR, infeasible_value: str = "minus_inf"
) -> list[Problem]:
    """Load a benchmark set (see module docstring) as a list of Problems."""
    if set_name not in BENCHMARK_SETS:
        raise ValueError(f"Unknown benchmark set {set_name!r}; choose from {BENCHMARK_SETS}")
    if set_name.startswith("mknapcb"):
        return _load_mknapcb(set_name, data_dir, infeasible_value)
    if set_name == "gk":
        return _load_gk(data_dir, infeasible_value)

    raw_problems = joblib.load(os.path.join(data_dir, set_name + ".joblib"))
    return [
        Problem(
            name=raw["problem_name"],
            dim=raw["num_of_items"],
            evaluate=MKPEvaluator(raw["profits"], raw["weights"], raw["capacity"], infeasible_value),
            known_optimum=raw["known_optimum"],
            optimization_type="max",
            extras={
                "num_of_constraints": raw["num_of_constraints"],
                # instance data for algorithms with problem-specific repair
                # operators (e.g. the Chu & Beasley GA)
                "profits": np.asarray(raw["profits"], dtype=np.float64),
                "weights": np.asarray(raw["weights"], dtype=np.float64),
                "capacity": np.asarray(raw["capacity"], dtype=np.float64),
            },
        )
        for raw in raw_problems
    ]
