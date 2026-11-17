"""
Set Covering Problem (SCP).

    minimize   costs . x
    subject to every row covered:  for each row i, sum_{j in cover(i)} x_j >= 1
               x in {0, 1}^n        (columns)

Benchmark instances come from Beasley's OR-Library (scp4x/scp5x/scp6x,
scpa-scpe, scpnre/nrf...). `download_scp_instances` fetches the raw text
files once into ``data/``; `load_scp_problems` parses them into Problem
objects.

OR-Library file format (1-based column indices, free whitespace/linebreaks):

    m n
    cost_1 ... cost_n
    then for each row i = 1..m:
        k_i  (how many columns cover row i)
        the k_i column indices

Unlike the MKP evaluator there is no ±inf masking: an uncovered solution
reports its true cost plus violation = number of uncovered rows, and the
framework's Deb's rule handles the rest. Note for SBABC: use
scout_bee_phase_version=1 — version 2's shrink-to-feasibility premise is
knapsack-specific and is exactly wrong here (dropping columns UNcovers rows).
"""

from __future__ import annotations

import os
import urllib.request

import numpy as np

from problems.base import Problem

_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
_ORLIB_URL = "https://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/{name}.txt"

# Proven optimal values for scp4/5/6 (Beasley 1987, EJOR 31; sets 4-6
# originally Balas & Ho) and scpa-e (Beasley 1990 lineage; all <= 400x4000
# solved to optimality — Caprara, Fischetti & Toth 1999, Oper. Res. 47(5)
# report them as optima). All 50 values verified instance-by-instance
# (2026-07-11) against two independent published tables: Adamo, Ghiani,
# Guerriero & Pareo 2023 (Algorithms 16(7):321, Tables 3-4) and Yu, Lam & Li
# 2015 (arXiv:1502.00199, Tables IV-V, which also mark these sets' optima as
# "Known"). The larger NRE-NRH / CLR / CYC sets have best-knowns only and are
# deliberately absent here.
KNOWN_OPTIMA = {
    "scp41": 429, "scp42": 512, "scp43": 516, "scp44": 494, "scp45": 512,
    "scp46": 560, "scp47": 430, "scp48": 492, "scp49": 641, "scp410": 514,
    "scp51": 253, "scp52": 302, "scp53": 226, "scp54": 242, "scp55": 211,
    "scp56": 213, "scp57": 293, "scp58": 288, "scp59": 279, "scp510": 265,
    "scp61": 138, "scp62": 146, "scp63": 145, "scp64": 131, "scp65": 161,
    "scpa1": 253, "scpa2": 252, "scpa3": 232, "scpa4": 234, "scpa5": 236,
    "scpb1": 69, "scpb2": 76, "scpb3": 80, "scpb4": 79, "scpb5": 72,
    "scpc1": 227, "scpc2": 219, "scpc3": 243, "scpc4": 219, "scpc5": 215,
    "scpd1": 60, "scpd2": 66, "scpd3": 72, "scpd4": 62, "scpd5": 61,
    "scpe1": 5, "scpe2": 5, "scpe3": 5, "scpe4": 5, "scpe5": 5,
}

# the default benchmark subset: 3 classes of increasing size + 2 harder classes
DEFAULT_INSTANCES = (
    "scp41", "scp42", "scp43",  # m=200,  n=1000, density 2%
    "scp51", "scp52", "scp53",  # m=200,  n=2000, density 2%
    "scp61", "scp62", "scp63",  # m=200,  n=1000, density 5%
    "scpa1", "scpa2", "scpa3",  # m=300,  n=3000, density 2%
    "scpc1", "scpc2", "scpc3",  # m=400,  n=4000, density 2%
)


class SCPEvaluator:
    """Evaluate SCP solutions; works on a single vector (dim,) or a batch (..., dim).

    Returns (value, violation): value is the total cost of the selected
    columns (always the true cost — no ±inf masking), violation is the number
    of uncovered rows (0 = feasible cover).
    """

    def __init__(self, costs, cover_matrix):
        self.costs = np.asarray(costs, dtype=np.float64)
        self.cover_matrix = np.asarray(cover_matrix, dtype=np.int8)  # (m, n): row i covered by col j

    def __call__(self, x):
        # int64 accumulator: int8 x int8 tensordot accumulates in int8 and
        # wraps at 256 covering columns (reachable on the dense NRE-NRH sets),
        # which would report a feasible cover as violated
        x = np.asarray(x, dtype=np.int64)
        cover_counts = np.tensordot(x, self.cover_matrix, axes=([x.ndim - 1], [1]))
        violation = np.sum(cover_counts == 0, axis=-1).astype(np.float64)
        value = x @ self.costs
        return value, violation


class RandomCoverSampler:
    """Random *feasible* initial solutions: visit the rows in random order
    and, whenever one is still uncovered, add a random column that covers it.

    Bernoulli(0.5) bit vectors select ~n/2 columns — an order of magnitude
    above any decent cover — so every algorithm would spend most of its
    budget just shedding columns. Randomized covers are the standard
    initialization in the SCP metaheuristics literature; attached to the
    problem (extras["init_sampler"]) they apply to all algorithms alike, so
    the comparison stays fair.
    """

    def __init__(self, cover_matrix):
        cover = np.asarray(cover_matrix, dtype=bool)
        self.cover_by_column = cover  # (m, n)
        self.columns_for_row = [np.flatnonzero(cover[i]) for i in range(cover.shape[0])]

    def __call__(self, rng, n: int) -> np.ndarray:
        m, dim = self.cover_by_column.shape
        solutions = np.zeros((n, dim), dtype=np.int8)
        for s in range(n):
            covered = np.zeros(m, dtype=bool)
            for row in rng.permutation(m):
                if not covered[row]:
                    column = rng.choice(self.columns_for_row[row])
                    solutions[s, column] = 1
                    covered |= self.cover_by_column[:, column]
            # drop redundant columns (in random order), as in Beasley &
            # Chu's SCP GA: the cover stays feasible, just leaner
            counts = self.cover_by_column @ solutions[s].astype(np.int64)
            for column in rng.permutation(np.flatnonzero(solutions[s])):
                rows = self.cover_by_column[:, column]
                if (counts[rows] >= 2).all():
                    solutions[s, column] = 0
                    counts[rows] -= 1
        return solutions


def parse_scp_file(path: str) -> tuple[np.ndarray, np.ndarray]:
    """(costs, cover_matrix) from an OR-Library SCP text file."""
    with open(path) as f:
        tokens = f.read().split()
    it = iter(tokens)
    m, n = int(next(it)), int(next(it))
    costs = np.array([float(next(it)) for _ in range(n)])

    cover = np.zeros((m, n), dtype=np.int8)
    for i in range(m):
        k = int(next(it))
        for _ in range(k):
            cover[i, int(next(it)) - 1] = 1  # file is 1-based
    leftovers = sum(1 for _ in it)
    if leftovers:
        raise ValueError(f"{path}: {leftovers} unparsed tokens — not OR-Library SCP format?")
    if (cover.sum(axis=1) == 0).any():
        raise ValueError(f"{path}: some rows are uncoverable")
    return costs, cover


def download_scp_instances(names=DEFAULT_INSTANCES, data_dir: str = _DATA_DIR) -> list[str]:
    """Fetch instance files from the OR-Library into `data_dir` (skips ones
    already present); returns the local paths. A 0-byte file is a failed
    earlier download, not a present instance — it gets re-fetched."""
    os.makedirs(data_dir, exist_ok=True)
    paths = []
    for name in names:
        path = os.path.join(data_dir, f"{name}.txt")
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            url = _ORLIB_URL.format(name=name)
            print(f"Downloading {url}")
            with urllib.request.urlopen(url, timeout=60) as response:
                data = response.read()
            with open(path, "wb") as f:
                f.write(data)
        paths.append(path)
    return paths


def load_scp_problems(
    names=DEFAULT_INSTANCES, data_dir: str = _DATA_DIR, init: str = "random_cover"
) -> list[Problem]:
    """Load OR-Library SCP instances as (minimization) Problem objects.

    Files must already be in `data_dir` — run `download_scp_instances` once
    (needs network) to put them there. `init` selects the population
    initializer every algorithm draws from: "random_cover" (default, see
    RandomCoverSampler) or "bernoulli" (uniform random bit vectors; the
    problem name gets a "_bern" suffix since results are not comparable).
    """
    if init not in ("random_cover", "bernoulli"):
        raise ValueError(f"init must be 'random_cover' or 'bernoulli', got {init!r}")

    problems = []
    for name in names:
        path = os.path.join(data_dir, f"{name}.txt")
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            raise FileNotFoundError(
                f"{path} missing or empty (an empty file is a failed download); "
                f"run problems.scp.download_scp_instances({[name]!r}) first"
            )
        costs, cover = parse_scp_file(path)
        extras = {"num_rows": cover.shape[0], "density": float(cover.mean())}
        if init == "random_cover":
            extras["init_sampler"] = RandomCoverSampler(cover)
        problems.append(
            Problem(
                name=name if init == "random_cover" else f"{name}_bern",
                dim=len(costs),
                evaluate=SCPEvaluator(costs, cover),
                known_optimum=KNOWN_OPTIMA.get(name),
                optimization_type="min",
                extras=extras,
            )
        )
    return problems
