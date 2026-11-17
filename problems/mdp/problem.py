"""
Maximum Diversity Problem (MDP, MaxSum model).

    maximize   sum_{i<j} d_ij x_i x_j          (total pairwise distance)
    subject to sum_i x_i = m                   (select exactly m of n elements)
               x in {0, 1}^n

A selection-type problem with a cardinality equality constraint and a
quadratic objective — structurally different from the knapsack family
(no resource weights, no covering coupling) yet a natural set problem:
a solution IS a subset. Benchmarks come from MDPLIB 2.0 (Marti, Duarte,
Martinez-Gavara & Sanchez-Oro 2021, https://www.uv.es/rmarti/paper/mdp.html;
archive at https://doi.org/10.5281/zenodo.13749316):

  SOM-b  integer distances in [0, 9], n = 100..500, m = n/10..2n/5
  GKD-c  Euclidean distances of random points in [0, 10]^10, n = 500, m = 50
  MDG-a  real distances in [0, 10],   n = 500, m = 50
  MDG-b  real distances in [0, 1000], n = 500, m = 50

KNOWN_BEST holds the library's best-known values (mdplib_2010_bestvalues.xls:
G_SS / ITS with 2 h CPU per instance) — best known, NOT proven optima, so
extras["optimum_proven"] is False and tables must say "best known".

Evaluation follows the SCP convention (no +-inf masking): value is the true
pairwise-distance sum of the selected elements, violation = |#selected - m|,
and Deb's rule does the rest. Initialization draws uniform random m-subsets
(extras["init_sampler"]), applied to every algorithm alike. Note for SBABC:
scout_bee_phase_version=2's shrink premise ("dropping elements can only
reduce violation") is false under an equality constraint — pin version 1.

The ~1 MB-per-instance distance files ship as compressed .npz (condensed
upper triangle, float64) built once by `prepare_mdp_data` from the MDPLIB
archive; the original MDPLIB text format is "n m" then one "i j d_ij" line
per pair (0-based, i < j).
"""

from __future__ import annotations

import os
import zipfile

import numpy as np

from problems.base import Problem

_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# MDPLIB best-known values (mdplib_2010_bestvalues.xls), n <= 500 instances
KNOWN_BEST = {
    "GKD-c_1_n500_m50": 19485.1875,
    "GKD-c_2_n500_m50": 19701.53711,
    "GKD-c_3_n500_m50": 19547.20703,
    "GKD-c_4_n500_m50": 19596.46875,
    "GKD-c_5_n500_m50": 19602.625,
    "GKD-c_6_n500_m50": 19421.55078,
    "GKD-c_7_n500_m50": 19534.30664,
    "GKD-c_8_n500_m50": 19487.32031,
    "GKD-c_9_n500_m50": 19221.63477,
    "GKD-c_10_n500_m50": 19703.35156,
    "GKD-c_11_n500_m50": 19587.12891,
    "GKD-c_12_n500_m50": 19360.23633,
    "GKD-c_13_n500_m50": 19366.69922,
    "GKD-c_14_n500_m50": 19458.56641,
    "GKD-c_15_n500_m50": 19422.15039,
    "GKD-c_16_n500_m50": 19680.20898,
    "GKD-c_17_n500_m50": 19331.38867,
    "GKD-c_18_n500_m50": 19461.39453,
    "GKD-c_19_n500_m50": 19477.32813,
    "GKD-c_20_n500_m50": 19604.84375,
    "SOM-b_1_n100_m10": 333,
    "SOM-b_2_n100_m20": 1195,
    "SOM-b_3_n100_m30": 2457,
    "SOM-b_4_n100_m40": 4142,
    "SOM-b_5_n200_m20": 1247,
    "SOM-b_6_n200_m40": 4450,
    "SOM-b_7_n200_m60": 9437,
    "SOM-b_8_n200_m80": 16225,
    "SOM-b_9_n300_m30": 2694,
    "SOM-b_10_n300_m60": 9689,
    "SOM-b_11_n300_m90": 20743,
    "SOM-b_12_n300_m120": 35881,
    "SOM-b_13_n400_m40": 4658,
    "SOM-b_14_n400_m80": 16956,
    "SOM-b_15_n400_m120": 36317,
    "SOM-b_16_n400_m160": 62487,
    "SOM-b_17_n500_m50": 7141,
    "SOM-b_18_n500_m100": 26258,
    "SOM-b_19_n500_m150": 56572,
    "SOM-b_20_n500_m200": 97344,
    "MDG-a_1_n500_m50": 7833.83252,
    "MDG-a_2_n500_m50": 7771.66162,
    "MDG-a_3_n500_m50": 7759.35986,
    "MDG-a_4_n500_m50": 7770.2417,
    "MDG-a_5_n500_m50": 7755.23096,
    "MDG-a_6_n500_m50": 7773.70996,
    "MDG-a_7_n500_m50": 7771.73096,
    "MDG-a_8_n500_m50": 7750.88135,
    "MDG-a_9_n500_m50": 7770.0708,
    "MDG-a_10_n500_m50": 7780.35059,
    "MDG-a_11_n500_m50": 7770.95068,
    "MDG-a_12_n500_m50": 7757.65039,
    "MDG-a_13_n500_m50": 7798.43164,
    "MDG-a_14_n500_m50": 7795.63135,
    "MDG-a_15_n500_m50": 7736.84033,
    "MDG-a_16_n500_m50": 7792.77197,
    "MDG-a_17_n500_m50": 7787.20068,
    "MDG-a_18_n500_m50": 7756.26172,
    "MDG-a_19_n500_m50": 7755.41064,
    "MDG-a_20_n500_m50": 7733.86133,
    "MDG-b_1_n500_m50": 778030.625,
    "MDG-b_2_n500_m50": 779963.6875,
    "MDG-b_3_n500_m50": 776768.4375,
    "MDG-b_4_n500_m50": 775394.625,
    "MDG-b_5_n500_m50": 775611.0625,
    "MDG-b_6_n500_m50": 775153.6875,
    "MDG-b_7_n500_m50": 777232.875,
    "MDG-b_8_n500_m50": 779168.75,
    "MDG-b_9_n500_m50": 774802.1875,
    "MDG-b_10_n500_m50": 774961.3125,
    "MDG-b_11_n500_m50": 777468.875,
    "MDG-b_12_n500_m50": 775492.9375,
    "MDG-b_13_n500_m50": 780191.875,
    "MDG-b_14_n500_m50": 782232.75,
    "MDG-b_15_n500_m50": 780300.375,
    "MDG-b_16_n500_m50": 775436.25,
    "MDG-b_17_n500_m50": 776619.125,
    "MDG-b_18_n500_m50": 775850.75,
    "MDG-b_19_n500_m50": 778802.9375,
    "MDG-b_20_n500_m50": 778644.8125,
}

# test set (15): mirrors the SCP campaign's size; disjoint from TUNING_INSTANCES
DEFAULT_INSTANCES = (
    "SOM-b_2_n100_m20", "SOM-b_6_n200_m40", "SOM-b_9_n300_m30",
    "SOM-b_13_n400_m40", "SOM-b_17_n500_m50",
    "GKD-c_1_n500_m50", "GKD-c_2_n500_m50", "GKD-c_3_n500_m50", "GKD-c_4_n500_m50",
    "MDG-a_1_n500_m50", "MDG-a_2_n500_m50", "MDG-a_3_n500_m50",
    "MDG-b_1_n500_m50", "MDG-b_2_n500_m50", "MDG-b_3_n500_m50",
)
TUNING_INSTANCES = (
    "SOM-b_3_n100_m30", "SOM-b_7_n200_m60", "SOM-b_11_n300_m90",
    "GKD-c_11_n500_m50", "MDG-a_11_n500_m50", "MDG-b_11_n500_m50",
)


class MDPEvaluator:
    """(value, violation) for one vector (dim,) or a batch (..., dim):
    value = sum of pairwise distances among selected elements,
    violation = |number selected - m| (0 = feasible)."""

    def __init__(self, distances, m: int):
        self.distances = np.asarray(distances, dtype=np.float64)  # (n, n) symmetric, zero diagonal
        self.m = int(m)

    def __call__(self, x):
        x = np.asarray(x, dtype=np.float64)
        value = 0.5 * np.sum(x * (x @ self.distances), axis=-1)
        violation = np.abs(np.sum(x, axis=-1) - self.m)
        return value, violation


class RandomSubsetSampler:
    """Uniform random m-subsets — feasible starts for every algorithm alike."""

    def __init__(self, n: int, m: int):
        self.n, self.m = int(n), int(m)

    def __call__(self, rng, count: int) -> np.ndarray:
        solutions = np.zeros((count, self.n), dtype=np.int8)
        for s in range(count):
            solutions[s, rng.choice(self.n, size=self.m, replace=False)] = 1
        return solutions


def parse_mdp_text(text: str) -> tuple[np.ndarray, int]:
    """(distance matrix, m) from MDPLIB's 'n m' + 'i j d' text format."""
    tokens = text.split()
    n, m = int(tokens[0]), int(tokens[1])
    triples = np.array(tokens[2:], dtype=np.float64).reshape(-1, 3)
    if len(triples) != n * (n - 1) // 2:
        raise ValueError(f"expected {n * (n - 1) // 2} pairs for n={n}, got {len(triples)}")
    distances = np.zeros((n, n))
    i, j = triples[:, 0].astype(int), triples[:, 1].astype(int)
    distances[i, j] = triples[:, 2]
    distances[j, i] = triples[:, 2]
    return distances, m


def prepare_mdp_data(zip_path: str, names=DEFAULT_INSTANCES + TUNING_INSTANCES,
                     data_dir: str = _DATA_DIR) -> list[str]:
    """Extract `names` from the MDPLIB archive (mdplib_2010.zip) into
    compressed .npz files. Needed once, only to (re)build the shipped data."""
    os.makedirs(data_dir, exist_ok=True)
    archive = zipfile.ZipFile(zip_path)
    members = archive.namelist()
    paths = []
    for name in names:
        (member,) = [entry for entry in members if entry.endswith(f"/{name}.txt")]
        distances, m = parse_mdp_text(archive.read(member).decode())
        n = distances.shape[0]
        path = os.path.join(data_dir, f"{name}.npz")
        np.savez_compressed(path, n=n, m=m, condensed=distances[np.triu_indices(n, 1)])
        paths.append(path)
    return paths


def load_mdp_problems(names=DEFAULT_INSTANCES, data_dir: str = _DATA_DIR) -> list[Problem]:
    """MDPLIB instances as (maximization) Problem objects."""
    problems = []
    for name in names:
        path = os.path.join(data_dir, f"{name}.npz")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} missing; build it with problems.mdp.prepare_mdp_data(<mdplib_2010.zip>)"
            )
        with np.load(path) as data:
            n, m, condensed = int(data["n"]), int(data["m"]), data["condensed"]
        distances = np.zeros((n, n))
        distances[np.triu_indices(n, 1)] = condensed
        distances += distances.T
        problems.append(
            Problem(
                name=name,
                dim=n,
                evaluate=MDPEvaluator(distances, m),
                known_optimum=KNOWN_BEST.get(name),
                optimization_type="max",
                extras={
                    "m": m,
                    "optimum_proven": False,  # MDPLIB best-knowns, not proven optima
                    "init_sampler": RandomSubsetSampler(n, m),
                },
            )
        )
    return problems
