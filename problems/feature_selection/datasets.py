"""
Benchmark datasets for the feature-selection domain.

Small/medium UCI classification sets spanning ~13-500 features, the range
used by the SBPSO feature-selection literature. Two sources:

  sklearn   bundled with scikit-learn, no network needed
  openml    fetched once via ``sklearn.datasets.fetch_openml`` and cached
            as a joblib file under ``data/``, so later runs (and the other
            PCs of the parallel runner) work offline once the cache is
            committed/copied

Categorical features are ordinal-encoded (integer codes) before z-scoring;
the standard k-NN FS benchmarks treat them the same way. Labels are mapped
to 0..C-1 by the evaluator.

`load_fs_problems()` is the experiment entry point, mirroring
`problems.mkp.load_mkp_problems`.
"""

from __future__ import annotations

import os

import joblib
import numpy as np

from problems.base import Problem
from problems.feature_selection.problem import make_fs_problem

_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# name -> (source, source_id, openml_version)
DATASETS = {
    # bundled with scikit-learn
    "wine": ("sklearn", "load_wine", None),
    "wdbc": ("sklearn", "load_breast_cancer", None),
    # OpenML (UCI originals), fetched once then cached
    "ionosphere": ("openml", "ionosphere", 1),
    "sonar": ("openml", "sonar", 1),
    "glass": ("openml", "glass", 1),
    "vehicle": ("openml", "vehicle", 1),
    "segment": ("openml", "segment", 1),
    "german_credit": ("openml", "credit-g", 1),
    # NOTE: OpenML's "musk" is the 6598-sample Musk *v2* data and carries
    # molecule/conformation ID columns that ordinal-encoding would leak into
    # the features — and its distance matrix is far too large for wrapper FS
    # at the campaign budgets. Deliberately not offered here.
    "hill_valley": ("openml", "hill-valley", 1),
    "madelon": ("openml", "madelon", 1),
}


def _fetch(name: str, data_dir: str) -> tuple[np.ndarray, np.ndarray]:
    """(X, y) for a dataset, from the joblib cache when available."""
    source, source_id, version = DATASETS[name]
    cache_path = os.path.join(data_dir, f"{name}.joblib")
    if os.path.exists(cache_path):
        cached = joblib.load(cache_path)
        return cached["X"], cached["y"]

    if source == "sklearn":
        from sklearn import datasets as skd

        bunch = getattr(skd, source_id)()
        X, y = np.asarray(bunch.data, dtype=np.float64), np.asarray(bunch.target)
    else:
        from sklearn.datasets import fetch_openml

        bunch = fetch_openml(source_id, version=version, as_frame=True, parser="auto")
        frame = bunch.data
        columns = []
        for col in frame.columns:  # ordinal-encode anything non-numeric
            series = frame[col]
            if series.dtype.kind in "biufc":
                columns.append(series.to_numpy(dtype=np.float64))
            else:
                _, codes = np.unique(series.astype(str).to_numpy(), return_inverse=True)
                columns.append(codes.astype(np.float64))
        X = np.column_stack(columns)
        y = bunch.target.astype(str).to_numpy()

    os.makedirs(data_dir, exist_ok=True)
    joblib.dump({"X": X, "y": y}, cache_path)
    return X, y


def load_fs_problems(
    names: list[str] | None = None,
    data_dir: str = _DATA_DIR,
    n_folds: int = 10,
    knn_k: int = 5,
    alpha: float = 0.9,
    fold_seed: int = 0,
) -> list[Problem]:
    """Build the FS benchmark as Problem objects (default: every dataset)."""
    names = list(DATASETS) if names is None else list(names)
    unknown = [n for n in names if n not in DATASETS]
    if unknown:
        raise ValueError(f"Unknown dataset(s) {unknown}; choose from {list(DATASETS)}")

    # Non-default evaluator settings are part of the problem's identity in
    # the results store (problems are keyed by name there), so encode them
    suffix = ""
    if (n_folds, knn_k, alpha, fold_seed) != (10, 5, 0.9, 0):
        suffix = f"_f{n_folds}k{knn_k}a{alpha:g}s{fold_seed}"

    problems = []
    for name in names:
        X, y = _fetch(name, data_dir)
        problems.append(make_fs_problem(f"fs_{name}{suffix}", X, y, n_folds, knn_k, alpha, fold_seed))
    return problems
