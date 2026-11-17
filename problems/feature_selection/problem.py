"""
Wrapper-based Feature Selection (FS).

    maximize   fit(S) = alpha * accuracy(S) + (1 - alpha) * (1 - |S| / n)
    subject to |S| >= 1
               S subset of the n features

A solution is a binary mask over features. `accuracy(S)` is k-NN
classification accuracy under stratified cross-validation with *fixed,
seeded folds*: every algorithm (and every evaluation) sees the identical
fold assignment, which preserves the common-random-numbers pairing the
Wilcoxon signed-rank comparisons rely on.

The k-NN runs directly on distance matrices (squared Euclidean over the
selected features, z-scored with training-fold statistics only) instead of
scikit-learn estimators: it is deterministic across machines — neighbor
order breaks distance ties by sample index (stable argsort), a tied vote
goes to the label of the nearest tied neighbor — and it lets small datasets
precompute per-feature squared differences once, making a subset evaluation
a masked sum. Datasets past a fixed memory threshold fall back to computing
the masked distance matrix per evaluation instead.

Wrapper evaluations are ~10^3 x more expensive than MKP evaluations, so the
evaluator memoizes subset -> (value, violation). Cache hits still pass
through the optimizer's counted-evaluate wrapper, so function-evaluation
budgets (`Config.max_fn_calls`) stay comparable across algorithms; the cache
only saves wall-clock time and is identical for everyone.
"""

from __future__ import annotations

import numpy as np

from problems.base import Problem

# Above this many bytes for the (n_val x n_train x n_features) fold tensors,
# distances are computed lazily per evaluation. A fixed constant — never a
# machine-dependent probe — so every machine picks the same code path and
# results stay bit-identical across the multi-PC runner.
PRECOMPUTE_LIMIT_BYTES = 200_000_000


def stratified_folds(y: np.ndarray, n_folds: int, seed: int) -> np.ndarray:
    """fold_of[i] = fold index of sample i; per-class round-robin after a
    seeded shuffle, so fold class proportions match the dataset's."""
    rng = np.random.default_rng(seed)
    fold_of = np.empty(len(y), dtype=np.int64)
    offset = 0  # stagger classes so fold sizes stay balanced overall
    for label in np.unique(y):
        members = np.flatnonzero(y == label)
        members = members[rng.permutation(len(members))]
        fold_of[members] = (np.arange(len(members)) + offset) % n_folds
        offset += len(members)
    return fold_of


class FeatureSelectionEvaluator:
    """Evaluate FS masks; works on a single vector (dim,) or a batch (..., dim).

    Returns (value, violation): value is the weighted fitness above,
    violation is 1.0 for the empty mask (the one infeasible solution).
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_folds: int = 10,
        knn_k: int = 5,
        alpha: float = 0.9,
        fold_seed: int = 0,
    ):
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        X = np.asarray(X, dtype=np.float64)
        _, y = np.unique(np.asarray(y), return_inverse=True)  # labels -> 0..C-1
        self.n_features = X.shape[1]
        self.n_classes = int(y.max()) + 1
        self.n_folds = n_folds
        self.knn_k = knn_k
        self.alpha = alpha
        self.fold_of = stratified_folds(y, n_folds, fold_seed)
        self.precomputed = X.shape[0] ** 2 * X.shape[1] * 8 <= PRECOMPUTE_LIMIT_BYTES
        self._X, self._y = X, y
        self._folds = None  # built on first use; workers rebuild instead of unpickling tensors
        self._cache: dict = {}

    def _build_folds(self) -> list:
        X, y = self._X, self._y
        folds = []
        for f in range(self.n_folds):
            val = np.flatnonzero(self.fold_of == f)
            train = np.flatnonzero(self.fold_of != f)
            mu = X[train].mean(axis=0)
            sd = X[train].std(axis=0)
            sd[sd == 0] = 1.0  # constant feature: contributes 0 distance
            Xn = (X - mu) / sd
            if self.precomputed:
                sq_diff = (Xn[val][:, None, :] - Xn[train][None, :, :]) ** 2
                folds.append((sq_diff, None, None, y[train], y[val]))
            else:
                folds.append((None, Xn[val], Xn[train], y[train], y[val]))
        return folds

    def _accuracy(self, mask: np.ndarray) -> float:
        if self._folds is None:
            self._folds = self._build_folds()
        correct = total = 0
        for sq_diff, Xv, Xt, y_train, y_val in self._folds:
            if sq_diff is not None:
                dist = sq_diff[:, :, mask].sum(axis=-1)
            else:
                A, B = Xv[:, mask], Xt[:, mask]
                dist = (A**2).sum(1)[:, None] + (B**2).sum(1)[None, :] - 2.0 * (A @ B.T)
            # stable sort => deterministic neighbor order under distance ties
            nearest = np.argsort(dist, axis=1, kind="stable")[:, : self.knn_k]
            votes = y_train[nearest]  # (n_val, k), column 0 = closest neighbor
            counts = (votes[:, :, None] == np.arange(self.n_classes)).sum(axis=1)
            is_winner = counts[np.arange(len(votes))[:, None], votes] == counts.max(axis=1)[:, None]
            pred = votes[np.arange(len(votes)), is_winner.argmax(axis=1)]
            correct += int((pred == y_val).sum())
            total += len(y_val)
        return correct / total

    def _evaluate_one(self, x: np.ndarray) -> tuple:
        key = x.astype(np.int8).tobytes()
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        mask = x.astype(bool)
        n_sel = int(mask.sum())
        if n_sel == 0:
            result = (0.0, 1.0)
        else:
            fit = self.alpha * self._accuracy(mask) + (1 - self.alpha) * (1 - n_sel / self.n_features)
            result = (float(fit), 0.0)
        self._cache[key] = result
        return result

    def __call__(self, x):
        x = np.asarray(x)
        if x.ndim == 1:
            return self._evaluate_one(x)
        flat = x.reshape(-1, x.shape[-1])
        pairs = [self._evaluate_one(row) for row in flat]
        values = np.array([p[0] for p in pairs], dtype=np.float64).reshape(x.shape[:-1])
        violations = np.array([p[1] for p in pairs], dtype=np.float64).reshape(x.shape[:-1])
        return values, violations

    def __getstate__(self):
        # ship only the dataset: fold tensors are O(n_samples² · n_features)
        # and rebuild deterministically on the other side; caches are per-process
        state = self.__dict__.copy()
        state["_folds"] = None
        state["_cache"] = {}
        return state


def make_fs_problem(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 10,
    knn_k: int = 5,
    alpha: float = 0.9,
    fold_seed: int = 0,
) -> Problem:
    evaluator = FeatureSelectionEvaluator(X, y, n_folds, knn_k, alpha, fold_seed)
    return Problem(
        name=name,
        dim=evaluator.n_features,
        evaluate=evaluator,
        known_optimum=None,  # no known optimum: report accuracy/|S| directly
        optimization_type="max",
        extras={
            "n_samples": int(np.asarray(y).shape[0]),
            "n_classes": evaluator.n_classes,
            "n_folds": n_folds,
            "knn_k": knn_k,
            "alpha": alpha,
            "fold_seed": fold_seed,
        },
    )
