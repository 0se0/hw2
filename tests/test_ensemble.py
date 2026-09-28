import numpy as np

from churn.ensemble import fit_blend_weights, nested_blend_scores
from churn.metrics import fast_auc


def _toy(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.3).astype(int)
    strong = rng.random(n) + 1.2 * y
    noise = rng.random(n)
    return np.column_stack([strong, noise]), y


def test_weights_are_a_valid_simplex_and_prefer_the_informative_model():
    M, y = _toy()
    w = fit_blend_weights(M, y, maxiter=30)
    assert np.all(w >= 0) and abs(w.sum() - 1) < 1e-9
    assert w[0] > w[1]


def test_blend_is_not_worse_than_equal_weights_in_sample():
    M, y = _toy()
    w = fit_blend_weights(M, y, maxiter=30)
    assert fast_auc(y, M @ w) >= fast_auc(y, M.mean(axis=1)) - 1e-9


def test_nested_scores_shapes_and_fair_score_not_above_insample():
    M, y = _toy()
    idx = np.arange(len(y))
    folds = [(np.setdiff1d(idx, v), v) for v in np.array_split(idx, 5)]
    nested_oof, nested_aucs, equal_aucs = nested_blend_scores(M, y, folds, maxiter=20)
    assert nested_oof.shape == y.shape and len(nested_aucs) == len(equal_aucs) == 5
    w = fit_blend_weights(M, y, maxiter=20)
    insample = np.mean([fast_auc(y[v], M[v] @ w) for _, v in folds])
    assert nested_aucs.mean() <= insample + 0.01
