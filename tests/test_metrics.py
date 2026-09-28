import numpy as np
from sklearn.metrics import roc_auc_score

from churn.metrics import fast_auc


def test_matches_sklearn_on_random_scores():
    rng = np.random.default_rng(0)
    y = (rng.random(2000) < 0.2).astype(int)
    s = rng.random(2000) + 0.3 * y
    assert abs(fast_auc(y, s) - roc_auc_score(y, s)) < 1e-12


def test_matches_sklearn_with_heavy_ties():
    rng = np.random.default_rng(1)
    y = (rng.random(1000) < 0.3).astype(int)
    s = np.round(rng.random(1000) + 0.2 * y, 1)
    assert abs(fast_auc(y, s) - roc_auc_score(y, s)) < 1e-12


def test_perfect_and_inverted_ranking():
    y = np.array([0, 0, 1, 1])
    assert fast_auc(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert fast_auc(y, np.array([0.9, 0.8, 0.2, 0.1])) == 0.0
