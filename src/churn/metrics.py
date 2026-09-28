"""Metrics."""

import numpy as np
from scipy.stats import rankdata


def fast_auc(y_true, scores) -> float:
    """ROC AUC via the rank-sum (Mann-Whitney) formula.

    Equivalent to ``sklearn.metrics.roc_auc_score`` (ties handled with average
    ranks) but several times faster, which matters because the blend-weight
    search and the bootstrap evaluate it tens of thousands of times.
    """
    y_true = np.asarray(y_true)
    ranks = rankdata(scores)
    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUC is undefined when only one class is present")
    return float((ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
