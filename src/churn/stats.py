"""Significance testing for comparing the ensemble against single models."""

import numpy as np
from scipy import stats

from .metrics import fast_auc


def corrected_resampled_ttest(diffs, n_train, n_test):
    """Nadeau & Bengio (2003) corrected resampled t-test.

    CV folds share training data, so their scores are not independent and the
    naive paired t-test underestimates the variance. The correction multiplies
    the variance term by (1/k + n_test/n_train).
    """
    diffs = np.asarray(diffs, dtype=float)
    k = len(diffs)
    var = diffs.var(ddof=1)
    if var == 0:
        return 0.0, 1.0
    t = diffs.mean() / np.sqrt((1 / k + n_test / n_train) * var)
    return float(t), float(2 * stats.t.sf(abs(t), df=k - 1))


def paired_bootstrap_auc_diff(y_true, ens_scores, model_scores, n_boot=1000, seed=42):
    """Bootstrap distribution of (ensemble AUC - model AUC) over evaluation rows.

    Rows are resampled once per iteration and shared across models (paired).
    Captures evaluation-set noise, not training-set variability.
    """
    y_true = np.asarray(y_true)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    diffs = {name: np.empty(n_boot) for name in model_scores}
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y_true[idx]
        ens_auc = fast_auc(yb, ens_scores[idx])
        for name, scores in model_scores.items():
            diffs[name][b] = ens_auc - fast_auc(yb, scores[idx])
    return diffs


def compare_to_ensemble(fold_aucs, ens_fold_aucs, y_true, ens_oof, model_oofs,
                        fold_indices, n_boot=1000, seed=42, alpha=0.05):
    """One row per model: naive p, corrected p, and bootstrap 95% CI."""
    n_train, n_test = len(fold_indices[0][0]), len(fold_indices[0][1])
    boot = paired_bootstrap_auc_diff(y_true, ens_oof, model_oofs, n_boot, seed)
    rows = {}
    for name, scores in fold_aucs.items():
        diffs = ens_fold_aucs - scores
        _, p_naive = stats.ttest_rel(ens_fold_aucs, scores)
        _, p_corr = corrected_resampled_ttest(diffs, n_train, n_test)
        lo, hi = np.percentile(boot[name], [2.5, 97.5])
        rows[name] = dict(
            mean_diff=float(diffs.mean()), p_naive=float(p_naive), p_corrected=p_corr,
            boot_ci=(float(lo), float(hi)),
            significant_corrected=bool(p_corr < alpha), ci_excludes_zero=bool(lo > 0),
        )
    return rows
