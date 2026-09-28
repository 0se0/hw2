import numpy as np
from scipy import stats

from churn.stats import corrected_resampled_ttest, paired_bootstrap_auc_diff


def test_corrected_p_is_larger_than_naive_p():
    diffs = np.array([0.0018, 0.0021, 0.0015, 0.0019, 0.0012])
    _, p_corr = corrected_resampled_ttest(diffs, n_train=132000, n_test=33000)
    _, p_naive = stats.ttest_1samp(diffs, 0.0)
    assert p_corr > p_naive


def test_corrected_t_matches_hand_formula():
    diffs = np.array([0.02, 0.01, 0.03, 0.015, 0.025])
    t, _ = corrected_resampled_ttest(diffs, n_train=80, n_test=20)
    expected = diffs.mean() / np.sqrt((1 / 5 + 20 / 80) * diffs.var(ddof=1))
    assert abs(t - expected) < 1e-12


def test_zero_variance_does_not_crash():
    assert corrected_resampled_ttest(np.zeros(5), 80, 20) == (0.0, 1.0)


def test_bootstrap_ci_excludes_zero_for_a_clearly_better_model():
    rng = np.random.default_rng(0)
    y = (rng.random(3000) < 0.3).astype(int)
    good = rng.random(3000) + 1.0 * y
    weak = rng.random(3000) + 0.2 * y
    diffs = paired_bootstrap_auc_diff(y, good, {"weak": weak}, n_boot=200)["weak"]
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    assert lo > 0
