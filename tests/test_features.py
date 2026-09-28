import numpy as np

from churn.features import compute_feature_state, engineer_features


def test_expected_columns_and_no_missing(train_df):
    state = compute_feature_state(train_df)
    X = engineer_features(train_df, state["balance_q80"])
    assert X.shape[1] == 27
    assert not X.isna().any().any()
    for col in ("id", "Exited", "CustomerId", "Surname"):
        assert col not in X.columns


def test_row_features_do_not_depend_on_other_rows(train_df):
    """Regression test for the old bug where Is_High_Value used each dataset's
    own Balance quantile, so the same customer got different features
    depending on which rows they were scored with."""
    q = compute_feature_state(train_df)["balance_q80"]
    full = engineer_features(train_df, q)
    single = engineer_features(train_df.iloc[[0]], q)
    np.testing.assert_allclose(single.iloc[0].to_numpy(), full.iloc[0].to_numpy())


def test_high_value_uses_the_supplied_threshold(train_df):
    X_low = engineer_features(train_df, 0.0)
    X_high = engineer_features(train_df, 1e12)
    assert X_high["Is_High_Value"].sum() == 0
    assert X_low["Is_High_Value"].sum() == (train_df["Balance"] > 0).sum()
