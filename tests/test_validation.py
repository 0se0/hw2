import pytest

from churn.features import InputValidationError, compute_feature_state, engineer_features, validate_input
from churn.predict import predict_frame
from churn.train import run_training


def test_valid_frames_pass(train_df, test_df):
    validate_input(train_df)
    validate_input(test_df)


def test_missing_column_is_reported_by_name(train_df):
    with pytest.raises(InputValidationError, match="Balance"):
        validate_input(train_df.drop(columns=["Balance"]))


def test_unseen_geography_is_rejected_not_scored_as_france(train_df):
    bad = train_df.copy()
    bad.loc[0, "Geography"] = "Italy"
    with pytest.raises(InputValidationError, match="Italy"):
        engineer_features(bad, compute_feature_state(train_df)["balance_q80"])


def test_unseen_gender_is_rejected(train_df):
    bad = train_df.copy()
    bad.loc[0, "Gender"] = "Other"
    with pytest.raises(InputValidationError, match="Gender"):
        validate_input(bad)


def test_missing_numeric_value_is_rejected(train_df):
    bad = train_df.copy()
    bad.loc[3, "CreditScore"] = None
    with pytest.raises(InputValidationError, match="CreditScore"):
        validate_input(bad)


def test_non_positive_age_is_rejected(train_df):
    bad = train_df.copy()
    bad.loc[0, "Age"] = 0
    with pytest.raises(InputValidationError, match="Age"):
        validate_input(bad)


def test_inference_rejects_bad_input_with_a_clear_error(train_df, test_df):
    out = run_training(train_df, test_df, ["rf", "lr"], {"rf": {"n_estimators": 10}}, n_folds=3,
                       blend_maxiter=3, blend_popsize=4, n_boot=10, log=lambda *_: None)
    bad = test_df.copy()
    bad.loc[0, "Geography"] = "Italy"
    with pytest.raises(InputValidationError, match="Italy"):
        predict_frame(out["artifact"], bad)
