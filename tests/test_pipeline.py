import json

import joblib
import numpy as np
import pandas as pd

from churn.predict import main as predict_main
from churn.predict import predict_frame
from churn.train import run_training

FAST = {"rf": {"n_estimators": 20}, "lr": {}}


def _train(train_df, test_df):
    return run_training(train_df, test_df, ["rf", "lr"], FAST, n_folds=3,
                        blend_maxiter=5, blend_popsize=4, n_boot=20, log=lambda *_: None)


def test_inference_reproduces_training_time_test_predictions(train_df, test_df):
    """Train/serve consistency: scoring test.csv through the saved artifact must
    give the same probabilities the training run produced."""
    out = _train(train_df, test_df)
    served = predict_frame(out["artifact"], test_df)
    np.testing.assert_allclose(served, out["test_pred"], rtol=1e-9, atol=1e-12)


def test_predictions_are_probabilities_and_metrics_are_serialisable(train_df, test_df):
    out = _train(train_df, test_df)
    assert out["test_pred"].min() >= 0 and out["test_pred"].max() <= 1
    assert abs(sum(out["metrics"]["blend_weights"].values()) - 1) < 1e-9
    json.dumps(out["metrics"])


def test_predict_cli_roundtrip(tmp_path, train_df, test_df):
    out = _train(train_df, test_df)
    model_path, in_path, out_path = tmp_path / "m.joblib", tmp_path / "in.csv", tmp_path / "out.csv"
    joblib.dump(out["artifact"], model_path)
    test_df.to_csv(in_path, index=False)
    predict_main(["--model", str(model_path), "--input", str(in_path), "--output", str(out_path)])
    preds = pd.read_csv(out_path)
    assert list(preds.columns) == ["id", "Exited"] and len(preds) == len(test_df)
    np.testing.assert_allclose(preds["Exited"].to_numpy(), out["test_pred"], rtol=1e-9)
