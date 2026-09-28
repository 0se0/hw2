"""Score new customers with a trained artifact.

    python -m churn.predict --model artifacts/model.joblib --input new.csv --output preds.csv
"""

import argparse

import joblib
import numpy as np
import pandas as pd

from .features import ID_COL, TARGET, engineer_features


def predict_frame(artifact: dict, df: pd.DataFrame) -> np.ndarray:
    """Churn probability per row: mean over fold models per base model, then the blend."""
    X = engineer_features(df, artifact["feature_state"]["balance_q80"])[artifact["feature_cols"]]
    per_model = [
        np.mean([m.predict_proba(X)[:, 1] for m in artifact["fold_models"][name]], axis=0)
        for name in artifact["model_names"]
    ]
    return np.column_stack(per_model) @ artifact["weights"]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--input", required=True, help="CSV with the same columns as test.csv")
    p.add_argument("--output", required=True)
    args = p.parse_args(argv)

    df = pd.read_csv(args.input)
    preds = predict_frame(joblib.load(args.model), df)
    out = pd.DataFrame({TARGET: preds})
    if ID_COL in df.columns:
        out.insert(0, ID_COL, df[ID_COL].to_numpy())
    out.to_csv(args.output, index=False)
    print(f"wrote {len(out):,} predictions to {args.output}")


if __name__ == "__main__":
    main()
