"""Feature engineering shared by training and inference.

Everything that depends on the training data (currently only the Balance
80th percentile) is computed once by ``compute_feature_state`` and passed in
explicitly, so train, test and any later inference call use the same values.
"""

import pandas as pd

TARGET = "Exited"
ID_COL = "id"
DROP_COLS = ["CustomerId", "Surname"]
GEO_MAPPING = {"France": 0, "Germany": 1, "Spain": 2}
GENDER_MAPPING = {"Female": 0, "Male": 1}
NUMERIC_COLUMNS = [
    "CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
    "HasCrCard", "IsActiveMember", "EstimatedSalary",
]
REQUIRED_COLUMNS = ["Geography", "Gender"] + NUMERIC_COLUMNS


class InputValidationError(ValueError):
    """The input frame cannot be scored safely."""


def validate_input(df: pd.DataFrame) -> None:
    """Fail loudly on input the model was never trained to handle.

    Without this, an unseen Geography such as "Italy" maps to NaN and is then
    silently filled with 0, i.e. scored as "France", and a missing column
    surfaces as an unrelated KeyError deep inside feature engineering.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise InputValidationError(f"missing required columns: {missing}")

    for col, mapping in (("Geography", GEO_MAPPING), ("Gender", GENDER_MAPPING)):
        unknown = sorted({str(v) for v in df.loc[~df[col].isin(mapping), col].unique()})
        if unknown:
            raise InputValidationError(
                f"unknown {col} values {unknown}; expected one of {sorted(mapping)}")

    for col in NUMERIC_COLUMNS:
        numeric = pd.to_numeric(df[col], errors="coerce")
        n_bad = int(numeric.isna().sum())
        if n_bad:
            raise InputValidationError(f"{col} has {n_bad} missing or non-numeric values")
    if (pd.to_numeric(df["Age"]) <= 0).any():
        raise InputValidationError("Age must be positive")


def compute_feature_state(train_df: pd.DataFrame) -> dict:
    return {"balance_q80": float(train_df["Balance"].quantile(0.8))}


def engineer_features(df: pd.DataFrame, balance_q80: float, validate: bool = True) -> pd.DataFrame:
    """Return the model-ready feature frame (no id / target columns)."""
    if validate:
        validate_input(df)
    df = df.copy().drop(columns=DROP_COLS, errors="ignore")

    df["Geography"] = df["Geography"].map(GEO_MAPPING)
    df["Gender"] = df["Gender"].map(GENDER_MAPPING)

    df["Balance_per_Product"] = df["Balance"] / (df["NumOfProducts"] + 1)
    df["Balance_per_Salary"] = df["Balance"] / (df["EstimatedSalary"] + 1)
    df["CreditScore_per_Age"] = df["CreditScore"] / df["Age"]
    df["Salary_per_Age"] = df["EstimatedSalary"] / df["Age"]
    df["Tenure_per_Age"] = df["Tenure"] / df["Age"]

    df["Age_CreditScore_Interaction"] = df["Age"] * df["CreditScore"] / 1000
    df["Balance_CreditScore_Interaction"] = df["Balance"] * df["CreditScore"] / 1000000
    df["Active_Card_Interaction"] = df["HasCrCard"] * df["IsActiveMember"]

    df["Is_High_Value"] = (df["Balance"] > balance_q80).astype(int)
    df["Is_Zero_Balance"] = (df["Balance"] == 0).astype(int)
    df["Is_Young"] = (df["Age"] < 35).astype(int)
    df["Is_Senior"] = (df["Age"] >= 55).astype(int)
    df["Is_High_Credit"] = (df["CreditScore"] > 700).astype(int)
    df["Is_Multi_Product"] = (df["NumOfProducts"] > 2).astype(int)
    df["Is_Germany"] = (df["Geography"] == 1).astype(int)

    df["Churn_Risk_Score"] = (
        df["Is_Germany"] * 3
        + df["Gender"] * 2
        + df["Is_Senior"] * 2
        + df["Is_Multi_Product"] * 3
        + (1 - df["IsActiveMember"]) * 2
        + df["Is_Zero_Balance"] * 1
    )

    df["Customer_Value_Score"] = (
        df["Balance"] / 100000
        + df["CreditScore"] / 1000
        + df["EstimatedSalary"] / 100000
        + df["NumOfProducts"] * 0.5
    )

    features = df.drop(columns=[TARGET, ID_COL], errors="ignore")
    for col in features.columns:
        features[col] = pd.to_numeric(features[col], errors="coerce").fillna(0)
    return features
