import numpy as np
import pandas as pd
import pytest


def make_frame(n, seed, with_target=True):
    rng = np.random.default_rng(seed)
    geo = rng.choice(["France", "Germany", "Spain"], size=n, p=[0.5, 0.25, 0.25])
    gender = rng.choice(["Male", "Female"], size=n)
    age = rng.integers(18, 80, size=n)
    products = rng.integers(1, 4, size=n)
    active = rng.integers(0, 2, size=n)
    balance = np.round(rng.uniform(0, 200000, size=n), 2)
    balance[rng.random(n) < 0.3] = 0.0
    df = pd.DataFrame({
        "id": np.arange(n),
        "CustomerId": 15000000 + np.arange(n),
        "Surname": [f"C{i}" for i in range(n)],
        "CreditScore": rng.integers(350, 850, size=n),
        "Geography": geo,
        "Gender": gender,
        "Age": age,
        "Tenure": rng.integers(0, 10, size=n),
        "Balance": balance,
        "NumOfProducts": products,
        "HasCrCard": rng.integers(0, 2, size=n),
        "IsActiveMember": active,
        "EstimatedSalary": np.round(rng.uniform(1000, 200000, size=n), 2),
    })
    if with_target:
        logit = -2 + (geo == "Germany") * 1.2 + (age > 50) * 0.8 - active + (products >= 3) * 0.9
        df["Exited"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return df


@pytest.fixture(scope="session")
def train_df():
    return make_frame(600, seed=0)


@pytest.fixture(scope="session")
def test_df():
    return make_frame(150, seed=1, with_target=False)
