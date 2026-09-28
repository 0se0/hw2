"""Model zoo: default hyperparameters and estimator construction."""

from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42

DEFAULT_PARAMS = {
    "rf": dict(n_estimators=200, max_depth=12, min_samples_split=20,
               min_samples_leaf=10, random_state=SEED, n_jobs=-1),
    "et": dict(n_estimators=200, max_depth=12, min_samples_split=20,
               min_samples_leaf=10, random_state=SEED, n_jobs=-1),
    "gb": dict(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=SEED),
    "lr": dict(random_state=SEED, max_iter=1000, C=0.1),
}
MODEL_CLASSES = {
    "rf": RandomForestClassifier,
    "et": ExtraTreesClassifier,
    "gb": GradientBoostingClassifier,
    "lr": LogisticRegression,
}

try:
    from lightgbm import LGBMClassifier

    DEFAULT_PARAMS["lgbm"] = dict(
        n_estimators=400, learning_rate=0.03, max_depth=6, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8, random_state=SEED, n_jobs=-1, verbose=-1,
    )
    MODEL_CLASSES["lgbm"] = LGBMClassifier
except ImportError:  # optional dependency
    pass

try:
    from xgboost import XGBClassifier

    DEFAULT_PARAMS["xgb"] = dict(
        n_estimators=400, learning_rate=0.03, max_depth=6, subsample=0.8,
        colsample_bytree=0.8, eval_metric="auc", random_state=SEED, n_jobs=-1,
    )
    MODEL_CLASSES["xgb"] = XGBClassifier
except ImportError:  # optional dependency
    pass

try:
    from catboost import CatBoostClassifier

    DEFAULT_PARAMS["catboost"] = dict(
        iterations=400, learning_rate=0.03, depth=6, random_state=SEED, verbose=False,
    )
    MODEL_CLASSES["catboost"] = CatBoostClassifier
except ImportError:  # optional dependency
    pass


def available_models() -> list:
    return list(MODEL_CLASSES)


def build_estimator(name: str, params: dict = None):
    """Fresh, unfitted estimator. Logistic regression is wrapped with a scaler
    so scaling is fit per training fold and travels with the model."""
    merged = {**DEFAULT_PARAMS[name], **(params or {})}
    estimator = MODEL_CLASSES[name](**merged)
    if name == "lr":
        return Pipeline([("scaler", StandardScaler()), ("model", estimator)])
    return estimator
