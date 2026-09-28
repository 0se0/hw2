"""Out-of-fold training and blend-weight fitting."""

import numpy as np
from scipy.optimize import differential_evolution
from sklearn.model_selection import StratifiedKFold

from .metrics import fast_auc
from .models import build_estimator


def generate_oof(X, y, X_test, model_names, tuned_params=None, n_folds=5, seed=42,
                 keep_models=False, log=print):
    """Train every model on each CV fold.

    Returns out-of-fold predictions for the training rows, fold-averaged
    predictions for the test rows (bagging), per-fold AUCs, the fold split, and
    optionally the fitted fold models (needed for inference).
    """
    tuned_params = tuned_params or {}
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_indices = list(cv.split(X, y))

    oof = {n: np.zeros(len(X)) for n in model_names}
    test = {n: np.zeros(len(X_test)) for n in model_names}
    fold_aucs = {n: [] for n in model_names}
    fold_models = {n: [] for n in model_names}

    for name in model_names:
        log(f"  {name}: {n_folds}-fold training")
        for tr_idx, val_idx in fold_indices:
            est = build_estimator(name, tuned_params.get(name))
            est.fit(X.iloc[tr_idx], y.iloc[tr_idx])
            val_pred = est.predict_proba(X.iloc[val_idx])[:, 1]
            oof[name][val_idx] = val_pred
            test[name] += est.predict_proba(X_test)[:, 1] / n_folds
            fold_aucs[name].append(fast_auc(y.iloc[val_idx].to_numpy(), val_pred))
            if keep_models:
                fold_models[name].append(est)
        fold_aucs[name] = np.array(fold_aucs[name])
        log(f"    OOF AUC {fast_auc(y.to_numpy(), oof[name]):.4f}  folds {np.round(fold_aucs[name], 4).tolist()}")

    return dict(oof=oof, test=test, fold_aucs=fold_aucs, fold_indices=fold_indices,
                fold_models=fold_models if keep_models else None)


def fit_blend_weights(oof_matrix, y, seed=42, maxiter=100, popsize=10):
    """Non-negative weights summing to 1 that maximise AUC of the blend.

    AUC is a step function of the weights (its gradient is ~0 almost
    everywhere), so gradient-based optimisers such as SLSQP stay at their
    starting point; differential evolution does not need gradients.
    """
    y = np.asarray(y)
    n_models = oof_matrix.shape[1]

    def neg_auc(weights):
        w = np.clip(weights, 0, None)
        total = w.sum()
        w = w / total if total > 0 else np.ones(n_models) / n_models
        return -fast_auc(y, oof_matrix @ w)

    result = differential_evolution(
        neg_auc, bounds=[(0, 1)] * n_models, seed=seed,
        maxiter=maxiter, popsize=popsize, tol=1e-8, polish=True,
    )
    w = np.clip(result.x, 0, None)
    return w / w.sum()


def nested_blend_scores(oof_matrix, y, fold_indices, seed=42, maxiter=100, popsize=10):
    """Fair blend score: weights for each fold are fit on the other folds only.

    Scoring the blend on the same OOF rows its weights were fit on is
    optimistic; this scores each fold with weights that never saw it.
    Also returns the equal-weight average as a no-fitting baseline.
    """
    y = np.asarray(y)
    nested_oof = np.zeros(len(y))
    for tr_idx, val_idx in fold_indices:
        w = fit_blend_weights(oof_matrix[tr_idx], y[tr_idx], seed, maxiter, popsize)
        nested_oof[val_idx] = oof_matrix[val_idx] @ w
    nested_aucs = np.array([fast_auc(y[v], nested_oof[v]) for _, v in fold_indices])
    equal_aucs = np.array([fast_auc(y[v], oof_matrix[v].mean(axis=1)) for _, v in fold_indices])
    return nested_oof, nested_aucs, equal_aucs
