"""Train the OOF ensemble, evaluate it fairly, write a submission and a model artifact.

    python -m churn.train --data-dir data/ --out-dir artifacts/
"""

import argparse
import json
import os
import time

import joblib
import numpy as np
import pandas as pd

from .ensemble import fit_blend_weights, generate_oof, nested_blend_scores
from .features import ID_COL, TARGET, compute_feature_state, engineer_features
from .metrics import fast_auc
from .models import available_models
from .stats import compare_to_ensemble

DEFAULT_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "params", "best_params.json")


def run_training(train_df, test_df, model_names, tuned_params=None, n_folds=5, seed=42,
                 blend_maxiter=100, blend_popsize=10, n_boot=1000, keep_models=True, log=print):
    state = compute_feature_state(train_df)
    X = engineer_features(train_df, state["balance_q80"])
    X_test = engineer_features(test_df, state["balance_q80"])[list(X.columns)]
    y = train_df[TARGET].reset_index(drop=True)
    X = X.reset_index(drop=True)
    y_arr = y.to_numpy()

    log(f"features: {X.shape[1]}, train rows: {len(X):,}, test rows: {len(X_test):,}")
    res = generate_oof(X, y, X_test, model_names, tuned_params, n_folds, seed, keep_models, log)

    oof_matrix = np.column_stack([res["oof"][n] for n in model_names])
    test_matrix = np.column_stack([res["test"][n] for n in model_names])

    log("fitting blend weights (final) and nested-CV blend score")
    weights = fit_blend_weights(oof_matrix, y_arr, seed, blend_maxiter, blend_popsize)
    nested_oof, nested_aucs, equal_aucs = nested_blend_scores(
        oof_matrix, y_arr, res["fold_indices"], seed, blend_maxiter, blend_popsize)
    insample_aucs = np.array([fast_auc(y_arr[v], oof_matrix[v] @ weights) for _, v in res["fold_indices"]])

    significance = compare_to_ensemble(
        res["fold_aucs"], nested_aucs, y_arr, nested_oof,
        {n: res["oof"][n] for n in model_names}, res["fold_indices"], n_boot, seed)

    metrics = dict(
        model_names=model_names,
        single_model_fold_mean_auc={n: float(res["fold_aucs"][n].mean()) for n in model_names},
        blend_weights=dict(zip(model_names, map(float, weights))),
        blend_nested_fold_mean_auc=float(nested_aucs.mean()),
        blend_nested_fold_std_auc=float(nested_aucs.std()),
        blend_insample_fold_mean_auc=float(insample_aucs.mean()),
        equal_weight_fold_mean_auc=float(equal_aucs.mean()),
        significance_vs_ensemble=significance,
    )
    artifact = dict(
        model_names=model_names, weights=weights, fold_models=res["fold_models"],
        feature_state=state, feature_cols=list(X.columns), seed=seed, n_folds=n_folds,
    )
    return dict(test_pred=test_matrix @ weights, metrics=metrics, artifact=artifact)


def _log_summary(metrics, log):
    log("\nsingle models (fold-mean AUC):")
    for n, v in metrics["single_model_fold_mean_auc"].items():
        log(f"  {n:<9}{v:.4f}")
    log(f"blend, nested CV (fair):   {metrics['blend_nested_fold_mean_auc']:.4f}"
        f" +- {metrics['blend_nested_fold_std_auc']:.4f}")
    log(f"blend, in-sample (biased): {metrics['blend_insample_fold_mean_auc']:.4f}")
    log(f"equal-weight average:      {metrics['equal_weight_fold_mean_auc']:.4f}")
    log("weights: " + ", ".join(f"{n}={w:.3f}" for n, w in metrics["blend_weights"].items()))
    log("\nensemble vs single model (mean AUC gain | naive p | corrected p | bootstrap 95% CI):")
    for n, r in metrics["significance_vs_ensemble"].items():
        lo, hi = r["boot_ci"]
        log(f"  {n:<9}{r['mean_diff']:+.4f} | {r['p_naive']:.4f} | {r['p_corrected']:.4f} | [{lo:+.5f}, {hi:+.5f}]")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=os.environ.get("CHURN_DATA_DIR", "."),
                   help="folder with train.csv and test.csv")
    p.add_argument("--out-dir", default="artifacts")
    p.add_argument("--params", default=DEFAULT_PARAMS_PATH, help="tuned hyperparameters (JSON)")
    p.add_argument("--models", nargs="*", default=None, help="subset of models (default: all available)")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--blend-maxiter", type=int, default=100)
    p.add_argument("--bootstrap", type=int, default=1000)
    args = p.parse_args(argv)

    train_df = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
    test_df = pd.read_csv(os.path.join(args.data_dir, "test.csv"))
    tuned = json.load(open(args.params)) if args.params and os.path.exists(args.params) else {}
    names = args.models or available_models()

    t0 = time.time()
    out = run_training(train_df, test_df, names, tuned, args.folds, args.seed,
                       args.blend_maxiter, n_boot=args.bootstrap)
    _log_summary(out["metrics"], print)

    os.makedirs(args.out_dir, exist_ok=True)
    pd.DataFrame({ID_COL: test_df[ID_COL], TARGET: out["test_pred"]}).to_csv(
        os.path.join(args.out_dir, "submission.csv"), index=False)
    joblib.dump(out["artifact"], os.path.join(args.out_dir, "model.joblib"))
    json.dump(out["metrics"], open(os.path.join(args.out_dir, "metrics.json"), "w"), indent=2)
    print(f"\nwrote submission.csv, model.joblib, metrics.json to {args.out_dir}/  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
