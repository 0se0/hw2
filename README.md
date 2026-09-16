# Bank Customer Churn Prediction

Binary classification to predict whether a bank customer
will churn, using an out-of-fold (OOF) ensemble of tree models,
linear models, and gradient boosting libraries.

## Results (5-fold OOF, real dataset — 165,034 train rows)

| Model | OOF AUC |
|-------|---------|
| Logistic Regression | 0.8781 ± 0.0011 |
| Extra Trees | 0.8857 ± 0.0013 |
| Random Forest | 0.8878 ± 0.0012 |
| Gradient Boosting | 0.8889 ± 0.0014 |
| LightGBM | 0.8894 ± 0.0015 |
| XGBoost | 0.8894 ± 0.0015 |
| **Ensemble (weighted blend)** | **0.8896 ± 0.0015** |

Optimal blend weights: LGBM 38.7% / XGB 34.7% / GB 14.1% / RF 7.3% / ET 5.0% / LR 0%
(a stacking meta-model was also tried and scored slightly lower — 0.8890 — so the
weighted blend was kept).

- 95% CI: [0.8883, 0.8909]
- Statistically significant improvement vs **all 6** individual models (p < 0.05) —
  up from 3/4 in the previous single-split version
- Predicted test churn rate: 21.12% (actual train churn rate: 21.16%)
- CatBoost was not installed in the environment this was run in, so it was skipped;
  installing it may push the ensemble slightly further.

## What changed in the rework
- **Added LightGBM / XGBoost / CatBoost** to the model zoo (used if installed,
  skipped otherwise) alongside RF / ExtraTrees / GradientBoosting / LogisticRegression.
- **Out-of-fold (OOF) evaluation** instead of a single 80/20 train/val split:
  every model now produces predictions for the *entire* training set via
  5-fold CV, so ensemble weights are fit on ~5x more validation data and are
  far less likely to overfit to one particular split.
- **Fold-averaged test predictions (bagging)**: test predictions are the
  average of 5 fold-models per base model, instead of a single model fit on
  80% of the data — this lowers prediction variance on the leaderboard set.
- **Ensemble weight search replaced**: the old approach was a brute-force
  grid over 6 discrete weight values. AUC is a rank-based, non-smooth
  objective, so a gradient-based optimizer (SLSQP) was tried and verified to
  get stuck at its initial guess; `scipy.optimize.differential_evolution`
  (derivative-free) is used instead and empirically finds real improvements.
- **Stacking added as an alternative** to weighted blending: a logistic
  regression meta-model trained on OOF predictions, evaluated via its own
  CV, is compared against the weighted blend, and whichever wins is used.
- **Fixed a train/test leakage-style bug**: `Is_High_Value` used to threshold
  each dataset against its own `Balance` 80th percentile, so train and test
  used different, inconsistent cutoffs. The threshold is now computed once on
  train and reused for both.
- **Statistical significance testing consolidated**: the old notebook
  retrained every model a second time (via `cross_val_score`) just to run the
  paired t-tests. The rework reuses the per-fold AUCs already produced during
  OOF generation — same statistical test, no duplicate training.
- **Colab dependency removed from the core pipeline**: the notebook reads
  `train.csv`/`test.csv` from disk (or `CHURN_DATA_DIR`) when available, and
  only falls back to `google.colab.files.upload()`/`download()` when actually
  running in Colab, so it also runs locally or in any other Jupyter environment.
- **Model persistence**: final models, scaler, weights/meta-model, and the
  feature list are saved to `churn_ensemble_artifact.joblib` for reuse without
  retraining.

The rework was smoke-tested end-to-end on synthetic data first, then run on
the real dataset (`train.csv`: 165,034 rows, `test.csv`: 110,023 rows) to
produce the numbers above.

## Feature Engineering
Expanded from 12 → 27 features:
- Ratio features (Balance/Product, CreditScore/Age)
- Interaction features (Age × CreditScore)
- Binary flags (Is_Germany, Is_Senior, Is_Multi_Product)
- Churn Risk Score (composite indicator)

## Top Features
1. NumOfProducts (0.2030)
2. Age (0.1554)
3. CreditScore_per_Age (0.0962)

## Tech
Python, Scikit-learn, LightGBM, XGBoost, (CatBoost if installed), SciPy,
Pandas, NumPy, Matplotlib, Seaborn, Joblib
