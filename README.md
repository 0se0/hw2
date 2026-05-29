# Bank Customer Churn Prediction

Binary classification to predict whether a bank customer 
will churn, using an optimized weighted ensemble model.

## Results

| Model | Validation AUC |
|-------|---------------|
| Random Forest | 0.8879 |
| Extra Trees | 0.8860 |
| Gradient Boosting | 0.8893 |
| Logistic Regression | 0.8767 |
| **Ensemble (RF 35% / ET 25% / GB 40%)** | **0.8889** |

- 5-fold Stratified CV AUC: **0.8889 ± 0.0013**
- 95% CI: [0.8878, 0.8901]
- Statistically significant vs 3/4 baselines (p < 0.05)

## Feature Engineering
Expanded from 12 → 27 features:
- Ratio features (Balance/Product, CreditScore/Age)
- Interaction features (Age × CreditScore)
- Binary flags (Is_Germany, Is_Senior, Is_Multi_Product)
- Churn Risk Score (composite indicator)

## Top Features
1. NumOfProducts (0.2028)
2. Age (0.1575)
3. CreditScore_per_Age (0.0945)

## Tech
Python, Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn
