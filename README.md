# DSCI-552 Homework 6 — Tree-Based Methods

Tianbo Hou · GitHub: houdan-hub · USC ID: 8943381210

## What this is

Machine-learning homework for DSCI 552 (USC). Part 1 works with the
[APS Failure at Scania Trucks](https://archive.ics.uci.edu/dataset/429/aps+failure+at+scania+trucks)
data set (60,000 training rows / 16,000 test rows, 170 numeric predictors,
1.67% positive class) and compares tree-based classifiers under a severe
class imbalance. Parts 2–5 are written answers to selected ISLR exercises.

## Contents

- `Hou_Tianbo_HW6.ipynb` — the full report: code, figures, and written
  discussion for every question. Executed top to bottom with no errors.
- `data/aps_failure_training_set.csv`, `data/aps_failure_test_set.csv` — the
  UCI data files (license header rows removed, `na` kept as the missing
  marker; the notebook reads them with `na_values='na'`).
- `requirements.txt` — Python packages needed to re-run the notebook.

## How to run

```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute Hou_Tianbo_HW6.ipynb
```

Everything uses relative paths under the repo root, so a fresh clone runs as
is. Full execution takes a few minutes (the SMOTE pipeline in 1(f) is the
slowest cell).

## Main results (test set, challenge cost = 10 × false alarms + 500 × misses)

| model | test acc | test AUC | misses | false alarms | cost |
|---|---|---|---|---|---|
| Random forest (no compensation) | 0.99244 | 0.99423 | 103 | 18 | 51,680 |
| Random forest (balanced weights) | 0.98887 | 0.99313 | 161 | 17 | 80,670 |
| XGBoost model tree (α = 1) | 0.99300 | 0.99259 | 89 | 23 | 44,730 |
| XGBoost + SMOTE (α = 0) | 0.99350 | 0.99197 | 63 | 41 | 31,910 |

SMOTE is applied only inside the CV folds and only to training data
(`imblearn.Pipeline`), never to the test set.

## References

- Scania CV AB. "APS Failure at Scania Trucks Data Set." *UCI Machine Learning Repository*, 2016.
- Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic Minority Over-sampling Technique. *Journal of Artificial Intelligence Research*, 16, 321–357.
- James, G., Witten, D., Tibshirani, R., & Hastie, T. (2023). *An Introduction to Statistical Learning: with Applications in Python*. Springer.
