# Results artifacts

This folder contains the evaluation outputs for the **DKA Antibiotics** Random Forest model.

## Files

- **classification_report.txt** — Precision/Recall/F1 per class and overall.
- **metrics.json** — Summary metrics (`roc_auc`, `average_precision`).
- **confusion_matrix.png** — 2×2 matrix of predicted vs actual labels.
- **roc_curve.png** — ROC curve on the test split.
- **pr_curve.png** — Precision–Recall curve.
- **calibration_curve.png** — Reliability diagram (how well probabilities are calibrated).
- **feature_importances.csv** — Permutation importances per feature on the test set.
- **feature_importances.png** — Top-20 feature importances bar chart.

## Reproduce

From the repo root, run:

```bash
python evaluate_rf_model.py \
  --model rf_antibiotics_model.pkl \
  --data "merged patients final with pcal_crp.xlsx" \
  --target "Antibiotics" \
  --exclude "VISIT #,PATIENT #,LENGTH OF STAY,CCI" \
  --outdir results
