# Historical 11-Input Semester-1 Model Evaluation

> **Superseded:** The following figures belong only to the previous 11-input model and are not results for the active system. The current 15-input evaluation is in `training_results_15_input.json` and summarized in `docs/14_input_migration.md`.

The final Random Forest retains the same 3,630 resolved-outcome cohort, stratified 80/20 split (`random_state=42`), preprocessing, hyperparameters, repeated 5-fold CV strategy, and end-of-Semester-1 leakage policy as the validated 12-input model. Admission Grade is excluded. Previous Qualification Grade is normalized during training as `grade / 190 × 5` and renamed `previous_academic_gpa_normalized`; observed support is exactly 2.50–5.00.

This transformation aligns numerical scales only. It does not imply academic equivalence between grading systems.

## Training cross-validation

| Metric | Mean | SD | Change from 12-input model |
|---|---:|---:|---:|
| Precision | 0.8564 | 0.0103 | −0.0028 |
| Recall | 0.8232 | 0.0268 | −0.0004 |
| F1 | 0.8393 | 0.0155 | −0.0016 |
| ROC-AUC | 0.9232 | 0.0162 | −0.0001 |
| PR-AUC | 0.9187 | 0.0162 | −0.0005 |

The change is negligible and stable across folds, so the usability-focused form passed the comparative gate. Out-of-fold training predictions again selected threshold 0.48.

## Holdout benchmark

| Metric | Result |
|---|---:|
| Accuracy | 0.8760 |
| Balanced accuracy | 0.8762 |
| Precision | 0.8191 |
| Recall | 0.8768 |
| F1 | 0.8469 |
| ROC-AUC | 0.9378 |
| PR-AUC | 0.9286 |
| Brier score | 0.0997 |

Confusion matrix: TN=387, FP=55, FN=35, TP=249. Because this holdout was consulted during earlier iterations, it is a final project benchmark rather than pristine external validation. No Target, Semester-2 field, final-outcome field, or banned administrative/economic field is used.
