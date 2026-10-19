# 15-input Semester-1 model migration

This document lives at its legacy `14_input_migration.md` path, but describes the completed **15-input** implementation.

## Active model schema

The app now has one active 15-input Random Forest for individual assessment and batch analysis. Its four added inputs are Mother's Education, Father's Education, Gender, and Marital Status. Attendance Percentage is not a feature because it is absent from the UCI source dataset.

The source feature vector also contains the approved-course count; the pipeline derives Semester Pass Rate from approved/enrolled, so the derived model schema remains 15 columns. GPA normalization remains `Previous qualification grade / 190 * 5`, with supported visible input range 2.50–5.00. The 0–4 Semester GPA form entry is normalized internally to the source model's grade scale.

The UI, preprocessing pipeline, CSV/Excel templates, validation, prediction, downloads, metadata, and result report all share the ordered schema in `src/uci_sem1_features.py`. `app.py` refuses an artifact unless it declares the exact 15 raw inputs and 15 final model features. Old training commands are compatibility shims to the 15-input trainer, preventing an accidental 11-input artifact overwrite.

## Training and evaluation

Run `python -m src.train_uci_sem1_15`. It writes the active artifact and metadata under `models/uci_sem1/` and the evaluation JSON at `reports/uci_sem1/training_results_15_input.json`.

- Cohort: 3,630 resolved outcomes; 2,209 Graduate, 1,421 Dropout; 794 Enrolled excluded.
- Split: stratified 80/20 with `random_state=42` (2,904 train / 726 holdout).
- CV: repeated stratified five-fold, two repeats, on training only.
- Threshold: selected from training-only five-fold out-of-fold predictions; highest F1 (balanced-accuracy tie-break) among candidates with recall ≥0.78 and precision ≥0.60. Current threshold: **0.52**.
- The holdout was consulted in previous project iterations. It is a project benchmark, not pristine external validation. Do not tune against it.

Current holdout results are Accuracy 87.9%, Precision 83.6%, Recall 85.9%, F1 84.7%, ROC-AUC 94.1%, PR-AUC 93.1%; TN=394, FP=48, FN=40, TP=244. See the JSON for exact full-precision results and CV intervals.

## Verification

The focused and full test suites should pass before release. Verify the dashboard and all three other pages; exercise individual assessment, CSV and Excel templates/uploads/results, and validation. Confirm exactly 15 student inputs (plus optional anonymous ID/semester metadata), model/metadata threshold agreement, and successful fresh-process artifact loading.

## Scope and limitations

This model is based on the Portuguese UCI “Predict Students' Dropout and Academic Success” dataset, covering multiple programs at one institution. It is not CSE-specific and is **not validated for Bangladesh**. Gender, marital status, and parental education can be sensitive/proxy attributes; subgroup performance must be reviewed and predictions must only support human-led assistance, never punitive or access-denial decisions. Local deployment requires consented longitudinal outcomes, local validation, and governance review.
