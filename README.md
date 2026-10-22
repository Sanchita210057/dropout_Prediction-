# Student Dropout Early Warning System

Project A's active model is a two-component **UCI + Bangladesh 18-input ensemble**. Model A supplies an academic signal learned from resolved Portuguese UCI outcomes; Model B uses all 18 Bangladesh questionnaire predictors; Model C combines them. The target the app estimates is **reported current Bangladesh dropout status**, not a validated future-dropout outcome.

## Active model contract

The active artifact is `models/combined_uci_bangladesh_18_v2/weighted_ensemble_dropout_status_18.joblib`. Both Individual Assessment and Batch Analysis use it and accept exactly these 18 inputs, in the order defined by `src/bangladesh_collection.py`:

1. HSC / Equivalent GPA
2. Age at enrollment
3. Mother's occupation
4. Father's occupation
5. Mother's highest education
6. Father's highest education
7. Gender
8. Marital status
9. Scholarship holder
10. Have any debt
11. Tuition fees up to date
12. Courses enrolled in the relevant semester
13. Evaluations completed in the relevant semester
14. Courses passed in the relevant semester
15. Semester GPA (native university scale)
16. Last-semester attendance percentage
17. Family financial condition
18. Student employment

The exact local parent occupation categories `Not Alive` and `Housewife` are preserved. `Current Academic Status` is the separate target, never an input: Dropout = 1; Running = 0; Passed = 0. Running is not proof of eventual persistence.

## Model A: UCI academic foundation

The UCI source contains 4,424 records. All **3,630 resolved** records (Graduate=0, Dropout=1) train Model A; 794 unresolved Enrolled records are excluded. After auditing all 15 predictors used by the original UCI model, ten were meaningfully aligned: prior qualification grade, age, gender, scholarship, debtor, tuition status, first-semester enrolled/evaluated/passed counts, and first-semester grade. The complete keep/exclude table and transformations are in `reports/combined_uci_bangladesh_18_v2/model_architecture_and_evaluation.md`.

UCI prior grade is linearly mapped from 0–190 to the interface's 0–5 range; first-semester grade is linearly mapped from the observed 0–18.875 range to 0–5. These are numerical range alignments only—not equivalence between Portuguese and Bangladeshi grading systems or qualifications. Binary gender/support values are mapped only to matching documented labels. Attendance, family income and employment do not exist in UCI and are not fabricated.

The shared 18-input assessment's parent-occupation dropdowns use exactly the Bangladesh questionnaire categories, including `Not Alive` and `Housewife`. The UCI academic component uses ten harmonized features and does not learn parental occupation categories. Portuguese UCI job-role categories are therefore not shown as assessment choices and are not crosswalked to Bangladesh occupation labels. The Bangladesh component receives the local categories unchanged; `Housewife` is in the mother's fitted category vocabulary but absent from the father's, where it is an unseen category. Research-response cleaning and historical responses retain their original categories.

Parent education remains source-specific: UCI qualification codes distinguish Portuguese completed/incomplete school years, technical programs and degree cycles, which cannot reliably be collapsed to the local questionnaire's education labels. Marital-status overlap is clearer: UCI single, married, widower, divorced and legally separated align with local Single / Never married, Married, Widowed, Divorced and Separated; de-facto union has no exact local equivalent. The active UCI model excludes parent education and marital status. Gender Female/Male and binary Yes/No fields retain exact-label mappings; other identities/unknowns are missing for the UCI component. UCI code definitions are from the [UCI dataset codebook](https://archive.ics.uci.edu/dataset/697/predict%2Bstudents%2Bdropout%2Band%2Bacademic%2Bsuccess).

## Model B: Bangladesh local component

Model B was trained on all **251** Bangladesh questionnaire records and all 18 approved features. The label distribution is 52 Dropout and 199 class-0 (Running/Passed) responses. Nineteen zero Semester GPA values that coincide with some courses passed but fewer than enrolled are treated as unavailable; rows and course counts are retained. Imputation and encoding remain inside training pipelines. The released workbook has no name, student-number, email, phone, or contact fields. It retains the original unique Google Forms submission timestamps; timestamps are not model inputs but may enable linkage with outside information.

## Model C: ensemble comparison and selection

The selected app architecture is a **calibrated fixed 5% UCI / 95% Bangladesh probability blend**, using the expanded ten-feature UCI foundation. The active threshold is **0.3040698561883773** (displayed as 30.4%). Calibration is applied to the clipped logit of the weighted blend, matching the training-time transform. The UCI share was not increased. A separate simple Logistic Regression stacker was trained only on Bangladesh out-of-fold base predictions; its coefficients and fold variation are recorded in the report. The stacker achieved slightly higher ROC-AUC, but lower PR-AUC and worse Brier score, with more false positives. Therefore the fixed 5% blend was selected over stacking. Inactive baseline and stacker artifacts have been moved to the external local archive; their comparison metrics remain documented in the evaluation report.

Results below are nested stratified five-fold CV on the 251 Bangladesh records. Selection, calibration/stacking, and thresholds use training-only folds; the outer fold is untouched. The threshold maximizes F1 with at most 50% of training cases flagged. A 63-record holdout had already been inspected in earlier work, so it is not reused or described as fresh validation.

| Candidate | Accuracy | Balanced accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Brier | TN / FP / FN / TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Bangladesh-only 18-input | 0.972 | 0.961 | 0.925 | 0.942 | 0.933 | 0.983 | 0.933 | 0.032 | 195 / 4 / 3 / 49 |
| UCI foundation, six numeric inputs | 0.904 | 0.847 | 0.780 | 0.750 | 0.765 | 0.915 | 0.750 | 0.100 | 188 / 11 / 13 / 39 |
| UCI foundation, ten aligned inputs | 0.908 | 0.829 | 0.837 | 0.692 | 0.758 | 0.939 | 0.810 | 0.073 | 192 / 7 / 16 / 36 |
| Archived six-feature 5%/95% blend | 0.972 | 0.961 | 0.925 | 0.942 | 0.933 | 0.985 | 0.911 | 0.032 | 195 / 4 / 3 / 49 |
| **Selected ten-feature 5%/95% blend** | **0.964** | **0.956** | **0.891** | **0.942** | **0.916** | **0.986** | **0.935** | **0.032** | **193 / 6 / 3 / 49** |
| OOF-selected weighted blend, ten-feature UCI | 0.968 | 0.959 | 0.907 | 0.942 | 0.925 | 0.986 | 0.933 | 0.033 | 194 / 5 / 3 / 49 |
| OOF Logistic Regression stacker | 0.952 | 0.949 | 0.845 | 0.942 | 0.891 | 0.987 | 0.919 | 0.037 | 190 / 9 / 3 / 49 |

The selected blend's PR-AUC is only about 0.002 above Bangladesh-only, with the same three false negatives but two more false positives. Brier scores for all CV candidates beat the constant-training-prevalence baseline of 0.164, but the very small local sample and nested selection make these estimates uncertain. The expanded UCI component appears to add a modest ranking signal; it does not establish a reliable gain or justify increasing its 5% share. Stack model coefficients, thresholds, fold-specific weights, confusion matrices, and prevalence Brier comparisons are in `reports/combined_uci_bangladesh_18_v2/model_architecture_and_evaluation.md` and `training_results.json`.

## Important limitations

- UCI labels are resolved graduate/dropout outcomes; Bangladesh labels are self-reported current status. They are not pooled or treated as identical.
- Bangladesh predictor timing relative to discontinuation is unknown. Attendance, semester results, tuition/debt, finances, and employment may be contemporaneous or retrospective. Do not describe this as prospective dropout prediction.
- The Bangladesh sample is a small convenience sample. There is no new independent Bangladesh evaluation cohort or external validation.
- Brier improvement over a prevalence baseline does not establish transportability or calibration in other institutions.
- Income ranges are provisional questionnaire brackets, not official poverty definitions. The system supports human review and should not be used for automated adverse decisions.
- Keep raw questionnaire responses private; do not publish microdata.

## Run and retrain

```powershell
python -m pip install -r requirements.txt
python -m pytest -q
python -m streamlit run app.py
```

Normal application predictions use only the saved artifacts and do not load either training dataset. The released repository includes both datasets for reproducibility. Training and data inspection default to these paths:

- `data/student_dropout.csv`
- `data/Bangladesh University Research Responses.xlsx`

To use authorized copies elsewhere, optionally override the paths. PowerShell:

```powershell
$env:CSE_UCI_DATASET_PATH = "E:\authorized-data\student_dropout.csv"
$env:CSE_BANGLADESH_DATASET_PATH = "E:\authorized-data\Bangladesh University Research Responses.xlsx"
```

Bash (including Linux):

```bash
export CSE_UCI_DATASET_PATH="/path/to/student_dropout.csv"
export CSE_BANGLADESH_DATASET_PATH="/path/to/Bangladesh University Research Responses.xlsx"
```

The application resolves its model and default data paths relative to `app.py` with `pathlib`, so it does not depend on a Windows drive letter. Streamlit Community Cloud currently defaults to Python 3.12; select a compatible Python version in the app's deployment settings and keep the pinned scikit-learn version from `requirements.txt` to load the saved artifact.

To reproduce the final-model evaluation, deliberately run `python -m src.train_combined_ensemble_18` after confirming both sources; it writes artifacts and reports under the active model paths, so it is not part of normal use. The active artifact contains both fitted components; no separate component model file is needed at inference.

The four public app pages are Dashboard, Individual Assessment, Batch Analysis, and About Project. A separate Bangladesh response-collection workflow remains in the codebase but is not exposed in the supervisor-facing navigation. Individual and batch assessment use the same 18 inputs, selected ensemble, threshold, and support guidance. CSV/Excel batch processing remains in memory. Project B is a separate local archive and is not part of this application.
