# Student Dropout Early Warning System Using Machine Learning

The active system uses one **15-input Random Forest** for Individual Assessment and Batch Analysis. Its four additional inputs are mother's education, father's education, gender, and marital status. Attendance percentage is not included because it is not present in the UCI source dataset. The model and user interface use the same strict 15-field schema; incompatible artifacts are rejected rather than silently loaded. See [the 15-input migration and limitations](docs/14_input_migration.md).

This academic project estimates elevated risk of later university dropout after a completed semester. It supports timely review—not automatic decisions or certainty about a student's future.

## Prediction point and target

The prediction point is the **end of the first semester**. The model uses background, financial, and Semester-1 performance information available then.

- Positive class: Dropout = 1
- Negative class: Graduate = 0
- Enrolled students are excluded because their final outcomes are unresolved
- All Semester-2 and final-outcome information is excluded

## Dataset

The project uses the UCI Machine Learning Repository dataset **Predict Students' Dropout and Academic Success**.

| Cohort | Students |
|---|---:|
| Original dataset | 4,424 |
| Graduate | 2,209 |
| Dropout | 1,421 |
| Enrolled excluded | 794 |
| Resolved modeling cohort | 3,630 |

Dropout prevalence is 39.15%.

## Active 15-input form

1. HSC / Equivalent GPA
2. Age at Enrollment
3. Mother's Occupation
4. Father's Occupation
5. Mother's Education
6. Father's Education
7. Gender
8. Marital Status
9. Scholarship Holder
10. Debtor
11. Tuition Fees Up to Date
12. Courses Enrolled
13. Evaluations Completed
14. Courses Passed
15. Semester GPA (0.00–4.00)

Admission Grade was removed. The source Previous Qualification Grade is normalized during training as `source grade / 190 × 5` and renamed `previous_academic_gpa_normalized`. Its exact observed support is 2.50–5.00, which becomes the website's HSC / Equivalent GPA range. This is a numerical scale normalization only; it does not claim academic equivalence between national grading systems.

Occupation, education, gender, and marital-status codes are presented using documented readable labels and mapped to source values. Semester pass rate is calculated inside the pipeline as `Courses Passed / Courses Enrolled`; zero enrolled courses produces a rate of zero.

For interface compatibility, the visible Semester GPA is converted internally using `source semester grade = semester GPA / 4 × 18.875`. The user never enters or sees the source 0–18.875 value. This numerical normalization does not imply equivalence between Portuguese and Bangladeshi grading systems.

## Model development and results

Preprocessing and feature engineering are contained in a scikit-learn pipeline. The active Random Forest was evaluated with repeated stratified cross-validation on training data. The holdout was not used for threshold selection; it had been consulted in earlier project iterations and is not a pristine external validation set.

The active model is **Random Forest**. The classification threshold **0.52** was selected using training-only out-of-fold probabilities, subject to the documented minimum precision/recall gate. The fixed holdout was not used for threshold selection.

| Held-out metric | Result |
|---|---:|
| Accuracy | 87.9% |
| Balanced accuracy | 87.5% |
| Precision | 83.6% |
| Recall | 85.9% |
| F1 | 84.7% |
| ROC-AUC | 94.1% |
| PR-AUC | 93.1% |
| Brier score | 0.1011 |

Confusion matrix (rows actual Graduate/Dropout; columns predicted Graduate/Dropout): `[[394, 48], [40, 244]]`. The same holdout had been consulted in earlier project iterations, so these results are a project benchmark rather than pristine external validation. Full CV and holdout metrics are recorded in `reports/uci_sem1/training_results_15_input.json`.

## Streamlit interface

The four pages are Dashboard, Individual Assessment, Batch Analysis, and About Project. The assessment page loads the frozen artifact from `models/uci_sem1/`, calls `predict_proba()`, reads the Dropout=1 probability, and applies the saved threshold. The application never retrains.


The source model was trained using first-semester academic performance. The prototype interface presents these fields as the student's most recently completed semester for easier local data collection. Local validation is required before operational use across different semester stages.

## Local data collection plan

A future Google Form should collect the same 15 user fields shown above, plus these research-only fields:

- Current Semester / Academic Year
- Anonymous Follow-Up ID

Unlabeled form responses cannot immediately retrain a supervised model. Future actual outcomes must be linked to each response through the anonymous ID before the data can support local validation, recalibration, or retraining.

## Leakage controls

The final model excludes Target, every Semester-2 variable, final-outcome information, Application Mode, Application Order, Course code, Nationality, unemployment, inflation, and GDP. No information occurring after the prediction point is used.

## Local use

Python 3.12 is the deployment target. The artifact was trained under Python 3.14.5 with scikit-learn 1.9.1; Python 3.12 must be tested independently before deployment.

```bash
python -m pip install -r requirements.txt
python -m pytest -q --basetemp=.pytest_tmp
python -m streamlit run app.py
```

## Limitations

- Data comes from Portuguese higher education and has not been validated in Bangladesh.
- Academic systems may differ between countries.
- Occupation categories originate from the source dataset.
- Probability calibration may require local recalibration.
- Predictions are estimates, not certainties.
- Outputs should support, not replace, human judgement.
- Operational deployment requires local longitudinal validation or retraining.

## Local dashboard and batch analysis

The dashboard has four pages: Dashboard, Individual Assessment, Batch Analysis, and About Project. Batch Analysis accepts in-memory CSV or Excel (.xlsx) exports from Google Forms/Sheets, validates every row against all 15 inputs, applies the same frozen 15-input Random Forest and 0.52 threshold, summarizes risk and warning indicators, and provides CSV or Excel results downloads.

Use **Download CSV Template** or **Download Excel Template** on the Batch Analysis page for the exact schema. `Anonymous Follow-Up ID` and `Current Semester / Academic Year` are metadata only; they never enter the model. Uploaded files are not written to disk or stored in a database. Avoid names, phone numbers, email addresses, and other unnecessary identifying data.

A five-record demonstration file is available at `data/sample_batch_students.csv`.
