# Future Semester-1 Website Form Plan

> **Historical and superseded:** This form plan documents the previous 11-input system and its 0.48 threshold. The active application uses the 15-input model with threshold 0.52. See `docs/14_input_migration.md` and `reports/uci_sem1/training_results_15_input.json` for current specifications and results.

## Workflow

Student completes Semester 1 → department enters available information → saved model estimates later dropout risk → result is **Lower Estimated Risk** or **Elevated Estimated Risk** → staff may consider supportive follow-up.

The future application must load `models/uci_sem1/dropout_early_warning_model.joblib`; it must not retrain. Migration is not part of this round.

## Exact 11 fields

### Student background

1. HSC / Equivalent GPA (2.50–5.00)
2. Age at Enrollment

### Family background

3. Mother's Occupation
4. Father's Occupation

### Financial status

5. Scholarship Holder
6. Debtor
7. Tuition Fees Up to Date

### First-semester performance

8. Courses Enrolled
9. Evaluations Completed
10. Courses Passed
11. Semester-1 GPA (0.00–4.00)

The source Previous Qualification Grade is normalized during training with `grade / 190 × 5`; it is not converted in the frontend. This aligns numerical scales only and does not claim cross-country academic equivalence. Admission Grade is excluded.

Courses Passed and Courses Enrolled are converted internally to Semester pass rate. Parent occupation codes must be rendered with the official UCI occupation descriptions; raw numeric codes must never be shown.

Semester-1 GPA is converted internally with `source semester grade = GPA / 4 × 18.875` for compatibility with the frozen model. The converted value is never displayed. This is numerical normalization only and does not imply equivalence between Portuguese and Bangladeshi grading systems.

## Result wording

The locked threshold is 0.48. A future interface may display the estimated percentage because the Brier score is 0.0967, but must say:

> This is an early-warning estimate based on patterns in the training dataset. It does not determine whether a student will actually drop out.

It must also display:

> The model was trained on a publicly available Portuguese higher-education dataset. It is an academic prototype. Before operational use in Bangladesh or another country, it should be externally validated or retrained using local longitudinal university data.

The output must never say that a student “will drop out,” and it must not trigger punitive or automatic decisions.
