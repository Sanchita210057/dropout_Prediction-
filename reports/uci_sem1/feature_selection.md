# Historical Feature-Usability Decision (11-input model)

> **Superseded:** This report documents the prior 11-input Random Forest feature decision. The active system is now the 15-input model. None of the 11-input metrics below are results for the active model; see `training_results_15_input.json` and `docs/14_input_migration.md` for current results.

The validated 12-input Random Forest was compared with an 11-input version using the same training partition and repeated cross-validation. The new form removes Admission Grade and replaces the raw 95–190 Previous Qualification Grade with a native 2.50–5.00 model input:

`previous_academic_gpa_normalized = previous_qualification_grade / 190 × 5`

Examples: 190→5.00, 152→4.00, and 95→2.50. This is only a linear numerical normalization, not a statement that source grades are academically equivalent to Bangladesh HSC GPAs.

Repeated-CV changes for the 11-input Random Forest were ROC-AUC −0.0001, PR-AUC −0.0005, precision −0.0028, recall −0.0004, and F1 −0.0016. These negligible changes support adopting the clearer form.

The 11 visible fields are HSC / Equivalent GPA, Age at Enrollment, both parent occupations, Scholarship Holder, Debtor, Tuition Fees Up to Date, Courses Enrolled, Evaluations Completed, Courses Passed, and Semester-1 GPA. Semester pass rate is derived safely as Passed/Enrolled, with zero used when Enrolled is zero. The visible 0–4 Semester-1 GPA is normalized internally to the frozen model's original 0–18.875 range; this is interface compatibility, not cross-country academic equivalence.
