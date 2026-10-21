# Combined UCI + Bangladesh 18-input model architecture and evaluation

## Cohorts and outcomes

- UCI: 3630 resolved records used to fit each final UCI component (Dropout=1, Graduate=0); 794 unresolved Enrolled records excluded.
- Bangladesh: 251 questionnaire records used for the local component and OOF fusion fit (Dropout=1; Running/Passed=0). There are 52 Dropout and 199 class-0 labels.
- The targets differ. UCI is eventual resolved outcome; Bangladesh is reported current status. These labels are not pooled or treated as equivalent. Running=0 is not proof of eventual persistence.
- No evaluation result is a fresh independent test: prior inspection of a 63-record holdout is disclosed and this comparison uses nested stratified CV across the 251 Bangladesh records.

## Audit of the 15 UCI model inputs

The academic component uses ten defensibly aligned features: six academic/numeric measures plus gender and three binary support/financial indicators. It does not map UCI parent education/occupation codes to Bangladesh survey categories. Unknown or unsupported local categories remain unrecognized by the UCI encoder, not coerced into a Portuguese code.

| Original UCI predictor | Bangladesh counterpart | Decision | Treatment / reason |
|---|---|---|---|
| Previous qualification (grade) | HSC / Equivalent GPA | Use | Linear numeric range alignment from UCI 0–190 to Bangladesh 0–5; no academic-system equivalence claimed. |
| Age at enrollment | Age at enrollment | Use | Direct numeric match. |
| Mother's occupation | Mother's occupation | Exclude | Portuguese occupation codes do not represent the Bangladesh survey categories. |
| Father's occupation | Father's occupation | Exclude | Portuguese occupation codes do not represent the Bangladesh survey categories. |
| Mother's qualification | Mother's highest education | Exclude | Detailed UCI education codes have no verified one-to-one Bangladesh mapping. |
| Father's qualification | Father's highest education | Exclude | Detailed UCI education codes have no verified one-to-one Bangladesh mapping. |
| Gender | Gender | Use | Map only documented UCI Female/Male values to matching labels; nonmatching local categories remain unseen, never coerced. |
| Marital status | Marital status | Exclude | Codes include distinctions such as de-facto union/legal separation that do not map exactly to local options. |
| Scholarship holder | Scholarship holder | Use | Map UCI binary 0/1 to No/Yes. |
| Debtor | Have any debt | Use | Map UCI binary 0/1 to No/Yes. |
| Tuition fees up to date | Tuition fees up to date | Use | Map UCI binary 0/1 to No/Yes. |
| Curricular units 1st sem (enrolled) | Courses enrolled | Use | Direct numeric count; domain ranges may differ. |
| Curricular units 1st sem (evaluations) | Evaluations completed | Use | Direct numeric count; domain ranges may differ. |
| Curricular units 1st sem (grade) | Semester GPA | Use | Linear range alignment from observed UCI 0–18.875 to Bangladesh model 0–5; not grading-system equivalence. |
| Curricular units 1st sem (approved) | Courses passed | Use | Direct numeric count. |

## Model C: nested comparison

Every outer fold refits the Bangladesh component selection, all calibration/meta layers, and thresholds using only the outer-training records. The fusion layer is trained on Bangladesh local-model OOF predictions only—never in-sample predictions from a local model trained on those same labels. The outer validation fold is untouched. Threshold selection uses nested cross-fitted training probabilities and caps alerts at 50%.

| Candidate | Accuracy | Balanced accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Brier | Prevalence Brier | TN / FP / FN / TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Bangladesh-only 18-input | 0.972 | 0.961 | 0.925 | 0.942 | 0.933 | 0.983 | 0.933 | 0.032 | 0.164 | 195 / 4 / 3 / 49 |
| UCI academic foundation (6 numeric) | 0.904 | 0.847 | 0.780 | 0.750 | 0.765 | 0.915 | 0.750 | 0.100 | 0.164 | 188 / 11 / 13 / 39 |
| UCI academic foundation (10 aligned) | 0.908 | 0.829 | 0.837 | 0.692 | 0.758 | 0.939 | 0.810 | 0.073 | 0.164 | 192 / 7 / 16 / 36 |
| Archived 5%/95% weighted blend (6-feature UCI) | 0.972 | 0.961 | 0.925 | 0.942 | 0.933 | 0.985 | 0.911 | 0.032 | 0.164 | 195 / 4 / 3 / 49 |
| Fixed 5%/95% blend (10-feature UCI) | 0.964 | 0.956 | 0.891 | 0.942 | 0.916 | 0.986 | 0.935 | 0.032 | 0.164 | 193 / 6 / 3 / 49 |
| OOF-selected weighted blend (10-feature UCI) | 0.968 | 0.959 | 0.907 | 0.942 | 0.925 | 0.986 | 0.933 | 0.033 | 0.164 | 194 / 5 / 3 / 49 |
| OOF Logistic Regression stacking (10-feature UCI) | 0.952 | 0.949 | 0.845 | 0.942 | 0.891 | 0.987 | 0.919 | 0.037 | 0.164 | 190 / 9 / 3 / 49 |

Previous archived 5%/95% model run (separately reported): accuracy=0.972, balanced_accuracy=0.961, precision=0.925, recall=0.942, f1=0.933, roc_auc=0.978, pr_auc_average_precision=0.885, brier_score=0.045.

## Fusion comparison, selected architecture and interpretation

Final meta-model coefficients on the two component scores: Bangladesh `3.6121`, UCI `1.6659`, intercept `-3.2976`. Fold-wise coefficient vectors are in `training_results.json`; their variability should be interpreted cautiously with only 251 local observations.
The selected active candidate is the fixed 5% UCI / 95% Bangladesh weighted blend with the ten-feature UCI foundation, threshold `0.3040698561883773`. Its nested-CV PR-AUC (0.935) and Brier score (0.032) were better than the OOF Logistic Regression stacker (PR-AUC 0.919; Brier 0.037). Stacking had slightly higher ROC-AUC but more false positives at its selected operating point. Therefore stacking is a documented comparison, not the selected app model; its inactive artifact is archived outside the project.
Model A was fit on 3630 UCI records; Model B was fit on 251 Bangladesh records. The stacker used OOF Bangladesh component scores; the selected weighted blend uses a fixed 5% UCI / 95% local signal and a training-only calibration layer. The original six-feature blend artifact is archived outside the project; its evaluation summary is retained above for historical comparison.
The expanded fixed blend's performance is only modestly different from the Bangladesh-only system and not an external validation. Different outcome definitions, 0–5 numerical grade normalization (not grading equivalence), uncertain predictor timing, small convenience sample and cross-country domain shift prevent a future-dropout prediction claim. `Current Academic Status` and Timestamp are excluded from all predictor matrices.
