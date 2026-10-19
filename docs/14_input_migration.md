# 14-input university model migration

**Status: staging branch. Not production-ready until retraining and tests pass.**

This change adds exactly three first-semester model inputs: mother's education, father's education, and gender. Attendance percentage and marital status are **not** part of this 14-input version. The previous 11-input artifact must be replaced with an actual retrained 14-input artifact **before merging this branch**.

## How to finish locally

1. In the Codex project folder, save uncommitted local changes and inspect `git status`. Fetch and integrate branch `feature/14-input-university-model` without overwriting any local experimental results.
2. Use the project's supported Python environment with the required `scikit-learn==1.9.1`.
3. Run `python -m src.train_uci_sem1_14` once. It trains and writes:
   - `models/uci_sem1/dropout_early_warning_model.joblib`
   - `models/uci_sem1/model_metadata.json`
   - `reports/uci_sem1/training_results_14_input.json`
4. Confirm `len(raw_input_features) == 14` and `artifact['threshold'] == metadata['threshold']`; verify classes 0=Graduate and 1=Dropout.
5. Run `python -m pytest -q --basetemp=.pytest_14_input`. Fix remaining outdated tests rather than deleting them. Test all four UI pages, individual assessment, CSV and Excel upload, template and exports. Run `python -m streamlit run app.py` for visual testing.
6. Compare CV and holdout metrics to the existing 11-input results without claiming improvement unless supported. Update dashboard/About with **new verified** numbers. Maintain the honest caveat about the previously consulted holdout and no validation for Bangladeshi students.
7. Commit the new model and reports to this branch (review artifact size); then merge after a successful local verification.

No automatic or silent fallback to the 11-input model is allowed because the assessment form now requires 14 inputs. The app refuses an artifact whose schema does not match.

The first-semester model does not use classroom attendance percentage. Gender is taken from the UCI's binary coding; its use requires documented subgroup/fairness scrutiny before any operational deployment. The input education-level labels should also be checked against the official codebook, especially rare category codes.

The existing Portugal model remains unsuitable for direct Bangladeshi institutional decision-making without external validation.
