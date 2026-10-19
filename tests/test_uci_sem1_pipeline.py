import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.train_uci_sem1_15 import DATA, METADATA, MODEL, training_frame
from src.uci_sem1_features import (FINAL_FEATURES, NORMALIZED_GPA_FEATURE,
    RAW_INPUT_FEATURES, SemesterFeatureBuilder, normalize_previous_grade)

TARGET = "Target"
FORBIDDEN = {"Admission grade", "Target", "Application mode", "Application order",
             "Course", "Nacionality", "Unemployment rate", "Inflation rate", "GDP"}


def test_dataset_and_resolved_target_cohort():
    raw = pd.read_csv(DATA, sep=";")
    raw.columns = raw.columns.str.strip()
    cohort, labels = training_frame()
    assert raw.shape == (4424, 37)
    assert raw[TARGET].value_counts().to_dict() == {
        "Graduate": 2209, "Dropout": 1421, "Enrolled": 794,
    }
    assert len(cohort) == 3630 and int(labels.sum()) == 1421
    assert set(labels.unique()) == {0, 1}


def test_exact_15_raw_inputs_and_no_leakage_fields():
    assert len(RAW_INPUT_FEATURES) == len(FINAL_FEATURES) == 15
    assert RAW_INPUT_FEATURES[:4] == [
        NORMALIZED_GPA_FEATURE, "Age at enrollment", "Mother's occupation", "Father's occupation",
    ]
    assert RAW_INPUT_FEATURES[4:8] == [
        "Mother's qualification", "Father's qualification", "Gender", "Marital status",
    ]
    assert not (set(RAW_INPUT_FEATURES) | set(FINAL_FEATURES)) & FORBIDDEN
    assert all(not feature.startswith("Curricular units 2nd sem")
               for feature in RAW_INPUT_FEATURES + FINAL_FEATURES)
    assert "Curricular units 1st sem (approved)" in RAW_INPUT_FEATURES
    assert "Curricular units 1st sem (approved)" not in FINAL_FEATURES
    assert "Semester pass rate" in FINAL_FEATURES


def test_derived_pass_rate_and_zero_denominator():
    frame = pd.DataFrame({
        "Curricular units 1st sem (enrolled)": [4, 0],
        "Curricular units 1st sem (approved)": [3, 0],
        **{feature: [1, 1] for feature in RAW_INPUT_FEATURES
           if feature not in {"Curricular units 1st sem (enrolled)",
                              "Curricular units 1st sem (approved)"}},
    })
    result = SemesterFeatureBuilder().transform(frame)
    assert result["Semester pass rate"].tolist() == [.75, 0.0]
    assert result.columns.tolist() == FINAL_FEATURES
    assert np.isfinite(result["Semester pass rate"]).all()


def test_normalized_previous_gpa_examples():
    assert normalize_previous_grade(pd.Series([190, 152, 95])).tolist() == [5.0, 4.0, 2.5]


def test_saved_15_input_artifact_and_metadata_round_trip():
    artifact = joblib.load(MODEL)
    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    cohort, _ = training_frame()
    row = cohort[RAW_INPUT_FEATURES].iloc[:2]
    probabilities = artifact["pipeline"].predict_proba(row)[:, 1]
    assert isinstance(artifact["pipeline"], Pipeline)
    assert np.all((probabilities >= 0) & (probabilities <= 1))
    assert artifact["threshold"] == metadata["threshold"]
    assert 0 < artifact["threshold"] < 1
    assert artifact["raw_input_features"] == RAW_INPUT_FEATURES
    assert metadata["raw_input_features"] == RAW_INPUT_FEATURES
    assert metadata["normalized_gpa_range"] == {
        "min": 2.5, "max": 5.0,
        "formula": "Previous qualification (grade) / 190 * 5",
    }
    assert artifact["features"] == FINAL_FEATURES


def test_training_results_record_training_only_threshold_and_holdout_once():
    result_path = Path(MODEL).parents[2] / "reports" / "uci_sem1" / "training_results_15_input.json"
    results = json.loads(result_path.read_text(encoding="utf-8"))
    assert results["threshold_selection"]["source"].startswith("Training-only")
    holdout = results["holdout_metrics"]
    assert sum(holdout[key] for key in ("tn", "fp", "fn", "tp")) == 726


def test_all_added_features_are_in_preprocessor_schema():
    artifact = joblib.load(MODEL)
    preprocessor = artifact["pipeline"].named_steps["preprocess"]
    assert preprocessor.feature_names_in_.tolist() == FINAL_FEATURES
    categorical = preprocessor.named_transformers_["categorical"]
    fitted_categories = categorical.named_steps["encoder"].categories_
    cat_fields = preprocessor.transformers_[0][2]
    categories_by_feature = dict(zip(cat_fields, fitted_categories))
    for feature in ["Mother's qualification", "Father's qualification", "Gender", "Marital status"]:
        assert feature in categories_by_feature
        assert len(categories_by_feature[feature]) > 1
