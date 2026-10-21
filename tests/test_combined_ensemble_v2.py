"""Tests for the expanded UCI foundation and OOF-fitted ensemble candidate."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.bangladesh_collection import PREDICTOR_FIELDS, load_bangladesh_training_data
from src.combined_ensemble_v2 import UCI_FOUNDATION_FEATURES, uci_harmonized_frame
from src.train_combined_ensemble_18 import load_uci
from src.training_data_paths import bangladesh_dataset_path
from src.train_combined_ensemble_18 import FEATURE_COMPATIBILITY
from src.bangladesh_collection import synthetic_example_rows

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "combined_uci_bangladesh_18_v2"
RESULTS_PATH = ROOT / "reports" / "combined_uci_bangladesh_18_v2" / "training_results.json"


class FixedProbabilityModel:
    classes_ = np.asarray([0, 1])

    def __init__(self, probability: float):
        self.probability = probability

    def predict_proba(self, frame):
        return np.tile([1.0 - self.probability, self.probability], (len(frame), 1))


def test_uci_foundation_uses_only_documented_harmonizable_features():
    result = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    artifact = joblib.load(MODEL_DIR / "weighted_ensemble_dropout_status_18.joblib")
    assert result["training_counts"]["uci_resolved_records"] == 3630
    assert result["uci_audit"]["excluded_enrolled_rows"] == 794
    assert len(FEATURE_COMPATIBILITY) == 15
    assert sum(item["decision"] == "Use" for item in FEATURE_COMPATIBILITY) == 10
    assert all(item["decision"] == "Exclude" for item in FEATURE_COMPATIBILITY if item["UCI predictor"] in {"Mother's occupation", "Father's occupation", "Mother's qualification", "Father's qualification", "Marital status"})
    assert list(artifact["pipeline"].uci_pipeline.feature_names_in_) == list(UCI_FOUNDATION_FEATURES)


def test_both_local_training_sources_are_readable_without_retraining():
    uci_X, uci_y, uci_audit = load_uci()
    bd_X, bd_y, _, bd_audit = load_bangladesh_training_data(bangladesh_dataset_path())
    assert len(uci_X) == len(uci_y) == uci_audit["resolved_rows_used"] == 3630
    assert len(bd_X) == len(bd_y) == bd_audit["records"] == 251
    assert list(bd_X.columns) == list(PREDICTOR_FIELDS)


def test_harmonizer_preserves_unmatched_categories_instead_of_coercing_them():
    frame = synthetic_example_rows().loc[:, list(PREDICTOR_FIELDS)].copy()
    frame.loc[0, "gender_bangladesh"] = "Another identity"
    output = uci_harmonized_frame(frame)
    assert output.loc[0, "gender_bangladesh"] == "Another identity"
    assert output.shape == (len(frame), 10)


def test_active_weighted_artifact_accepts_exactly_18_inputs_and_both_signals():
    weighted = joblib.load(MODEL_DIR / "weighted_ensemble_dropout_status_18.joblib")
    values = synthetic_example_rows().iloc[[0]].loc[:, list(PREDICTOR_FIELDS)]
    pipeline = weighted["pipeline"]
    assert weighted["input_feature_count"] == 18
    assert list(pipeline.feature_names_in_) == list(PREDICTOR_FIELDS)
    local, uci = pipeline.component_probabilities(values)
    assert local.shape == uci.shape == (1,)
    assert np.isfinite(local[0]) and np.isfinite(uci[0])
    probability = pipeline.predict_proba(values)[0, 1]
    assert 0 <= probability <= 1
    assert weighted["pipeline"].blend_weight_uci == .05


def test_final_probability_depends_on_uci_signal_for_each_ensemble():
    weighted = joblib.load(MODEL_DIR / "weighted_ensemble_dropout_status_18.joblib")["pipeline"]
    frame = synthetic_example_rows().iloc[[0]].loc[:, list(PREDICTOR_FIELDS)]
    real_uci = weighted.uci_pipeline
    try:
        weighted.uci_pipeline = FixedProbabilityModel(.01)
        lower = weighted.predict_proba(frame)[0, 1]
        weighted.uci_pipeline = FixedProbabilityModel(.99)
        upper = weighted.predict_proba(frame)[0, 1]
    finally:
        weighted.uci_pipeline = real_uci
    assert upper != lower


def test_metadata_documents_cohorts_oof_fusion_threshold_and_distinct_targets():
    result = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    metadata = json.loads((MODEL_DIR / "model_metadata.json").read_text(encoding="utf-8"))
    assert result["training_counts"]["uci_resolved_records"] == 3630
    assert result["training_counts"]["bangladesh_records"] == 251
    assert "OOF" in result["stacker_fit_procedure"]
    assert "reported current status" in result["warning"].lower()
    assert metadata["active_model"] == "weighted_ensemble_dropout_status_18.joblib"
    assert result["threshold"] == metadata["threshold"]
