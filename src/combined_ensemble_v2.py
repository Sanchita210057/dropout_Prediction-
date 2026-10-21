"""Inference architecture for the experimental UCI + Bangladesh stacker."""
from __future__ import annotations

import numpy as np
import pandas as pd


UCI_NUMERIC_FEATURES = (
    "hsc_equivalent_gpa", "age_at_enrollment", "courses_enrolled",
    "evaluations_completed", "courses_passed", "semester_gpa_native",
)
UCI_CATEGORICAL_FEATURES = (
    "gender_bangladesh", "scholarship_holder", "debtor", "tuition_fees_up_to_date",
)
UCI_FOUNDATION_FEATURES = (*UCI_NUMERIC_FEATURES, *UCI_CATEGORICAL_FEATURES)


def uci_harmonized_frame(bangladesh_frame: pd.DataFrame) -> pd.DataFrame:
    """Project the 18 Bangladesh inputs onto ten defensibly shared UCI fields.

    Numeric grade ranges are normalized for model compatibility only. A
    Bangladeshi category unsupported by UCI stays unchanged and is treated as
    an unseen category by the foundation encoder rather than fabricated.
    """
    data = {field: pd.to_numeric(bangladesh_frame[field], errors="coerce") for field in UCI_NUMERIC_FEATURES}
    data.update({field: bangladesh_frame[field].astype(object) for field in UCI_CATEGORICAL_FEATURES})
    return pd.DataFrame(data, index=bangladesh_frame.index).loc[:, UCI_FOUNDATION_FEATURES]


class StackedEnsemblePipeline:
    """Run both component models and fuse their OOF-trained score signals."""

    def __init__(self, bangladesh_pipeline, uci_pipeline, fusion_model, features):
        self.bangladesh_pipeline = bangladesh_pipeline
        self.uci_pipeline = uci_pipeline
        self.fusion_model = fusion_model
        self.feature_names_in_ = np.asarray(list(features), dtype=object)
        self.classes_ = np.asarray([0, 1])

    def component_probabilities(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        if list(frame.columns) != list(self.feature_names_in_):
            raise ValueError("Stacked ensemble requires the exact 18-input Bangladesh schema.")
        local = self.bangladesh_pipeline.predict_proba(frame.loc[:, self.feature_names_in_])[:, 1]
        uci = self.uci_pipeline.predict_proba(uci_harmonized_frame(frame))[:, 1]
        return local, uci

    def fusion_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        local, uci = self.component_probabilities(frame)
        return pd.DataFrame({"bangladesh_signal": local, "uci_academic_signal": uci}, index=frame.index)

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        if list(frame.columns) != list(self.feature_names_in_):
            raise ValueError("Stacked ensemble requires the exact 18-input Bangladesh schema.")
        probability = self.fusion_model.predict_proba(self.fusion_features(frame))[:, 1]
        return np.column_stack([1.0 - probability, probability])


class WeightedEnsemblePipeline:
    """Calibrated fixed-weight blend using both fitted component models."""

    def __init__(self, bangladesh_pipeline, uci_pipeline, blend_weight, calibrator, features):
        self.bangladesh_pipeline = bangladesh_pipeline
        self.uci_pipeline = uci_pipeline
        self.blend_weight = float(blend_weight)
        self.blend_weight_uci = self.blend_weight
        self.calibrator = calibrator
        self.feature_names_in_ = np.asarray(list(features), dtype=object)
        self.classes_ = np.asarray([0, 1])

    def component_probabilities(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        if list(frame.columns) != list(self.feature_names_in_):
            raise ValueError("Weighted ensemble requires the exact 18-input Bangladesh schema.")
        local = self.bangladesh_pipeline.predict_proba(frame.loc[:, self.feature_names_in_])[:, 1]
        uci = self.uci_pipeline.predict_proba(uci_harmonized_frame(frame))[:, 1]
        return local, uci

    def blended_probability(self, frame: pd.DataFrame) -> np.ndarray:
        local, uci = self.component_probabilities(frame)
        return (1.0 - self.blend_weight) * local + self.blend_weight * uci

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        if list(frame.columns) != list(self.feature_names_in_):
            raise ValueError("Weighted ensemble requires the exact 18-input Bangladesh schema.")
        blended = np.clip(self.blended_probability(frame), 1e-6, 1.0 - 1e-6)
        calibrated = self.calibrator.predict_proba(blended.reshape(-1, 1))[:, 1]
        return np.column_stack([1.0 - calibrated, calibrated])
