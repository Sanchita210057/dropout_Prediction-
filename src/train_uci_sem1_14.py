"""Retrain the single active 14-input UCI Semester-1 Random Forest.

Run from the repository root:
    python -m src.train_uci_sem1_14

This replaces the active model artifact only when the user explicitly runs it.
It does NOT establish validation on Bangladeshi students.
"""
from __future__ import annotations

import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.model_selection import (
    RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict,
    cross_validate, train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.uci_sem1_features import (
    FINAL_FEATURES, NORMALIZED_GPA_FEATURE, RAW_INPUT_FEATURES,
    SemesterFeatureBuilder, normalize_previous_grade,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/student_dropout.csv"
REPORT = ROOT / "reports/uci_sem1/training_results_14_input.json"
MODEL = ROOT / "models/uci_sem1/dropout_early_warning_model.joblib"
METADATA = ROOT / "models/uci_sem1/model_metadata.json"
CATEGORICAL = [
    "Mother's occupation", "Father's occupation",
    "Mother's qualification", "Father's qualification", "Gender",
    "Scholarship holder", "Debtor", "Tuition fees up to date",
]
SCORING = {
    "roc_auc": "roc_auc", "average_precision": "average_precision",
    "precision": "precision", "recall": "recall", "f1": "f1",
    "balanced_accuracy": "balanced_accuracy",
}

def training_frame():
    data = pd.read_csv(DATA, sep=";")
    data.columns = data.columns.str.strip()
    cohort = data.loc[data["Target"].isin(("Graduate", "Dropout"))].copy()
    cohort[NORMALIZED_GPA_FEATURE] = normalize_previous_grade(
        cohort["Previous qualification (grade)"]
    )
    labels = cohort["Target"].map({"Graduate": 0, "Dropout": 1}).astype(int)
    return cohort, labels

def make_pipeline():
    numerical = [f for f in FINAL_FEATURES if f not in CATEGORICAL]
    preprocessor = ColumnTransformer([
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]), CATEGORICAL),
        ("numerical", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numerical),
    ])
    model = RandomForestClassifier(
        n_estimators=500, max_depth=12, min_samples_leaf=5,
        class_weight="balanced_subsample", n_jobs=-1, random_state=42,
    )
    return Pipeline([
        ("derive", SemesterFeatureBuilder()),
        ("preprocess", preprocessor),
        ("model", model),
    ])

def performance(y, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predictions, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predictions)),
        "precision": float(precision_score(y, predictions, zero_division=0)),
        "recall": float(recall_score(y, predictions, zero_division=0)),
        "f1": float(f1_score(y, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "average_precision": float(average_precision_score(y, probabilities)),
        "brier_score": float(brier_score_loss(y, probabilities)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }

def main():
    cohort, labels = training_frame()
    if len(RAW_INPUT_FEATURES) != 14 or len(FINAL_FEATURES) != 14:
        raise RuntimeError("Training requires exactly 14 raw and derived features.")
    xtrain, xtest, ytrain, ytest = train_test_split(
        cohort[RAW_INPUT_FEATURES], labels,
        test_size=0.20, stratify=labels, random_state=42,
    )
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)
    scores = cross_validate(make_pipeline(), xtrain, ytrain,
                            cv=cv, scoring=SCORING, n_jobs=1)
    summary = {
        key: {
            "mean": float(np.mean(scores["test_" + key])),
            "std": float(np.std(scores["test_" + key], ddof=1)),
        } for key in SCORING
    }
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = cross_val_predict(
        make_pipeline(), xtrain, ytrain, cv=folds,
        method="predict_proba", n_jobs=1,
    )[:, 1]
    candidates = [performance(ytrain, oof, round(float(t), 2))
                  for t in np.arange(0.20, 0.71, 0.02)]
    eligible = [r for r in candidates if r["recall"] >= .78 and r["precision"] >= .60]
    selected = max(eligible or candidates,
                   key=lambda r: (r["f1"], r["balanced_accuracy"]))
    threshold = selected["threshold"]

    fitted = make_pipeline()
    fitted.fit(xtrain, ytrain)
    probabilities = fitted.predict_proba(xtest)[:, list(fitted.classes_).index(1)]
    holdout = performance(ytest, probabilities, threshold)

    result = {
        "model": "Random Forest", "raw_input_features": RAW_INPUT_FEATURES,
        "derived_features": FINAL_FEATURES, "training_rows": int(len(ytrain)),
        "holdout_rows": int(len(ytest)), "cv": summary,
        "selected_threshold_training_metrics": selected,
        "holdout_metrics": holdout,
        "holdout_note": "Previously consulted by earlier project iterations: "
                        "this is not a pristine external validation set.",
        "scope_note": "Portugal university students only; not validated for Bangladesh.",
        "versions": {"python": platform.python_version(),
                     "scikit_learn": sklearn.__version__},
    }
    metadata = {
        "title": "Student Dropout Early Warning System Using Machine Learning",
        "model": "Random Forest", "prediction_point": "End of first semester",
        "threshold": threshold, "features": FINAL_FEATURES,
        "raw_input_features": RAW_INPUT_FEATURES,
        "normalized_gpa_range": {
            "min": 2.5, "max": 5.0,
            "formula": "Previous qualification (grade) / 190 * 5",
        },
        "target_mapping": {"Graduate": 0, "Dropout": 1},
        "excluded_outcome": "Enrolled", "versions": result["versions"],
        "limitation": result["scope_note"],
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    # Only write the active artifacts after all training and evaluation succeeds.
    joblib.dump({
        "pipeline": fitted, "threshold": threshold,
        "features": FINAL_FEATURES, "raw_input_features": RAW_INPUT_FEATURES,
    }, MODEL)
    REPORT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    METADATA.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"threshold": threshold, "cv": summary,
                      "holdout_metrics": holdout}, indent=2))

if __name__ == "__main__":
    main()
