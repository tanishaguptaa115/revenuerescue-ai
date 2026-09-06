"""
RevenueRescue AI - ML Inference Layer

Loads the already-trained Recovery and Risk/Fraud models and applies
the SAME feature encoding/order used during training.

This module does NOT train models.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Dict, Any

import joblib
import numpy as np
import pandas as pd

from ml.models.data_preparation import (
    CATEGORICAL_COLUMNS,
    BOOLEAN_COLUMNS,
    RECOVERY_FEATURE_SPEC,
    RISK_FEATURE_SPEC,
    prepare_datasets,
)


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RECOVERY_MODEL_PATH = os.path.join(
    BASE_DIR, "output", "models", "recovery", "recovery_model.joblib"
)
RECOVERY_FEATURES_PATH = os.path.join(
    BASE_DIR, "output", "models", "recovery", "feature_names.json"
)

RISK_MODEL_PATH = os.path.join(
    BASE_DIR, "output", "models", "risk", "risk_model.joblib"
)
RISK_FEATURES_PATH = os.path.join(
    BASE_DIR, "output", "models", "risk", "feature_names.json"
)


def _load_feature_names(path: str) -> list[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Feature artifact not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, list):
        return payload

    if isinstance(payload, dict):
        if "feature_names" in payload:
            return payload["feature_names"]

    raise ValueError(f"Unsupported feature_names.json format: {path}")


def _build_matrix(
    row: pd.DataFrame,
    logical_features: list[str],
    feature_names: list[str],
    encoder,
) -> np.ndarray:
    """
    Reproduce the exact column ordering used by data_preparation.py:

        categorical one-hot columns
        boolean columns
        numeric columns

    The encoder is already fitted on training data.
    """

    cat_cols = [
        c for c in logical_features
        if c in CATEGORICAL_COLUMNS
    ]

    bool_cols = [
        c for c in logical_features
        if c in BOOLEAN_COLUMNS
    ]

    numeric_cols = [
        c for c in logical_features
        if c not in CATEGORICAL_COLUMNS
        and c not in BOOLEAN_COLUMNS
    ]

    blocks = []

    if cat_cols:
        blocks.append(
            encoder.transform(row[cat_cols])
        )

    if bool_cols:
        blocks.append(
            row[bool_cols].astype(int).to_numpy(dtype=float)
        )

    if numeric_cols:
        blocks.append(
            row[numeric_cols].to_numpy(dtype=float)
        )

    if not blocks:
        X = np.empty((len(row), 0))
    else:
        X = np.hstack(blocks)

    if X.shape[1] != len(feature_names):
        raise ValueError(
            "Inference feature mismatch: "
            f"model expects {len(feature_names)} encoded features, "
            f"but preprocessing produced {X.shape[1]}."
        )

    return X


@lru_cache(maxsize=1)
def _runtime():
    """
    Load models and reconstruct the train-fitted preprocessors exactly once.

    prepare_datasets() does not train models. It only rebuilds the
    deterministic train/val/test preprocessing objects and fitted encoders.
    """

    required_paths = [
        RECOVERY_MODEL_PATH,
        RECOVERY_FEATURES_PATH,
        RISK_MODEL_PATH,
        RISK_FEATURES_PATH,
    ]

    missing = [p for p in required_paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "Missing ML artifacts:\n" + "\n".join(missing)
        )

    recovery_model = joblib.load(RECOVERY_MODEL_PATH)
    risk_model = joblib.load(RISK_MODEL_PATH)

    recovery_feature_names = _load_feature_names(
        RECOVERY_FEATURES_PATH
    )

    risk_feature_names = _load_feature_names(
        RISK_FEATURES_PATH
    )

    # Recreate the exact train-only fitted preprocessors.
    prepared = prepare_datasets()

    return {
        "recovery_model": recovery_model,
        "risk_model": risk_model,
        "recovery_encoder": prepared.recovery_preprocessor,
        "risk_encoder": prepared.risk_preprocessor,
        "recovery_feature_names": recovery_feature_names,
        "risk_feature_names": risk_feature_names,
    }


def _validate_probability(value: float, name: str) -> float:
    value = float(value)

    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite.")

    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1.")

    return value


def score_transaction(
    *,
    recovery_features: Dict[str, Any] | None,
    risk_features: Dict[str, Any],
    payment_failed: bool,
) -> Dict[str, Any]:
    """
    Score one transaction.

    Recovery model is evaluated only for failed payments.
    Risk model is always evaluated.

    Feature dictionaries must contain the logical model features expected
    by the trained feature specifications.
    """

    runtime = _runtime()

    # -----------------------------
    # Risk / Fraud score
    # -----------------------------

    risk_row = pd.DataFrame([risk_features])

    missing_risk = [
        feature
        for feature in RISK_FEATURE_SPEC
        if feature not in risk_row.columns
    ]

    if missing_risk:
        raise ValueError(
            f"Missing risk features: {missing_risk}"
        )

    risk_row = risk_row[RISK_FEATURE_SPEC]

    risk_X = _build_matrix(
        risk_row,
        RISK_FEATURE_SPEC,
        runtime["risk_feature_names"],
        runtime["risk_encoder"],
    )

    risk_score = float(
        runtime["risk_model"].predict_proba(risk_X)[0, 1]
    )

    risk_score = _validate_probability(
        risk_score,
        "risk_score",
    )

    # -----------------------------
    # Recovery score
    # -----------------------------

    recovery_score = None

    if payment_failed:
        if recovery_features is None:
            raise ValueError(
                "recovery_features are required when payment_failed=True."
            )

        recovery_row = pd.DataFrame([recovery_features])

        missing_recovery = [
            feature
            for feature in RECOVERY_FEATURE_SPEC
            if feature not in recovery_row.columns
        ]

        if missing_recovery:
            raise ValueError(
                f"Missing recovery features: {missing_recovery}"
            )

        recovery_row = recovery_row[RECOVERY_FEATURE_SPEC]

        recovery_X = _build_matrix(
            recovery_row,
            RECOVERY_FEATURE_SPEC,
            runtime["recovery_feature_names"],
            runtime["recovery_encoder"],
        )

        recovery_score = float(
            runtime["recovery_model"].predict_proba(recovery_X)[0, 1]
        )

        recovery_score = _validate_probability(
            recovery_score,
            "recovery_score",
        )

    return {
        "risk_score": risk_score,
        "recovery_score": recovery_score,
        "models": {
            "risk": "risk_model.joblib",
            "recovery": (
                "recovery_model.joblib"
                if payment_failed
                else None
            ),
        },
    }


def warmup_models() -> Dict[str, Any]:
    """
    Load and validate ML artifacts at application startup.
    """

    runtime = _runtime()

    return {
        "status": "ready",
        "recovery_features": len(
            runtime["recovery_feature_names"]
        ),
        "risk_features": len(
            runtime["risk_feature_names"]
        ),
    }
