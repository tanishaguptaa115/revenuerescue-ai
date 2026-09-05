"""
risk_model.py

Trains and evaluates the RevenueRescue AI Risk/Fraud Model, which predicts
risk_label (fraudulent=1 / legitimate=0) for ALL transactions.

This module consumes the Risk train/val/test matrices EXACTLY as returned
by data_preparation.prepare_datasets() - it does not reconstruct the
split, does not perform any new random split, and does not touch
data_preparation.py, any generator module, main_generator.py, or
recovery_model.py. It is fully self-contained (duplicates the small
evaluation/interpretability helpers it needs rather than importing from
recovery_model.py) so the two model modules stay independent.

Two models are trained:
  1. Baseline: LogisticRegression, on an imputed + scaled copy of the
     prepared feature matrix (Logistic Regression cannot consume NaN).
  2. Main model: HistGradientBoostingClassifier, on the RAW prepared
     feature matrix, with days_since_last_successful_payment's missing
     values passed through untouched - HGB natively learns a split
     direction for missing values, so no imputation is applied or wanted.

Fraud prevalence here is roughly 4-6%, so PR-AUC is treated as the
PRIMARY discrimination metric throughout (never accuracy). The
validation set is used to compare the two models and to choose a
decision threshold; the test set is touched exactly once, at the very
end, to report final metrics for the selected model only.
"""

import json
import os
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data_preparation import FORBIDDEN_FEATURE_COLUMNS, DataPreparationResult, prepare_datasets

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RANDOM_SEED: int = 42
OUTPUT_DIR: str = "output/models/risk"

# Threshold at/below which the minority class is considered imbalanced
# enough to justify class_weight="balanced" for the baseline model. Fraud
# prevalence here (~4-6%) is expected to fall well under this, unlike the
# Recovery Model's much more balanced target.
IMBALANCE_THRESHOLD: float = 0.20

# Candidate thresholds explicitly requested for the risk-sensitive
# operating-point discussion, plus a finer grid used to actually select
# the best-F1 threshold.
REPORT_THRESHOLDS: List[float] = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
FINE_THRESHOLD_GRID: np.ndarray = np.round(np.arange(0.02, 0.96, 0.01), 2)

# Above this validation PR-AUC, print an explicit honesty/artifact-check
# note - PR-AUC (not ROC-AUC) is the right metric to gate this check on,
# since ROC-AUC is inflated by definition under heavy class imbalance.
SUSPICIOUSLY_HIGH_PR_AUC: float = 0.80

# Number of leakage-safe internal CV folds used by CalibratedClassifierCV
# to fit each calibration mapping: for each fold, a clone of the base HGB
# is trained on the OTHER folds and calibrated on the held-out fold, so
# the calibration mapping is always fit on out-of-fold predictions -
# TRAIN data only, never validation or test.
CALIBRATION_CV_FOLDS: int = 5

# Two validation-set PR-AUC scores within this absolute margin of each
# other are treated as "effectively tied" for model-selection purposes,
# at which point the candidate with the lower (better) Brier score wins
# instead. This is what turns "prefer trustworthy probabilities" into a
# concrete, defensible rule rather than a vague preference.
PR_AUC_TIE_EPSILON: float = 0.01


# ---------------------------------------------------------------------------
# Pre-training validation
# ---------------------------------------------------------------------------

def _run_pretraining_checks(result: DataPreparationResult) -> None:
    """
    Structural/statistical checks required BEFORE any model is fit.
    Raises on hard failures.
    """
    train, val, test = result.risk_train, result.risk_val, result.risk_test

    print("--- Pre-training checks ---")
    print(f"  risk_train.X shape: {train.X.shape}")
    print(f"  risk_val.X shape:   {val.X.shape}")
    print(f"  risk_test.X shape:  {test.X.shape}")

    assert set(np.unique(train.y)) == {0, 1}, "risk_train must contain both classes."
    for name, split in [("val", val), ("test", test)]:
        classes = set(np.unique(split.y))
        if classes != {0, 1}:
            warnings.warn(f"risk_{name} does not contain both classes: {classes}", stacklevel=2)
    print(f"  train classes: {sorted(set(np.unique(train.y)))}")
    print(f"  val classes:   {sorted(set(np.unique(val.y)))}")
    print(f"  test classes:  {sorted(set(np.unique(test.y)))}")

    for name, split in [("train", train), ("val", val), ("test", test)]:
        assert set(np.unique(split.y)).issubset({0, 1}), f"risk_{name}.y is not binary."

    train_c, val_c, test_c = set(train.customer_ids), set(val.customer_ids), set(test.customer_ids)
    assert not (train_c & val_c), "Customer overlap between risk train and val."
    assert not (train_c & test_c), "Customer overlap between risk train and test."
    assert not (val_c & test_c), "Customer overlap between risk val and test."
    print(f"  customer overlap (train/val/test): "
          f"{len(train_c & val_c)}/{len(train_c & test_c)}/{len(val_c & test_c)} (all must be 0)")

    feature_names = set(train.feature_names)
    forbidden_present = feature_names & FORBIDDEN_FEATURE_COLUMNS
    assert not forbidden_present, f"Forbidden features present: {forbidden_present}"
    assert "recovery_label" not in feature_names and "risk_label" not in feature_names, (
        "Target columns leaked into features."
    )
    print(f"  forbidden features present: {sorted(forbidden_present)} (must be empty)")
    print(f"  recovery_label present in risk features: {'recovery_label' in feature_names} (must be False)")
    print("  ALL PRE-TRAINING CHECKS PASSED\n")


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def _decide_class_weight(y_train: np.ndarray) -> Optional[str]:
    """
    Data-driven decision: only use class_weight="balanced" if the
    minority class fraction in the TRAINING split falls at or below
    IMBALANCE_THRESHOLD. Printed and returned so the choice is auditable.
    """
    minority_fraction = min(y_train.mean(), 1 - y_train.mean())
    print(f"  Training minority-class (fraud) fraction: {minority_fraction:.4f} "
          f"(imbalance threshold: {IMBALANCE_THRESHOLD})")
    if minority_fraction <= IMBALANCE_THRESHOLD:
        print("  -> class_weight='balanced' IS justified: fraud is a small minority "
              "of training rows, and an unweighted LogisticRegression would be "
              "dominated by the legitimate class, pushing nearly all predicted "
              "probabilities toward 0 regardless of signal.")
        return "balanced"
    print("  -> class_weight='balanced' is NOT used - training data is not "
          "meaningfully imbalanced.")
    return None


def build_baseline_model(y_train: np.ndarray) -> Pipeline:
    """
    LogisticRegression baseline. Cannot consume NaN, so this pipeline
    imputes (median, fit on train only) and scales (fit on train only)
    before the classifier. Both steps live inside the Pipeline so that
    fitting the Pipeline on train data alone guarantees no val/test
    leakage into either the imputer or the scaler.

    No SMOTE or other resampling is used: with ~700 fraud rows in
    training, synthetic oversampling risks manufacturing unrealistic
    interpolated fraud patterns for a first implementation, and
    class_weight="balanced" already gives the minority class proportional
    influence without touching the data distribution itself.
    """
    class_weight = _decide_class_weight(y_train)
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=1000, class_weight=class_weight, random_state=RANDOM_SEED,
            )),
        ]
    )


def build_main_model() -> HistGradientBoostingClassifier:
    """
    HistGradientBoostingClassifier main model. Consumes the RAW prepared
    matrix directly - NaN in days_since_last_successful_payment is left
    exactly as data_preparation.py produced it; HGB natively learns which
    branch a missing value should take at each split.

    class_weight="balanced" is supported natively by HGB and is applied
    here for the same reason as the baseline: fraud is a small minority
    of training rows. min_samples_leaf is kept relatively high given how
    few positive examples exist, to reduce the chance of the model
    carving out tiny leaves that memorize individual fraud rows rather
    than learning generalizable structure.
    """
    return HistGradientBoostingClassifier(
        loss="log_loss",
        max_iter=300,
        learning_rate=0.05,
        max_depth=6,
        min_samples_leaf=30,
        l2_regularization=1.0,
        class_weight="balanced",
        early_stopping=False,  # we do our own validation-based comparison
        random_state=RANDOM_SEED,
    )


def build_calibrated_model(method: str) -> CalibratedClassifierCV:
    """
    Build a leakage-safe probability-calibrated version of the main HGB
    model. CalibratedClassifierCV(cv=K) never fits on the full training
    set directly: internally, for each of the K stratified folds, it
    trains a FRESH clone of the base estimator on the other K-1 folds and
    fits the calibration mapping (sigmoid/Platt or isotonic) on that
    fold's held-out predictions - i.e. genuinely out-of-fold training
    predictions, never in-sample predictions from an already-fitted
    model. At inference time the K fold-specific (base model, calibrator)
    pairs are averaged.

    A FRESH, unfitted build_main_model() must be passed in here (never
    the already-fitted `main_model` instance) - CalibratedClassifierCV
    with an integer `cv` clones and refits the estimator itself.

    Args:
        method: "sigmoid" (Platt scaling) or "isotonic".
    """
    cv = StratifiedKFold(n_splits=CALIBRATION_CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    return CalibratedClassifierCV(estimator=build_main_model(), method=method, cv=cv)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    roc_auc: Optional[float]
    pr_auc: Optional[float]
    brier: float
    prob_stats: Dict
    prevalence: float
    n: int


def _prob_distribution_stats(y_prob: np.ndarray) -> Dict:
    return {
        "min": float(np.min(y_prob)),
        "max": float(np.max(y_prob)),
        "mean": float(np.mean(y_prob)),
        "std": float(np.std(y_prob)),
        "p25": float(np.percentile(y_prob, 25)),
        "p50": float(np.percentile(y_prob, 50)),
        "p75": float(np.percentile(y_prob, 75)),
        "p95": float(np.percentile(y_prob, 95)),
        "p99": float(np.percentile(y_prob, 99)),
    }


def evaluate_probabilistic(y_true: np.ndarray, y_prob: np.ndarray) -> EvalResult:
    """Threshold-independent metrics: ROC-AUC, PR-AUC, Brier, prob stats, prevalence."""
    has_both_classes = len(set(np.unique(y_true))) == 2
    roc_auc = float(roc_auc_score(y_true, y_prob)) if has_both_classes else None
    pr_auc = float(average_precision_score(y_true, y_prob)) if has_both_classes else None
    brier = float(brier_score_loss(y_true, y_prob))
    return EvalResult(
        roc_auc=roc_auc, pr_auc=pr_auc, brier=brier,
        prob_stats=_prob_distribution_stats(y_prob),
        prevalence=float(np.mean(y_true)), n=len(y_true),
    )


def evaluate_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict:
    """Threshold-dependent metrics: precision, recall, F1, confusion matrix."""
    y_pred = (y_prob >= threshold).astype(int)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "threshold": threshold,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": cm.tolist(),  # [[TN, FP], [FN, TP]]
    }


def print_threshold_table(y_true: np.ndarray, y_prob: np.ndarray, thresholds: List[float]) -> None:
    print(f"  {'threshold':>10} {'precision':>10} {'recall':>10} {'f1':>10}")
    for t in thresholds:
        m = evaluate_at_threshold(y_true, y_prob, t)
        print(f"  {t:>10.2f} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f}")


def select_best_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray, grid: np.ndarray) -> Tuple[float, Dict]:
    """Grid-search the threshold maximizing F1 on the given (validation) data."""
    best_t = 0.5
    best_metrics = None
    best_f1 = -1.0
    for t in grid:
        m = evaluate_at_threshold(y_true, y_prob, float(t))
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_t = float(t)
            best_metrics = m
    return best_t, best_metrics


def select_candidate_by_pr_auc_then_brier(
    candidates: Dict[str, "EvalResult"], tie_epsilon: float = PR_AUC_TIE_EPSILON
) -> Tuple[str, str]:
    """
    Defensible, ROC-AUC-blind selection rule across an arbitrary set of
    named candidates: PR-AUC is primary; any candidate within
    `tie_epsilon` (absolute) of the best PR-AUC is treated as tied, and
    among those the one with the LOWEST Brier score wins. ROC-AUC is
    never consulted here.

    Returns:
        (selected_name, rationale_string)
    """
    best_pr_auc = max(c.pr_auc for c in candidates.values())
    tied_names = [
        name for name, c in candidates.items() if (best_pr_auc - c.pr_auc) <= tie_epsilon
    ]
    selected_name = min(tied_names, key=lambda name: candidates[name].brier)

    if len(tied_names) == 1:
        rationale = (
            f"'{selected_name}' has the single best validation PR-AUC "
            f"({candidates[selected_name].pr_auc:.4f}), no tie-break needed."
        )
    else:
        rationale = (
            f"Candidates {tied_names} are within {tie_epsilon} PR-AUC of the best "
            f"score ({best_pr_auc:.4f}) - treated as effectively tied. Among them, "
            f"'{selected_name}' has the lowest (best) Brier score "
            f"({candidates[selected_name].brier:.4f}), so it is selected for "
            f"trustworthy probabilities without sacrificing discrimination."
        )
    return selected_name, rationale



def reliability_table(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> List[Dict]:
    """Numeric calibration/reliability table: mean predicted prob vs. observed
    fraction of positives, per bin."""
    try:
        frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="quantile")
    except ValueError:
        frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
    return [
        {"mean_predicted": float(mp), "observed_fraction_positive": float(fp)}
        for mp, fp in zip(mean_pred, frac_pos)
    ]


def save_reliability_plot(y_true: np.ndarray, y_prob: np.ndarray, path: str, title: str) -> None:
    try:
        frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="quantile")
    except ValueError:
        frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="uniform")
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfectly calibrated")
    ax.plot(mean_pred, frac_pos, marker="o", label="model")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed fraction of positives")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_pr_curve_plot(y_true: np.ndarray, y_prob: np.ndarray, path: str, title: str) -> None:
    """Precision-recall curve - the more informative curve than ROC under
    heavy class imbalance, included alongside the required reliability plot."""
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(recall, precision, label=f"PR-AUC = {pr_auc:.4f}")
    ax.axhline(y=float(np.mean(y_true)), linestyle="--", color="gray", label="no-skill baseline")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Interpretability
# ---------------------------------------------------------------------------

def baseline_coefficient_importance(pipeline: Pipeline, feature_names: List[str], top_n: int = 10) -> List[Dict]:
    """
    Coefficient-magnitude importance for the LogisticRegression baseline.
    Coefficients are read from the fitted classifier AFTER StandardScaler,
    so magnitudes are already on a comparable scale across features.
    """
    coefs = pipeline.named_steps["clf"].coef_[0]
    order = np.argsort(-np.abs(coefs))[:top_n]
    return [
        {"feature": feature_names[i], "coefficient": float(coefs[i]), "abs_coefficient": float(abs(coefs[i]))}
        for i in order
    ]


def main_model_permutation_importance(
    model: HistGradientBoostingClassifier,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: List[str],
    top_n: int = 10,
) -> List[Dict]:
    """
    Permutation importance for the HistGradientBoosting main model,
    computed on the VALIDATION set only (never test) - measures the drop
    in average precision when each feature is independently shuffled.
    Average precision (= PR-AUC) is used as the scoring function rather
    than accuracy or ROC-AUC, consistent with PR-AUC being the primary
    metric for this imbalanced target.
    """
    result = permutation_importance(
        model, X_val, y_val, scoring="average_precision",
        n_repeats=20, random_state=RANDOM_SEED, n_jobs=-1,
    )
    order = np.argsort(-result.importances_mean)[:top_n]
    return [
        {
            "feature": feature_names[i],
            "importance_mean": float(result.importances_mean[i]),
            "importance_std": float(result.importances_std[i]),
        }
        for i in order
    ]


# ---------------------------------------------------------------------------
# Post-training validation
# ---------------------------------------------------------------------------

def _run_posttraining_checks(y_prob_val: np.ndarray, y_prob_test: np.ndarray) -> None:
    print("--- Post-training checks ---")
    for name, probs in [("validation", y_prob_val), ("test", y_prob_test)]:
        assert np.all(np.isfinite(probs)), f"Non-finite predicted probabilities found on {name}."
        assert np.all((probs >= 0.0) & (probs <= 1.0)), f"Predicted probabilities out of [0,1] on {name}."
        print(f"  {name}: probabilities finite and within [0,1] - OK")
    print("  Test set was touched exactly once, after model+threshold selection on validation - OK")
    print("  Selected model and threshold were both derived from validation data only - OK\n")


# ---------------------------------------------------------------------------
# Artifact saving
# ---------------------------------------------------------------------------

def save_artifacts(
    output_dir: str,
    final_model,
    threshold: float,
    feature_names: List[str],
    metrics: Dict,
) -> Dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join(output_dir, "risk_model.joblib")
    joblib.dump(final_model, model_path)

    threshold_path = os.path.join(output_dir, "threshold.json")
    with open(threshold_path, "w") as f:
        json.dump({"selected_threshold": threshold}, f, indent=2)

    feature_names_path = os.path.join(output_dir, "feature_names.json")
    with open(feature_names_path, "w") as f:
        json.dump({"feature_names": feature_names}, f, indent=2)

    metrics_path = os.path.join(output_dir, "evaluation_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    return {
        "model": model_path,
        "threshold": threshold_path,
        "feature_names": feature_names_path,
        "metrics": metrics_path,
    }


# ---------------------------------------------------------------------------
# Smoke test / main training run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    np.random.seed(RANDOM_SEED)  # only affects this script's own local scope

    print("=" * 70)
    print("RISK / FRAUD MODEL - TRAINING AND EVALUATION")
    print("=" * 70)

    result = prepare_datasets()
    train, val, test = result.risk_train, result.risk_val, result.risk_test
    feature_names = train.feature_names

    _run_pretraining_checks(result)

    print("--- Class imbalance summary ---")
    print(f"  Fraud prevalence - train: {train.y.mean():.4f}  "
          f"val: {val.y.mean():.4f}  test: {test.y.mean():.4f}")
    print("  PR-AUC will be treated as the PRIMARY discrimination metric "
          "throughout (never accuracy) because of this imbalance.\n")

    # --- Train both models on TRAIN ONLY -----------------------------------
    print("--- Training baseline (LogisticRegression) ---")
    baseline = build_baseline_model(train.y)
    baseline.fit(train.X, train.y)
    print("  fit complete\n")

    print("--- Training main model (HistGradientBoostingClassifier) ---")
    print("  class_weight='balanced' used (same imbalance justification as baseline).")
    main_model = build_main_model()
    main_model.fit(train.X, train.y)
    print("  fit complete\n")

    # --- Add leakage-safe calibration candidates -----------------------------
    # Each is fit on TRAIN ONLY. CalibratedClassifierCV(cv=K) internally
    # trains a FRESH clone of the base HGB per fold and fits the
    # calibration mapping on that fold's held-out (out-of-fold) predictions
    # - never on in-sample predictions from an already-fitted model, and
    # never touching validation or test.
    print(f"--- Training calibrated candidates (HGB + sigmoid / isotonic, "
          f"{CALIBRATION_CV_FOLDS}-fold out-of-fold calibration on TRAIN only) ---")
    hgb_sigmoid = build_calibrated_model("sigmoid")
    hgb_sigmoid.fit(train.X, train.y)
    print("  hgb_sigmoid fit complete")

    hgb_isotonic = build_calibrated_model("isotonic")
    hgb_isotonic.fit(train.X, train.y)
    print("  hgb_isotonic fit complete\n")

    # --- Evaluate ALL candidates on VALIDATION --------------------------------
    baseline_val_prob = baseline.predict_proba(val.X)[:, 1]
    main_val_prob = main_model.predict_proba(val.X)[:, 1]
    sigmoid_val_prob = hgb_sigmoid.predict_proba(val.X)[:, 1]
    isotonic_val_prob = hgb_isotonic.predict_proba(val.X)[:, 1]

    candidate_models = {
        "baseline_lr": baseline,
        "raw_hgb": main_model,
        "hgb_sigmoid": hgb_sigmoid,
        "hgb_isotonic": hgb_isotonic,
    }
    candidate_val_probs = {
        "baseline_lr": baseline_val_prob,
        "raw_hgb": main_val_prob,
        "hgb_sigmoid": sigmoid_val_prob,
        "hgb_isotonic": isotonic_val_prob,
    }
    candidate_val_evals = {
        name: evaluate_probabilistic(val.y, prob) for name, prob in candidate_val_probs.items()
    }

    print("--- Calibration comparison (validation) ---")
    print(f"  {'candidate':>15} {'ROC-AUC':>10} {'PR-AUC':>10} {'Brier':>10} {'mean_pred_prob':>15}")
    for name, ev in candidate_val_evals.items():
        print(f"  {name:>15} {ev.roc_auc:>10.4f} {ev.pr_auc:>10.4f} {ev.brier:>10.4f} "
              f"{ev.prob_stats['mean']:>15.4f}")
    print(f"  (actual validation fraud prevalence: {val.y.mean():.4f} - compare against "
          f"mean predicted prob above; a well-calibrated model's mean prediction should "
          f"be close to this number)\n")

    # --- Select final candidate: PR-AUC primary, Brier tie-break, no ROC-AUC ---
    selected_name, selection_rationale = select_candidate_by_pr_auc_then_brier(candidate_val_evals)
    selected_model = candidate_models[selected_name]
    selected_val_prob = candidate_val_probs[selected_name]
    selected_val_eval = candidate_val_evals[selected_name]

    is_calibrated_selection = selected_name in {"hgb_sigmoid", "hgb_isotonic"}
    calibration_honesty_note = None
    if not is_calibrated_selection:
        calibration_honesty_note = (
            "Calibration did not materially improve validation probability "
            "quality; raw model retained."
        )

    print(f"--- Selected calibrated model: {selected_name} ---")
    print(f"  Why: {selection_rationale}")
    print(f"  Validation metrics for {selected_name}: ROC-AUC={selected_val_eval.roc_auc:.4f}, "
          f"PR-AUC={selected_val_eval.pr_auc:.4f}, Brier={selected_val_eval.brier:.4f}, "
          f"mean predicted prob={selected_val_eval.prob_stats['mean']:.4f} "
          f"(actual prevalence={val.y.mean():.4f})")
    if calibration_honesty_note:
        print(f"  {calibration_honesty_note}")
    print()

    # --- Threshold selection on VALIDATION ONLY, for the SELECTED candidate --
    print("--- Threshold table (validation, selected candidate) ---")
    print_threshold_table(val.y, selected_val_prob, REPORT_THRESHOLDS)
    best_threshold, best_threshold_metrics = select_best_f1_threshold(
        val.y, selected_val_prob, FINE_THRESHOLD_GRID
    )
    print(f"\n  Selected threshold (max F1 on validation): {best_threshold:.2f}")
    print(f"  At this threshold - precision={best_threshold_metrics['precision']:.4f}, "
          f"recall={best_threshold_metrics['recall']:.4f}, f1={best_threshold_metrics['f1']:.4f}")
    print(
        "  Rationale: for fraud detection, false positives (blocking a legitimate\n"
        "  customer) and false negatives (letting fraud through) both carry real\n"
        "  cost, and neither is obviously worse without a business-specified cost\n"
        "  ratio. Maximizing F1 on validation gives a defensible, reproducible\n"
        "  starting operating point that balances the two; the eventual Decision\n"
        "  Engine can move this threshold once merchant-specific risk tolerance\n"
        "  and false-positive cost are defined, without needing to retrain the\n"
        "  model itself.\n"
    )

    # --- Interpretability (validation data only) ------------------------------
    # Kept exactly as before: coefficients describe the linear baseline's
    # learned signal, and permutation importance describes the RAW (base)
    # HGB's learned signal - both are properties of the underlying model,
    # independent of whether a calibration wrapper is layered on top for
    # deployment, so they remain informative regardless of which candidate
    # was ultimately selected.
    print("--- Feature importance: baseline (LogisticRegression coefficients) ---")
    baseline_importance = baseline_coefficient_importance(baseline, feature_names, top_n=10)
    for row in baseline_importance:
        print(f"  {row['feature']:<40} coef={row['coefficient']:+.4f}")

    print("\n--- Feature importance: main model (HGB permutation importance, validation) ---")
    main_importance = main_model_permutation_importance(main_model, val.X, val.y, feature_names, top_n=10)
    for row in main_importance:
        print(f"  {row['feature']:<40} importance={row['importance_mean']:+.4f} (+/-{row['importance_std']:.4f})")
    print()

    # --- FINAL evaluation on TEST - touched exactly once, here ---------------
    print("--- FINAL evaluation on untouched TEST set ---")
    test_prob = selected_model.predict_proba(test.X)[:, 1]
    _run_posttraining_checks(selected_val_prob, test_prob)

    test_prob_eval = evaluate_probabilistic(test.y, test_prob)
    test_threshold_eval = evaluate_at_threshold(test.y, test_prob, best_threshold)

    print(f"  n={test_prob_eval.n}, prevalence={test_prob_eval.prevalence:.4f}")
    print(f"  ROC-AUC: {test_prob_eval.roc_auc}")
    print(f"  PR-AUC:  {test_prob_eval.pr_auc}  <-- PRIMARY metric")
    print(f"  Brier:   {test_prob_eval.brier:.4f}")
    print(f"  Predicted probability stats: {test_prob_eval.prob_stats}")
    print(f"  At threshold={best_threshold:.2f}: precision={test_threshold_eval['precision']:.4f}, "
          f"recall={test_threshold_eval['recall']:.4f}, f1={test_threshold_eval['f1']:.4f}")
    print(f"  Confusion matrix [[TN, FP], [FN, TP]]: {test_threshold_eval['confusion_matrix']}")

    test_reliability = reliability_table(test.y, test_prob, n_bins=10)
    print("  Reliability table (test):")
    for row in test_reliability:
        print(f"    predicted={row['mean_predicted']:.3f}  observed={row['observed_fraction_positive']:.3f}")

    # --- Honesty check: flag suspiciously high performance --------------------
    print("\n--- Performance sanity check ---")
    if (test_prob_eval.pr_auc or 0) >= SUSPICIOUSLY_HIGH_PR_AUC:
        print(
            f"  NOTE: test PR-AUC ({test_prob_eval.pr_auc:.4f}) is very high for a "
            f"~5% prevalence fraud task. This dataset is SYNTHETIC, and risk_label "
            f"was generated by label_generator.py from a formula whose dominant "
            f"inputs (archetype-driven base rate, velocity/diversity interactions, "
            f"device/IP flags, chargeback history) are largely the SAME features "
            f"this model was trained on. High separability here most likely "
            f"reflects the generator's label formula being learnable, not "
            f"evidence that a real-world fraud model would perform this well - "
            f"real fraud signals are noisier, adversarially adapting, and far "
            f"less cleanly tied to a handful of observable features. This should "
            f"be reported as a synthetic-data characteristic, not oversold as "
            f"production-grade fraud-detection performance."
        )
    else:
        print("  Test PR-AUC is not in the suspiciously-high range for this "
              "prevalence; no additional artifact investigation triggered by "
              "this heuristic.")

    # --- Save artifacts ---------------------------------------------------------
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_reliability_plot(
        test.y, test_prob, os.path.join(OUTPUT_DIR, "reliability_curve_test.png"),
        title=f"Risk Model ({selected_name}) - Test Reliability",
    )
    save_pr_curve_plot(
        test.y, test_prob, os.path.join(OUTPUT_DIR, "pr_curve_test.png"),
        title=f"Risk Model ({selected_name}) - Test Precision-Recall Curve",
    )

    calibration_comparison_payload = {
        "candidates_validation": {
            name: {
                "roc_auc": ev.roc_auc, "pr_auc": ev.pr_auc, "brier": ev.brier,
                "prob_stats": ev.prob_stats, "prevalence": ev.prevalence, "n": ev.n,
            }
            for name, ev in candidate_val_evals.items()
        },
        "validation_actual_prevalence": float(val.y.mean()),
        "selection_rule": (
            f"PR-AUC primary; candidates within {PR_AUC_TIE_EPSILON} PR-AUC of the "
            f"best are treated as tied, and the tied candidate with the lowest "
            f"Brier score is selected. ROC-AUC is never used to select."
        ),
        "selected_candidate": selected_name,
        "selection_rationale": selection_rationale,
        "calibration_honesty_note": calibration_honesty_note,
    }
    calibration_comparison_path = os.path.join(OUTPUT_DIR, "calibration_comparison.json")
    with open(calibration_comparison_path, "w") as f:
        json.dump(calibration_comparison_payload, f, indent=2)

    metrics_payload = {
        "selected_model": selected_name,
        "calibration_method": (
            selected_name.replace("hgb_", "") if is_calibrated_selection else "none (raw model retained)"
        ),
        "calibration_honesty_note": calibration_honesty_note,
        "selected_threshold": best_threshold,
        "class_imbalance": {
            "train_prevalence": float(train.y.mean()),
            "val_prevalence": float(val.y.mean()),
            "test_prevalence": float(test.y.mean()),
            "class_weight_used": "balanced",
        },
        "validation": {
            "candidates": calibration_comparison_payload["candidates_validation"],
            "selected_model_threshold_table": [
                evaluate_at_threshold(val.y, selected_val_prob, t) for t in REPORT_THRESHOLDS
            ],
            "selected_model_best_f1_threshold_metrics": best_threshold_metrics,
        },
        "test": {
            "roc_auc": test_prob_eval.roc_auc, "pr_auc": test_prob_eval.pr_auc,
            "brier": test_prob_eval.brier, "prob_stats": test_prob_eval.prob_stats,
            "prevalence": test_prob_eval.prevalence, "n": test_prob_eval.n,
            "at_selected_threshold": test_threshold_eval,
            "reliability_table": test_reliability,
        },
        "feature_importance": {
            "baseline_lr_coefficients_top10": baseline_importance,
            "main_hgb_permutation_importance_top10": main_importance,
        },
    }

    artifact_paths = save_artifacts(OUTPUT_DIR, selected_model, best_threshold, feature_names, metrics_payload)
    print("\n--- Artifacts saved ---")
    for name, path in artifact_paths.items():
        print(f"  {name}: {path}")
    print(f"  reliability_plot: {os.path.join(OUTPUT_DIR, 'reliability_curve_test.png')}")
    print(f"  pr_curve_plot: {os.path.join(OUTPUT_DIR, 'pr_curve_test.png')}")
    print(f"  calibration_comparison: {calibration_comparison_path}")

    print("\nRISK MODEL TRAINING COMPLETE.")

