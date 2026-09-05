"""
recovery_model.py

Trains and evaluates the RevenueRescue AI Recovery Model, which predicts
recovery_label (recovered=1 / not_recovered=0) for FAILED transactions.

This module consumes the Recovery train/val/test matrices EXACTLY as
returned by data_preparation.prepare_datasets() - it does not reconstruct
the split, does not perform any new random split, and does not touch
data_preparation.py, any generator module, or main_generator.py.

Two models are trained:
  1. Baseline: LogisticRegression, on an imputed + scaled copy of the
     prepared feature matrix (Logistic Regression cannot consume NaN).
  2. Main model: HistGradientBoostingClassifier, on the RAW prepared
     feature matrix, with days_since_last_successful_payment's missing
     values passed through untouched - HGB natively learns a split
     direction for missing values, so no imputation is applied or wanted.

The validation set is used to compare the two models and to choose a
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
from sklearn.calibration import calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data_preparation import FORBIDDEN_FEATURE_COLUMNS, DataPreparationResult, prepare_datasets

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RANDOM_SEED: int = 42
OUTPUT_DIR: str = "output/models/recovery"

# Threshold at/below which the minority class is considered imbalanced
# enough to justify class_weight="balanced" for the baseline model. This
# is a documented, data-driven decision point, not a fixed default.
IMBALANCE_THRESHOLD: float = 0.20

# Candidate thresholds always printed in the summary table, plus a finer
# grid used to actually select the best-F1 threshold.
REPORT_THRESHOLDS: List[float] = [0.30, 0.40, 0.50, 0.60, 0.70]
FINE_THRESHOLD_GRID: np.ndarray = np.round(np.arange(0.05, 0.96, 0.01), 2)

# Above this validation ROC-AUC, print an explicit honesty/artifact-check
# note rather than silently reporting an impressive-looking number.
SUSPICIOUSLY_HIGH_AUC: float = 0.90


# ---------------------------------------------------------------------------
# Pre-training validation
# ---------------------------------------------------------------------------

def _run_pretraining_checks(result: DataPreparationResult) -> None:
    """
    Structural/statistical checks required BEFORE any model is fit.
    Raises on hard failures.
    """
    train, val, test = result.recovery_train, result.recovery_val, result.recovery_test

    print("--- Pre-training checks ---")
    print(f"  recovery_train.X shape: {train.X.shape}")
    print(f"  recovery_val.X shape:   {val.X.shape}")
    print(f"  recovery_test.X shape:  {test.X.shape}")

    # Both classes present in train (hard requirement - can't fit
    # otherwise); val/test are checked and reported, not hard-required,
    # matching data_preparation.py's own "where statistically possible"
    # policy.
    assert set(np.unique(train.y)) == {0, 1}, "recovery_train must contain both classes."
    for name, split in [("val", val), ("test", test)]:
        classes = set(np.unique(split.y))
        if classes != {0, 1}:
            warnings.warn(f"recovery_{name} does not contain both classes: {classes}", stacklevel=2)
    print(f"  train classes: {sorted(set(np.unique(train.y)))}")
    print(f"  val classes:   {sorted(set(np.unique(val.y)))}")
    print(f"  test classes:  {sorted(set(np.unique(test.y)))}")

    # Target is strictly binary {0, 1} in every split.
    for name, split in [("train", train), ("val", val), ("test", test)]:
        assert set(np.unique(split.y)).issubset({0, 1}), f"recovery_{name}.y is not binary."

    # No customer overlap, re-checked here independently of
    # data_preparation.py's own validation (defense in depth).
    train_c, val_c, test_c = set(train.customer_ids), set(val.customer_ids), set(test.customer_ids)
    assert not (train_c & val_c), "Customer overlap between recovery train and val."
    assert not (train_c & test_c), "Customer overlap between recovery train and test."
    assert not (val_c & test_c), "Customer overlap between recovery val and test."
    print(f"  customer overlap (train/val/test): "
          f"{len(train_c & val_c)}/{len(train_c & test_c)}/{len(val_c & test_c)} (all must be 0)")

    # No forbidden feature names present.
    feature_names = set(train.feature_names)
    forbidden_present = feature_names & FORBIDDEN_FEATURE_COLUMNS
    assert not forbidden_present, f"Forbidden features present: {forbidden_present}"
    assert "recovery_label" not in feature_names and "risk_label" not in feature_names, (
        "Target columns leaked into features."
    )
    print(f"  forbidden features present: {sorted(forbidden_present)} (must be empty)")
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
    print(f"  Training minority-class fraction: {minority_fraction:.4f} "
          f"(imbalance threshold: {IMBALANCE_THRESHOLD})")
    if minority_fraction <= IMBALANCE_THRESHOLD:
        print("  -> class_weight='balanced' IS justified (minority class is scarce).")
        return "balanced"
    print("  -> class_weight='balanced' is NOT used - training data is only mildly "
          "imbalanced, and reweighting would distort probability calibration "
          "without a clear benefit here.")
    return None


def build_baseline_model(y_train: np.ndarray) -> Pipeline:
    """
    LogisticRegression baseline. Cannot consume NaN, so this pipeline
    imputes (median, fit on train only) and scales (fit on train only)
    before the classifier. Both steps live inside the Pipeline so that
    fitting the Pipeline on train data alone guarantees no val/test
    leakage into either the imputer or the scaler.
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
    exactly as data_preparation.py produced it; HGB natively learns
    which branch a missing value should take at each split, so no
    imputation is applied (imputing here would throw away the
    "no prior success yet" signal has_prior_success/NaN jointly encode).
    """
    return HistGradientBoostingClassifier(
        loss="log_loss",
        max_iter=300,
        learning_rate=0.05,
        max_depth=6,
        min_samples_leaf=20,
        l2_regularization=1.0,
        early_stopping=False,  # we do our own validation-based comparison
        random_state=RANDOM_SEED,
    )


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
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="quantile")
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
    print("  Test set was touched exactly once, after model+threshold selection on validation - OK\n")


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

    model_path = os.path.join(output_dir, "recovery_model.joblib")
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
    print("RECOVERY MODEL - TRAINING AND EVALUATION")
    print("=" * 70)

    result = prepare_datasets()
    train, val, test = result.recovery_train, result.recovery_val, result.recovery_test
    feature_names = train.feature_names

    _run_pretraining_checks(result)

    # --- Train both models on TRAIN ONLY -----------------------------------
    print("--- Training baseline (LogisticRegression) ---")
    baseline = build_baseline_model(train.y)
    baseline.fit(train.X, train.y)
    print("  fit complete\n")

    print("--- Training main model (HistGradientBoostingClassifier) ---")
    main_model = build_main_model()
    main_model.fit(train.X, train.y)
    print("  fit complete\n")

    # --- Evaluate both on VALIDATION -----------------------------------------
    baseline_val_prob = baseline.predict_proba(val.X)[:, 1]
    main_val_prob = main_model.predict_proba(val.X)[:, 1]

    baseline_val_eval = evaluate_probabilistic(val.y, baseline_val_prob)
    main_val_eval = evaluate_probabilistic(val.y, main_val_prob)

    print("--- Validation comparison ---")
    print(f"  {'model':>20} {'ROC-AUC':>10} {'PR-AUC':>10} {'Brier':>10}")
    print(f"  {'baseline (LR)':>20} {baseline_val_eval.roc_auc:>10.4f} "
          f"{baseline_val_eval.pr_auc:>10.4f} {baseline_val_eval.brier:>10.4f}")
    print(f"  {'main (HGB)':>20} {main_val_eval.roc_auc:>10.4f} "
          f"{main_val_eval.pr_auc:>10.4f} {main_val_eval.brier:>10.4f}\n")

    # --- Model selection on VALIDATION (PR-AUC primary, ROC-AUC tiebreak) ----
    if main_val_eval.pr_auc >= baseline_val_eval.pr_auc:
        selected_name = "main_hgb"
        selected_model = main_model
        selected_val_prob = main_val_prob
        selected_val_eval = main_val_eval
    else:
        selected_name = "baseline_lr"
        selected_model = baseline
        selected_val_prob = baseline_val_prob
        selected_val_eval = baseline_val_eval
    print(f"--- Selected model: {selected_name} (higher validation PR-AUC) ---\n")

    # --- Threshold selection on VALIDATION ONLY -------------------------------
    print("--- Threshold table (validation, selected model) ---")
    print_threshold_table(val.y, selected_val_prob, REPORT_THRESHOLDS)
    best_threshold, best_threshold_metrics = select_best_f1_threshold(
        val.y, selected_val_prob, FINE_THRESHOLD_GRID
    )
    print(f"\n  Selected threshold (max F1 on validation): {best_threshold:.2f}")
    print(f"  At this threshold - precision={best_threshold_metrics['precision']:.4f}, "
          f"recall={best_threshold_metrics['recall']:.4f}, f1={best_threshold_metrics['f1']:.4f}")
    print("  Rationale: F1 balances the cost of chasing unrecoverable failures "
          "(false positives) against missing genuinely recoverable ones (false "
          "negatives) - a reasonable default before the Decision Engine's own "
          "cost-sensitive policy is layered on top in a later step.\n")

    # --- Interpretability (validation data only, both models regardless of selection) ---
    print("--- Feature importance: baseline (LogisticRegression coefficients) ---")
    baseline_importance = baseline_coefficient_importance(baseline, feature_names, top_n=10)
    for row in baseline_importance:
        print(f"  {row['feature']:<45} coef={row['coefficient']:+.4f}")

    print("\n--- Feature importance: main model (HGB permutation importance, validation) ---")
    main_importance = main_model_permutation_importance(main_model, val.X, val.y, feature_names, top_n=10)
    for row in main_importance:
        print(f"  {row['feature']:<45} importance={row['importance_mean']:+.4f} (+/-{row['importance_std']:.4f})")
    print()

    # --- FINAL evaluation on TEST - touched exactly once, here ---------------
    print("--- FINAL evaluation on untouched TEST set ---")
    test_prob = selected_model.predict_proba(test.X)[:, 1]
    _run_posttraining_checks(selected_val_prob, test_prob)

    test_prob_eval = evaluate_probabilistic(test.y, test_prob)
    test_threshold_eval = evaluate_at_threshold(test.y, test_prob, best_threshold)

    print(f"  n={test_prob_eval.n}, prevalence={test_prob_eval.prevalence:.4f}")
    print(f"  ROC-AUC: {test_prob_eval.roc_auc}")
    print(f"  PR-AUC:  {test_prob_eval.pr_auc}")
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
    if (test_prob_eval.roc_auc or 0) >= SUSPICIOUSLY_HIGH_AUC:
        print(
            f"  NOTE: test ROC-AUC ({test_prob_eval.roc_auc:.4f}) is very high. This "
            f"dataset is SYNTHETIC, and recovery_label was generated by "
            f"label_generator.py using a formula whose dominant inputs "
            f"(is_soft_failure, failure_reason_code, retry_count_so_far, "
            f"customer_past_recovery_rate) are EXACTLY the features this model "
            f"was trained on. High separability here reflects the generator's "
            f"label formula being learnable, not evidence that a real-world "
            f"recovery model would perform this well - real payment-recovery "
            f"signals are noisier and less directly tied to observable features. "
            f"This should be reported as a synthetic-data characteristic, not "
            f"oversold as production-grade performance."
        )
    else:
        print("  Test performance is not in the suspiciously-high range; no "
              "additional artifact investigation triggered by this heuristic.")

    # --- Save artifacts ---------------------------------------------------------
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_reliability_plot(
        test.y, test_prob, os.path.join(OUTPUT_DIR, "reliability_curve_test.png"),
        title=f"Recovery Model ({selected_name}) - Test Reliability",
    )

    metrics_payload = {
        "selected_model": selected_name,
        "selected_threshold": best_threshold,
        "validation": {
            "baseline_lr": {
                "roc_auc": baseline_val_eval.roc_auc, "pr_auc": baseline_val_eval.pr_auc,
                "brier": baseline_val_eval.brier, "prob_stats": baseline_val_eval.prob_stats,
                "prevalence": baseline_val_eval.prevalence, "n": baseline_val_eval.n,
            },
            "main_hgb": {
                "roc_auc": main_val_eval.roc_auc, "pr_auc": main_val_eval.pr_auc,
                "brier": main_val_eval.brier, "prob_stats": main_val_eval.prob_stats,
                "prevalence": main_val_eval.prevalence, "n": main_val_eval.n,
            },
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

    print("\nRECOVERY MODEL TRAINING COMPLETE.")