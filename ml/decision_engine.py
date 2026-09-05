"""
decision_engine.py

RevenueRescue AI - Decision Engine.

This module is the BUSINESS-RULE layer that sits on top of the two
trained ML models (Recovery Model, Risk/Fraud Model). It is deliberately
split into two independent levels, per the project's architecture:

  1. Model-scoring functions (score_risk / score_recovery and their
     artifact loaders) - thin wrappers around the already-trained,
     already-calibrated joblib artifacts. They do NOT retrain, fit, or
     alter anything; they only load and run inference on an
     already-prepared, correctly-ordered feature vector.

  2. The business decision function (decide()) - a pure, deterministic
     function of already-computed scores (risk_score, recovery_score)
     plus a small amount of transaction/merchant context. It has NO
     dependency on scikit-learn, joblib, or any model artifact, which is
     exactly what makes it independently unit-testable (see the test
     section below) and safe to reason about without an ML environment.

IMPORTANT DESIGN NOTE ON THRESHOLDS:
The Risk Model's own saved threshold.json (its F1-optimal classification
cutoff, selected purely to evaluate the model's own precision/recall) is
NOT the same thing as this engine's BLOCK/REVIEW boundaries. This engine
derives its operating boundaries from `merchant_risk_tolerance` - a
business input representing how much fraud risk a given merchant is
willing to accept - because that is the appropriate input for an
automated action layer, not a single global metric-optimizing cutoff.
The Risk Model's threshold is still loaded and reported for reference
(e.g. "this is what the model itself would call fraud"), but it does not
drive BLOCK/REVIEW decisions here.

This module has NO side effects on import: no file writes, no network
calls, no database access, no randomness. Model artifacts are only read
from disk when a caller explicitly asks for them (load_recovery_artifacts
/ load_risk_artifacts), and only in the demo section at the bottom of
this file when run as __main__.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import numpy as np

# joblib is only needed for the model-scoring layer, imported lazily
# inside the loader functions so that importing this module (and running
# its business-logic tests) never requires scikit-learn/joblib to be
# importable at all - keeping the business-rule layer genuinely
# independent of the ML stack, as required.


# ---------------------------------------------------------------------------
# Configurable policy constants
#
# These are the ONLY numbers that define the engine's behavior. Keeping
# them as named constants (rather than embedding literals in decide())
# means the policy can be tuned or discussed without touching logic.
# ---------------------------------------------------------------------------

# How far ABOVE a merchant's own risk tolerance the risk score must climb
# before the engine escalates from REVIEW to an outright BLOCK.
BLOCK_MARGIN_ABOVE_TOLERANCE: float = 0.25

# Absolute safety rails on the computed block threshold, regardless of
# how strict or lenient a merchant's stated tolerance is:
#   - ABSOLUTE_BLOCK_FLOOR: even an extremely strict merchant (tolerance
#     near 0) should not have transactions auto-blocked below this risk
#     level - that band is handled by REVIEW instead, to avoid an
#     over-aggressive engine blocking too readily.
#   - ABSOLUTE_BLOCK_CEILING: even an extremely lenient merchant
#     (tolerance near 1.0) must still have a hard safety ceiling above
#     which the engine blocks automatically - "very high risk -> BLOCK"
#     is a non-negotiable floor of the business principle that fraud
#     safety takes priority over revenue.
ABSOLUTE_BLOCK_FLOOR: float = 0.50
ABSOLUTE_BLOCK_CEILING: float = 0.90

# Fallback recovery-probability cutoff used only if the caller does not
# supply one (e.g. the Recovery Model's own saved threshold.json value).
# Matches the Recovery Model's F1-optimal validation threshold from the
# most recent training run, kept here only as a sane, documented default.
DEFAULT_RECOVERY_SCORE_THRESHOLD: float = 0.32

# Once a failed payment has already been retried this many times, further
# automated recovery attempts have strongly diminishing returns (this
# mirrors the synthetic generator's own RECOVERY_RETRY_DECAY_FACTOR
# concept) - the engine stops recommending RECOVER past this point even
# if the model's recovery probability still looks acceptable.
MAX_RETRY_ATTEMPTS_FOR_RECOVERY: int = 3


def compute_block_threshold(merchant_risk_tolerance: float) -> float:
    """
    The risk level at/above which the engine BLOCKs outright, adapted to
    a merchant's own risk tolerance but always clamped into
    [ABSOLUTE_BLOCK_FLOOR, ABSOLUTE_BLOCK_CEILING].
    """
    return min(
        ABSOLUTE_BLOCK_CEILING,
        max(ABSOLUTE_BLOCK_FLOOR, merchant_risk_tolerance + BLOCK_MARGIN_ABOVE_TOLERANCE),
    )


# ---------------------------------------------------------------------------
# Actions and reason codes
# ---------------------------------------------------------------------------

class Action(str, Enum):
    """The four possible engine outputs. String-valued so results are
    trivially JSON-serializable for an API/audit-trail layer later."""

    ALLOW = "ALLOW"
    RECOVER = "RECOVER"
    REVIEW = "REVIEW"
    BLOCK = "BLOCK"


class ReasonCode(str, Enum):
    """Deterministic, machine-readable reason for every decision."""

    HIGH_FRAUD_RISK = "HIGH_FRAUD_RISK"
    ABOVE_MERCHANT_RISK_TOLERANCE = "ABOVE_MERCHANT_RISK_TOLERANCE"
    RECOVERY_OPPORTUNITY = "RECOVERY_OPPORTUNITY"
    LOW_RISK_NO_FAILURE = "LOW_RISK_NO_FAILURE"
    LOW_RECOVERY_PROBABILITY = "LOW_RECOVERY_PROBABILITY"
    HARD_FAILURE_NO_RECOVERY = "HARD_FAILURE_NO_RECOVERY"


# ---------------------------------------------------------------------------
# Typed input / output
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DecisionInput:
    """
    Everything the business decision layer needs. Deliberately does NOT
    include raw model features, transaction_id, customer_id, or
    timestamp - those belong to the model-scoring layer or an outer
    orchestration layer, not to the decision rules themselves.

    risk_score and recovery_score are ALREADY-COMPUTED probabilities
    (e.g. from score_risk()/score_recovery() below, or from any other
    inference pipeline) - decide() never calls a model itself.
    """

    risk_score: float
    recovery_score: Optional[float]
    payment_failed: bool
    merchant_risk_tolerance: float
    is_soft_failure: Optional[bool] = None
    retry_count_so_far: int = 0
    # amount and payment_method are not currently branched on by any rule
    # below; they are carried through into DecisionResult.metadata purely
    # for auditability and as a documented extension point (e.g. a future
    # amount-tiered policy), consistent with this project's original
    # "complete audit trail" requirement.
    amount: float = 0.0
    payment_method: str = "unknown"


@dataclass(frozen=True)
class DecisionResult:
    """The engine's output. `metadata` carries the computed thresholds and
    pass-through context so every decision is fully explainable after the
    fact without needing to re-derive anything."""

    action: Action
    risk_score: float
    recovery_score: Optional[float]
    merchant_risk_tolerance: float
    reason_code: ReasonCode
    human_readable_reason: str
    priority: str
    metadata: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_decision_input(inp: DecisionInput) -> None:
    if not (0.0 <= inp.risk_score <= 1.0):
        raise ValueError(f"risk_score must be in [0, 1], got {inp.risk_score}.")
    if inp.recovery_score is not None and not (0.0 <= inp.recovery_score <= 1.0):
        raise ValueError(f"recovery_score must be in [0, 1] or None, got {inp.recovery_score}.")
    if not (0.0 <= inp.merchant_risk_tolerance <= 1.0):
        raise ValueError(f"merchant_risk_tolerance must be in [0, 1], got {inp.merchant_risk_tolerance}.")
    if inp.retry_count_so_far < 0:
        raise ValueError(f"retry_count_so_far must be >= 0, got {inp.retry_count_so_far}.")
    if inp.amount < 0:
        raise ValueError(f"amount must be >= 0, got {inp.amount}.")


def validate_decision_result(result: DecisionResult) -> None:
    """
    Standalone post-hoc validator, usable by callers (and by the test
    section below) independently of decide() itself.
    """
    if result.action not in Action:
        raise ValueError(f"action must be one of {[a.value for a in Action]}, got {result.action}.")
    if not (0.0 <= result.risk_score <= 1.0):
        raise ValueError(f"risk_score must be in [0, 1], got {result.risk_score}.")
    if result.recovery_score is not None and not (0.0 <= result.recovery_score <= 1.0):
        raise ValueError(f"recovery_score must be in [0, 1] or None, got {result.recovery_score}.")
    if not (0.0 <= result.merchant_risk_tolerance <= 1.0):
        raise ValueError(
            f"merchant_risk_tolerance must be in [0, 1], got {result.merchant_risk_tolerance}."
        )


# ---------------------------------------------------------------------------
# The business decision function
# ---------------------------------------------------------------------------

def decide(
    inp: DecisionInput,
    recovery_score_threshold: float = DEFAULT_RECOVERY_SCORE_THRESHOLD,
) -> DecisionResult:
    """
    Deterministic, side-effect-free business decision.

    Rule order (each rule only applies if none of the earlier ones fired):

      1. FRAUD SAFETY GATE (always evaluated first, unconditionally):
         - risk_score >= block_threshold           -> BLOCK
         - risk_score >= merchant_risk_tolerance    -> REVIEW
         Recovery is NEVER reachable from here - this is the concrete
         implementation of "fraud safety takes priority over revenue
         recovery" and "never RECOVER if fraud risk is above the
         merchant's allowed risk tolerance".

      2. Only once risk_score < merchant_risk_tolerance (the gate has
         passed) does recovery logic run at all:
         - payment succeeded                        -> ALLOW
         - payment failed for a confirmed hard reason -> ALLOW
           (retrying a hard/terminal failure is unlikely to help)
         - already retried >= MAX_RETRY_ATTEMPTS_FOR_RECOVERY times
                                                     -> ALLOW
           (diminishing returns; avoid endless automated retries)
         - recovery_score >= recovery_score_threshold -> RECOVER
         - otherwise (low/unknown recovery probability) -> ALLOW
    """
    _validate_decision_input(inp)

    block_threshold = compute_block_threshold(inp.merchant_risk_tolerance)
    review_threshold = inp.merchant_risk_tolerance

    metadata: Dict = {
        "block_threshold": block_threshold,
        "review_threshold": review_threshold,
        "recovery_score_threshold": recovery_score_threshold,
        "is_soft_failure": inp.is_soft_failure,
        "retry_count_so_far": inp.retry_count_so_far,
        "amount": inp.amount,
        "payment_method": inp.payment_method,
    }

    # --- 1. Fraud safety gate - evaluated BEFORE anything about recovery ---
    if inp.risk_score >= block_threshold:
        result = DecisionResult(
            action=Action.BLOCK,
            risk_score=inp.risk_score,
            recovery_score=inp.recovery_score,
            merchant_risk_tolerance=inp.merchant_risk_tolerance,
            reason_code=ReasonCode.HIGH_FRAUD_RISK,
            human_readable_reason=(
                f"Fraud risk {inp.risk_score:.2f} is at/above the block threshold "
                f"{block_threshold:.2f}; blocking automatically regardless of any "
                f"recovery opportunity."
            ),
            priority="critical",
            metadata=metadata,
        )
        validate_decision_result(result)
        return result

    if inp.risk_score >= review_threshold:
        result = DecisionResult(
            action=Action.REVIEW,
            risk_score=inp.risk_score,
            recovery_score=inp.recovery_score,
            merchant_risk_tolerance=inp.merchant_risk_tolerance,
            reason_code=ReasonCode.ABOVE_MERCHANT_RISK_TOLERANCE,
            human_readable_reason=(
                f"Fraud risk {inp.risk_score:.2f} is at/above this merchant's own "
                f"tolerance {inp.merchant_risk_tolerance:.2f}; routing for manual "
                f"review rather than auto-approving or auto-recovering."
            ),
            priority="high",
            metadata=metadata,
        )
        validate_decision_result(result)
        return result

    # --- 2. Fraud gate has passed - recovery logic only reachable here ---
    if not inp.payment_failed:
        result = DecisionResult(
            action=Action.ALLOW,
            risk_score=inp.risk_score,
            recovery_score=inp.recovery_score,
            merchant_risk_tolerance=inp.merchant_risk_tolerance,
            reason_code=ReasonCode.LOW_RISK_NO_FAILURE,
            human_readable_reason=(
                "Transaction succeeded and fraud risk is within this merchant's "
                "tolerance; no action needed."
            ),
            priority="low",
            metadata=metadata,
        )
        validate_decision_result(result)
        return result

    if inp.is_soft_failure is False:
        result = DecisionResult(
            action=Action.ALLOW,
            risk_score=inp.risk_score,
            recovery_score=inp.recovery_score,
            merchant_risk_tolerance=inp.merchant_risk_tolerance,
            reason_code=ReasonCode.HARD_FAILURE_NO_RECOVERY,
            human_readable_reason=(
                "Payment failed for a confirmed hard/terminal reason; an automated "
                "retry is unlikely to succeed, so no recovery action is taken."
            ),
            priority="low",
            metadata=metadata,
        )
        validate_decision_result(result)
        return result

    if inp.retry_count_so_far >= MAX_RETRY_ATTEMPTS_FOR_RECOVERY:
        metadata["retry_limit_reached"] = True
        result = DecisionResult(
            action=Action.ALLOW,
            risk_score=inp.risk_score,
            recovery_score=inp.recovery_score,
            merchant_risk_tolerance=inp.merchant_risk_tolerance,
            reason_code=ReasonCode.LOW_RECOVERY_PROBABILITY,
            human_readable_reason=(
                f"This payment has already been retried {inp.retry_count_so_far} "
                f"times; further automated attempts have diminishing returns, so "
                f"no further recovery action is taken."
            ),
            priority="low",
            metadata=metadata,
        )
        validate_decision_result(result)
        return result

    if inp.recovery_score is not None and inp.recovery_score >= recovery_score_threshold:
        result = DecisionResult(
            action=Action.RECOVER,
            risk_score=inp.risk_score,
            recovery_score=inp.recovery_score,
            merchant_risk_tolerance=inp.merchant_risk_tolerance,
            reason_code=ReasonCode.RECOVERY_OPPORTUNITY,
            human_readable_reason=(
                f"Fraud risk {inp.risk_score:.2f} is acceptable and recovery "
                f"probability {inp.recovery_score:.2f} meets the threshold "
                f"{recovery_score_threshold:.2f}; attempting automated recovery "
                f"(e.g. retry or customer nudge)."
            ),
            priority="medium",
            metadata=metadata,
        )
        validate_decision_result(result)
        return result

    result = DecisionResult(
        action=Action.ALLOW,
        risk_score=inp.risk_score,
        recovery_score=inp.recovery_score,
        merchant_risk_tolerance=inp.merchant_risk_tolerance,
        reason_code=ReasonCode.LOW_RECOVERY_PROBABILITY,
        human_readable_reason=(
            "Fraud risk is acceptable, but recovery probability is too low (or "
            "unknown) to justify an automated recovery attempt."
        ),
        priority="low",
        metadata=metadata,
    )
    validate_decision_result(result)
    return result


# ---------------------------------------------------------------------------
# Model-scoring layer (separate from, and independent of, decide() above)
# ---------------------------------------------------------------------------

@dataclass
class LoadedModel:
    """A trained model plus the metadata needed to use it correctly."""

    model: object
    threshold: float
    feature_names: List[str]


def _load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def _load_model_artifacts(model_dir: str, model_filename: str) -> LoadedModel:
    """
    Shared loader used by both load_recovery_artifacts and
    load_risk_artifacts. Reads existing artifacts only - never trains,
    fits, or writes anything.
    """
    import joblib  # local import - see module docstring

    model_path = os.path.join(model_dir, model_filename)
    threshold_path = os.path.join(model_dir, "threshold.json")
    feature_names_path = os.path.join(model_dir, "feature_names.json")

    for path in (model_path, threshold_path, feature_names_path):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Required model artifact not found: '{path}'. This engine loads "
                f"already-trained artifacts only - it does not train or regenerate "
                f"them. Run the corresponding training script first."
            )

    model = joblib.load(model_path)
    threshold = _load_json(threshold_path)["selected_threshold"]
    feature_names = _load_json(feature_names_path)["feature_names"]
    return LoadedModel(model=model, threshold=threshold, feature_names=feature_names)


def load_recovery_artifacts(model_dir: str = "output/models/recovery") -> LoadedModel:
    """Load the trained Recovery Model, its selected threshold, and its
    expected feature order. Never retrains."""
    return _load_model_artifacts(model_dir, "recovery_model.joblib")


def load_risk_artifacts(model_dir: str = "output/models/risk") -> LoadedModel:
    """
    Load the trained Risk Model, its selected threshold, and its expected
    feature order. Never retrains. Whatever calibration layer was
    selected during training (raw / sigmoid / isotonic) is already baked
    into the saved joblib object - this loader does not need to know or
    care which one it is.
    """
    return _load_model_artifacts(model_dir, "risk_model.joblib")


def _validate_feature_order(X: np.ndarray, feature_names: List[str]) -> None:
    """
    Explicit feature-order validation. This can only check COUNT, not
    semantic column identity (a bare numpy array carries no column
    labels) - callers are responsible for building X with columns in
    exactly the order given by `feature_names`. The error message makes
    that expectation unambiguous rather than failing silently or letting
    the model silently score garbage.
    """
    if X.ndim != 2:
        raise ValueError(f"X must be a 2D array of shape (n_samples, n_features); got ndim={X.ndim}.")
    if X.shape[1] != len(feature_names):
        raise ValueError(
            f"Feature count mismatch: X has {X.shape[1]} columns but this model "
            f"expects {len(feature_names)} features, in this exact order: "
            f"{feature_names}. Build X's columns in this order."
        )


def score_risk(loaded: LoadedModel, X: np.ndarray) -> np.ndarray:
    """Batch fraud-probability scoring. X must already have columns in
    `loaded.feature_names` order (see _validate_feature_order)."""
    _validate_feature_order(X, loaded.feature_names)
    return loaded.model.predict_proba(X)[:, 1]


def score_recovery(loaded: LoadedModel, X: np.ndarray) -> np.ndarray:
    """Batch recovery-probability scoring. X must already have columns in
    `loaded.feature_names` order (see _validate_feature_order)."""
    _validate_feature_order(X, loaded.feature_names)
    return loaded.model.predict_proba(X)[:, 1]


def score_risk_single(loaded: LoadedModel, ordered_feature_values: List[float]) -> float:
    """Convenience single-row wrapper around score_risk()."""
    X = np.array([ordered_feature_values], dtype=float)
    return float(score_risk(loaded, X)[0])


def score_recovery_single(loaded: LoadedModel, ordered_feature_values: List[float]) -> float:
    """Convenience single-row wrapper around score_recovery()."""
    X = np.array([ordered_feature_values], dtype=float)
    return float(score_recovery(loaded, X)[0])


# ---------------------------------------------------------------------------
# Deterministic unit-style tests for the business decision layer
# ---------------------------------------------------------------------------

def _run_tests() -> bool:
    """
    Tests decide() directly - no model loading, no file I/O, no
    randomness. Prints PASS/FAIL per scenario and returns overall success.
    """
    print("=" * 70)
    print("DECISION ENGINE - UNIT TESTS")
    print("=" * 70)

    all_passed = True

    def check(name: str, inp: DecisionInput, predicate, note: str, **decide_kwargs):
        nonlocal all_passed
        result = decide(inp, **decide_kwargs)
        validate_decision_result(result)
        passed = predicate(result)
        all_passed = all_passed and passed
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {name}: action={result.action.value}, "
              f"reason={result.reason_code.value}  ({note})")
        if not passed:
            print(f"         -> unexpected result: {result}")
        return result

    # A. LOW RISK + FAILED + HIGH RECOVERY -> RECOVER
    check(
        "A. low risk, failed, high recovery",
        DecisionInput(risk_score=0.05, recovery_score=0.80, payment_failed=True,
                      merchant_risk_tolerance=0.50, is_soft_failure=True, retry_count_so_far=0),
        lambda r: r.action == Action.RECOVER,
        "expected RECOVER",
    )

    # B. LOW RISK + FAILED + LOW RECOVERY -> ALLOW (documented rule: no
    # fraud concern and low recovery odds -> no automated action, rather
    # than escalating to REVIEW, which is reserved for fraud ambiguity)
    check(
        "B. low risk, failed, low recovery",
        DecisionInput(risk_score=0.05, recovery_score=0.05, payment_failed=True,
                      merchant_risk_tolerance=0.50, is_soft_failure=True, retry_count_so_far=0),
        lambda r: r.action == Action.ALLOW and r.reason_code == ReasonCode.LOW_RECOVERY_PROBABILITY,
        "documented rule: ALLOW, reason=LOW_RECOVERY_PROBABILITY",
    )

    # C. HIGH RISK -> BLOCK
    check(
        "C. high risk",
        DecisionInput(risk_score=0.95, recovery_score=0.90, payment_failed=True,
                      merchant_risk_tolerance=0.50, is_soft_failure=True, retry_count_so_far=0),
        lambda r: r.action == Action.BLOCK,
        "expected BLOCK regardless of recovery score",
    )

    # D. MODERATE RISK -> REVIEW
    check(
        "D. moderate risk (above tolerance, below block)",
        DecisionInput(risk_score=0.60, recovery_score=0.80, payment_failed=True,
                      merchant_risk_tolerance=0.50, is_soft_failure=True, retry_count_so_far=0),
        lambda r: r.action == Action.REVIEW,
        "expected REVIEW",
    )

    # E. LOW RISK + NO FAILURE -> ALLOW
    check(
        "E. low risk, no failure",
        DecisionInput(risk_score=0.05, recovery_score=None, payment_failed=False,
                      merchant_risk_tolerance=0.50),
        lambda r: r.action == Action.ALLOW and r.reason_code == ReasonCode.LOW_RISK_NO_FAILURE,
        "expected ALLOW / LOW_RISK_NO_FAILURE",
    )

    # F. RISK ABOVE MERCHANT TOLERANCE -> never RECOVER, even with a
    # very high recovery score.
    check(
        "F. risk above merchant tolerance, high recovery score",
        DecisionInput(risk_score=0.55, recovery_score=0.99, payment_failed=True,
                      merchant_risk_tolerance=0.50, is_soft_failure=True, retry_count_so_far=0),
        lambda r: r.action != Action.RECOVER,
        "expected NOT RECOVER (fraud gate takes priority)",
    )

    # G. HARD FAILURE -> no unsafe recovery action
    check(
        "G. hard failure, low risk",
        DecisionInput(risk_score=0.05, recovery_score=0.95, payment_failed=True,
                      merchant_risk_tolerance=0.50, is_soft_failure=False, retry_count_so_far=0),
        lambda r: r.action != Action.RECOVER and r.reason_code == ReasonCode.HARD_FAILURE_NO_RECOVERY,
        "expected NOT RECOVER, reason=HARD_FAILURE_NO_RECOVERY",
    )

    # H. Borderline threshold values (exact equality)
    tolerance = 0.40
    check(
        "H1. risk_score exactly == merchant_risk_tolerance",
        DecisionInput(risk_score=tolerance, recovery_score=0.90, payment_failed=True,
                      merchant_risk_tolerance=tolerance, is_soft_failure=True, retry_count_so_far=0),
        lambda r: r.action == Action.REVIEW,
        "boundary is inclusive (>=) -> expected REVIEW",
    )

    block_threshold = compute_block_threshold(tolerance)
    check(
        "H2. risk_score exactly == block_threshold",
        DecisionInput(risk_score=block_threshold, recovery_score=0.90, payment_failed=True,
                      merchant_risk_tolerance=tolerance, is_soft_failure=True, retry_count_so_far=0),
        lambda r: r.action == Action.BLOCK,
        f"block_threshold={block_threshold:.2f}, boundary is inclusive (>=) -> expected BLOCK",
    )

    rec_threshold = DEFAULT_RECOVERY_SCORE_THRESHOLD
    check(
        "H3. recovery_score exactly == recovery_score_threshold",
        DecisionInput(risk_score=0.05, recovery_score=rec_threshold, payment_failed=True,
                      merchant_risk_tolerance=0.50, is_soft_failure=True, retry_count_so_far=0),
        lambda r: r.action == Action.RECOVER,
        f"recovery_threshold={rec_threshold:.2f}, boundary is inclusive (>=) -> expected RECOVER",
    )

    # Extra: input validation guards actually reject out-of-range values.
    validation_ok = True
    for bad_kwargs in (
        dict(risk_score=1.5), dict(recovery_score=-0.1), dict(merchant_risk_tolerance=2.0),
    ):
        base = dict(risk_score=0.1, recovery_score=0.1, payment_failed=True, merchant_risk_tolerance=0.5)
        base.update(bad_kwargs)
        try:
            decide(DecisionInput(**base))
            validation_ok = False
        except ValueError:
            pass
    status = "PASS" if validation_ok else "FAIL"
    print(f"[{status}] I. input validation rejects out-of-range scores")
    all_passed = all_passed and validation_ok

    print()
    print("ALL TESTS PASSED" if all_passed else "SOME TESTS FAILED")
    print()
    return all_passed


# ---------------------------------------------------------------------------
# Hackathon-demo examples
# ---------------------------------------------------------------------------

def _print_decision(label: str, result: DecisionResult) -> None:
    print(f"\n{label}")
    print(f"  action:            {result.action.value}")
    print(f"  reason_code:       {result.reason_code.value}")
    print(f"  human_readable:    {result.human_readable_reason}")
    print(f"  priority:          {result.priority}")
    print(f"  risk_score:        {result.risk_score:.2f}")
    print(f"  recovery_score:    {result.recovery_score}")
    print(f"  merchant_tolerance:{result.merchant_risk_tolerance:.2f}")
    print(f"  metadata:          {result.metadata}")


def _run_demo_examples() -> None:
    print("=" * 70)
    print("DECISION ENGINE - DEMO EXAMPLES")
    print("=" * 70)

    _print_decision(
        "1) Safe, successful transaction",
        decide(DecisionInput(
            risk_score=0.03, recovery_score=None, payment_failed=False,
            merchant_risk_tolerance=0.50, amount=1200.0, payment_method="UPI",
        )),
    )

    _print_decision(
        "2) Failed, low-risk transaction with strong recovery chance",
        decide(DecisionInput(
            risk_score=0.08, recovery_score=0.78, payment_failed=True,
            merchant_risk_tolerance=0.50, is_soft_failure=True, retry_count_so_far=0,
            amount=850.0, payment_method="card",
        )),
    )

    _print_decision(
        "3) Suspicious transaction (moderate risk)",
        decide(DecisionInput(
            risk_score=0.62, recovery_score=0.70, payment_failed=True,
            merchant_risk_tolerance=0.50, is_soft_failure=True, retry_count_so_far=0,
            amount=15000.0, payment_method="netbanking",
        )),
    )

    _print_decision(
        "4) High fraud-risk transaction",
        decide(DecisionInput(
            risk_score=0.93, recovery_score=0.85, payment_failed=True,
            merchant_risk_tolerance=0.50, is_soft_failure=True, retry_count_so_far=0,
            amount=9800.0, payment_method="wallet",
        )),
    )
    print()


def _run_live_inference_demo() -> None:
    """
    Optional, best-effort demonstration of the model-scoring layer
    against real saved artifacts. Skips gracefully (rather than crashing
    the whole script) if the artifacts aren't present in this
    environment - the business-logic tests above never depend on this.
    """
    print("=" * 70)
    print("DECISION ENGINE - LIVE ARTIFACT LOADING (best-effort)")
    print("=" * 70)
    try:
        recovery = load_recovery_artifacts()
        print(f"Loaded Recovery Model. Saved threshold={recovery.threshold}, "
              f"{len(recovery.feature_names)} features expected.")
    except FileNotFoundError as e:
        print(f"Recovery artifacts not available in this environment: {e}")
        recovery = None

    try:
        risk = load_risk_artifacts()
        print(f"Loaded Risk Model. Saved threshold={risk.threshold} "
              f"(this is the MODEL's own F1-optimal cutoff, not the engine's "
              f"merchant-tolerance-based BLOCK/REVIEW boundary), "
              f"{len(risk.feature_names)} features expected.")
    except FileNotFoundError as e:
        print(f"Risk artifacts not available in this environment: {e}")
        risk = None

    if risk is not None:
        # A dummy, arbitrarily-chosen but correctly-ordered feature row -
        # purely to demonstrate score_risk_single()'s explicit
        # feature-order contract, not a real transaction.
        dummy_row = [0.0] * len(risk.feature_names)
        try:
            score = score_risk_single(risk, dummy_row)
            print(f"Example score_risk_single() call on an all-zero row: {score:.4f}")
        except ValueError as e:
            print(f"score_risk_single() feature-order validation triggered: {e}")
    print()


if __name__ == "__main__":
    tests_passed = _run_tests()
    _run_demo_examples()
    _run_live_inference_demo()

    if not tests_passed:
        raise SystemExit(1)