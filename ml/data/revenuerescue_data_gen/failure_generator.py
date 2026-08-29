"""
failure_generator.py

Decides, for each transaction, whether it fails at all, and if so, samples
a failure_reason_code, derives is_soft_failure deterministically from that
reason, and samples retry_count_so_far.

This module deliberately stops at failure context. No risk signals,
velocity features, fatigue calculations, or recovery/risk labels are
produced here - those depend on information (or targets) this module has
no business touching yet.
"""

from typing import Dict

import numpy as np
import pandas as pd

from .config import FAILURE_REASON_CODES, RANDOM_SEED, SOFT_FAILURE_REASONS

# ---------------------------------------------------------------------------
# Local, module-specific parameters
#
# These are failure-modeling parameters, not duplicates of anything in
# archetypes.py (which has no failure-rate or retry fields) or config.py.
# They live here because they are specific to this module's job.
# ---------------------------------------------------------------------------

# Base probability that a transaction fails at all, before payment-method,
# amount-anomaly, and history adjustments. Reflects each archetype's
# general reliability - NOT a deterministic outcome, just a starting point
# for a probabilistic draw.
_BASE_FAILURE_RATE_BY_ARCHETYPE: Dict[str, float] = {
    "loyal_low_risk": 0.05,
    "normal": 0.12,
    "new_customer": 0.18,
    "financially_constrained": 0.30,
    "suspicious": 0.35,
    "account_takeover_like": 0.25,
}

# Multiplier on failure probability by payment method, reflecting that
# some rails are structurally more failure-prone (e.g. netbanking's
# bank-timeout exposure) than others (e.g. UPI's simpler flow).
_METHOD_FAILURE_MULTIPLIER: Dict[str, float] = {
    "UPI": 0.9,
    "card": 1.1,
    "netbanking": 1.2,
    "wallet": 1.0,
}

# If a transaction's amount is well above the customer's own typical
# amount, it is more likely to trip bank-side scrutiny / decline - this is
# the "transaction context" component of failure probability.
_AMOUNT_ANOMALY_RATIO_THRESHOLD: float = 2.0
_AMOUNT_ANOMALY_MULTIPLIER: float = 1.3

# Bounds so no archetype/method/context combination pushes failure
# probability to an unrealistic extreme.
_MIN_FAILURE_PROB: float = 0.01
_MAX_FAILURE_PROB: float = 0.90

# Per-archetype base probability distribution over the eight configured
# failure_reason_code values. These are PROBABILITIES, not a deterministic
# archetype -> reason mapping - every archetype can produce every reason,
# just with different likelihoods. Keys must match FAILURE_REASON_CODES.
#
# Note: config.py's FAILURE_REASON_CODES does not include an "expired_card"
# or "account_closed" code. "Card lifecycle" issues are represented via
# card_declined_issuer, and "account issues" are represented via
# fraud_suspected_by_bank / customer_cancelled - the closest available
# codes - rather than inventing new categories outside the approved config.
_FAILURE_REASON_WEIGHTS_BY_ARCHETYPE: Dict[str, Dict[str, float]] = {
    "loyal_low_risk": {
        "insufficient_funds": 0.10, "bank_timeout": 0.25, "otp_expired": 0.25,
        "card_declined_issuer": 0.10, "network_error": 0.20, "invalid_cvv": 0.05,
        "fraud_suspected_by_bank": 0.02, "customer_cancelled": 0.03,
    },
    "normal": {
        "insufficient_funds": 0.15, "bank_timeout": 0.15, "otp_expired": 0.20,
        "card_declined_issuer": 0.15, "network_error": 0.15, "invalid_cvv": 0.10,
        "fraud_suspected_by_bank": 0.03, "customer_cancelled": 0.07,
    },
    "new_customer": {
        "insufficient_funds": 0.10, "bank_timeout": 0.10, "otp_expired": 0.30,
        "card_declined_issuer": 0.20, "network_error": 0.10, "invalid_cvv": 0.15,
        "fraud_suspected_by_bank": 0.02, "customer_cancelled": 0.03,
    },
    "financially_constrained": {
        "insufficient_funds": 0.55, "bank_timeout": 0.10, "otp_expired": 0.10,
        "card_declined_issuer": 0.10, "network_error": 0.08, "invalid_cvv": 0.03,
        "fraud_suspected_by_bank": 0.01, "customer_cancelled": 0.03,
    },
    "suspicious": {
        "insufficient_funds": 0.05, "bank_timeout": 0.05, "otp_expired": 0.05,
        "card_declined_issuer": 0.30, "network_error": 0.05, "invalid_cvv": 0.10,
        "fraud_suspected_by_bank": 0.35, "customer_cancelled": 0.05,
    },
    "account_takeover_like": {
        "insufficient_funds": 0.03, "bank_timeout": 0.05, "otp_expired": 0.10,
        "card_declined_issuer": 0.25, "network_error": 0.05, "invalid_cvv": 0.07,
        "fraud_suspected_by_bank": 0.40, "customer_cancelled": 0.05,
    },
}

# Payment-method-conditioned multipliers layered on top of the archetype
# distribution above, so failure REASON depends on both archetype and the
# rail actually used - e.g. netbanking failures skew toward bank/network
# issues regardless of archetype.
_METHOD_REASON_MULTIPLIERS: Dict[str, Dict[str, float]] = {
    "netbanking": {"bank_timeout": 1.5, "network_error": 1.5},
    "card": {"invalid_cvv": 1.4, "card_declined_issuer": 1.4},
    "UPI": {"otp_expired": 1.3},
    "wallet": {"otp_expired": 1.2},
}

# Mean (lambda) of a Poisson draw for retry_count_so_far on FAILED
# transactions, per archetype. Kept low (<1) for most archetypes so most
# failures show 0-1 prior retries, with heavier tails for archetypes whose
# behavior plausibly involves more repeated attempts (financially
# constrained retrying the same payment, suspicious/ATO card-testing-like
# repeated attempts).
_RETRY_LAMBDA_BY_ARCHETYPE: Dict[str, float] = {
    "loyal_low_risk": 0.3,
    "normal": 0.5,
    "new_customer": 0.4,
    "financially_constrained": 0.9,
    "suspicious": 1.3,
    "account_takeover_like": 1.1,
}
_MAX_RETRY_COUNT: int = 5

# Multiplier on the archetype-based retry lambda, conditioned on WHY the
# transaction failed - this is the "failure context" half of retry
# propensity, layered on top of the "customer behavior" half (archetype).
# Transient/technical failures invite more retries (the customer or system
# just tries again); terminal/adverse failures invite far fewer, since
# retrying rarely helps and may not even be attempted.
_RETRY_REASON_MULTIPLIER: Dict[str, float] = {
    "bank_timeout": 1.4,           # transient infra hiccup - retry readily
    "network_error": 1.4,          # transient infra hiccup - retry readily
    "insufficient_funds": 1.3,     # customer plausibly retries after top-up
    "otp_expired": 1.0,            # simple UX slip - neutral retry propensity
    "card_declined_issuer": 0.6,   # often needs a different card, not a retry
    "invalid_cvv": 0.6,            # input error - retrying blindly rarely helps
    "customer_cancelled": 0.2,     # customer chose not to pay - retry unlikely
    "fraud_suspected_by_bank": 0.2,  # bank-blocked - retry unlikely/undesirable
}


def _compute_failure_probabilities(
    merged: pd.DataFrame,
) -> np.ndarray:
    """
    Compute a per-row failure probability from archetype base rate,
    payment-method multiplier, amount-anomaly multiplier, and a customer
    history multiplier - all combined multiplicatively and clipped to a
    sane range. No future/outcome information is used.
    """
    base = merged["archetype"].map(_BASE_FAILURE_RATE_BY_ARCHETYPE).to_numpy()
    method_mult = merged["payment_method"].map(_METHOD_FAILURE_MULTIPLIER).to_numpy()

    amount_ratio = merged["amount"] / merged["avg_transaction_amount_customer"]
    amount_mult = np.where(
        amount_ratio > _AMOUNT_ANOMALY_RATIO_THRESHOLD, _AMOUNT_ANOMALY_MULTIPLIER, 1.0
    )

    # Customers with lower historical success are somewhat more likely to
    # fail again - a mild multiplier (1.0 to 1.5), not a dominant effect,
    # since archetype already captures most of this signal.
    history_mult = 1.0 + (1.0 - merged["customer_past_success_rate"].to_numpy()) * 0.5

    probs = base * method_mult * amount_mult * history_mult
    return np.clip(probs, _MIN_FAILURE_PROB, _MAX_FAILURE_PROB)


def _sample_failure_reasons(
    archetypes: pd.Series, payment_methods: pd.Series, rng: np.random.Generator
) -> np.ndarray:
    """
    Sample a failure_reason_code for each (already-decided-failed) row,
    using an archetype-based probability distribution adjusted by the
    transaction's payment method, then renormalized. Always a probabilistic
    draw - never a deterministic archetype -> reason lookup.
    """
    reasons = np.empty(len(archetypes), dtype=object)
    for i, (arch, method) in enumerate(zip(archetypes, payment_methods)):
        weights = dict(_FAILURE_REASON_WEIGHTS_BY_ARCHETYPE[arch])
        method_boost = _METHOD_REASON_MULTIPLIERS.get(method, {})
        for reason, boost in method_boost.items():
            weights[reason] = weights[reason] * boost

        codes = list(weights.keys())
        probs = np.array([weights[c] for c in codes], dtype=float)
        probs = probs / probs.sum()
        reasons[i] = rng.choice(codes, p=probs)
    return reasons


def _sample_retry_counts(
    archetypes: pd.Series,
    failure_reason_code: np.ndarray,
    is_failed: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Sample retry_count_so_far for failed rows via a Poisson draw whose
    mean depends on BOTH customer behavior (archetype-based lambda) and
    failure context (a per-reason multiplier on that lambda), capped at
    _MAX_RETRY_COUNT. Successful transactions get 0 (no retries were
    needed). Uses only archetype and failure_reason_code - never
    recovery_label or risk_label, which do not exist at this stage.
    """
    archetype_lambda = archetypes.map(_RETRY_LAMBDA_BY_ARCHETYPE).to_numpy()
    # Non-failed rows have no reason code; default multiplier of 1.0 is
    # irrelevant for them since they're zeroed out below regardless.
    reason_multiplier = np.array(
        [_RETRY_REASON_MULTIPLIER.get(r, 1.0) for r in failure_reason_code]
    )
    combined_lambda = archetype_lambda * reason_multiplier

    raw_counts = rng.poisson(lam=np.clip(combined_lambda, 0.01, None))
    raw_counts = np.clip(raw_counts, 0, _MAX_RETRY_COUNT)
    return np.where(is_failed, raw_counts, 0).astype(int)


def generate_failure_context(
    transactions: pd.DataFrame,
    customer_profiles: pd.DataFrame,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Add failure_reason_code, is_soft_failure, and retry_count_so_far to a
    transaction-context DataFrame.

    Every existing row and column from `transactions` is preserved
    unchanged; exactly three new columns are appended. Transactions that
    are decided (probabilistically) not to fail get failure_reason_code
    and is_soft_failure set to null (pd.NA) and retry_count_so_far = 0.

    Args:
        transactions: output of transaction_context_generator.generate_transaction_context.
        customer_profiles: output of customer_generator.generate_customer_profiles,
            used only to look up each transaction's customer archetype and
            history - never persisted onto the returned DataFrame.
        seed: seed for a local, isolated random generator.

    Returns:
        pandas.DataFrame identical to `transactions` plus failure_reason_code,
        is_soft_failure, and retry_count_so_far.
    """
    if len(transactions) == 0:
        raise ValueError("transactions must contain at least one row.")

    rng = np.random.default_rng(seed)

    # Join in archetype and history purely for probability computation -
    # these columns are dropped before returning.
    merge_cols = ["customer_id", "archetype", "customer_past_success_rate",
                  "avg_transaction_amount_customer"]
    merged = transactions.merge(customer_profiles[merge_cols], on="customer_id", how="left")

    failure_probs = _compute_failure_probabilities(merged)
    is_failed = rng.random(len(merged)) < failure_probs

    failure_reason_code = np.full(len(merged), None, dtype=object)
    if is_failed.any():
        failure_reason_code[is_failed] = _sample_failure_reasons(
            merged.loc[is_failed, "archetype"],
            merged.loc[is_failed, "payment_method"],
            rng,
        )

    # is_soft_failure is DERIVED, never independently sampled - a
    # definitional lookup against config.SOFT_FAILURE_REASONS.
    is_soft_failure = np.array(
        [SOFT_FAILURE_REASONS[r] if r is not None else pd.NA for r in failure_reason_code],
        dtype=object,
    )

    retry_count_so_far = _sample_retry_counts(
        merged["archetype"], failure_reason_code, is_failed, rng
    )

    result = transactions.copy()
    result["failure_reason_code"] = failure_reason_code
    result["is_soft_failure"] = pd.array(is_soft_failure, dtype="boolean")
    result["retry_count_so_far"] = retry_count_so_far

    _validate_failure_context(result, expected_rows=len(transactions))
    return result


def _validate_failure_context(df: pd.DataFrame, expected_rows: int) -> None:
    """Structural/statistical sanity checks. Raises AssertionError on failure."""
    assert len(df) == expected_rows, f"Expected {expected_rows} rows, got {len(df)}."

    non_null_reasons = df["failure_reason_code"].dropna()
    assert non_null_reasons.isin(FAILURE_REASON_CODES).all(), (
        "Found failure_reason_code values outside the configured set."
    )

    # is_soft_failure must exactly match the deterministic mapping for
    # every failed row, and must be null wherever there was no failure.
    for reason, soft_flag in zip(df["failure_reason_code"], df["is_soft_failure"]):
        if reason is None or (isinstance(reason, float) and pd.isna(reason)):
            assert pd.isna(soft_flag), "Non-failed row must have null is_soft_failure."
        else:
            assert soft_flag == SOFT_FAILURE_REASONS[reason], (
                f"is_soft_failure mismatch for reason '{reason}'."
            )

    assert (df["retry_count_so_far"] >= 0).all(), "retry_count_so_far must be non-negative."
    assert df["retry_count_so_far"].dtype.kind in "iu", "retry_count_so_far must be integer."

    required_fields = [
        "transaction_id", "customer_id", "merchant_id",
        "timestamp", "amount", "currency", "payment_method", "retry_count_so_far",
    ]
    assert not df[required_fields].isnull().any().any(), (
        "Missing values found in required (non-failure-dependent) fields."
    )


if __name__ == "__main__":
    from .customer_generator import generate_customer_profiles
    from .transaction_context_generator import generate_transaction_context

    small_customers = generate_customer_profiles(num_customers=30, seed=RANDOM_SEED)
    tx = generate_transaction_context(small_customers, num_transactions=100, seed=RANDOM_SEED)
    tx_with_failures = generate_failure_context(tx, small_customers, seed=RANDOM_SEED)

    failed_mask = tx_with_failures["failure_reason_code"].notna()
    print("Failure rate:", round(failed_mask.mean(), 3))

    print("\nFailure reason distribution:")
    print(tx_with_failures.loc[failed_mask, "failure_reason_code"].value_counts())

    print("\nSoft vs hard failure distribution:")
    print(tx_with_failures.loc[failed_mask, "is_soft_failure"].value_counts())

    print("\nRetry count distribution:")
    print(tx_with_failures["retry_count_so_far"].value_counts().sort_index())