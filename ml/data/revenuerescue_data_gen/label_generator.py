"""
label_generator.py

Generates the two target labels for the RevenueRescue AI synthetic
dataset: recovery_label (did a failed payment later recover) and
risk_label (is this transaction a synthetic fraud/risk outcome).

Both labels are produced from LARGELY INDEPENDENT feature families and
independent noise draws, so that recovery and risk are never derivable
from one another - this is what makes the required "high recovery + high
risk" and "low recovery + low risk" style conflicts possible.

risk_label is a SYNTHETIC, illustrative fraud/risk outcome for this
hackathon prototype - it is not, and does not claim to be, real Razorpay
fraud data.
"""

from typing import Dict

import numpy as np
import pandas as pd
from numpy.random import SeedSequence, default_rng

from .config import (
    LABEL_NOISE_RATE,
    RANDOM_SEED,
    RECOVERY_BASE_RATE_HARD_FAILURE,
    RECOVERY_BASE_RATE_SOFT_FAILURE,
    RECOVERY_FATIGUE_PENALTY_WEIGHT,
    RECOVERY_HISTORY_WEIGHT,
    RECOVERY_RETRY_DECAY_FACTOR,
    RISK_BASE_RATE_BY_ARCHETYPE,
    RISK_INTERACTION_BOOST_WEIGHT,
)

# ---------------------------------------------------------------------------
# Local, module-specific parameters
#
# These add realistic nuance beyond what the shared config.py weights
# cover, without duplicating anything already defined there.
# ---------------------------------------------------------------------------

# Mild per-payment-method recovery multiplier - simpler rails (UPI,
# wallet) are slightly easier to successfully retry than card/netbanking,
# which often require more manual steps. Kept small so payment_method is
# a minor, not dominant, recovery factor.
_METHOD_RECOVERY_MULTIPLIER: Dict[str, float] = {
    "UPI": 1.05, "wallet": 1.05, "card": 0.95, "netbanking": 0.95,
}

# Per-reason nuance layered on top of the soft/hard base rate split.
# insufficient_funds sits slightly below the generic soft-failure rate -
# reflecting that it typically needs a delay, not an instant retry - but
# still well above the hard-failure rate, so it is never "never recovers".
# fraud_suspected_by_bank and customer_cancelled sit below the generic
# hard-failure rate, since retrying rarely helps in either case.
_REASON_RECOVERY_MULTIPLIER: Dict[str, float] = {
    "bank_timeout": 1.05, "network_error": 1.05, "otp_expired": 1.0,
    "insufficient_funds": 0.95, "card_declined_issuer": 1.0, "invalid_cvv": 1.0,
    "fraud_suspected_by_bank": 0.5, "customer_cancelled": 0.6,
}

# A transaction far above the customer's own typical amount gets a MILD
# recovery penalty - deliberately gentle, so high-value transactions are
# never automatically unrecoverable.
_AMOUNT_ANOMALY_RECOVERY_RATIO_THRESHOLD: float = 3.0
_AMOUNT_ANOMALY_RECOVERY_MULTIPLIER: float = 0.92

# Overall dampening applied to the final blended recovery probability.
# Without this, the configured base rates alone (0.70 for soft failures,
# which make up most failures in this dataset) push average recovery well
# above realistic levels: a raw "would this customer respond favorably"
# probability is not the same as "this failed payment actually ends up
# recovered", since many attempts stall, get abandoned, or never
# complete even under favorable conditions. This constant brings the
# dataset's overall recovery-among-failed rate into
# EXPECTED_RECOVERY_PREVALENCE_RANGE (config.py) - calibrated empirically
# against the full 20,000-transaction generated dataset - without
# altering the RELATIVE ordering driven by is_soft_failure, retry decay,
# history, fatigue, or amount (it is applied uniformly, after those
# effects, as the last step before clipping).
_RECOVERY_OVERALL_SCALE: float = 0.90

# Recovery probability is always kept strictly probabilistic - never
# certain, never impossible.
_RECOVERY_PROB_MIN: float = 0.03
_RECOVERY_PROB_MAX: float = 0.95

# Risk-side interaction thresholds: what counts as "high" diversity /
# "high" 1h velocity for the purposes of the required interaction boost.
_HIGH_DIVERSITY_THRESHOLD: int = 3
_HIGH_VELOCITY_1H_THRESHOLD: int = 2

# Chargeback history contributes a meaningful, capped, additive risk
# increase - each prior chargeback adds risk, but with diminishing
# marginal relevance beyond a handful of occurrences.
_CHARGEBACK_RISK_WEIGHT_PER_COUNT: float = 0.07
_CHARGEBACK_RISK_COUNT_CAP: int = 4

# A transaction far above the customer's own typical amount also nudges
# risk probability up slightly - independent of, and much smaller than,
# the archetype/behavioral-signal contributions.
_AMOUNT_ANOMALY_RISK_RATIO_THRESHOLD: float = 3.0
_AMOUNT_ANOMALY_RISK_BOOST: float = 0.05

# Risk probability is always kept strictly probabilistic - even
# suspicious/account_takeover_like customers are not always fraudulent.
_RISK_PROB_MIN: float = 0.003
_RISK_PROB_MAX: float = 0.95


def _compute_recovery_probabilities(merged: pd.DataFrame) -> np.ndarray:
    """
    Compute a per-row recovery probability for FAILED transactions, from:
    is_soft_failure, failure_reason_code, retry_count_so_far,
    customer_past_recovery_rate, nudge_ignore_tendency (fatigue proxy),
    payment_method, and amount vs the customer's typical amount.

    Values for non-failed rows are meaningless and are never used by the
    caller (recovery_label is set to null for those rows regardless).
    """
    is_soft = merged["is_soft_failure"].fillna(False).to_numpy(dtype=bool)
    base = np.where(is_soft, RECOVERY_BASE_RATE_SOFT_FAILURE, RECOVERY_BASE_RATE_HARD_FAILURE)

    reason_mult = merged["failure_reason_code"].map(_REASON_RECOVERY_MULTIPLIER).fillna(1.0).to_numpy()
    method_mult = merged["payment_method"].map(_METHOD_RECOVERY_MULTIPLIER).fillna(1.0).to_numpy()

    # Diminishing returns: each additional prior retry multiplies
    # probability down by RECOVERY_RETRY_DECAY_FACTOR.
    retry_decay = RECOVERY_RETRY_DECAY_FACTOR ** merged["retry_count_so_far"].to_numpy()

    prob_context = base * reason_mult * method_mult * retry_decay

    # Blend the context-driven probability with the customer's own
    # historical recovery behavior - strong personal history pulls
    # probability toward that customer's own track record.
    history = merged["customer_past_recovery_rate"].to_numpy()
    prob_blended = prob_context * (1 - RECOVERY_HISTORY_WEIGHT) + history * RECOVERY_HISTORY_WEIGHT

    # Fatigue suppresses nudge-driven recovery - customers who tend to
    # ignore nudges recover less often, all else equal.
    fatigue = merged["nudge_ignore_tendency"].to_numpy()
    prob_fatigue = prob_blended * (1 - RECOVERY_FATIGUE_PENALTY_WEIGHT * fatigue)

    amount_ratio = merged["amount"].to_numpy() / merged["avg_transaction_amount_customer"].to_numpy()
    amount_mult = np.where(
        amount_ratio > _AMOUNT_ANOMALY_RECOVERY_RATIO_THRESHOLD,
        _AMOUNT_ANOMALY_RECOVERY_MULTIPLIER,
        1.0,
    )

    prob_final = prob_fatigue * amount_mult * _RECOVERY_OVERALL_SCALE
    return np.clip(prob_final, _RECOVERY_PROB_MIN, _RECOVERY_PROB_MAX)


def _compute_risk_probabilities(merged: pd.DataFrame) -> np.ndarray:
    """
    Compute a per-row risk probability for EVERY transaction, from:
    archetype-specific base rate, method diversity x 1h velocity
    interaction, new-customer x IP-mismatch interaction, device-change x
    IP-mismatch interaction, chargeback history, and amount anomaly.

    Uses a feature family (velocity, diversity, device/IP flags,
    chargebacks) almost entirely disjoint from the recovery features
    above (failure reason, retries, recovery history, fatigue) - the two
    labels are not derivable from one another.
    """
    base = merged["archetype"].map(RISK_BASE_RATE_BY_ARCHETYPE).to_numpy()

    high_diversity = merged["num_payment_methods_used_recently"].to_numpy() >= _HIGH_DIVERSITY_THRESHOLD
    high_velocity = merged["velocity_txn_count_1h"].to_numpy() >= _HIGH_VELOCITY_1H_THRESHOLD
    diversity_velocity_boost = np.where(
        high_diversity & high_velocity, RISK_INTERACTION_BOOST_WEIGHT, 0.0
    )

    is_new = merged["is_new_customer"].to_numpy(dtype=bool)
    ip_mismatch = merged["ip_country_mismatch"].to_numpy(dtype=bool)
    device_change = merged["device_change_flag"].to_numpy(dtype=bool)

    new_ip_boost = np.where(is_new & ip_mismatch, RISK_INTERACTION_BOOST_WEIGHT, 0.0)
    device_ip_boost = np.where(device_change & ip_mismatch, RISK_INTERACTION_BOOST_WEIGHT, 0.0)

    chargeback_count = np.clip(
        merged["chargeback_history_count"].to_numpy(), 0, _CHARGEBACK_RISK_COUNT_CAP
    )
    chargeback_boost = chargeback_count * _CHARGEBACK_RISK_WEIGHT_PER_COUNT

    amount_ratio = merged["amount"].to_numpy() / merged["avg_transaction_amount_customer"].to_numpy()
    amount_boost = np.where(
        amount_ratio > _AMOUNT_ANOMALY_RISK_RATIO_THRESHOLD, _AMOUNT_ANOMALY_RISK_BOOST, 0.0
    )

    prob = base + diversity_velocity_boost + new_ip_boost + device_ip_boost + chargeback_boost + amount_boost
    return np.clip(prob, _RISK_PROB_MIN, _RISK_PROB_MAX)


def generate_labels(
    transactions: pd.DataFrame,
    customer_profiles: pd.DataFrame,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Add recovery_label and risk_label to a fully-featured transaction
    DataFrame (post velocity_engine.compute_velocity_features).

    recovery_label is "recovered" / "not_recovered" for FAILED
    transactions only (failure_reason_code not null), and null for
    successful transactions - a normal successful payment is never
    treated as a "recovered failure". risk_label ("fraudulent" /
    "legitimate") is generated for every transaction, failed or not.

    Both labels are sampled probabilistically (never via deterministic
    if/else rules) from largely independent feature families, with
    ~LABEL_NOISE_RATE independent label noise applied to each.

    Args:
        transactions: output of velocity_engine.compute_velocity_features.
        customer_profiles: output of customer_generator.generate_customer_profiles.
        seed: seed for a local, isolated random generator.

    Returns:
        pandas.DataFrame identical to `transactions` plus recovery_label
        and risk_label.
    """
    if len(transactions) == 0:
        raise ValueError("transactions must contain at least one row.")

    # IMPORTANT: every generator module in this pipeline independently
    # builds np.random.default_rng(seed) with the same default seed. In
    # isolation that's fine, but it means two modules' FIRST rng.random(n)
    # call, over arrays of the same length and row order, produce the
    # IDENTICAL underlying random numbers. Concretely: failure_generator's
    # is_failed decision is exactly "rng.random(n) < failure_probs" as its
    # first draw - reusing a plain default_rng(seed) here would make this
    # module's first draw for recovered_draw use those SAME numbers,
    # artificially correlating "was this row selected as failed" with
    # "is this row now selected as recovered" (failed rows are, by
    # construction, rows with a low random draw relative to their failure
    # probability - re-testing that same low draw against a DIFFERENT
    # threshold here would bias recovery upward for no genuine reason).
    # A SeedSequence with a module-specific tag produces a well-mixed,
    # independent stream from the same `seed` input, eliminating this
    # cross-module collision while remaining fully deterministic and
    # reproducible for a given seed.
    rng = default_rng(SeedSequence([seed, 0x4C4142]))  # 0x4C4142 = "LAB"

    merge_cols = [
        "customer_id", "archetype", "customer_past_success_rate",
        "customer_past_recovery_rate", "avg_transaction_amount_customer",
        "chargeback_history_count", "nudge_ignore_tendency",
    ]
    merged = transactions.merge(customer_profiles[merge_cols], on="customer_id", how="left")

    is_failed = merged["failure_reason_code"].notna().to_numpy()
    n = len(merged)

    # --- recovery_label ------------------------------------------------------
    recovery_prob = _compute_recovery_probabilities(merged)
    recovered_draw = rng.random(n) < recovery_prob
    noise_flip_recovery = rng.random(n) < LABEL_NOISE_RATE
    recovered_final = np.where(noise_flip_recovery, ~recovered_draw, recovered_draw)

    recovery_label = np.full(n, None, dtype=object)
    recovery_label[is_failed] = np.where(
        recovered_final[is_failed], "recovered", "not_recovered"
    )

    # --- risk_label ------------------------------------------------------------
    risk_prob = _compute_risk_probabilities(merged)
    fraud_draw = rng.random(n) < risk_prob
    noise_flip_risk = rng.random(n) < LABEL_NOISE_RATE
    fraud_final = np.where(noise_flip_risk, ~fraud_draw, fraud_draw)
    risk_label = np.where(fraud_final, "fraudulent", "legitimate")

    result = transactions.copy()
    result["recovery_label"] = recovery_label
    result["risk_label"] = risk_label

    _validate_labels(result, expected_rows=n)
    return result


def _validate_labels(df: pd.DataFrame, expected_rows: int) -> None:
    """Structural/type sanity checks. Raises AssertionError on failure."""
    assert len(df) == expected_rows, f"Expected {expected_rows} rows, got {len(df)}."
    assert "recovery_label" in df.columns and "risk_label" in df.columns

    non_null_recovery = df["recovery_label"].dropna()
    assert non_null_recovery.isin(["recovered", "not_recovered"]).all(), (
        "Found invalid recovery_label values."
    )
    # recovery_label must be null EXACTLY where there was no failure.
    assert (df["recovery_label"].isna() == df["failure_reason_code"].isna()).all(), (
        "recovery_label nullness does not match failure_reason_code nullness."
    )

    assert df["risk_label"].notna().all(), "risk_label must never be null."
    assert df["risk_label"].isin(["fraudulent", "legitimate"]).all(), (
        "Found invalid risk_label values."
    )


# ---------------------------------------------------------------------------
# Targeted scenario test: builds four crafted feature profiles designed to
# independently target {high, low} recovery x {high, low} risk, replicates
# each many times, and confirms all four combinations are actually
# achievable in the sampled output.
# ---------------------------------------------------------------------------

def _build_scenario_batch(
    scenario_name: str,
    n_replicas: int,
    *,
    archetype: str,
    is_soft_failure: bool,
    failure_reason_code: str,
    retry_count_so_far: int,
    customer_past_recovery_rate: float,
    customer_past_success_rate: float,
    nudge_ignore_tendency: float,
    chargeback_history_count: int,
    num_payment_methods_used_recently: int,
    velocity_txn_count_1h: int,
    velocity_txn_count_24h: int,
    ip_country_mismatch: bool,
    device_change_flag: bool,
    is_new_customer: bool,
    amount_ratio: float = 1.0,
) -> "tuple[pd.DataFrame, pd.DataFrame]":
    """Build a matched (transactions, customer_profiles) pair for one scenario."""
    customer_ids = [f"{scenario_name}_cust_{i:04d}" for i in range(n_replicas)]
    avg_amount = 1000.0

    customer_profiles = pd.DataFrame(
        {
            "customer_id": customer_ids,
            "archetype": archetype,
            "customer_past_success_rate": customer_past_success_rate,
            "customer_past_recovery_rate": customer_past_recovery_rate,
            "avg_transaction_amount_customer": avg_amount,
            "chargeback_history_count": chargeback_history_count,
            "nudge_ignore_tendency": nudge_ignore_tendency,
        }
    )
    transactions = pd.DataFrame(
        {
            "transaction_id": [f"{scenario_name}_txn_{i:04d}" for i in range(n_replicas)],
            "customer_id": customer_ids,
            "amount": avg_amount * amount_ratio,
            "payment_method": "UPI",
            "failure_reason_code": failure_reason_code,
            "is_soft_failure": is_soft_failure,
            "retry_count_so_far": retry_count_so_far,
            "num_payment_methods_used_recently": num_payment_methods_used_recently,
            "ip_country_mismatch": ip_country_mismatch,
            "device_change_flag": device_change_flag,
            "is_new_customer": is_new_customer,
            "velocity_txn_count_1h": velocity_txn_count_1h,
            "velocity_txn_count_24h": velocity_txn_count_24h,
        }
    )
    return transactions, customer_profiles


def _run_targeted_scenario_test(seed: int = RANDOM_SEED) -> None:
    """Build and evaluate the four required recovery x risk combinations."""
    n_replicas = 300

    scenarios = {
        "A_high_recovery_low_risk": _build_scenario_batch(
            "A", n_replicas, archetype="loyal_low_risk", is_soft_failure=True,
            failure_reason_code="bank_timeout", retry_count_so_far=0,
            customer_past_recovery_rate=0.90, customer_past_success_rate=0.90,
            nudge_ignore_tendency=0.05, chargeback_history_count=0,
            num_payment_methods_used_recently=1, velocity_txn_count_1h=0,
            velocity_txn_count_24h=0, ip_country_mismatch=False,
            device_change_flag=False, is_new_customer=False,
        ),
        "B_high_recovery_high_risk": _build_scenario_batch(
            "B", n_replicas, archetype="account_takeover_like", is_soft_failure=True,
            failure_reason_code="bank_timeout", retry_count_so_far=0,
            customer_past_recovery_rate=0.90, customer_past_success_rate=0.90,
            nudge_ignore_tendency=0.05, chargeback_history_count=3,
            num_payment_methods_used_recently=4, velocity_txn_count_1h=3,
            velocity_txn_count_24h=6, ip_country_mismatch=True,
            device_change_flag=True, is_new_customer=True,
        ),
        "C_low_recovery_low_risk": _build_scenario_batch(
            "C", n_replicas, archetype="loyal_low_risk", is_soft_failure=False,
            failure_reason_code="invalid_cvv", retry_count_so_far=3,
            customer_past_recovery_rate=0.10, customer_past_success_rate=0.10,
            nudge_ignore_tendency=0.80, chargeback_history_count=0,
            num_payment_methods_used_recently=1, velocity_txn_count_1h=0,
            velocity_txn_count_24h=0, ip_country_mismatch=False,
            device_change_flag=False, is_new_customer=False,
        ),
        "D_low_recovery_high_risk": _build_scenario_batch(
            "D", n_replicas, archetype="suspicious", is_soft_failure=False,
            failure_reason_code="invalid_cvv", retry_count_so_far=3,
            customer_past_recovery_rate=0.10, customer_past_success_rate=0.10,
            nudge_ignore_tendency=0.80, chargeback_history_count=3,
            num_payment_methods_used_recently=4, velocity_txn_count_1h=3,
            velocity_txn_count_24h=6, ip_country_mismatch=True,
            device_change_flag=True, is_new_customer=True,
        ),
    }

    print("Targeted scenario test (A/B/C/D x", n_replicas, "replicas each)")
    all_labeled = []
    for name, (tx, profiles) in scenarios.items():
        labeled = generate_labels(tx, profiles, seed=seed)
        recovery_rate = (labeled["recovery_label"] == "recovered").mean()
        fraud_rate = (labeled["risk_label"] == "fraudulent").mean()
        print(f"  {name}: recovery_rate={recovery_rate:.3f}  fraud_rate={fraud_rate:.3f}")
        all_labeled.append(labeled)

    combined = pd.concat(all_labeled, ignore_index=True)
    combined_failed = combined[combined["recovery_label"].notna()]
    crosstab = pd.crosstab(combined_failed["recovery_label"], combined_failed["risk_label"])
    print("\nCombined cross-tab (recovery_label x risk_label), all scenarios pooled:")
    print(crosstab)

    all_four_present = crosstab.shape == (2, 2) and (crosstab.to_numpy() > 0).all()
    print("\nAll four recovery x risk combinations observed:", all_four_present)
    assert all_four_present, "Not all four recovery/risk combinations occurred."
    print()


if __name__ == "__main__":
    from .customer_generator import generate_customer_profiles
    from .failure_generator import generate_failure_context
    from .risk_signal_generator import generate_risk_signals
    from .transaction_context_generator import generate_transaction_context
    from .velocity_engine import compute_velocity_features

    _run_targeted_scenario_test()

    small_customers = generate_customer_profiles(num_customers=30, seed=RANDOM_SEED)
    tx = generate_transaction_context(small_customers, num_transactions=100, seed=RANDOM_SEED)
    tx = generate_failure_context(tx, small_customers, seed=RANDOM_SEED)
    tx = generate_risk_signals(tx, small_customers, seed=RANDOM_SEED)
    tx = compute_velocity_features(tx)
    tx = generate_labels(tx, small_customers, seed=RANDOM_SEED)

    print("Shape:", tx.shape)

    print("\nRecovery label distribution (including null for non-failed):")
    print(tx["recovery_label"].value_counts(dropna=False))

    print("\nRisk label distribution:")
    print(tx["risk_label"].value_counts())

    failed_count = tx["failure_reason_code"].notna().sum()
    print(f"\nFailed transactions: {failed_count}  Non-failed transactions: {len(tx) - failed_count}")

    failed_only = tx[tx["recovery_label"].notna()]
    recovery_rate = (failed_only["recovery_label"] == "recovered").mean()
    print(f"\nRecovery rate among failed transactions: {recovery_rate:.3f}")

    fraud_rate = (tx["risk_label"] == "fraudulent").mean()
    print(f"Fraud/risk prevalence (all transactions): {fraud_rate:.3f}")

    print("\nCross-tab recovery_label x risk_label (failed transactions only):")
    print(pd.crosstab(failed_only["recovery_label"], failed_only["risk_label"]))