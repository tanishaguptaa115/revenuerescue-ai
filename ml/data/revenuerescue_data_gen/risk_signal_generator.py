"""
risk_signal_generator.py

Adds transaction-level risk signals - method diversity, IP/country
mismatch, device change, and new-customer status - to a transaction
context that already carries failure information.

This module deliberately does NOT compute velocity_txn_count_1h/24h.
Those require true chronological rolling-window logic across every
customer's full transaction history and are the sole responsibility of
velocity_engine.py. It also does not compute any score, label, fatigue
metric, or policy decision - those belong to later modules.
"""

from typing import List

import numpy as np
import pandas as pd

from .config import PAYMENT_METHODS, RANDOM_SEED

# ---------------------------------------------------------------------------
# Local, module-specific parameters
# ---------------------------------------------------------------------------

# Number of distinct payment methods that exist at all - the hard ceiling
# for method diversity, since a customer cannot have used more distinct
# methods than exist in the system.
_MAX_METHOD_DIVERSITY: int = len(PAYMENT_METHODS)

# How many of the customer's own most-recent transactions (in chronological
# order, current transaction inclusive) are considered when counting
# distinct payment methods "used recently". Kept small and local to this
# customer's own history - not a global velocity computation, and never
# looks at transactions that happen AFTER the current one.
_DIVERSITY_LOOKBACK_WINDOW: int = 5

# A customer with high method_switching_tendency occasionally gets a small
# extra bump to their observed diversity, representing method attempts not
# fully captured when a customer has very few transactions on record. This
# is a PROBABILITY scale factor, not a hard addition - it keeps diversity
# tied to the tendency without making it a free-standing random draw.
_METHOD_SWITCH_BONUS_SCALE: float = 0.5

# Tenure threshold (in days) below which a customer is flagged as "new".
# Matches the exact cold-start threshold used in customer_generator.py's
# zero-history logic, so this module's is_new_customer definition stays
# consistent with (not a contradiction of) that earlier concept.
_NEW_CUSTOMER_TENURE_THRESHOLD_DAYS: int = 3

# Archetypes for which device_change_flag and ip_country_mismatch should
# be correlated (both firing together more often), representing a single
# underlying compromise-like event - required interaction effect. The two
# fields remain independently stored columns; only their sampling is
# correlated for these archetypes.
_JOINT_RISK_ARCHETYPES = {"suspicious", "account_takeover_like"}

# How much the average of a customer's device/IP tendencies is boosted
# when deciding the probability of a SHARED compromise-like event, for
# customers in _JOINT_RISK_ARCHETYPES only.
_JOINT_EVENT_BOOST_FACTOR: float = 1.6

# Hard ceiling on the joint-event probability. Without this, archetypes
# with high combined tendencies (e.g. account_takeover_like, whose average
# device/IP tendency is already ~0.65) could have their boosted
# probability exceed 1.0 and saturate to certainty - which would make the
# signal effectively deterministic for that archetype, violating the
# requirement that these remain genuinely probabilistic.
_MAX_JOINT_EVENT_PROB: float = 0.90


def _rolling_distinct_count(values: np.ndarray, window: int) -> np.ndarray:
    """
    For a chronologically-ordered array of categorical values, return, for
    each position, the number of distinct values seen within the trailing
    `window` positions (current position inclusive). Uses only values at
    or before the current position - never looks ahead.
    """
    out = np.empty(len(values), dtype=int)
    for i in range(len(values)):
        start = max(0, i - window + 1)
        out[i] = len(set(values[start : i + 1]))
    return out


def _compute_recent_method_diversity(transactions: pd.DataFrame) -> np.ndarray:
    """
    Compute, for every transaction row, the number of distinct payment
    methods this customer has used within their own trailing
    _DIVERSITY_LOOKBACK_WINDOW transactions (current one included),
    grounded in the customer's ACTUAL payment_method history rather than
    an independently sampled number. Leak-safe: each row only considers
    that same customer's transactions at or before it in time.

    Returns an array aligned with `transactions`' original row order.
    """
    work = transactions[["customer_id", "timestamp", "payment_method"]].copy()
    work["_orig_pos"] = np.arange(len(work))
    # Sort by customer then time so the rolling window is chronologically
    # correct per customer, independent of the DataFrame's overall
    # (globally time-sorted, cross-customer) row order.
    work_sorted = work.sort_values(["customer_id", "timestamp"], kind="mergesort")

    diversity_sorted = work_sorted.groupby("customer_id")["payment_method"].transform(
        lambda s: _rolling_distinct_count(s.to_numpy(), _DIVERSITY_LOOKBACK_WINDOW)
    )

    result = np.empty(len(transactions), dtype=int)
    result[work_sorted["_orig_pos"].to_numpy()] = diversity_sorted.to_numpy()
    return result


def _sample_device_and_ip_signals(
    merged: pd.DataFrame, rng: np.random.Generator
) -> "tuple[np.ndarray, np.ndarray]":
    """
    Sample device_change_flag and ip_country_mismatch probabilistically
    from each customer's own tendencies, with a correlated "shared
    compromise-like event" boost for suspicious / account_takeover_like
    customers so the two signals co-occur more often for them - without
    ever collapsing them into a single column.
    """
    p_device = merged["device_change_tendency"].to_numpy()
    p_ip = merged["ip_mismatch_tendency"].to_numpy()
    is_joint_archetype = merged["archetype"].isin(_JOINT_RISK_ARCHETYPES).to_numpy()

    joint_prob = np.clip(((p_device + p_ip) / 2.0) * _JOINT_EVENT_BOOST_FACTOR, 0.0, _MAX_JOINT_EVENT_PROB)
    joint_event = rng.random(len(merged)) < joint_prob

    independent_device = rng.random(len(merged)) < p_device
    independent_ip = rng.random(len(merged)) < p_ip

    fires_joint = is_joint_archetype & joint_event
    device_change_flag = np.where(fires_joint, True, independent_device)
    ip_country_mismatch = np.where(fires_joint, True, independent_ip)
    return device_change_flag, ip_country_mismatch


def generate_risk_signals(
    transactions: pd.DataFrame,
    customer_profiles: pd.DataFrame,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Add num_payment_methods_used_recently, ip_country_mismatch,
    device_change_flag, and is_new_customer to a transaction context that
    already includes failure information.

    Every existing row and column from `transactions` is preserved
    unchanged; exactly four new columns are appended. Customer-level
    tendency fields are merged in only for computing these signals and
    are dropped before returning.

    Args:
        transactions: output of failure_generator.generate_failure_context
            (or transaction_context_generator.generate_transaction_context).
        customer_profiles: output of customer_generator.generate_customer_profiles.
        seed: seed for a local, isolated random generator.

    Returns:
        pandas.DataFrame identical to `transactions` plus
        num_payment_methods_used_recently, ip_country_mismatch,
        device_change_flag, and is_new_customer.
    """
    if len(transactions) == 0:
        raise ValueError("transactions must contain at least one row.")

    rng = np.random.default_rng(seed)

    merge_cols: List[str] = [
        "customer_id", "archetype", "method_switching_tendency",
        "device_change_tendency", "ip_mismatch_tendency", "customer_tenure_days",
    ]
    merged = transactions.merge(customer_profiles[merge_cols], on="customer_id", how="left")

    # --- num_payment_methods_used_recently ----------------------------------
    base_diversity = _compute_recent_method_diversity(transactions)
    switch_bonus_prob = merged["method_switching_tendency"].to_numpy() * _METHOD_SWITCH_BONUS_SCALE
    switch_bonus = (rng.random(len(merged)) < switch_bonus_prob).astype(int)
    num_payment_methods_used_recently = np.clip(
        base_diversity + switch_bonus, 1, _MAX_METHOD_DIVERSITY
    ).astype(int)

    # --- device_change_flag / ip_country_mismatch ---------------------------
    device_change_flag, ip_country_mismatch = _sample_device_and_ip_signals(merged, rng)

    # --- is_new_customer -----------------------------------------------------
    is_new_customer = merged["customer_tenure_days"].to_numpy() < _NEW_CUSTOMER_TENURE_THRESHOLD_DAYS

    result = transactions.copy()
    result["num_payment_methods_used_recently"] = num_payment_methods_used_recently
    result["ip_country_mismatch"] = pd.array(ip_country_mismatch, dtype="boolean")
    result["device_change_flag"] = pd.array(device_change_flag, dtype="boolean")
    result["is_new_customer"] = pd.array(is_new_customer, dtype="boolean")

    _validate_risk_signals(result, customer_profiles, expected_rows=len(transactions))
    return result


def _validate_risk_signals(
    df: pd.DataFrame, customer_profiles: pd.DataFrame, expected_rows: int
) -> None:
    """Structural/statistical sanity checks. Raises AssertionError on failure."""
    assert len(df) == expected_rows, f"Expected {expected_rows} rows, got {len(df)}."

    valid_customer_ids = set(customer_profiles["customer_id"])
    assert set(df["customer_id"]).issubset(valid_customer_ids), (
        "Found customer_id values not present in customer_profiles."
    )

    assert df["num_payment_methods_used_recently"].dtype.kind in "iu", (
        "num_payment_methods_used_recently must be integer."
    )
    assert (df["num_payment_methods_used_recently"] >= 1).all(), (
        "num_payment_methods_used_recently must be >= 1."
    )
    assert (df["num_payment_methods_used_recently"] <= _MAX_METHOD_DIVERSITY).all(), (
        f"num_payment_methods_used_recently must be <= {_MAX_METHOD_DIVERSITY}."
    )

    for flag_col in ["ip_country_mismatch", "device_change_flag", "is_new_customer"]:
        assert df[flag_col].notna().all(), f"'{flag_col}' must not contain nulls."
        assert df[flag_col].isin([True, False]).all(), f"'{flag_col}' contains invalid values."

    tenure_lookup = customer_profiles.set_index("customer_id")["customer_tenure_days"]
    expected_is_new = df["customer_id"].map(tenure_lookup) < _NEW_CUSTOMER_TENURE_THRESHOLD_DAYS
    assert (df["is_new_customer"].to_numpy() == expected_is_new.to_numpy()).all(), (
        "is_new_customer does not logically match customer_tenure_days."
    )

    required_fields = [
        "transaction_id", "customer_id", "merchant_id", "timestamp", "amount",
        "currency", "payment_method", "retry_count_so_far",
        "num_payment_methods_used_recently", "ip_country_mismatch",
        "device_change_flag", "is_new_customer",
    ]
    assert not df[required_fields].isnull().any().any(), (
        "Missing values found in required fields."
    )


if __name__ == "__main__":
    from .customer_generator import generate_customer_profiles
    from .failure_generator import generate_failure_context
    from .transaction_context_generator import generate_transaction_context

    small_customers = generate_customer_profiles(num_customers=30, seed=RANDOM_SEED)
    tx = generate_transaction_context(small_customers, num_transactions=100, seed=RANDOM_SEED)
    tx_failed = generate_failure_context(tx, small_customers, seed=RANDOM_SEED)
    tx_risk = generate_risk_signals(tx_failed, small_customers, seed=RANDOM_SEED)

    print("Shape:", tx_risk.shape)

    print("\nnum_payment_methods_used_recently distribution:")
    print(tx_risk["num_payment_methods_used_recently"].value_counts().sort_index())

    print("\nIP mismatch rate:", round(tx_risk["ip_country_mismatch"].mean(), 3))
    print("Device change rate:", round(tx_risk["device_change_flag"].mean(), 3))
    print("New customer rate:", round(tx_risk["is_new_customer"].mean(), 3))

    joint_rate = (tx_risk["ip_country_mismatch"] & tx_risk["device_change_flag"]).mean()
    print("Joint device_change + IP mismatch rate:", round(joint_rate, 3))

    merged_view = tx_risk.merge(
        small_customers[["customer_id", "archetype"]], on="customer_id"
    )
    print("\nRates by archetype:")
    print(
        merged_view.groupby("archetype")[
            ["ip_country_mismatch", "device_change_flag", "is_new_customer"]
        ].mean().round(3)
    )