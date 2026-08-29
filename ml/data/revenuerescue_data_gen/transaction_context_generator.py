"""
transaction_context_generator.py

Generates transaction-level context rows (identifiers, timestamp, amount,
currency, payment method) conditioned on the customer-level profiles
produced by customer_generator.py.

This module deliberately stops short of anything that requires knowledge
of failure/risk/outcome: no failure_reason_code, no retry_count_so_far,
no risk signals, no velocity counts, and no labels are produced here.
Those belong to later modules that consume this module's output.
"""

from typing import Dict, List

import numpy as np
import pandas as pd

from .config import (
    AMOUNT_LOGNORMAL_SIGMA,
    AMOUNT_MAX,
    AMOUNT_MIN,
    DATE_RANGE_END,
    DATE_RANGE_START,
    PAYMENT_METHOD_WEIGHTS,
    PAYMENT_METHODS,
    RANDOM_SEED,
)

# ---------------------------------------------------------------------------
# Local, module-specific constants
#
# These are generation-mechanism details (how many merchants exist, how
# frequency/clustering are shaped) rather than business parameters, so they
# live here rather than in config.py, per the instruction not to modify
# other files in this step.
# ---------------------------------------------------------------------------

# Small, fixed, clearly-synthetic merchant pool. Kept small relative to the
# customer/transaction volume so merchant_id repeats naturally, and each
# customer only ever touches a limited subset of it (requirement 9).
_NUM_MERCHANTS: int = 120
_MERCHANT_POOL: List[str] = [f"merch_{i:04d}" for i in range(_NUM_MERCHANTS)]

# Relative transaction-frequency weight per archetype: how many
# transactions a customer of this archetype tends to generate, relative to
# others. This is INDEPENDENT of ARCHETYPE_WEIGHTS (which controls customer
# population share) - it controls activity level, not headcount.
_BASE_FREQUENCY_WEIGHT: Dict[str, float] = {
    "loyal_low_risk": 1.4,
    "normal": 1.0,
    "new_customer": 0.4,
    "financially_constrained": 0.9,
    "suspicious": 1.1,       # bursty card-testing activity, not just loyal usage
    "account_takeover_like": 0.6,
}

# Day-of-week weighting (Mon=0 ... Sun=6): mild lift toward the end of the
# week / weekend, reflecting typical consumer payment activity patterns.
_DAY_OF_WEEK_WEIGHTS: np.ndarray = np.array([1.0, 1.0, 1.0, 1.05, 1.15, 1.3, 1.2])

# Multiplier applied to late-month days (day-of-month >= 25) specifically
# for financially_constrained customers, per requirement 6.
_MONTH_END_BOOST: float = 1.8
_MONTH_END_DAY_THRESHOLD: int = 25

# Hour-of-day weighting (index = hour 0-23): low overnight, rising through
# the morning, peaking in the evening - a simple, generic activity curve.
_HOUR_WEIGHTS: np.ndarray = np.array(
    [0.2, 0.15, 0.1, 0.1, 0.1, 0.15,
     0.3, 0.5, 0.7, 0.9, 1.0, 1.1,
     1.2, 1.2, 1.1, 1.1, 1.2, 1.3,
     1.4, 1.5, 1.4, 1.2, 0.8, 0.4]
)
_HOUR_WEIGHTS_NORM: np.ndarray = _HOUR_WEIGHTS / _HOUR_WEIGHTS.sum()


def _assign_transaction_counts(
    customer_profiles: pd.DataFrame, num_transactions: int, rng: np.random.Generator
) -> np.ndarray:
    """
    Assign an integer transaction count to each customer such that counts
    sum EXACTLY to num_transactions, are non-uniform, and are biased by
    archetype activity level plus per-customer jitter.
    """
    base_weights = customer_profiles["archetype"].map(_BASE_FREQUENCY_WEIGHT).to_numpy()
    jitter = rng.lognormal(mean=0.0, sigma=0.5, size=len(customer_profiles))
    weights = base_weights * jitter
    probs = weights / weights.sum()
    # multinomial guarantees the counts sum exactly to num_transactions,
    # while probs (derived from weights) keep the distribution non-uniform.
    return rng.multinomial(num_transactions, probs)


def _build_calendar() -> pd.DatetimeIndex:
    """All calendar days (midnight-aligned) in the configured date range."""
    return pd.date_range(start=DATE_RANGE_START, end=DATE_RANGE_END, freq="D")


def _day_weights_for_archetype(calendar_days: pd.DatetimeIndex, archetype: str) -> np.ndarray:
    """
    Per-day sampling weights for a given archetype: day-of-week pattern,
    with an added month-end boost for financially_constrained customers.
    """
    weights = _DAY_OF_WEEK_WEIGHTS[calendar_days.weekday]
    if archetype == "financially_constrained":
        month_end_mask = calendar_days.day >= _MONTH_END_DAY_THRESHOLD
        weights = np.where(month_end_mask, weights * _MONTH_END_BOOST, weights)
    return weights / weights.sum()


def _add_time_of_day(day: pd.Timestamp, rng: np.random.Generator) -> pd.Timestamp:
    """Attach a realistically-weighted hour/minute/second to a calendar day."""
    hour = int(rng.choice(24, p=_HOUR_WEIGHTS_NORM))
    minute = int(rng.integers(0, 60))
    second = int(rng.integers(0, 60))
    return pd.Timestamp(day) + pd.Timedelta(hours=hour, minutes=minute, seconds=second)


def _sample_customer_timestamps(
    archetype: str,
    velocity_tendency: float,
    count: int,
    calendar_days: pd.DatetimeIndex,
    day_weights: np.ndarray,
    rng: np.random.Generator,
) -> List[pd.Timestamp]:
    """
    Sample `count` timestamps for one customer, clustered in time according
    to that customer's velocity_tendency: higher tendency -> fewer, tighter
    clusters (bursty behavior); lower tendency -> transactions spread more
    independently across the date range.
    """
    if count == 0:
        return []
    if count == 1:
        day = rng.choice(calendar_days, p=day_weights)
        return [_add_time_of_day(day, rng)]

    # Higher velocity_tendency -> smaller fraction -> fewer distinct clusters
    # -> transactions bunch together more tightly (card-testing-like bursts).
    frac = 1.0 - velocity_tendency
    num_clusters = int(np.clip(round(count * frac), 1, count))

    cluster_days = rng.choice(calendar_days, size=num_clusters, p=day_weights, replace=True)
    cluster_assignment = rng.integers(0, num_clusters, size=count)

    timestamps: List[pd.Timestamp] = []
    for c_idx in range(num_clusters):
        n_in_cluster = int(np.sum(cluster_assignment == c_idx))
        if n_in_cluster == 0:
            continue
        base_day = cluster_days[c_idx]
        # A SINGLE base timestamp anchors the whole cluster - one weighted
        # hour-of-day draw per cluster, not per transaction. Every other
        # transaction in this cluster is placed as a strictly increasing
        # offset from this one anchor, which is what makes the cluster
        # genuinely ordered and tightly grouped in time rather than a set
        # of independently-scattered hours with an offset added on top.
        base_ts = _add_time_of_day(base_day, rng)
        if n_in_cluster == 1:
            timestamps.append(base_ts)
            continue
        # Max gap between consecutive same-cluster transactions shrinks as
        # velocity_tendency rises, tightening the burst.
        max_gap_minutes = 240 * (1 - velocity_tendency) + 5
        gaps = rng.uniform(1.0, max_gap_minutes, size=n_in_cluster - 1)
        offsets_minutes = np.concatenate(([0.0], np.cumsum(gaps)))
        for offset_minutes in offsets_minutes:
            timestamps.append(base_ts + pd.Timedelta(minutes=float(offset_minutes)))
    return timestamps


def _sample_amounts(
    avg_amount: float, count: int, rng: np.random.Generator
) -> np.ndarray:
    """
    Sample `count` transaction amounts scattered around a customer's own
    typical amount, using a long-tailed lognormal multiplier so most
    transactions stay near their usual spend but occasional legitimate
    high-value transactions can still occur.
    """
    # A fraction of the dataset-wide amount spread parameter is reused here
    # to control per-customer variation around their own typical amount -
    # deliberately smaller than the global spread, since a single
    # customer's spend varies less than the population as a whole.
    per_customer_sigma = AMOUNT_LOGNORMAL_SIGMA * 0.5
    multipliers = rng.lognormal(mean=0.0, sigma=per_customer_sigma, size=count)
    amounts = avg_amount * multipliers
    return np.clip(np.round(amounts, 2), AMOUNT_MIN, AMOUNT_MAX)


def _sample_payment_methods(
    method_switching_tendency: float, favorite_method: str, count: int, rng: np.random.Generator
) -> List[str]:
    """
    Sample `count` payment methods for one customer: mostly their personal
    favorite method, occasionally switching to another method sampled from
    the global PAYMENT_METHOD_WEIGHTS distribution. The switching
    probability is driven by method_switching_tendency, so archetype
    influences this only indirectly (through that tendency), never via a
    hardcoded archetype -> method mapping.
    """
    switches = rng.random(count) < method_switching_tendency
    global_methods = list(PAYMENT_METHOD_WEIGHTS.keys())
    global_probs = np.array(list(PAYMENT_METHOD_WEIGHTS.values()))
    global_probs = global_probs / global_probs.sum()
    switched_choices = rng.choice(global_methods, size=count, p=global_probs)
    return [switched_choices[i] if switches[i] else favorite_method for i in range(count)]


def generate_transaction_context(
    customer_profiles: pd.DataFrame,
    num_transactions: int,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Generate a transaction-context DataFrame with exactly `num_transactions`
    rows, each tied to an existing customer_id from `customer_profiles`.

    Only transaction context is produced: identifiers, timestamp, amount,
    currency, and payment_method. No failure, risk, velocity, or label
    fields are computed here - later modules own those, and this function
    uses no information beyond each customer's own static profile, so
    nothing here can leak future/outcome information.

    Args:
        customer_profiles: output of customer_generator.generate_customer_profiles.
        num_transactions: exact number of transaction rows to generate.
        seed: seed for a local, isolated random generator.

    Returns:
        pandas.DataFrame with columns: transaction_id, customer_id,
        merchant_id, timestamp, amount, currency, payment_method - sorted
        chronologically by timestamp.
    """
    if num_transactions <= 0:
        raise ValueError("num_transactions must be a positive integer.")

    rng = np.random.default_rng(seed)
    calendar_days = _build_calendar()

    # Precompute per-archetype day-of-week/month-end weighting once, rather
    # than recomputing per customer.
    archetype_day_weights = {
        arch: _day_weights_for_archetype(calendar_days, arch)
        for arch in customer_profiles["archetype"].unique()
    }

    counts = _assign_transaction_counts(customer_profiles, num_transactions, rng)

    rows: List[Dict] = []
    for (_, customer), count in zip(customer_profiles.iterrows(), counts):
        if count == 0:
            continue

        # Each customer gets a small, fixed, personal merchant subset -
        # not a fresh random merchant per transaction.
        num_preferred_merchants = int(rng.integers(1, 5))
        preferred_merchants = rng.choice(
            _MERCHANT_POOL, size=num_preferred_merchants, replace=False
        )

        # Each customer gets one personal favorite payment method, sampled
        # from the global distribution once, then mostly reused.
        global_methods = list(PAYMENT_METHOD_WEIGHTS.keys())
        global_probs = np.array(list(PAYMENT_METHOD_WEIGHTS.values()))
        global_probs = global_probs / global_probs.sum()
        favorite_method = rng.choice(global_methods, p=global_probs)

        timestamps = _sample_customer_timestamps(
            archetype=customer["archetype"],
            velocity_tendency=customer["velocity_tendency"],
            count=int(count),
            calendar_days=calendar_days,
            day_weights=archetype_day_weights[customer["archetype"]],
            rng=rng,
        )
        amounts = _sample_amounts(
            avg_amount=customer["avg_transaction_amount_customer"], count=int(count), rng=rng
        )
        methods = _sample_payment_methods(
            method_switching_tendency=customer["method_switching_tendency"],
            favorite_method=favorite_method,
            count=int(count),
            rng=rng,
        )
        merchants = rng.choice(preferred_merchants, size=int(count), replace=True)

        for i in range(int(count)):
            rows.append(
                {
                    "customer_id": customer["customer_id"],
                    "merchant_id": merchants[i],
                    "timestamp": timestamps[i],
                    "amount": float(amounts[i]),
                    "currency": "INR",
                    "payment_method": methods[i],
                }
            )

    df = pd.DataFrame(rows)

    # Clip any timestamps that drifted past the configured range due to
    # intra-cluster time offsets near the boundary, then sort chronologically.
    range_start = pd.Timestamp(DATE_RANGE_START)
    range_end = pd.Timestamp(DATE_RANGE_END) + pd.Timedelta(hours=23, minutes=59, seconds=59)
    df["timestamp"] = df["timestamp"].clip(lower=range_start, upper=range_end)
    df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)

    # Assign transaction_id AFTER chronological sorting, so IDs reflect
    # arrival order - consistent with how real transaction IDs behave.
    df.insert(0, "transaction_id", [f"txn_{i:07d}" for i in range(len(df))])

    _validate_transaction_context(df, expected_rows=num_transactions, customer_profiles=customer_profiles)
    return df


def _validate_transaction_context(
    df: pd.DataFrame, expected_rows: int, customer_profiles: pd.DataFrame
) -> None:
    """Structural/statistical sanity checks. Raises AssertionError on failure."""
    assert len(df) == expected_rows, f"Expected {expected_rows} rows, got {len(df)}."
    assert df["transaction_id"].is_unique, "transaction_id values must be unique."

    valid_customer_ids = set(customer_profiles["customer_id"])
    assert set(df["customer_id"]).issubset(valid_customer_ids), (
        "Found customer_id values not present in customer_profiles."
    )

    range_start = pd.Timestamp(DATE_RANGE_START)
    range_end = pd.Timestamp(DATE_RANGE_END) + pd.Timedelta(days=1)
    assert df["timestamp"].between(range_start, range_end).all(), (
        "Found timestamps outside the configured date range."
    )

    assert (df["amount"] > 0).all(), "All amounts must be positive."
    assert df["amount"].between(AMOUNT_MIN, AMOUNT_MAX).all(), (
        "Found amounts outside configured global bounds."
    )
    assert df["payment_method"].isin(PAYMENT_METHODS).all(), "Found invalid payment_method values."
    assert (df["currency"] == "INR").all(), "All currency values must be INR."

    required_fields = [
        "transaction_id", "customer_id", "merchant_id",
        "timestamp", "amount", "currency", "payment_method",
    ]
    assert not df[required_fields].isnull().any().any(), "Missing values in required fields."


if __name__ == "__main__":
    from .customer_generator import generate_customer_profiles

    small_customers = generate_customer_profiles(num_customers=30, seed=RANDOM_SEED)
    tx_df = generate_transaction_context(small_customers, num_transactions=100, seed=RANDOM_SEED)

    print("Shape:", tx_df.shape)

    print("\nTransaction counts per customer:")
    print(tx_df["customer_id"].value_counts())

    print("\nPayment method distribution:")
    print(tx_df["payment_method"].value_counts(normalize=True))

    print("\nAmount stats:")
    print(
        f"  min={tx_df['amount'].min():.2f}  "
        f"median={tx_df['amount'].median():.2f}  "
        f"max={tx_df['amount'].max():.2f}"
    )

    print("\nTimestamp range:")
    print(f"  earliest={tx_df['timestamp'].min()}  latest={tx_df['timestamp'].max()}")