"""
customer_generator.py

Generates the customer-level table for the RevenueRescue AI synthetic
dataset: one row per unique customer, carrying their assigned archetype
and the static, latent attributes that later modules (transaction context,
failure, risk signals, labels) will condition on.

This module produces CUSTOMERS only - no transactions, no timestamps, no
payment methods, no failure reasons, no velocity counts, and no labels.
That separation matters: customer-level "latent truth" must exist before
any transaction is generated, otherwise transaction features would have
nothing consistent to be conditioned on.
"""

import numpy as np
import pandas as pd

from .archetypes import ARCHETYPES
from .config import ARCHETYPE_WEIGHTS, RANDOM_SEED

# Small jitter applied to per-customer behavioral tendencies so that
# customers of the same archetype are not exact statistical clones of one
# another. This is a local implementation detail (not a config constant)
# because it is purely a "make sampling realistic" mechanism, not a
# tunable business parameter.
_TENDENCY_JITTER_STD: float = 0.03

# Tenure (in days) below which a customer is considered to have
# effectively no payment history yet - used to force zero historical
# rates and zero chargebacks, satisfying the "brand-new customer with
# zero historical data" requirement deterministically rather than leaving
# it to chance.
_ZERO_HISTORY_TENURE_THRESHOLD_DAYS: int = 3


def _clip01(values: np.ndarray) -> np.ndarray:
    """Clip an array of probabilities/tendencies into the valid [0, 1] range."""
    return np.clip(values, 0.0, 1.0)


def _sample_archetypes(num_customers: int, rng: np.random.Generator) -> np.ndarray:
    """Assign one archetype to each customer using ARCHETYPE_WEIGHTS."""
    names = list(ARCHETYPE_WEIGHTS.keys())
    probs = np.array(list(ARCHETYPE_WEIGHTS.values()), dtype=float)
    probs = probs / probs.sum()  # defensive re-normalization
    return rng.choice(names, size=num_customers, p=probs)


def _sample_field_per_archetype(
    archetype_labels: np.ndarray,
    rng: np.random.Generator,
    range_getter,
) -> np.ndarray:
    """
    Sample a uniform value within each customer's archetype-specific range.

    `range_getter` is a function: CustomerArchetype -> (low, high) tuple,
    read directly from archetypes.py - no ranges are duplicated here.
    Sampling is done per-customer (not per-archetype-group) so every
    customer gets an independently drawn value within their archetype's
    bounds, rather than all customers in an archetype sharing one value.
    """
    out = np.empty(len(archetype_labels), dtype=float)
    for i, arch_name in enumerate(archetype_labels):
        low, high = range_getter(ARCHETYPES[arch_name])
        out[i] = rng.uniform(low, high)
    return out


def _sample_tendency_with_jitter(
    archetype_labels: np.ndarray,
    rng: np.random.Generator,
    tendency_getter,
) -> np.ndarray:
    """
    Sample a per-customer tendency value centered on the archetype's base
    tendency (from archetypes.py), with small Gaussian jitter, clipped to
    [0, 1]. This is what keeps same-archetype customers from being exact
    clones of one another on behavioral fields.
    """
    base = np.array(
        [tendency_getter(ARCHETYPES[a]) for a in archetype_labels], dtype=float
    )
    jitter = rng.normal(loc=0.0, scale=_TENDENCY_JITTER_STD, size=base.shape)
    return _clip01(base + jitter)


def generate_customer_profiles(
    num_customers: int,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Generate a customer-level DataFrame with exactly `num_customers` rows.

    Each customer is assigned one archetype (via ARCHETYPE_WEIGHTS) and a
    set of static attributes sampled within that archetype's declared
    ranges/tendencies from archetypes.py. No transaction-level, timestamp,
    payment-method, failure, velocity, or label data is produced here.

    Args:
        num_customers: exact number of unique customers to generate.
        seed: seed for a local, isolated random generator. Does NOT touch
            NumPy's global random state, so this call is safe to use
            alongside other generators without cross-contamination.

    Returns:
        pandas.DataFrame with one row per customer and the columns:
        customer_id, archetype, customer_tenure_days,
        customer_past_success_rate, customer_past_recovery_rate,
        avg_transaction_amount_customer, chargeback_history_count,
        device_change_tendency, ip_mismatch_tendency,
        method_switching_tendency, velocity_tendency,
        nudge_ignore_tendency, opt_out_tendency.
    """
    if num_customers <= 0:
        raise ValueError("num_customers must be a positive integer.")

    # Local, isolated random generator - does not mutate global NumPy state,
    # so this function is safe to call repeatedly or alongside other
    # generation modules without one call affecting another's randomness.
    rng = np.random.default_rng(seed)

    archetype_labels = _sample_archetypes(num_customers, rng)

    # --- Tenure -------------------------------------------------------
    tenure_days = _sample_field_per_archetype(
        archetype_labels, rng, lambda a: a.tenure_days_range
    )
    tenure_days = np.round(tenure_days).astype(int)
    tenure_days = np.clip(tenure_days, 0, None)  # tenure can never be negative

    is_zero_history = tenure_days < _ZERO_HISTORY_TENURE_THRESHOLD_DAYS

    # --- Historical success / recovery rates -------------------------------
    past_success_rate = _sample_field_per_archetype(
        archetype_labels, rng, lambda a: a.historical_success_rate_range
    )
    past_recovery_rate = _sample_field_per_archetype(
        archetype_labels, rng, lambda a: a.historical_recovery_rate_range
    )
    # Force zero history for brand-new customers rather than leaving it to
    # the sampled range alone - this guarantees the required zero-history
    # edge case deterministically, instead of inventing history that
    # couldn't exist yet.
    past_success_rate = np.where(is_zero_history, 0.0, past_success_rate)
    past_recovery_rate = np.where(is_zero_history, 0.0, past_recovery_rate)
    past_success_rate = _clip01(past_success_rate)
    past_recovery_rate = _clip01(past_recovery_rate)

    # --- Typical transaction amount ----------------------------------------
    avg_amount = _sample_field_per_archetype(
        archetype_labels, rng, lambda a: a.typical_amount_range
    )
    avg_amount = np.round(np.clip(avg_amount, a_min=1.0, a_max=None), 2)

    # --- Chargeback history count -------------------------------------------
    # Sampled PROBABILISTICALLY from each customer's archetype
    # chargeback_tendency - never a deterministic function of the
    # archetype label itself.
    chargeback_tendencies = np.array(
        [ARCHETYPES[a].chargeback_tendency for a in archetype_labels]
    )
    has_chargeback = rng.random(num_customers) < chargeback_tendencies
    # When a customer does have chargeback history, draw a small positive
    # count (1 to 4) rather than an unbounded value.
    chargeback_magnitude = 1 + rng.poisson(lam=0.6, size=num_customers)
    chargeback_magnitude = np.clip(chargeback_magnitude, 1, 4)
    chargeback_history_count = np.where(has_chargeback, chargeback_magnitude, 0)
    # Brand-new customers cannot have chargeback history regardless of
    # archetype tendency - there has been no time to accumulate one.
    chargeback_history_count = np.where(is_zero_history, 0, chargeback_history_count)
    chargeback_history_count = chargeback_history_count.astype(int)

    # --- Behavioral tendencies (carried forward per-customer, with jitter) --
    device_change_tendency = _sample_tendency_with_jitter(
        archetype_labels, rng, lambda a: a.device_change_tendency
    )
    ip_mismatch_tendency = _sample_tendency_with_jitter(
        archetype_labels, rng, lambda a: a.ip_mismatch_tendency
    )
    method_switching_tendency = _sample_tendency_with_jitter(
        archetype_labels, rng, lambda a: a.method_switching_tendency
    )
    velocity_tendency = _sample_tendency_with_jitter(
        archetype_labels, rng, lambda a: a.velocity_tendency
    )
    nudge_ignore_tendency = _sample_tendency_with_jitter(
        archetype_labels, rng, lambda a: a.nudge_ignore_tendency
    )
    opt_out_tendency = _sample_tendency_with_jitter(
        archetype_labels, rng, lambda a: a.opt_out_tendency
    )

    customer_ids = [f"cust_{i:06d}" for i in range(num_customers)]

    df = pd.DataFrame(
        {
            "customer_id": customer_ids,
            "archetype": archetype_labels,
            "customer_tenure_days": tenure_days,
            "customer_past_success_rate": past_success_rate,
            "customer_past_recovery_rate": past_recovery_rate,
            "avg_transaction_amount_customer": avg_amount,
            "chargeback_history_count": chargeback_history_count,
            "device_change_tendency": device_change_tendency,
            "ip_mismatch_tendency": ip_mismatch_tendency,
            "method_switching_tendency": method_switching_tendency,
            "velocity_tendency": velocity_tendency,
            "nudge_ignore_tendency": nudge_ignore_tendency,
            "opt_out_tendency": opt_out_tendency,
        }
    )

    _validate_customer_profiles(df, expected_rows=num_customers)
    return df


def _validate_customer_profiles(df: pd.DataFrame, expected_rows: int) -> None:
    """
    Run structural/statistical sanity checks on the generated customer
    table. Raises AssertionError on any violation - generation should
    fail loudly rather than silently produce a malformed dataset.
    """
    assert len(df) == expected_rows, (
        f"Expected {expected_rows} customer rows, got {len(df)}."
    )
    assert df["customer_id"].is_unique, "customer_id values must be unique."

    required_fields = df.columns.tolist()
    assert not df[required_fields].isnull().any().any(), (
        "Missing values found in required customer fields."
    )

    assert set(df["archetype"].unique()).issubset(set(ARCHETYPES.keys())), (
        "Unexpected archetype label found in generated customers."
    )

    probability_fields = [
        "customer_past_success_rate",
        "customer_past_recovery_rate",
        "device_change_tendency",
        "ip_mismatch_tendency",
        "method_switching_tendency",
        "velocity_tendency",
        "nudge_ignore_tendency",
        "opt_out_tendency",
    ]
    for field in probability_fields:
        assert df[field].between(0.0, 1.0).all(), (
            f"Field '{field}' contains values outside [0, 1]."
        )

    assert (df["customer_tenure_days"] >= 0).all(), (
        "customer_tenure_days must be non-negative."
    )
    assert (df["chargeback_history_count"] >= 0).all(), (
        "chargeback_history_count must be non-negative."
    )
    assert (df["avg_transaction_amount_customer"] > 0).all(), (
        "avg_transaction_amount_customer must be positive."
    )

    # Amount and tenure must respect each customer's OWN archetype bounds
    # (not just be globally positive/non-negative) - checked per row
    # against archetypes.py, the single source of truth for these ranges.
    for arch_name, arch in ARCHETYPES.items():
        subset = df[df["archetype"] == arch_name]
        if subset.empty:
            continue
        tenure_low, tenure_high = arch.tenure_days_range
        assert subset["customer_tenure_days"].between(tenure_low, tenure_high).all(), (
            f"'{arch_name}' customers have tenure outside {arch.tenure_days_range}."
        )
        amount_low, amount_high = arch.typical_amount_range
        assert subset["avg_transaction_amount_customer"].between(
            amount_low, amount_high
        ).all(), (
            f"'{arch_name}' customers have avg amount outside {arch.typical_amount_range}."
        )


if __name__ == "__main__":
    # Small smoke test: generate 30 customers and inspect the output.
    sample_df = generate_customer_profiles(num_customers=30, seed=RANDOM_SEED)

    print("Shape:", sample_df.shape)

    print("\nArchetype distribution:")
    print(sample_df["archetype"].value_counts())

    print("\nFirst 5 rows:")
    print(sample_df.head(5).to_string())

    print("\nTenure range:")
    print(
        f"  min={sample_df['customer_tenure_days'].min()}  "
        f"max={sample_df['customer_tenure_days'].max()}"
    )

    zero_history_count = (
        sample_df["customer_past_success_rate"].eq(0.0)
        & sample_df["customer_past_recovery_rate"].eq(0.0)
    ).sum()
    print(f"\nZero-history customer count: {zero_history_count}")