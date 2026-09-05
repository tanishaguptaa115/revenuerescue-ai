
"""
velocity_engine.py

Computes TRUE chronological rolling-window transaction-velocity features
per customer: how many of that customer's own PRIOR transactions fall
within the trailing 1-hour and 24-hour windows before each transaction.

This is the one module in the pipeline that requires genuine time-ordered,
cross-row computation. Every other generator module works row-by-row (or
customer-by-customer, independent of other customers' transactions);
velocity is inherently about a customer's transaction SEQUENCE, which is
exactly why it was deliberately deferred out of
transaction_context_generator.py and risk_signal_generator.py until here.

No archetype information is used anywhere in this module - velocity must
emerge purely from actual timestamps, never from a customer's assigned
behavioral profile.
"""

import numpy as np
import pandas as pd

# Window sizes expressed in nanoseconds (pandas/numpy datetime64[ns]
# resolution), so comparisons can be done as plain integer arithmetic
# rather than repeated Timedelta operations - important for performance
# at 20,000+ rows.
_ONE_HOUR_NS: np.int64 = np.int64(60 * 60 * 1_000_000_000)
_TWENTY_FOUR_HOUR_NS: np.int64 = np.int64(24 * 60 * 60 * 1_000_000_000)


def _rolling_prior_count(times_ns: np.ndarray, window_ns: np.int64) -> np.ndarray:
    """
    Given a chronologically-sorted (ascending, ties allowed) array of
    timestamps in integer nanoseconds for a SINGLE customer, return, for
    every position i, the number of PRIOR positions j < i such that
    times_ns[i] - times_ns[j] <= window_ns.

    This is a vectorized sliding-window count: since the array is sorted
    ascending, for each i the set of qualifying j's is exactly the
    contiguous range [left_bound(i), i - 1], where left_bound(i) is the
    first index whose timestamp is >= times_ns[i] - window_ns. That left
    boundary is found via a single vectorized np.searchsorted call across
    all positions at once - O(n log n), not O(n^2).

    The current position i is never included (only j < i are considered),
    and no position > i is ever examined (times_ns is only ever compared
    to earlier entries in the same sorted array), so this can never look
    into the future.
    """
    n = len(times_ns)
    thresholds = times_ns - window_ns
    # side='left': first index with times_ns[idx] >= threshold. Since
    # times_ns is sorted ascending, everything in [left_bound, i-1] is
    # both >= threshold (within the window) and < position i (prior).
    left_bound = np.searchsorted(times_ns, thresholds, side="left")
    positions = np.arange(n)
    return positions - left_bound


def _days_since_last_success(times_ns: np.ndarray, is_success: np.ndarray) -> np.ndarray:
    """
    Given a chronologically-sorted array of timestamps (nanoseconds) and a
    same-length boolean array marking which of those transactions were
    successful (not failed) for a SINGLE customer, return, for every
    position i, the number of days since that customer's most recent
    PRIOR successful transaction - or NaN if no prior success exists yet.

    Leak-safe by construction: for each position i, the "prior success"
    tracker is only updated with position i's own success status AFTER
    that position's output value has already been written - so a
    transaction's own outcome can never influence its own feature value,
    and no later position is ever consulted.
    """
    n = len(times_ns)
    out = np.full(n, np.nan, dtype=float)
    ns_per_day = 24 * 60 * 60 * 1_000_000_000
    last_success_ns = None
    for i in range(n):
        if last_success_ns is not None:
            out[i] = (times_ns[i] - last_success_ns) / ns_per_day
        # Only AFTER recording this row's own output do we allow this
        # row's own outcome to become "prior" information for later rows.
        if is_success[i]:
            last_success_ns = times_ns[i]
    return out


def compute_velocity_features(transactions: pd.DataFrame) -> pd.DataFrame:
    """
    Add velocity_txn_count_1h, velocity_txn_count_24h, and
    days_since_last_successful_payment to a transaction DataFrame,
    computed as true chronological, leak-safe features of each
    customer's own transaction sequence.

    Works regardless of the input DataFrame's row order: internally sorts
    by (customer_id, timestamp, original row position) to get a fully
    deterministic chronological ordering per customer - the original row
    position breaks ties between same-timestamp transactions safely,
    without ever needing to look at rows that come later in time. Results
    are returned in the SAME row order as the input.

    Args:
        transactions: any DataFrame containing at least 'customer_id',
            'timestamp', and 'failure_reason_code' columns (e.g. the
            output of risk_signal_generator.generate_risk_signals, which
            itself runs after failure_generator.generate_failure_context).

    Returns:
        pandas.DataFrame identical to `transactions`, in the same row
        order, plus velocity_txn_count_1h, velocity_txn_count_24h, and
        days_since_last_successful_payment (null where no prior
        successful transaction exists for that customer yet).
    """
    if len(transactions) == 0:
        raise ValueError("transactions must contain at least one row.")
    if "failure_reason_code" not in transactions.columns:
        raise ValueError(
            "transactions must include 'failure_reason_code' to compute "
            "days_since_last_successful_payment - run failure_generator "
            "before velocity_engine."
        )

    n = len(transactions)
    work = transactions[["customer_id", "timestamp"]].copy()
    work["_orig_pos"] = np.arange(n)
    # is_success is read once here, purely to compute a leak-safe HISTORY
    # feature (days since a PRIOR success) - it is never used to decide
    # anything about the current row's own velocity counts above.
    work["_is_success"] = transactions["failure_reason_code"].isna().to_numpy()

    # Deterministic chronological order per customer, regardless of the
    # input's original row order. mergesort is stable, but we still add
    # _orig_pos as an explicit sort key (rather than relying on stability
    # alone) so tie-breaking is well-defined even if the input arrives in
    # an arbitrary order.
    work_sorted = work.sort_values(
        ["customer_id", "timestamp", "_orig_pos"], kind="mergesort"
    ).reset_index(drop=True)

    times_ns_all = work_sorted["timestamp"].to_numpy().astype("datetime64[ns]").astype(np.int64)
    is_success_all = work_sorted["_is_success"].to_numpy()

    velocity_1h_sorted = np.empty(n, dtype=np.int64)
    velocity_24h_sorted = np.empty(n, dtype=np.int64)
    days_since_success_sorted = np.empty(n, dtype=float)

    # Each customer's rows are contiguous in work_sorted because we sorted
    # by customer_id first - group positions are simple index ranges.
    group_positions = work_sorted.groupby("customer_id", sort=False).indices
    for positions in group_positions.values():
        positions = np.sort(positions)  # ensure ascending (chronological) order
        group_times = times_ns_all[positions]
        velocity_1h_sorted[positions] = _rolling_prior_count(group_times, _ONE_HOUR_NS)
        velocity_24h_sorted[positions] = _rolling_prior_count(group_times, _TWENTY_FOUR_HOUR_NS)
        days_since_success_sorted[positions] = _days_since_last_success(
            group_times, is_success_all[positions]
        )

    # Map results back to the ORIGINAL input row order via _orig_pos.
    orig_positions = work_sorted["_orig_pos"].to_numpy()
    velocity_1h = np.empty(n, dtype=np.int64)
    velocity_24h = np.empty(n, dtype=np.int64)
    days_since_success = np.empty(n, dtype=float)
    velocity_1h[orig_positions] = velocity_1h_sorted
    velocity_24h[orig_positions] = velocity_24h_sorted
    days_since_success[orig_positions] = days_since_success_sorted

    result = transactions.copy()
    result["velocity_txn_count_1h"] = velocity_1h.astype(int)
    result["velocity_txn_count_24h"] = velocity_24h.astype(int)
    result["days_since_last_successful_payment"] = days_since_success

    _validate_velocity_features(result, expected_rows=n, expected_columns=transactions.columns)
    return result


def _validate_velocity_features(
    df: pd.DataFrame, expected_rows: int, expected_columns: pd.Index
) -> None:
    """Structural/statistical sanity checks. Raises AssertionError on failure."""
    assert len(df) == expected_rows, f"Expected {expected_rows} rows, got {len(df)}."
    assert set(expected_columns).issubset(set(df.columns)), (
        "Original columns were not fully preserved."
    )

    for col in ["velocity_txn_count_1h", "velocity_txn_count_24h"]:
        assert df[col].dtype.kind in "iu", f"{col} must be integer."
        assert (df[col] >= 0).all(), f"{col} must be non-negative."

    # A shorter window can never contain more prior transactions than a
    # longer window - a basic internal-consistency check that would catch
    # a broken window computation.
    assert (df["velocity_txn_count_1h"] <= df["velocity_txn_count_24h"]).all(), (
        "velocity_txn_count_1h exceeds velocity_txn_count_24h for some row - "
        "the 1h window cannot contain more prior transactions than 24h."
    )

    # Each customer's chronologically-FIRST transaction must have zero
    # prior transactions in both windows - directly checks that no future
    # transaction and no self-count ever leaks into the result.
    work = df[["customer_id", "timestamp"]].copy()
    work["_orig_pos"] = np.arange(len(work))
    first_per_customer = (
        work.sort_values(["customer_id", "timestamp", "_orig_pos"], kind="mergesort")
        .groupby("customer_id")
        .head(1)["_orig_pos"]
        .to_numpy()
    )
    assert (df.iloc[first_per_customer]["velocity_txn_count_1h"] == 0).all(), (
        "A customer's first transaction must have velocity_txn_count_1h == 0."
    )
    assert (df.iloc[first_per_customer]["velocity_txn_count_24h"] == 0).all(), (
        "A customer's first transaction must have velocity_txn_count_24h == 0."
    )

    # days_since_last_successful_payment: never negative where present.
    days_col = df["days_since_last_successful_payment"]
    assert (days_col.dropna() >= 0).all(), (
        "days_since_last_successful_payment must be non-negative where present."
    )

    # Per customer (chronological order): every row up to AND INCLUDING
    # that customer's first-ever success must be null (no prior success
    # can exist yet, and a row is never allowed to reference its own
    # outcome); every row AFTER the first success must be non-null. If a
    # customer never has any success, every row for that customer is null.
    work2 = df[["customer_id", "timestamp"]].copy()
    work2["_orig_pos"] = np.arange(len(work2))
    work2["_is_success"] = df["failure_reason_code"].isna().to_numpy()
    ordered = work2.sort_values(["customer_id", "timestamp", "_orig_pos"], kind="mergesort")
    for _, group in ordered.groupby("customer_id", sort=False):
        success_positions = np.where(group["_is_success"].to_numpy())[0]
        orig_pos = group["_orig_pos"].to_numpy()
        values = days_col.iloc[orig_pos].to_numpy()
        if len(success_positions) == 0:
            assert np.isnan(values).all(), (
                "A customer with no successful transactions must have "
                "days_since_last_successful_payment null for all rows."
            )
        else:
            first_success_idx = success_positions[0]
            assert np.isnan(values[: first_success_idx + 1]).all(), (
                "Rows up to and including a customer's first success must "
                "have null days_since_last_successful_payment."
            )
            assert not np.isnan(values[first_success_idx + 1 :]).any(), (
                "Rows after a customer's first success must have a "
                "non-null days_since_last_successful_payment."
            )


def _brute_force_prior_counts(
    customer_ids: np.ndarray, timestamps: np.ndarray, window: pd.Timedelta
) -> np.ndarray:
    """
    Slow, obviously-correct O(n^2) reference implementation used only in
    tests to cross-check the vectorized algorithm above - never used in
    the actual generation pipeline.
    """
    n = len(customer_ids)
    counts = np.zeros(n, dtype=int)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if customer_ids[j] != customer_ids[i]:
                continue
            # "Prior" per this module's tie-break convention: earlier
            # timestamp, or equal timestamp with a smaller original
            # position (j provided in original input order here).
            is_prior = timestamps[j] < timestamps[i] or (
                timestamps[j] == timestamps[i] and j < i
            )
            if is_prior and (timestamps[i] - timestamps[j]) <= window:
                counts[i] += 1
    return counts


def _run_targeted_edge_case_test() -> None:
    """
    Deliberately constructed timestamps with hand-checkable expected
    counts, plus a cross-check against the brute-force reference
    implementation for the same data.
    """
    df = pd.DataFrame(
        {
            "customer_id": ["cust_A"] * 4,
            "timestamp": pd.to_datetime(
                ["2026-01-01 10:00", "2026-01-01 10:10", "2026-01-01 10:30", "2026-01-01 11:05"]
            ),
            # All successes here - this test targets velocity counts only;
            # days_since_last_successful_payment has its own dedicated
            # targeted test below.
            "failure_reason_code": [None, None, None, None],
        }
    )
    result = compute_velocity_features(df)

    print("Targeted edge-case test (Customer A: 10:00, 10:10, 10:30, 11:05)")
    print(result[["timestamp", "velocity_txn_count_1h", "velocity_txn_count_24h"]].to_string())

    # Hand-computed ground truth (gaps to 11:05 are 65/55/35 minutes for
    # 10:00/10:10/10:30 respectively - the 65-minute gap falls OUTSIDE the
    # 60-minute window, so exactly two prior transactions (10:10, 10:30)
    # qualify for the 11:05 row, not one).
    expected_1h = [0, 1, 2, 2]
    expected_24h = [0, 1, 2, 3]

    brute_1h = _brute_force_prior_counts(
        df["customer_id"].to_numpy(), df["timestamp"].to_numpy(), pd.Timedelta(hours=1)
    )
    brute_24h = _brute_force_prior_counts(
        df["customer_id"].to_numpy(), df["timestamp"].to_numpy(), pd.Timedelta(hours=24)
    )

    assert result["velocity_txn_count_1h"].tolist() == expected_1h == list(brute_1h), (
        f"1h mismatch: got {result['velocity_txn_count_1h'].tolist()}, "
        f"expected {expected_1h}, brute-force {list(brute_1h)}"
    )
    assert result["velocity_txn_count_24h"].tolist() == expected_24h == list(brute_24h), (
        f"24h mismatch: got {result['velocity_txn_count_24h'].tolist()}, "
        f"expected {expected_24h}, brute-force {list(brute_24h)}"
    )
    print("PASSED: matches hand-computed expectations and brute-force reference.\n")


def _run_targeted_days_since_success_test() -> None:
    """
    Deliberately constructed FAIL/SUCCESS sequence with hand-checkable
    expected days_since_last_successful_payment values.

    Sequence for Customer B:
      2026-01-01 10:00  FAIL     -> null (no prior success)
      2026-01-01 10:30  SUCCESS  -> null (no prior success BEFORE this row -
                                     this row's own success only becomes
                                     "prior" information for LATER rows)
      2026-01-02 10:00  FAIL     -> 23h30m since 01-01 10:30 = 0.979167 days
      2026-01-03 10:00  SUCCESS -> 47h30m since 01-01 10:30 = 1.979167 days
                                     (this row's own success then becomes
                                     the new reference point for row 5)
      2026-01-04 10:00  FAIL     -> 24h since 01-03 10:00   = 1.0 days
    """
    df = pd.DataFrame(
        {
            "customer_id": ["cust_B"] * 5,
            "timestamp": pd.to_datetime(
                [
                    "2026-01-01 10:00", "2026-01-01 10:30", "2026-01-02 10:00",
                    "2026-01-03 10:00", "2026-01-04 10:00",
                ]
            ),
            "failure_reason_code": [
                "bank_timeout", None, "bank_timeout", None, "bank_timeout",
            ],
        }
    )
    result = compute_velocity_features(df)

    print("Targeted days_since_last_successful_payment test (Customer B: FAIL/SUCCESS sequence)")
    print(
        result[
            ["timestamp", "failure_reason_code", "days_since_last_successful_payment"]
        ].to_string()
    )

    expected = [np.nan, np.nan, 23.5 / 24, 47.5 / 24, 1.0]
    actual = result["days_since_last_successful_payment"].to_numpy()

    for i, (exp, act) in enumerate(zip(expected, actual)):
        if np.isnan(exp):
            assert np.isnan(act), f"Row {i}: expected null, got {act}"
        else:
            assert np.isclose(act, exp, atol=1e-9), f"Row {i}: expected {exp}, got {act}"

    print("PASSED: matches hand-computed expectations.\n")


if __name__ == "__main__":
    from .customer_generator import generate_customer_profiles
    from .failure_generator import generate_failure_context
    from .risk_signal_generator import generate_risk_signals
    from .transaction_context_generator import generate_transaction_context
    from .config import RANDOM_SEED

    _run_targeted_edge_case_test()
    _run_targeted_days_since_success_test()

    small_customers = generate_customer_profiles(num_customers=30, seed=RANDOM_SEED)
    tx = generate_transaction_context(small_customers, num_transactions=100, seed=RANDOM_SEED)
    tx_failed = generate_failure_context(tx, small_customers, seed=RANDOM_SEED)
    tx_risk = generate_risk_signals(tx_failed, small_customers, seed=RANDOM_SEED)
    tx_velocity = compute_velocity_features(tx_risk)

    print("Shape:", tx_velocity.shape)

    print("\n1h velocity distribution:")
    print(tx_velocity["velocity_txn_count_1h"].value_counts().sort_index())

    print("\n24h velocity distribution:")
    print(tx_velocity["velocity_txn_count_24h"].value_counts().sort_index())

    print("\ndays_since_last_successful_payment summary:")
    print(tx_velocity["days_since_last_successful_payment"].describe())
    print("Null count (no prior success yet):", tx_velocity["days_since_last_successful_payment"].isna().sum())

    print("\nMax 1h velocity:", tx_velocity["velocity_txn_count_1h"].max())
    print("Max 24h velocity:", tx_velocity["velocity_txn_count_24h"].max())

    print("\nTop 5 highest-velocity transactions (by 24h count):")
    top = tx_velocity.sort_values("velocity_txn_count_24h", ascending=False).head(5)
    print(
        top[
            ["customer_id", "timestamp", "velocity_txn_count_1h", "velocity_txn_count_24h"]
        ].to_string()
    )
