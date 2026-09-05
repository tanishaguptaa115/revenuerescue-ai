"""
main_generator.py

Orchestrates the full RevenueRescue AI synthetic dataset generation
pipeline:

    customer_generator
        -> transaction_context_generator
        -> failure_generator
        -> risk_signal_generator
        -> velocity_engine
        -> label_generator

This module contains NO generation logic of its own - it only calls the
existing, already-validated modules in the proven order and passes the
same `seed` through every stage (each module now derives its own
decorrelated internal stream via a module-specific SeedSequence tag, so
no cross-module seed-spawning is needed here).

`generate_dataset()` is a pure function: given the same arguments, it
always returns the same two DataFrames, with no side effects. File
writing (CSV / report output) is handled separately by `save_outputs()`,
so the core generation logic stays fully testable in isolation.
"""

import os
from typing import Dict, List, Tuple

import pandas as pd

from .archetypes import ARCHETYPES
from .config import (
    EXPECTED_FRAUD_PREVALENCE_RANGE,
    EXPECTED_RECOVERY_PREVALENCE_RANGE,
    NUM_CUSTOMERS,
    NUM_TRANSACTIONS,
    RANDOM_SEED,
)
from .customer_generator import generate_customer_profiles
from .failure_generator import generate_failure_context
from .label_generator import generate_labels
from .risk_signal_generator import generate_risk_signals
from .transaction_context_generator import generate_transaction_context
from .velocity_engine import compute_velocity_features

# ---------------------------------------------------------------------------
# Expected final schema - used only for a holistic, whole-dataset sanity
# check after every stage has run. Each individual module already
# validates its own additions; this check exists to catch anything that
# could only go wrong at the ASSEMBLED level (e.g. a stage silently
# dropping a column, or a required demo scenario failing to appear at
# all across the full dataset).
# ---------------------------------------------------------------------------

_EXPECTED_FINAL_COLUMNS: List[str] = [
    "transaction_id", "customer_id", "merchant_id", "timestamp", "amount",
    "currency", "payment_method", "failure_reason_code", "is_soft_failure",
    "retry_count_so_far", "num_payment_methods_used_recently",
    "ip_country_mismatch", "device_change_flag", "is_new_customer",
    "velocity_txn_count_1h", "velocity_txn_count_24h",
    "days_since_last_successful_payment", "recovery_label", "risk_label",
]


def generate_dataset(
    num_customers: int = NUM_CUSTOMERS,
    num_transactions: int = NUM_TRANSACTIONS,
    seed: int = RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the full generation pipeline and return the finished dataset.

    Pure function: no file I/O, no printing, no global state mutation.
    Every stage is called in the exact order already proven correct in
    each module's own smoke tests and full-scale validation.

    Args:
        num_customers: number of unique customers to generate.
        num_transactions: number of transaction rows to generate.
        seed: single seed passed to every stage. Each module derives its
            own decorrelated internal random stream from this seed via a
            module-specific SeedSequence tag, so passing the same seed to
            every stage here does not risk the cross-module collisions
            that were found and fixed earlier in this project.

    Returns:
        (transactions, customer_profiles) - the two DataFrames produced
        by the pipeline. `transactions` carries the full 19-column
        feature + label schema; `customer_profiles` carries the
        customer-level archetype and latent attributes.
    """
    customer_profiles = generate_customer_profiles(num_customers, seed=seed)

    transactions = generate_transaction_context(customer_profiles, num_transactions, seed=seed)
    transactions = generate_failure_context(transactions, customer_profiles, seed=seed)
    transactions = generate_risk_signals(transactions, customer_profiles, seed=seed)
    transactions = compute_velocity_features(transactions)
    transactions = generate_labels(transactions, customer_profiles, seed=seed)

    _run_final_checks(transactions, customer_profiles, expected_rows=num_transactions)
    return transactions, customer_profiles


def _run_final_checks(
    transactions: pd.DataFrame, customer_profiles: pd.DataFrame, expected_rows: int
) -> None:
    """
    Holistic, whole-dataset checks that only make sense to run AFTER every
    stage has completed - schema completeness and the presence of the
    demo-critical scenarios this dataset was designed to support. Each
    stage's own internal validation (already run inside that stage) is
    not repeated here.
    """
    assert len(transactions) == expected_rows, (
        f"Expected {expected_rows} final rows, got {len(transactions)}."
    )
    assert list(transactions.columns) == _EXPECTED_FINAL_COLUMNS, (
        f"Final schema mismatch.\nExpected: {_EXPECTED_FINAL_COLUMNS}\n"
        f"Got: {list(transactions.columns)}"
    )

    # All six archetypes must actually appear in the customer population.
    assert set(customer_profiles["archetype"].unique()) == set(ARCHETYPES.keys()), (
        "Not all six archetypes are present in the generated customer population."
    )

    # Both failed and non-failed transactions must exist.
    is_failed = transactions["failure_reason_code"].notna()
    assert is_failed.any() and (~is_failed).any(), (
        "Dataset must contain both failed and non-failed transactions."
    )

    # Both recovery outcomes must exist among failed transactions.
    failed = transactions[is_failed]
    assert set(failed["recovery_label"].unique()) == {"recovered", "not_recovered"}, (
        "Both 'recovered' and 'not_recovered' must appear among failed transactions."
    )

    # Both risk outcomes must exist across the whole dataset.
    assert set(transactions["risk_label"].unique()) == {"fraudulent", "legitimate"}, (
        "Both 'fraudulent' and 'legitimate' must appear in risk_label."
    )

    # All four recovery x risk combinations must be observable among
    # failed transactions - the core "risk-aware, not just recovery-aware"
    # requirement this whole dataset exists to demonstrate.
    crosstab = pd.crosstab(failed["recovery_label"], failed["risk_label"])
    assert crosstab.shape == (2, 2) and (crosstab.to_numpy() > 0).all(), (
        "All four recovery_label x risk_label combinations must be observed "
        "among failed transactions."
    )

    # Velocity internal consistency (re-asserted here as a whole-dataset
    # cross-check; velocity_engine already validates this at its own
    # stage, but confirming it survived unchanged through label_generator
    # is a legitimate final-assembly check).
    assert (transactions["velocity_txn_count_1h"] <= transactions["velocity_txn_count_24h"]).all(), (
        "velocity_txn_count_1h must never exceed velocity_txn_count_24h."
    )

    # days_since_last_successful_payment: both null and non-null values
    # must exist (cold-start customers and customers with a success
    # history both need to be representable).
    days_col = transactions["days_since_last_successful_payment"]
    assert days_col.isna().any() and days_col.notna().any(), (
        "days_since_last_successful_payment must contain both null and "
        "non-null values."
    )


def generate_summary(transactions: pd.DataFrame, customer_profiles: pd.DataFrame) -> Dict:
    """
    Compute a structured summary of the generated dataset: row counts,
    archetype distribution, failure/recovery/risk rates, and velocity
    stats. Returns a plain dict (JSON/markdown-friendly) rather than
    printing directly, so callers can format or log it however they like.
    """
    is_failed = transactions["failure_reason_code"].notna()
    failed = transactions[is_failed]

    return {
        "num_customers": len(customer_profiles),
        "num_transactions": len(transactions),
        "archetype_distribution": customer_profiles["archetype"]
        .value_counts(normalize=True)
        .round(4)
        .to_dict(),
        "failure_rate": round(is_failed.mean(), 4),
        "failure_reason_distribution": failed["failure_reason_code"]
        .value_counts(normalize=True)
        .round(4)
        .to_dict(),
        "soft_failure_rate": round(failed["is_soft_failure"].mean(), 4) if len(failed) else None,
        "recovery_rate_among_failed": round((failed["recovery_label"] == "recovered").mean(), 4)
        if len(failed)
        else None,
        "recovery_rate_target_range": EXPECTED_RECOVERY_PREVALENCE_RANGE,
        "fraud_prevalence": round((transactions["risk_label"] == "fraudulent").mean(), 4),
        "fraud_prevalence_target_range": EXPECTED_FRAUD_PREVALENCE_RANGE,
        "velocity_1h_max": int(transactions["velocity_txn_count_1h"].max()),
        "velocity_24h_max": int(transactions["velocity_txn_count_24h"].max()),
        "velocity_1h_mean": round(transactions["velocity_txn_count_1h"].mean(), 4),
        "velocity_24h_mean": round(transactions["velocity_txn_count_24h"].mean(), 4),
        "days_since_last_success_null_count": int(
            transactions["days_since_last_successful_payment"].isna().sum()
        ),
        "days_since_last_success_median": round(
            transactions["days_since_last_successful_payment"].median(), 4
        ),
    }


def _format_summary_markdown(summary: Dict) -> str:
    """Render the summary dict as a small, readable markdown report."""
    lines = [
        "# RevenueRescue AI - Synthetic Dataset Generation Report",
        "",
        f"- Customers: {summary['num_customers']}",
        f"- Transactions: {summary['num_transactions']}",
        f"- Failure rate: {summary['failure_rate']:.2%}",
        f"- Recovery rate among failed transactions: "
        f"{summary['recovery_rate_among_failed']:.2%} "
        f"(target {summary['recovery_rate_target_range']})",
        f"- Fraud/risk prevalence: {summary['fraud_prevalence']:.2%} "
        f"(target {summary['fraud_prevalence_target_range']})",
        f"- Max velocity (1h / 24h): {summary['velocity_1h_max']} / {summary['velocity_24h_max']}",
        f"- days_since_last_successful_payment: "
        f"{summary['days_since_last_success_null_count']} null "
        f"(median where present: {summary['days_since_last_success_median']} days)",
        "",
        "## Archetype distribution",
        "",
    ]
    for archetype, share in summary["archetype_distribution"].items():
        lines.append(f"- {archetype}: {share:.2%}")

    lines += ["", "## Failure reason distribution (among failed transactions)", ""]
    for reason, share in summary["failure_reason_distribution"].items():
        lines.append(f"- {reason}: {share:.2%}")

    return "\n".join(lines) + "\n"


def save_outputs(
    transactions: pd.DataFrame,
    customer_profiles: pd.DataFrame,
    output_dir: str = "output",
    sample_size: int = 200,
    seed: int = RANDOM_SEED,
) -> Dict[str, str]:
    """
    Write the four generation deliverables to disk. Kept entirely
    separate from generate_dataset() so the generation logic itself stays
    a pure, side-effect-free function.

    Writes:
      - revenuerescue_dataset.csv: the full transactions dataset.
      - revenuerescue_sample.csv: a small sample, lightly stratified by
        (recovery_label, risk_label) so it shows varied scenarios rather
        than an arbitrary random slice.
      - generation_report.md: the summary report in markdown.
      - customer_profiles.csv: the complete customer_profiles DataFrame
        exactly as generated (archetype and all customer-level latent
        attributes), persisted so downstream consumers (e.g. the ML
        data-preparation layer) can join customer-level fields without
        needing to regenerate the dataset themselves.

    Returns a dict mapping each deliverable name to its written file path.
    """
    os.makedirs(output_dir, exist_ok=True)

    dataset_path = os.path.join(output_dir, "revenuerescue_dataset.csv")
    transactions.to_csv(dataset_path, index=False)

    # Stratify the sample by (recovery_label, risk_label) so small demo
    # slices naturally include a mix of scenarios rather than whatever a
    # plain random sample happens to draw.
    strata_key = transactions["recovery_label"].fillna("no_failure") + "|" + transactions["risk_label"]
    per_stratum = max(1, sample_size // strata_key.nunique())
    sample = (
        transactions.groupby(strata_key, group_keys=False)
        .apply(lambda g: g.sample(n=min(per_stratum, len(g)), random_state=seed))
        .sample(frac=1.0, random_state=seed)  # shuffle stratum order
        .head(sample_size)
    )
    sample_path = os.path.join(output_dir, "revenuerescue_sample.csv")
    sample.to_csv(sample_path, index=False)

    summary = generate_summary(transactions, customer_profiles)
    report_path = os.path.join(output_dir, "generation_report.md")
    with open(report_path, "w") as f:
        f.write(_format_summary_markdown(summary))

    # Persist customer_profiles exactly as generated - no filtering,
    # reordering, or transformation of any column.
    customer_profiles_path = os.path.join(output_dir, "customer_profiles.csv")
    customer_profiles.to_csv(customer_profiles_path, index=False)

    return {
        "dataset": dataset_path,
        "sample": sample_path,
        "report": report_path,
        "customer_profiles": customer_profiles_path,
    }


if __name__ == "__main__":
    transactions, customer_profiles = generate_dataset()

    summary = generate_summary(transactions, customer_profiles)
    print("Generation summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    paths = save_outputs(transactions, customer_profiles, output_dir="output")
    print("\nFiles written:")
    for name, path in paths.items():
        print(f"  {name}: {path}")