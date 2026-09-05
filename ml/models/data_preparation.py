"""
data_preparation.py

ML data-preparation layer for RevenueRescue AI's Recovery and Risk models.

This module does NOT train any model. It:
  1. Loads the ACTUAL local files (source of truth - never regenerated):
       - output/revenuerescue_dataset.csv
       - output/customer_profiles.csv
  2. Builds a single customer-level, grouped-chronological train/val/test
     split assignment and reuses it identically for both models.
  3. Assembles the Recovery-model population (failed transactions only)
     and the Risk-model population (all transactions), with their
     respective approved feature sets.
  4. Enforces the approved leakage exclusions - including deliberately
     excluding `archetype`, which exists in customer_profiles.csv but is
     never used as a feature: it was a generation-time latent variable
     that drove the synthetic labels directly, so using it here would be
     an answer-key feature no real deployment would ever have access to.
  5. Fits all preprocessing (one-hot encoding) ONLY on the training split.
  6. Runs strong structural/statistical validation on the result.

Both output/revenuerescue_dataset.csv and output/customer_profiles.csv
are required, real, local files as of this step. This module reads them
directly - it never calls into the synthetic-data generator package and
never fabricates a value for a column that isn't present in either file.
"""

import os
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_DATASET_PATH = "output/revenuerescue_dataset.csv"
DEFAULT_CUSTOMER_PROFILES_PATH = "output/customer_profiles.csv"

# Fallback locations checked ONLY if the default customer_profiles path
# does not exist - read-only discovery, never a write target.
CUSTOMER_PROFILES_FALLBACK_PATHS: List[str] = [
    "ml/data/revenuerescue_data_gen/output/customer_profiles.csv",
    "ml/data/output/customer_profiles.csv",
    "data/customer_profiles.csv",
    "customer_profiles.csv",
]

# Grouped chronological split fractions, applied over CUSTOMERS (by first
# transaction timestamp), not over rows. Test fraction is implied
# (1 - TRAIN_FRACTION - VAL_FRACTION).
TRAIN_FRACTION: float = 0.70
VAL_FRACTION: float = 0.15

EXPECTED_TRANSACTION_ROWS: int = 20_000
EXPECTED_CUSTOMER_PROFILE_ROWS: int = 4_000

# Columns that must NEVER be used as predictive features, regardless of
# model. `archetype` is explicitly excluded even though it IS present in
# customer_profiles.csv - see module docstring.
FORBIDDEN_FEATURE_COLUMNS = {"transaction_id", "customer_id", "timestamp", "archetype"}

# Customer-level fields joined from customer_profiles.csv.
CUSTOMER_PROFILE_FIELDS: List[str] = [
    "avg_transaction_amount_customer",
    "customer_past_success_rate",
    "customer_past_recovery_rate",
    "nudge_ignore_tendency",
    "chargeback_history_count",
]

# Approved feature specifications.
RECOVERY_FEATURE_SPEC: List[str] = [
    "amount",
    "amount_to_customer_avg_ratio",
    "payment_method",
    "failure_reason_code",
    "is_soft_failure",
    "retry_count_so_far",
    "days_since_last_successful_payment",
    "has_prior_success",
    "is_new_customer",
    "customer_past_success_rate",
    "customer_past_recovery_rate",
    "nudge_ignore_tendency",
]

RISK_FEATURE_SPEC: List[str] = [
    "amount",
    "amount_to_customer_avg_ratio",
    "payment_method",
    "num_payment_methods_used_recently",
    "ip_country_mismatch",
    "device_change_flag",
    "is_new_customer",
    "velocity_txn_count_1h",
    "velocity_txn_count_24h",
    "days_since_last_successful_payment",
    "has_prior_success",
    "chargeback_history_count",
]

CATEGORICAL_COLUMNS = {"payment_method", "failure_reason_code"}
BOOLEAN_COLUMNS = {
    "is_soft_failure", "ip_country_mismatch", "device_change_flag",
    "is_new_customer", "has_prior_success",
}

REQUIRED_RAW_COLUMNS = [
    "transaction_id", "customer_id", "merchant_id", "timestamp", "amount",
    "currency", "payment_method", "failure_reason_code", "is_soft_failure",
    "retry_count_so_far", "num_payment_methods_used_recently",
    "ip_country_mismatch", "device_change_flag", "is_new_customer",
    "velocity_txn_count_1h", "velocity_txn_count_24h",
    "days_since_last_successful_payment", "recovery_label", "risk_label",
]

REQUIRED_CUSTOMER_PROFILE_COLUMNS = ["customer_id"] + CUSTOMER_PROFILE_FIELDS


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class PreparedSplit:
    """One (train/val/test) split of a model's prepared data."""

    X: np.ndarray
    y: np.ndarray
    feature_names: List[str]
    transaction_ids: pd.Series
    customer_ids: pd.Series


@dataclass
class DataPreparationResult:
    """Everything data_preparation.py produces, in one place."""

    recovery_train: PreparedSplit
    recovery_val: PreparedSplit
    recovery_test: PreparedSplit
    risk_train: PreparedSplit
    risk_val: PreparedSplit
    risk_test: PreparedSplit

    recovery_feature_names: List[str]
    risk_feature_names: List[str]
    recovery_logical_features: List[str]
    risk_logical_features: List[str]
    recovery_features_dropped: List[str]
    risk_features_dropped: List[str]

    customer_split_map: pd.Series  # customer_id -> 'train'/'val'/'test'
    split_summary: Dict
    target_distributions: Dict
    missing_dependencies: Dict
    recovery_preprocessor: OneHotEncoder
    risk_preprocessor: OneHotEncoder


# ---------------------------------------------------------------------------
# Step 1: load the two ACTUAL local files (source of truth)
# ---------------------------------------------------------------------------

def _load_dataset(dataset_path: str) -> pd.DataFrame:
    """
    Load the local, already-generated transaction dataset. Performs NO
    generation and NO regeneration - fails loudly if the file is missing
    rather than silently falling back to the synthetic generator.
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Dataset not found at '{dataset_path}'. This module reads the "
            f"already-generated dataset as the source of truth and will "
            f"NOT regenerate it. Run the existing generation pipeline "
            f"(ml.data.revenuerescue_data_gen.main_generator) separately "
            f"first if this file is genuinely missing."
        )

    df = pd.read_csv(dataset_path)

    missing_cols = [c for c in REQUIRED_RAW_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Dataset at '{dataset_path}' is missing expected columns: "
            f"{missing_cols}. Refusing to proceed with an unexpected schema."
        )

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def _load_customer_profiles(customer_profiles_path: Optional[str]) -> Tuple[pd.DataFrame, Dict]:
    """
    Load the local, already-generated customer_profiles table. This is a
    REQUIRED, real, local file as of this step - this function reads it
    directly. It never calls into the generator package and never
    fabricates a value for any column that is genuinely absent.

    If the given/default path is missing, a small fixed set of fallback
    locations is checked (read-only discovery only). If none exist, this
    raises loudly - it does NOT silently proceed or invent values.
    """
    report: Dict = {"path_used": None, "paths_checked": [], "missing_fields": []}

    candidates = [customer_profiles_path or DEFAULT_CUSTOMER_PROFILES_PATH] + \
        CUSTOMER_PROFILES_FALLBACK_PATHS

    for candidate in candidates:
        report["paths_checked"].append(candidate)
        if os.path.exists(candidate):
            df = pd.read_csv(candidate)
            report["path_used"] = candidate
            missing_cols = [c for c in REQUIRED_CUSTOMER_PROFILE_COLUMNS if c not in df.columns]
            if missing_cols:
                raise ValueError(
                    f"customer_profiles file at '{candidate}' is missing "
                    f"required columns: {missing_cols}. Refusing to "
                    f"fabricate values for them."
                )
            report["missing_fields"] = []
            return df, report

    raise FileNotFoundError(
        f"customer_profiles.csv was not found at any of: {candidates}. "
        f"This module requires the ACTUAL local customer_profiles.csv "
        f"(produced by ml.data.revenuerescue_data_gen.main_generator's "
        f"save_outputs()). It will NOT fabricate customer-level fields "
        f"and will NOT regenerate the synthetic dataset to obtain them. "
        f"Run the generation pipeline's main_generator separately first "
        f"if this file is genuinely missing."
    )


# ---------------------------------------------------------------------------
# Step 2: customer-level grouped chronological split assignment
# ---------------------------------------------------------------------------

def _build_customer_split_assignment(
    df: pd.DataFrame, train_fraction: float, val_fraction: float
) -> pd.Series:
    """
    Assign every customer_id to exactly one of {'train', 'val', 'test'},
    using the grouped chronological policy: customers are ordered by
    their OWN first transaction timestamp, and the earliest
    `train_fraction` of customers go to train, the next `val_fraction` to
    val, and the remainder to test. ALL of a customer's transactions
    follow that customer into their assigned split - this is the single
    shared mapping reused identically for both the Recovery and Risk
    populations.

    Returns:
        pandas.Series indexed by customer_id, values in {'train','val','test'}.
    """
    first_seen = df.groupby("customer_id")["timestamp"].min().sort_values(kind="mergesort")
    customers_sorted = first_seen.index.to_numpy()
    n = len(customers_sorted)

    n_train = int(round(n * train_fraction))
    n_val = int(round(n * val_fraction))
    n_val = min(n_val, n - n_train)  # guard against rounding edge cases

    train_customers = customers_sorted[:n_train]
    val_customers = customers_sorted[n_train : n_train + n_val]
    test_customers = customers_sorted[n_train + n_val :]

    assignment = pd.Series(index=customers_sorted, dtype=object)
    assignment.loc[train_customers] = "train"
    assignment.loc[val_customers] = "val"
    assignment.loc[test_customers] = "test"
    assignment.index.name = "customer_id"
    return assignment


def _validate_no_customer_overlap(customer_split_map: pd.Series) -> None:
    train_ids = set(customer_split_map[customer_split_map == "train"].index)
    val_ids = set(customer_split_map[customer_split_map == "val"].index)
    test_ids = set(customer_split_map[customer_split_map == "test"].index)

    assert not (train_ids & val_ids), "Customer overlap found between train and val."
    assert not (train_ids & test_ids), "Customer overlap found between train and test."
    assert not (val_ids & test_ids), "Customer overlap found between val and test."


# ---------------------------------------------------------------------------
# Step 3: feature engineering common to both models
# ---------------------------------------------------------------------------

def _engineer_common_columns(df: pd.DataFrame, customer_profiles: pd.DataFrame) -> pd.DataFrame:
    """
    Add engineered columns that do not depend on the train/val/test split
    (pure per-row transforms, or joins of STATIC pre-existing customer
    history - never anything computed from labels or other rows'
    outcomes), so this is leak-safe to do before splitting.
    """
    out = df.copy()

    # has_prior_success is always computable from a column already in
    # the transaction CSV.
    out["has_prior_success"] = out["days_since_last_successful_payment"].notna().astype(int)

    join_cols = ["customer_id"] + CUSTOMER_PROFILE_FIELDS
    out = out.merge(customer_profiles[join_cols], on="customer_id", how="left")

    out["amount_to_customer_avg_ratio"] = out["amount"] / out["avg_transaction_amount_customer"]

    return out


def _select_available_features(
    df: pd.DataFrame, feature_spec: List[str]
) -> Tuple[List[str], List[str]]:
    """
    Split a requested feature list into (available, dropped). Anything in
    FORBIDDEN_FEATURE_COLUMNS is excluded even if it happens to be
    present in `df` (defensive backstop) - and any column genuinely
    absent from `df` is dropped, never fabricated.
    """
    available, dropped = [], []
    for feat in feature_spec:
        if feat in FORBIDDEN_FEATURE_COLUMNS:
            dropped.append(feat)
            continue
        if feat in df.columns:
            available.append(feat)
        else:
            dropped.append(feat)
    return available, dropped


# ---------------------------------------------------------------------------
# Step 4: preprocessing (fit ONLY on train)
# ---------------------------------------------------------------------------

def _prepare_feature_matrix(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], OneHotEncoder]:
    """
    Build feature matrices for train/val/test from a shared, pre-selected
    feature column list:
      - categorical columns -> OneHotEncoder(handle_unknown="ignore"),
        fit ONLY on train_df, then applied unchanged to val/test.
      - boolean columns -> cast to 0/1 integers.
      - remaining numeric columns -> passed through as float, WITHOUT
        imputing days_since_last_successful_payment's missingness. NaN
        is preserved deliberately (see module docstring); any tree-based
        model consuming this matrix must support native NaN handling
        (HistGradientBoosting, LightGBM, XGBoost).

    No scaling is applied - this preprocessing targets tree-based models,
    which are scale-invariant.
    """
    cat_cols = [c for c in feature_cols if c in CATEGORICAL_COLUMNS]
    bool_cols = [c for c in feature_cols if c in BOOLEAN_COLUMNS]
    numeric_cols = [c for c in feature_cols if c not in CATEGORICAL_COLUMNS and c not in BOOLEAN_COLUMNS]

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    if cat_cols:
        encoder.fit(train_df[cat_cols])  # fit ONLY on train
        cat_feature_names = list(encoder.get_feature_names_out(cat_cols))
    else:
        cat_feature_names = []

    def _build(split_df: pd.DataFrame) -> np.ndarray:
        blocks = []
        if cat_cols:
            blocks.append(encoder.transform(split_df[cat_cols]))
        if bool_cols:
            blocks.append(split_df[bool_cols].astype(int).to_numpy(dtype=float))
        if numeric_cols:
            blocks.append(split_df[numeric_cols].to_numpy(dtype=float))
        if not blocks:
            return np.empty((len(split_df), 0))
        return np.hstack(blocks)

    X_train = _build(train_df)
    X_val = _build(val_df)
    X_test = _build(test_df)

    output_feature_names = cat_feature_names + bool_cols + numeric_cols
    return X_train, X_val, X_test, output_feature_names, encoder


# ---------------------------------------------------------------------------
# Step 5: assemble one model's population into PreparedSplit objects
# ---------------------------------------------------------------------------

def _assemble_model_splits(
    population_df: pd.DataFrame,
    customer_split_map: pd.Series,
    feature_spec: List[str],
    target_column: str,
    positive_label: str,
) -> Tuple[PreparedSplit, PreparedSplit, PreparedSplit, List[str], List[str], OneHotEncoder]:
    """
    Given a model's eligible row population and the shared customer split
    map, produce (train, val, test) PreparedSplit objects, plus the
    logical feature lists (used/dropped) and the fitted OneHotEncoder
    (fit ONLY on the training rows of this population) so callers can
    apply identical preprocessing to any future data.
    """
    df = population_df.copy()
    df["split"] = df["customer_id"].map(customer_split_map)

    unmapped = df["split"].isna().sum()
    if unmapped:
        raise AssertionError(
            f"{unmapped} rows have a customer_id not present in the "
            f"customer split map - every customer must be assigned."
        )

    available_features, dropped_features = _select_available_features(df, feature_spec)

    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]

    X_train, X_val, X_test, feature_names, encoder = _prepare_feature_matrix(
        train_df, val_df, test_df, available_features
    )

    def _y(split_df: pd.DataFrame) -> np.ndarray:
        return (split_df[target_column] == positive_label).astype(int).to_numpy()

    train_split = PreparedSplit(
        X=X_train, y=_y(train_df), feature_names=feature_names,
        transaction_ids=train_df["transaction_id"].reset_index(drop=True),
        customer_ids=train_df["customer_id"].reset_index(drop=True),
    )
    val_split = PreparedSplit(
        X=X_val, y=_y(val_df), feature_names=feature_names,
        transaction_ids=val_df["transaction_id"].reset_index(drop=True),
        customer_ids=val_df["customer_id"].reset_index(drop=True),
    )
    test_split = PreparedSplit(
        X=X_test, y=_y(test_df), feature_names=feature_names,
        transaction_ids=test_df["transaction_id"].reset_index(drop=True),
        customer_ids=test_df["customer_id"].reset_index(drop=True),
    )

    return train_split, val_split, test_split, available_features, dropped_features, encoder


# ---------------------------------------------------------------------------
# Step 6: validation
# ---------------------------------------------------------------------------

def _validate_result(
    df: pd.DataFrame,
    customer_profiles: pd.DataFrame,
    customer_split_map: pd.Series,
    recovery_population: pd.DataFrame,
    risk_population: pd.DataFrame,
    recovery_feature_names: List[str],
    risk_feature_names: List[str],
    recovery_splits: Tuple[PreparedSplit, PreparedSplit, PreparedSplit],
    risk_splits: Tuple[PreparedSplit, PreparedSplit, PreparedSplit],
) -> None:
    """Strong structural/statistical validation. Raises on hard failures,
    warns on soft ("where statistically possible") concerns."""

    # --- exact row-count expectations ------------------------------------
    assert len(df) == EXPECTED_TRANSACTION_ROWS, (
        f"Expected {EXPECTED_TRANSACTION_ROWS} transaction rows, got {len(df)}."
    )
    assert len(customer_profiles) == EXPECTED_CUSTOMER_PROFILE_ROWS, (
        f"Expected {EXPECTED_CUSTOMER_PROFILE_ROWS} customer_profiles rows, "
        f"got {len(customer_profiles)}."
    )

    # --- expected columns exist -------------------------------------------
    missing_cols = [c for c in REQUIRED_RAW_COLUMNS if c not in df.columns]
    assert not missing_cols, f"Dataset is missing expected columns: {missing_cols}"

    # --- unique customer_id in customer_profiles ---------------------------
    assert customer_profiles["customer_id"].is_unique, (
        "customer_id is not unique in customer_profiles.csv."
    )

    # --- every transaction customer_id exists in customer_profiles ---------
    tx_ids = set(df["customer_id"])
    profile_ids = set(customer_profiles["customer_id"])
    missing_ids = tx_ids - profile_ids
    assert not missing_ids, (
        f"{len(missing_ids)} transaction customer_id values are not present "
        f"in customer_profiles.csv, e.g. {list(missing_ids)[:5]}."
    )

    # --- no customer overlap across splits ----------------------------------
    _validate_no_customer_overlap(customer_split_map)

    # --- recovery contains failed transactions only -------------------------
    assert recovery_population["recovery_label"].notna().all(), (
        "Recovery population contains rows with a null recovery_label."
    )
    assert recovery_population["failure_reason_code"].notna().all(), (
        "Recovery population contains rows with no failure_reason_code - "
        "only failed transactions should be eligible."
    )

    # --- risk contains all transactions --------------------------------------
    assert len(risk_population) == len(df), (
        f"Risk population has {len(risk_population)} rows but the full "
        f"dataset has {len(df)} - every transaction must be eligible."
    )

    # --- no target/identifier leakage -----------------------------------------
    assert "risk_label" not in recovery_feature_names, "risk_label leaked into recovery features."
    assert "recovery_label" not in risk_feature_names, "recovery_label leaked into risk features."
    for forbidden in FORBIDDEN_FEATURE_COLUMNS:
        assert forbidden not in recovery_feature_names, f"'{forbidden}' leaked into recovery features."
        assert forbidden not in risk_feature_names, f"'{forbidden}' leaked into risk features."

    # --- preprocessing fitted only on training data ---------------------------
    for name, (tr, va, te) in [("recovery", recovery_splits), ("risk", risk_splits)]:
        assert tr.X.shape[1] == va.X.shape[1] == te.X.shape[1], (
            f"{name}: train/val/test feature matrices have inconsistent "
            f"column counts - preprocessing was not applied consistently."
        )

    # --- both classes present where statistically possible (soft check) -------
    for name, (tr, va, te) in [("recovery", recovery_splits), ("risk", risk_splits)]:
        for split_name, split_obj in [("train", tr), ("val", va), ("test", te)]:
            if len(split_obj.y) == 0:
                warnings.warn(f"{name}/{split_name} split has ZERO rows.", stacklevel=2)
                continue
            unique_classes = set(np.unique(split_obj.y))
            if unique_classes != {0, 1}:
                warnings.warn(
                    f"{name}/{split_name} split does not contain both classes "
                    f"(found {unique_classes}) - metrics requiring both "
                    f"classes will be undefined for this split.",
                    stacklevel=2,
                )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def prepare_datasets(
    dataset_path: str = DEFAULT_DATASET_PATH,
    customer_profiles_path: Optional[str] = None,
    train_fraction: float = TRAIN_FRACTION,
    val_fraction: float = VAL_FRACTION,
) -> DataPreparationResult:
    """
    Build the full Recovery-model and Risk-model prepared datasets from
    the two ACTUAL local files. Trains NOTHING - this is purely the
    data-preparation layer.

    Args:
        dataset_path: path to the local revenuerescue_dataset.csv.
        customer_profiles_path: path to the local customer_profiles.csv.
            Defaults to output/customer_profiles.csv; a small fixed set
            of fallback locations is checked if that default is absent.
        train_fraction / val_fraction: grouped-chronological split
            fractions, applied over customers (not rows).

    Returns:
        DataPreparationResult with prepared train/val/test objects for
        both models, feature name lists, the shared customer split map,
        and summary/diagnostic reports.
    """
    df = _load_dataset(dataset_path)
    customer_profiles, cp_report = _load_customer_profiles(customer_profiles_path)

    df = _engineer_common_columns(df, customer_profiles)

    customer_split_map = _build_customer_split_assignment(df, train_fraction, val_fraction)
    _validate_no_customer_overlap(customer_split_map)

    recovery_population = df[df["recovery_label"].notna()].copy()
    risk_population = df.copy()

    recovery_train, recovery_val, recovery_test, recovery_features, recovery_dropped, recovery_encoder = (
        _assemble_model_splits(
            recovery_population, customer_split_map, RECOVERY_FEATURE_SPEC,
            target_column="recovery_label", positive_label="recovered",
        )
    )
    risk_train, risk_val, risk_test, risk_features, risk_dropped, risk_encoder = _assemble_model_splits(
        risk_population, customer_split_map, RISK_FEATURE_SPEC,
        target_column="risk_label", positive_label="fraudulent",
    )

    _validate_result(
        df, customer_profiles, customer_split_map, recovery_population, risk_population,
        recovery_train.feature_names, risk_train.feature_names,
        (recovery_train, recovery_val, recovery_test),
        (risk_train, risk_val, risk_test),
    )

    split_summary = {
        "total_rows": len(df),
        "total_customers": df["customer_id"].nunique(),
        "customers_per_split": {
            "train": int((customer_split_map == "train").sum()),
            "val": int((customer_split_map == "val").sum()),
            "test": int((customer_split_map == "test").sum()),
        },
        "transactions_per_split": {
            "train": int((df["customer_id"].map(customer_split_map) == "train").sum()),
            "val": int((df["customer_id"].map(customer_split_map) == "val").sum()),
            "test": int((df["customer_id"].map(customer_split_map) == "test").sum()),
        },
        "recovery_rows_per_split": {
            "train": len(recovery_train.y), "val": len(recovery_val.y), "test": len(recovery_test.y),
        },
        "risk_rows_per_split": {
            "train": len(risk_train.y), "val": len(risk_val.y), "test": len(risk_test.y),
        },
    }

    target_distributions = {
        "recovery": {
            "train": _class_balance(recovery_train.y),
            "val": _class_balance(recovery_val.y),
            "test": _class_balance(recovery_test.y),
        },
        "risk": {
            "train": _class_balance(risk_train.y),
            "val": _class_balance(risk_val.y),
            "test": _class_balance(risk_test.y),
        },
    }

    return DataPreparationResult(
        recovery_train=recovery_train, recovery_val=recovery_val, recovery_test=recovery_test,
        risk_train=risk_train, risk_val=risk_val, risk_test=risk_test,
        recovery_feature_names=recovery_train.feature_names,
        risk_feature_names=risk_train.feature_names,
        recovery_logical_features=recovery_features, risk_logical_features=risk_features,
        recovery_features_dropped=recovery_dropped, risk_features_dropped=risk_dropped,
        customer_split_map=customer_split_map, split_summary=split_summary,
        target_distributions=target_distributions, missing_dependencies=cp_report,
        recovery_preprocessor=recovery_encoder, risk_preprocessor=risk_encoder,
    )


def _class_balance(y: np.ndarray) -> Dict:
    if len(y) == 0:
        return {"n": 0, "positive": 0, "negative": 0, "positive_rate": None}
    positive = int(y.sum())
    negative = int(len(y) - positive)
    return {
        "n": int(len(y)),
        "positive": positive,
        "negative": negative,
        "positive_rate": round(positive / len(y), 4),
    }


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    result = prepare_datasets()

    print("=" * 70)
    print("DATA PREPARATION SMOKE TEST (actual local files)")
    print("=" * 70)

    print(f"\ncustomer_profiles.csv used: {result.missing_dependencies['path_used']}")

    print("\n--- Totals ---")
    print(f"Total transactions: {result.split_summary['total_rows']}")
    print(f"Total customers: {result.split_summary['total_customers']}")

    print("\n--- Customers per split ---")
    for split, count in result.split_summary["customers_per_split"].items():
        print(f"  {split}: {count}")

    print("\n--- Transactions per split ---")
    for split, count in result.split_summary["transactions_per_split"].items():
        print(f"  {split}: {count}")

    print("\n--- Recovery rows per split ---")
    for split, count in result.split_summary["recovery_rows_per_split"].items():
        print(f"  {split}: {count}")

    print("\n--- Risk rows per split ---")
    for split, count in result.split_summary["risk_rows_per_split"].items():
        print(f"  {split}: {count}")

    print("\n--- Recovery positive rate (recovered=1) per split ---")
    for split, bal in result.target_distributions["recovery"].items():
        print(f"  {split}: n={bal['n']}, positive_rate={bal['positive_rate']}")

    print("\n--- Risk positive rate (fraudulent=1) per split ---")
    for split, bal in result.target_distributions["risk"].items():
        print(f"  {split}: n={bal['n']}, positive_rate={bal['positive_rate']}")

    print("\n--- Customer-overlap check ---")
    train_ids = set(result.customer_split_map[result.customer_split_map == "train"].index)
    val_ids = set(result.customer_split_map[result.customer_split_map == "val"].index)
    test_ids = set(result.customer_split_map[result.customer_split_map == "test"].index)
    print(f"  train ∩ val:  {len(train_ids & val_ids)} (must be 0)")
    print(f"  train ∩ test: {len(train_ids & test_ids)} (must be 0)")
    print(f"  val ∩ test:   {len(val_ids & test_ids)} (must be 0)")

    print("\n--- Final logical feature counts ---")
    print(f"  Recovery: {len(result.recovery_logical_features)} used, "
          f"{len(result.recovery_features_dropped)} dropped {result.recovery_features_dropped}")
    print(f"  Risk:     {len(result.risk_logical_features)} used, "
          f"{len(result.risk_features_dropped)} dropped {result.risk_features_dropped}")

    print("\n--- Final encoded feature counts (post one-hot expansion) ---")
    print(f"  Recovery: {len(result.recovery_feature_names)} columns -> {result.recovery_feature_names}")
    print(f"  Risk:     {len(result.risk_feature_names)} columns -> {result.risk_feature_names}")

    print("\n--- Matrix shapes ---")
    print(f"  recovery_train.X: {result.recovery_train.X.shape}")
    print(f"  recovery_val.X:   {result.recovery_val.X.shape}")
    print(f"  recovery_test.X:  {result.recovery_test.X.shape}")
    print(f"  risk_train.X:     {result.risk_train.X.shape}")
    print(f"  risk_val.X:       {result.risk_val.X.shape}")
    print(f"  risk_test.X:      {result.risk_test.X.shape}")

    print("\n--- Preprocessing objects returned ---")
    print(f"  recovery_preprocessor: {type(result.recovery_preprocessor).__name__} "
          f"(categories learned: {result.recovery_preprocessor.categories_ if hasattr(result.recovery_preprocessor, 'categories_') else 'n/a'})")
    print(f"  risk_preprocessor:     {type(result.risk_preprocessor).__name__} "
          f"(categories learned: {result.risk_preprocessor.categories_ if hasattr(result.risk_preprocessor, 'categories_') else 'n/a'})")

    print("\nSMOKE TEST COMPLETE - no model was trained.")