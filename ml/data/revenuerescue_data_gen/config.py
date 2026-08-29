"""
config.py

Central configuration for the RevenueRescue AI synthetic dataset generator.

Design principle:
This file contains ONLY constants and parameter definitions. It contains
NO generation logic, NO randomness, and NO row-level computation. Every
other module imports from here so that a single change (e.g. adjusting
the fraud prevalence target) propagates consistently through the entire
pipeline without hunting through generation code.

This also serves as a "knobs panel" for the hackathon demo: if a judge
asks "what if fraud were rarer" or "what if you had more data", you
change a number here, not the generation logic.
"""

from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

# Fixed seed so the dataset (and every downstream model trained on it) is
# reproducible run-to-run. This matters for a hackathon submission because
# judges may re-run your generator, and non-reproducible data undermines
# trust in your reported metrics.
RANDOM_SEED: int = 42


# ---------------------------------------------------------------------------
# Dataset scale
# ---------------------------------------------------------------------------

# Total number of transaction rows to generate. Chosen (see Step 3A design
# discussion) as large enough to support stable model training and a
# meaningful stratified split, small enough to generate/debug quickly.
NUM_TRANSACTIONS: int = 20_000

# Recommended number of unique customers. Deliberately smaller than
# NUM_TRANSACTIONS so that customers naturally have multiple transactions
# (required for rolling/velocity features and for customer-level history
# features like customer_past_success_rate to be meaningful). ~4,000
# customers over 20,000 transactions gives an average of 5 transactions
# per customer, with the actual per-customer count drawn from a
# distribution (heavier for loyal/normal, near-1 for new_customer).
NUM_CUSTOMERS: int = 4_000


# ---------------------------------------------------------------------------
# Time range
# ---------------------------------------------------------------------------

# Synthetic transactions span a 180-day (~6 month) window. Long enough to
# express tenure, seasonality (e.g., month-end insufficient-funds patterns),
# and repeat-customer behavior; short enough that "recent" velocity windows
# (1h/24h) remain meaningful relative to the whole dataset.
DATE_RANGE_START: str = "2025-09-01"
DATE_RANGE_END: str = "2026-02-28"


# ---------------------------------------------------------------------------
# Customer archetype population weights
# ---------------------------------------------------------------------------

# Proportion of the CUSTOMER population (not transactions) belonging to
# each archetype. Must sum to 1.0.
#
# Rationale for the specific weights:
# - loyal_low_risk and normal dominate, because most real payment traffic
#   is legitimate, low-drama activity.
# - new_customer is meaningfully sized (cold-start is a required edge case,
#   not a rare curiosity) but still a minority.
# - financially_constrained is a real, common, NON-fraud segment - this
#   population is what generates realistic "insufficient funds" failures.
# - suspicious and account_takeover_like are DELIBERATELY small minorities
#   (~1.5% and ~1% of customers respectively) to reflect realistic fraud
#   prevalence. Making them larger would make the risk model's job
#   artificially easy and the dataset unrealistic.
ARCHETYPE_WEIGHTS: Dict[str, float] = {
    "loyal_low_risk": 0.32,
    "normal": 0.40,
    "new_customer": 0.15,
    "financially_constrained": 0.105,
    "suspicious": 0.015,
    "account_takeover_like": 0.01,
}


# ---------------------------------------------------------------------------
# Payment methods
# ---------------------------------------------------------------------------

# Kept to four rails - enough to express method-dependent failure/recovery
# behavior (a core required scenario) without exploding categorical
# cardinality for a small hackathon dataset.
PAYMENT_METHODS: List[str] = ["UPI", "card", "netbanking", "wallet"]

# Relative popularity of each method in the overall transaction mix.
# UPI dominance reflects realistic Indian payment volume distribution.
PAYMENT_METHOD_WEIGHTS: Dict[str, float] = {
    "UPI": 0.50,
    "card": 0.30,
    "netbanking": 0.12,
    "wallet": 0.08,
}


# ---------------------------------------------------------------------------
# Failure reason codes
# ---------------------------------------------------------------------------

FAILURE_REASON_CODES: List[str] = [
    "insufficient_funds",
    "bank_timeout",
    "otp_expired",
    "card_declined_issuer",
    "network_error",
    "invalid_cvv",
    "fraud_suspected_by_bank",
    "customer_cancelled",
]

# Deterministic mapping: soft (transient, plausibly recoverable via retry)
# vs hard (terminal, retry alone won't help - needs customer action or
# should not be pursued). This mapping is a DEFINITIONAL fact about each
# reason code, not a prediction target, so it is legitimately hard-coded
# rather than sampled. failure_generator.py will derive is_soft_failure
# by looking up this dictionary, never by random draw.
SOFT_FAILURE_REASONS: Dict[str, bool] = {
    "insufficient_funds": True,   # transient - can resolve after payday/top-up
    "bank_timeout": True,         # transient - infra hiccup
    "otp_expired": True,          # transient - simple UX retry
    "network_error": True,        # transient - infra hiccup
    "card_declined_issuer": False,   # often needs updated card/customer action
    "invalid_cvv": False,            # customer input error, retry rarely helps
    "fraud_suspected_by_bank": False,  # terminal by design - should not auto-retry
    "customer_cancelled": False,       # customer intent, not a technical failure
}


# ---------------------------------------------------------------------------
# Transaction amount distribution
# ---------------------------------------------------------------------------

# Amounts are long-tailed (many small transactions, a few large ones), so we
# model them with a lognormal distribution rather than a uniform or normal
# one - this matches real payment amount distributions far better.
# Parameters are for the underlying normal distribution in log-space
# (numpy.random.lognormal(mean, sigma)).
AMOUNT_LOGNORMAL_MEAN: float = 6.5   # exp(6.5) ~= 665 -> realistic median-ish anchor
AMOUNT_LOGNORMAL_SIGMA: float = 1.0  # controls spread / long tail

# Hard floor/ceiling to keep generated amounts within a plausible range
# (avoids absurd outliers like ₹0.02 or ₹5,00,00,000 from the tail of the
# lognormal distribution).
AMOUNT_MIN: float = 10.0
AMOUNT_MAX: float = 50_000.0


# ---------------------------------------------------------------------------
# Recovery base-rate parameters
# ---------------------------------------------------------------------------

# Base probability of recovery conditioned on failure softness. These are
# STARTING points that label_generator.py will further adjust using
# customer history, retry count, and fatigue signals - they are not the
# final per-row probabilities.
RECOVERY_BASE_RATE_SOFT_FAILURE: float = 0.70
RECOVERY_BASE_RATE_HARD_FAILURE: float = 0.15

# Multiplicative penalty applied per additional retry beyond the first,
# capturing diminishing returns of repeated retry attempts (a required
# scenario: "repeated failures with diminishing recovery probability").
RECOVERY_RETRY_DECAY_FACTOR: float = 0.75

# How strongly customer_past_recovery_rate shifts the base recovery
# probability. Kept as an explicit weight so it's tunable/explainable
# rather than buried in formula constants inside label_generator.py.
RECOVERY_HISTORY_WEIGHT: float = 0.35

# How strongly recent nudge fatigue (ignored/unsubscribed responses,
# high nudge frequency) suppresses recovery probability specifically for
# the nudge-driven recovery path.
RECOVERY_FATIGUE_PENALTY_WEIGHT: float = 0.25


# ---------------------------------------------------------------------------
# Risk / fraud base-rate parameters
# ---------------------------------------------------------------------------

# Overall target fraud prevalence range in the final dataset. Matches
# realistic real-world card-not-present fraud rates - deliberately rare,
# so the risk model must be evaluated with imbalance-aware metrics
# (PR-AUC, recall at fixed FPR) rather than plain accuracy.
EXPECTED_FRAUD_PREVALENCE_RANGE: Tuple[float, float] = (0.02, 0.05)

# Base risk probability by archetype "family" - used as a starting point
# before per-row signal adjustments (velocity, IP mismatch, device change,
# chargeback history) are layered on in label_generator.py.
RISK_BASE_RATE_BY_ARCHETYPE: Dict[str, float] = {
    "loyal_low_risk": 0.002,
    "normal": 0.01,
    "new_customer": 0.02,
    "financially_constrained": 0.01,
    "suspicious": 0.35,
    "account_takeover_like": 0.55,
}

# How strongly joint risk-signal interactions (e.g. high velocity AND
# multiple payment methods AND device change) compound risk probability,
# multiplicatively rather than additively - required to realistically
# represent card-testing and account-takeover patterns.
RISK_INTERACTION_BOOST_WEIGHT: float = 0.30


# ---------------------------------------------------------------------------
# Label noise
# ---------------------------------------------------------------------------

# Both labels get a small amount of injected noise (probability that the
# sampled label is flipped after being drawn). Real-world outcome labels
# are never perfectly clean - injecting noise avoids suspiciously perfect
# separability, which would make later model evaluation metrics look
# unrealistic to judges/evaluators.
LABEL_NOISE_RATE: float = 0.025  # 2.5%, within the requested 2-3% range


# ---------------------------------------------------------------------------
# Expected prevalence ranges (used by validation.py, not by generation)
# ---------------------------------------------------------------------------

# These are SANITY-CHECK targets, not generation inputs. validation.py will
# assert the realized dataset falls within these bounds after generation,
# catching cases where an upstream parameter change accidentally produces
# an unrealistic dataset (e.g. 40% fraud rate due to a bug).
EXPECTED_RECOVERY_PREVALENCE_RANGE: Tuple[float, float] = (0.30, 0.50)
EXPECTED_RISK_PREVALENCE_RANGE: Tuple[float, float] = (0.02, 0.05)