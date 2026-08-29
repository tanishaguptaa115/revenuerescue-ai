"""
archetypes.py

Declarative definitions of the six customer archetypes used to generate
RevenueRescue AI's synthetic dataset.

Design principle:
This module is pure DATA, not logic. It defines, for each archetype, the
*distributions and parameters* that later modules (customer_generator.py,
transaction_context_generator.py, risk_signal_generator.py, etc.) will
sample from. No random numbers are drawn here, and no rows are produced
here - this file should be inspectable at a glance, and a reviewer should
be able to see exactly what assumptions drive each archetype's behavior
without reading any generation code.

Each numeric field is expressed as a (min, max) range or a probability,
representing the parameters of a distribution to be sampled from later
(e.g. "tenure_days_range" is the support of a uniform/skewed draw, not a
fixed value). This keeps every customer within an archetype distinct,
rather than archetypes collapsing into six fixed "clones."
"""

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class CustomerArchetype:
    """
    Declarative parameter set for one customer archetype.

    All *_range fields are (min, max) tuples describing the support of a
    distribution that customer_generator.py will sample from - they are
    NOT fixed values. All *_tendency / *_rate fields are probabilities
    (0.0-1.0) describing how likely a given behavior is for a customer of
    this archetype, again to be sampled per-customer/per-transaction by
    downstream modules, not applied deterministically.
    """

    name: str

    # --- Tenure -------------------------------------------------------
    # Range of how long (in days) a customer of this archetype has existed.
    # new_customer explicitly allows a minimum of 0 days, enabling the
    # required "brand-new customer with zero historical data" edge case.
    tenure_days_range: Tuple[int, int]

    # --- Historical performance ----------------------------------------
    # Range for customer_past_success_rate: the fraction of this
    # customer's past payments (not just recoveries) that succeeded
    # outright. High for loyal customers, deliberately undefined/zero-ish
    # for brand-new customers with no history.
    historical_success_rate_range: Tuple[float, float]

    # Range for customer_past_recovery_rate: of this customer's past
    # FAILED payments, what fraction were later recovered. Kept separate
    # from historical_success_rate because a customer can have high
    # first-attempt success but poor recovery behavior when they do fail
    # (or vice versa) - collapsing these into one number would remove a
    # genuinely useful, independent signal.
    historical_recovery_rate_range: Tuple[float, float]

    # --- Spend behavior --------------------------------------------------
    # Range for this archetype's typical (mean) transaction amount in INR.
    # Used to seed avg_transaction_amount_customer per customer, which in
    # turn conditions individual transaction amounts. Wide, high range for
    # loyal_low_risk supports the required "high-value legitimate
    # customer" scenario.
    typical_amount_range: Tuple[float, float]

    # --- Chargeback tendency ---------------------------------------------
    # Probability that a given customer of this archetype has ANY
    # chargeback history at all (chargeback_history_count > 0). Near-zero
    # for legitimate archetypes, elevated for suspicious/account-takeover.
    chargeback_tendency: float

    # --- Payment-method switching ------------------------------------
    # Probability weight controlling how many distinct payment methods
    # this archetype tends to use in a short recent window
    # (num_payment_methods_used_recently). High switching is a classic
    # card-testing / fraud signal, so this is deliberately elevated for
    # suspicious and account_takeover_like archetypes.
    method_switching_tendency: float

    # --- Velocity tendency -------------------------------------------
    # Probability weight controlling how bursty this archetype's
    # transaction timing tends to be (feeds velocity_txn_count_1h/24h via
    # velocity_engine.py, which computes the actual rolling counts from
    # generated timestamps - this field only biases how tightly-clustered
    # those timestamps are for a given archetype).
    velocity_tendency: float

    # --- Device change tendency ------------------------------------------
    # Probability that a given transaction shows a device_change_flag.
    # Sharply elevated for account_takeover_like, since a new/unrecognized
    # device is one of the clearest ATO signals.
    device_change_tendency: float

    # --- IP/country mismatch tendency --------------------------------
    # Probability that a given transaction shows ip_country_mismatch.
    # Also sharply elevated for account_takeover_like; kept independent
    # from device_change_tendency so the two can be jointly boosted by
    # risk_signal_generator.py to create realistic interaction effects
    # rather than always co-occurring by construction.
    ip_mismatch_tendency: float

    # --- Communication / nudge behavior -------------------------------
    # Probability that this archetype tends to IGNORE recovery nudges
    # (feeds last_nudge_response sampling). Elevated for
    # financially_constrained (fatigue from repeated unaffordable
    # requests) and suspicious archetypes.
    nudge_ignore_tendency: float

    # Probability that a customer of this archetype has opted out of
    # communication entirely. Kept low and roughly archetype-independent,
    # since opt-out is mostly a preference, not a risk/recovery signal -
    # but must be nonzero everywhere so the "opted-out customer" required
    # scenario can appear across different recovery-probability profiles.
    opt_out_tendency: float


# ---------------------------------------------------------------------------
# Archetype registry
# ---------------------------------------------------------------------------

ARCHETYPES: Dict[str, CustomerArchetype] = {
    "loyal_low_risk": CustomerArchetype(
        name="loyal_low_risk",
        tenure_days_range=(365, 2000),
        historical_success_rate_range=(0.85, 0.99),
        historical_recovery_rate_range=(0.70, 0.95),
        typical_amount_range=(500.0, 20_000.0),  # includes high-value legitimate customers
        chargeback_tendency=0.005,
        method_switching_tendency=0.05,
        velocity_tendency=0.05,
        device_change_tendency=0.03,
        ip_mismatch_tendency=0.01,
        nudge_ignore_tendency=0.10,
        opt_out_tendency=0.05,
    ),
    "normal": CustomerArchetype(
        name="normal",
        tenure_days_range=(90, 1200),
        historical_success_rate_range=(0.65, 0.90),
        historical_recovery_rate_range=(0.40, 0.70),
        typical_amount_range=(200.0, 5_000.0),
        chargeback_tendency=0.01,
        method_switching_tendency=0.15,
        velocity_tendency=0.15,
        device_change_tendency=0.08,
        ip_mismatch_tendency=0.03,
        nudge_ignore_tendency=0.30,
        opt_out_tendency=0.08,
    ),
    "new_customer": CustomerArchetype(
        name="new_customer",
        # Explicitly allows 0 - enables the required "zero historical
        # data" cold-start edge case.
        tenure_days_range=(0, 30),
        # Wide, low-anchored ranges reflect genuine lack of history rather
        # than assuming new customers are good or bad by default.
        historical_success_rate_range=(0.0, 0.60),
        historical_recovery_rate_range=(0.0, 0.50),
        typical_amount_range=(150.0, 3_000.0),
        chargeback_tendency=0.01,
        method_switching_tendency=0.25,  # exploring different rails is normal for new users
        velocity_tendency=0.10,
        device_change_tendency=0.10,  # single new device is expected, not yet suspicious
        ip_mismatch_tendency=0.03,
        nudge_ignore_tendency=0.35,  # no established trust/engagement yet
        opt_out_tendency=0.06,
    ),
    "financially_constrained": CustomerArchetype(
        name="financially_constrained",
        tenure_days_range=(60, 1500),
        historical_success_rate_range=(0.45, 0.75),
        # Recovery rate here is notably decent DESPITE lower success rate -
        # this archetype often just needs a delay (e.g., payday timing),
        # which is the required "delay is better than immediate retry"
        # insufficient_funds scenario.
        historical_recovery_rate_range=(0.35, 0.65),
        typical_amount_range=(100.0, 2_000.0),
        chargeback_tendency=0.008,
        method_switching_tendency=0.10,
        velocity_tendency=0.08,
        device_change_tendency=0.05,
        ip_mismatch_tendency=0.02,
        # Elevated: repeated insufficient-funds nudges around the same
        # time each month plausibly cause fatigue/annoyance.
        nudge_ignore_tendency=0.45,
        opt_out_tendency=0.10,
    ),
    "suspicious": CustomerArchetype(
        name="suspicious",
        # Kept to a SMALL MINORITY of the customer population via
        # ARCHETYPE_WEIGHTS in config.py, not via any field here.
        tenure_days_range=(0, 180),
        historical_success_rate_range=(0.30, 0.70),
        historical_recovery_rate_range=(0.20, 0.55),
        typical_amount_range=(50.0, 1_500.0),  # card-testing favors small amounts
        chargeback_tendency=0.20,
        method_switching_tendency=0.60,  # classic card-testing signal
        velocity_tendency=0.55,          # rapid repeated attempts
        device_change_tendency=0.25,
        ip_mismatch_tendency=0.20,
        nudge_ignore_tendency=0.50,
        opt_out_tendency=0.05,
    ),
    "account_takeover_like": CustomerArchetype(
        name="account_takeover_like",
        # Also a SMALL MINORITY (config.py ARCHETYPE_WEIGHTS). Note this
        # archetype is layered onto what looks like an otherwise-real
        # account (tenure can be nontrivial), which is the defining
        # feature of account takeover versus fresh synthetic fraud.
        tenure_days_range=(30, 900),
        historical_success_rate_range=(0.60, 0.90),  # account had a normal history before takeover
        historical_recovery_rate_range=(0.40, 0.70),
        typical_amount_range=(500.0, 10_000.0),  # takeover attempts often go for higher value
        chargeback_tendency=0.15,
        method_switching_tendency=0.30,
        velocity_tendency=0.35,
        # Sharply elevated - the two clearest ATO signals, intentionally
        # the highest values in the registry.
        device_change_tendency=0.70,
        ip_mismatch_tendency=0.60,
        nudge_ignore_tendency=0.40,
        opt_out_tendency=0.05,
    ),
}


def get_archetype(name: str) -> CustomerArchetype:
    """
    Retrieve an archetype definition by name.

    Raises a KeyError with a clear message if an invalid archetype name
    is requested, rather than silently returning None - generation code
    should fail loudly on a typo'd archetype key instead of producing
    customers with missing parameters.
    """
    if name not in ARCHETYPES:
        raise KeyError(
            f"Unknown archetype '{name}'. Valid archetypes are: "
            f"{list(ARCHETYPES.keys())}"
        )
    return ARCHETYPES[name]