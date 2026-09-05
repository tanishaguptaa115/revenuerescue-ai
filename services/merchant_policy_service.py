"""
Merchant policy service for RevenueRescue AI.

Reads merchant policy configuration from Supabase and converts the
database representation into a form that the Decision Engine can use.

This module:
- reads existing data only
- does not modify database schema
- does not insert/update/delete rows
- performs no model training
"""

from dataclasses import dataclass
from typing import Optional

from .supabase_client import get_supabase_client


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Database stores merchant_risk_tolerance as text:
# low / medium / high
#
# Decision Engine expects a numeric value in [0, 1].
RISK_TOLERANCE_MAP = {
    "low": 0.30,
    "medium": 0.50,
    "high": 0.70,
}


# ---------------------------------------------------------------------------
# Typed policy object
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MerchantPolicy:
    merchant_id: str
    max_auto_retries: int
    max_auto_action_amount: float
    max_risk_score_for_auto_action: float
    min_recovery_probability_for_auto_action: float
    max_fatigue_score_for_auto_action: float
    cooldown_minutes: int
    auto_nudge_enabled: bool
    auto_retry_enabled: bool
    merchant_risk_tolerance: str

    @property
    def risk_tolerance_score(self) -> float:
        """
        Convert the database text value into the numeric value expected
        by the Decision Engine.
        """
        value = self.merchant_risk_tolerance.strip().lower()

        if value not in RISK_TOLERANCE_MAP:
            raise ValueError(
                f"Unsupported merchant_risk_tolerance '{self.merchant_risk_tolerance}'. "
                f"Expected one of: {list(RISK_TOLERANCE_MAP)}"
            )

        return RISK_TOLERANCE_MAP[value]


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _parse_policy(row: dict) -> MerchantPolicy:
    """
    Convert one Supabase row into a validated MerchantPolicy object.
    """
    required_fields = [
        "merchant_id",
        "max_auto_retries",
        "max_auto_action_amount",
        "max_risk_score_for_auto_action",
        "min_recovery_probability_for_auto_action",
        "max_fatigue_score_for_auto_action",
        "cooldown_minutes",
        "auto_nudge_enabled",
        "auto_retry_enabled",
        "merchant_risk_tolerance",
    ]

    missing = [field for field in required_fields if field not in row]

    if missing:
        raise ValueError(
            f"Merchant policy row is missing required fields: {missing}"
        )

    policy = MerchantPolicy(
        merchant_id=str(row["merchant_id"]),
        max_auto_retries=int(row["max_auto_retries"]),
        max_auto_action_amount=float(row["max_auto_action_amount"]),
        max_risk_score_for_auto_action=float(
            row["max_risk_score_for_auto_action"]
        ),
        min_recovery_probability_for_auto_action=float(
            row["min_recovery_probability_for_auto_action"]
        ),
        max_fatigue_score_for_auto_action=float(
            row["max_fatigue_score_for_auto_action"]
        ),
        cooldown_minutes=int(row["cooldown_minutes"]),
        auto_nudge_enabled=bool(row["auto_nudge_enabled"]),
        auto_retry_enabled=bool(row["auto_retry_enabled"]),
        merchant_risk_tolerance=str(row["merchant_risk_tolerance"]),
    )

    # Validate the mapped tolerance immediately.
    _ = policy.risk_tolerance_score

    return policy


# ---------------------------------------------------------------------------
# Supabase read operations
# ---------------------------------------------------------------------------

def get_merchant_policy(merchant_id: str) -> MerchantPolicy:
    """
    Fetch one merchant's policy from Supabase.

    Raises:
        ValueError: if merchant_id is empty or no valid policy is found.
        RuntimeError: if the Supabase request fails.
    """
    if not merchant_id or not merchant_id.strip():
        raise ValueError("merchant_id must be a non-empty string.")

    client = get_supabase_client()

    try:
        response = (
            client.table("merchant_policies")
            .select(
                "merchant_id,"
                "max_auto_retries,"
                "max_auto_action_amount,"
                "max_risk_score_for_auto_action,"
                "min_recovery_probability_for_auto_action,"
                "max_fatigue_score_for_auto_action,"
                "cooldown_minutes,"
                "auto_nudge_enabled,"
                "auto_retry_enabled,"
                "merchant_risk_tolerance"
            )
            .eq("merchant_id", merchant_id)
            .limit(1)
            .execute()
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed to read merchant policy from Supabase."
        ) from exc

    rows = response.data or []

    if not rows:
        raise ValueError(
            f"No merchant policy found for merchant_id='{merchant_id}'."
        )

    return _parse_policy(rows[0])


def list_merchant_policies(limit: int = 10) -> list[MerchantPolicy]:
    """
    Read up to `limit` merchant policies.

    Useful for development/admin inspection.
    """
    if limit < 1:
        raise ValueError("limit must be >= 1.")

    client = get_supabase_client()

    try:
        response = (
            client.table("merchant_policies")
            .select(
                "merchant_id,"
                "max_auto_retries,"
                "max_auto_action_amount,"
                "max_risk_score_for_auto_action,"
                "min_recovery_probability_for_auto_action,"
                "max_fatigue_score_for_auto_action,"
                "cooldown_minutes,"
                "auto_nudge_enabled,"
                "auto_retry_enabled,"
                "merchant_risk_tolerance"
            )
            .limit(limit)
            .execute()
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed to read merchant policies from Supabase."
        ) from exc

    return [_parse_policy(row) for row in (response.data or [])]


# ---------------------------------------------------------------------------
# Local verification
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    policies = list_merchant_policies(limit=10)

    print("=" * 70)
    print("MERCHANT POLICY SERVICE - VERIFICATION")
    print("=" * 70)
    print(f"Policies found: {len(policies)}")

    for policy in policies:
        print("\nMerchant:")
        print(f"  merchant_id: {policy.merchant_id}")
        print(
            f"  risk tolerance: {policy.merchant_risk_tolerance} "
            f"-> {policy.risk_tolerance_score:.2f}"
        )
        print(f"  max auto retries: {policy.max_auto_retries}")
        print(f"  max auto action amount: {policy.max_auto_action_amount:.2f}")
        print(
            "  max risk score for auto action: "
            f"{policy.max_risk_score_for_auto_action:.2f}"
        )
        print(
            "  min recovery probability for auto action: "
            f"{policy.min_recovery_probability_for_auto_action:.2f}"
        )
        print(
            "  max fatigue score for auto action: "
            f"{policy.max_fatigue_score_for_auto_action:.2f}"
        )
        print(f"  cooldown: {policy.cooldown_minutes} minutes")
        print(f"  auto nudge enabled: {policy.auto_nudge_enabled}")
        print(f"  auto retry enabled: {policy.auto_retry_enabled}")

    print("\nMerchant policy service verification complete.")