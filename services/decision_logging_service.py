"""
RevenueRescue AI - Decision Persistence and Audit Logging Service.

This module persists transaction and decision information to the
existing Supabase database schema.

Tables used:
    - transactions
    - payment_attempts
    - agent_decisions
    - audit_events

Important:
    - No database schema changes are performed here.
    - No model training is performed here.
    - No feature engineering is performed here.
    - Business actions are mapped to the database's allowed values.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from .supabase_client import get_supabase_client


# ============================================================================
# DATA OBJECTS
# ============================================================================


@dataclass(frozen=True)
class TransactionRecord:
    """Persisted transaction information."""

    id: str
    merchant_id: str


@dataclass(frozen=True)
class DecisionLogRecord:
    """IDs produced by the complete persistence flow."""

    transaction_id: str
    payment_attempt_id: Optional[str]
    decision_id: str
    audit_event_id: str


# ============================================================================
# VALIDATION HELPERS
# ============================================================================


def _validate_non_empty(value: str, field_name: str) -> str:
    """Validate that a required string is non-empty."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")

    return value.strip()


def _validate_positive(value: float, field_name: str) -> float:
    """
    Validate a strictly positive numeric value.

    The existing `transactions` table requires amount > 0.
    """

    numeric_value = float(value)

    if numeric_value <= 0:
        raise ValueError(f"{field_name} must be > 0.")

    return numeric_value


# ============================================================================
# BUSINESS ACTION -> DATABASE ACTION
# ============================================================================


def _map_business_action_to_db_action(action: str) -> str:
    """
    Map RevenueRescue business actions to the values allowed by
    `agent_decisions.recommended_action`.

    Business action:
        RECOVER
        ALLOW
        REVIEW
        BLOCK

    Database action:
        auto_retry
        do_not_retry
        human_review
        blocked
    """

    mapping = {
        "RECOVER": "auto_retry",
        "ALLOW": "do_not_retry",
        "REVIEW": "human_review",
        "BLOCK": "blocked",
    }

    normalized = action.strip().upper()

    if normalized not in mapping:
        raise ValueError(
            f"Unsupported business action for agent_decisions: {action}"
        )

    return mapping[normalized]


# ============================================================================
# BUSINESS ACTION -> DATABASE POLICY RESULT
# ============================================================================


def _map_business_action_to_policy_result(action: str) -> str:
    """
    Map RevenueRescue business actions to the values allowed by
    `agent_decisions.policy_result`.

    Database allows:
        approved
        blocked
        requires_review
    """

    mapping = {
        "RECOVER": "approved",
        "ALLOW": "approved",
        "REVIEW": "requires_review",
        "BLOCK": "blocked",
    }

    normalized = action.strip().upper()

    if normalized not in mapping:
        raise ValueError(
            f"Unsupported business action for policy_result: {action}"
        )

    return mapping[normalized]


# ============================================================================
# TRANSACTION PERSISTENCE
# ============================================================================


def create_transaction(
    *,
    merchant_id: str,
    amount: float,
    currency: str = "INR",
    payment_method: Optional[str] = None,
    payment_failed: bool = False,
    failure_reason: Optional[str] = None,
    external_transaction_id: Optional[str] = None,
    customer_id: Optional[str] = None,
) -> TransactionRecord:
    """
    Create a transaction in the existing `transactions` table.

    The database generates the transaction UUID.

    Status mapping:
        payment_failed=True  -> failed
        payment_failed=False -> captured
    """

    merchant_id = _validate_non_empty(
        merchant_id,
        "merchant_id",
    )

    amount = _validate_positive(
        amount,
        "amount",
    )

    currency = _validate_non_empty(
        currency,
        "currency",
    )

    payload: dict[str, Any] = {
        "merchant_id": merchant_id,
        "amount": amount,
        "currency": currency,
        "payment_method": payment_method,
        "status": "failed" if payment_failed else "captured",
        "failure_reason": failure_reason,
        "external_transaction_id": external_transaction_id,
        "customer_id": customer_id,
    }

    # Omit optional fields when they are not supplied.
    payload = {
        key: value
        for key, value in payload.items()
        if value is not None
    }

    client = get_supabase_client()

    try:
        response = (
            client.table("transactions")
            .insert(payload)
            .execute()
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed to create transaction in Supabase."
        ) from exc

    rows = response.data or []

    if not rows:
        raise RuntimeError(
            "Supabase created no transaction record."
        )

    row = rows[0]

    return TransactionRecord(
        id=str(row["id"]),
        merchant_id=str(row["merchant_id"]),
    )


# ============================================================================
# PAYMENT ATTEMPT PERSISTENCE
# ============================================================================


def create_payment_attempt(
    *,
    transaction_id: str,
    attempt_number: int,
    payment_method: Optional[str],
    status: str,
    failure_reason: Optional[str] = None,
) -> str:
    """
    Create a payment attempt linked to an existing transaction.
    """

    transaction_id = _validate_non_empty(
        transaction_id,
        "transaction_id",
    )

    if attempt_number < 1:
        raise ValueError(
            "attempt_number must be >= 1."
        )

    status = _validate_non_empty(
        status,
        "status",
    )

    payload: dict[str, Any] = {
        "transaction_id": transaction_id,
        "attempt_number": attempt_number,
        "payment_method": payment_method,
        "status": status,
        "failure_reason": failure_reason,
    }

    payload = {
        key: value
        for key, value in payload.items()
        if value is not None
    }

    client = get_supabase_client()

    try:
        response = (
            client.table("payment_attempts")
            .insert(payload)
            .execute()
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed to create payment attempt in Supabase."
        ) from exc

    rows = response.data or []

    if not rows:
        raise RuntimeError(
            "Supabase created no payment attempt record."
        )

    return str(rows[0]["id"])


# ============================================================================
# AGENT DECISION PERSISTENCE
# ============================================================================


def create_agent_decision(
    *,
    transaction_id: str,
    recovery_probability: Optional[float],
    risk_score: Optional[float],
    fatigue_score: Optional[float],
    opportunity_score: Optional[float],
    recommended_action: str,
    policy_result: str,
    reasoning: dict[str, Any],
    explanation: Optional[str] = None,
    llm_provider: Optional[str] = None,
    llm_model: Optional[str] = None,
) -> str:
    """
    Persist a RevenueRescue business decision in `agent_decisions`.

    The public/business action is translated into the exact database
    values required by the existing CHECK constraints.

    The original business-level policy result is preserved inside
    the JSON reasoning field.
    """

    transaction_id = _validate_non_empty(
        transaction_id,
        "transaction_id",
    )

    recommended_action = _validate_non_empty(
        recommended_action,
        "recommended_action",
    )

    policy_result = _validate_non_empty(
        policy_result,
        "policy_result",
    )

    db_recommended_action = _map_business_action_to_db_action(
        recommended_action
    )

    db_policy_result = _map_business_action_to_policy_result(
        recommended_action
    )

    # Preserve the original RevenueRescue policy result in the JSON field
    # while storing the database-compatible policy_result in the constrained
    # text column.
    stored_reasoning = dict(reasoning)
    stored_reasoning["business_policy_result"] = policy_result
    stored_reasoning["database_recommended_action"] = (
        db_recommended_action
    )
    stored_reasoning["database_policy_result"] = db_policy_result

    payload: dict[str, Any] = {
        "transaction_id": transaction_id,
        "recovery_probability": recovery_probability,
        "risk_score": risk_score,
        "fatigue_score": fatigue_score,
        "opportunity_score": opportunity_score,
        "recommended_action": db_recommended_action,
        "policy_result": db_policy_result,
        "reasoning": stored_reasoning,
        "explanation": explanation,
        "llm_provider": llm_provider,
        "llm_model": llm_model,
    }

    payload = {
        key: value
        for key, value in payload.items()
        if value is not None
    }

    client = get_supabase_client()

    try:
        response = (
            client.table("agent_decisions")
            .insert(payload)
            .execute()
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed to create agent decision in Supabase."
        ) from exc

    rows = response.data or []

    if not rows:
        raise RuntimeError(
            "Supabase created no agent decision record."
        )

    return str(rows[0]["id"])


# ============================================================================
# AUDIT EVENT PERSISTENCE
# ============================================================================


def create_audit_event(
    *,
    merchant_id: str,
    transaction_id: str,
    decision_id: str,
    event_type: str,
    actor_type: str,
    message: Optional[str],
    payload: dict[str, Any],
    action_id: Optional[str] = None,
) -> str:
    """
    Persist one audit event in the existing `audit_events` table.
    """

    merchant_id = _validate_non_empty(
        merchant_id,
        "merchant_id",
    )

    transaction_id = _validate_non_empty(
        transaction_id,
        "transaction_id",
    )

    decision_id = _validate_non_empty(
        decision_id,
        "decision_id",
    )

    event_type = _validate_non_empty(
        event_type,
        "event_type",
    )

    actor_type = _validate_non_empty(
        actor_type,
        "actor_type",
    )

    audit_payload: dict[str, Any] = {
        "merchant_id": merchant_id,
        "transaction_id": transaction_id,
        "decision_id": decision_id,
        "action_id": action_id,
        "event_type": event_type,
        "actor_type": actor_type,
        "message": message,
        "payload": payload,
    }

    audit_payload = {
        key: value
        for key, value in audit_payload.items()
        if value is not None
    }

    client = get_supabase_client()

    try:
        response = (
            client.table("audit_events")
            .insert(audit_payload)
            .execute()
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed to create audit event in Supabase."
        ) from exc

    rows = response.data or []

    if not rows:
        raise RuntimeError(
            "Supabase created no audit event record."
        )

    return str(rows[0]["id"])


# ============================================================================
# COMPLETE DECISION PERSISTENCE FLOW
# ============================================================================


def persist_decision_flow(
    *,
    merchant_id: str,
    amount: float,
    currency: str,
    payment_method: Optional[str],
    payment_failed: bool,
    failure_reason: Optional[str],
    recovery_probability: Optional[float],
    risk_score: Optional[float],
    fatigue_score: Optional[float],
    opportunity_score: Optional[float],
    recommended_action: str,
    policy_result: str,
    reasoning: dict[str, Any],
    explanation: Optional[str] = None,
    retry_count_so_far: int = 0,
    event_type: str = "DECISION_CREATED",
    actor_type: str = "agent",
) -> DecisionLogRecord:
    """
    Persist one complete RevenueRescue decision flow.

    Order:
        1. transaction
        2. payment attempt, when payment failed
        3. agent decision
        4. audit event

    Existing foreign-key relationships are respected.
    """

    transaction = create_transaction(
        merchant_id=merchant_id,
        amount=amount,
        currency=currency,
        payment_method=payment_method,
        payment_failed=payment_failed,
        failure_reason=failure_reason,
    )

    payment_attempt_id: Optional[str] = None

    if payment_failed:
        payment_attempt_id = create_payment_attempt(
            transaction_id=transaction.id,
            attempt_number=max(1, retry_count_so_far + 1),
            payment_method=payment_method,
            status="failed",
            failure_reason=failure_reason,
        )

    decision_id = create_agent_decision(
        transaction_id=transaction.id,
        recovery_probability=recovery_probability,
        risk_score=risk_score,
        fatigue_score=fatigue_score,
        opportunity_score=opportunity_score,
        recommended_action=recommended_action,
        policy_result=policy_result,
        reasoning=reasoning,
        explanation=explanation,
    )

    audit_event_id = create_audit_event(
        merchant_id=merchant_id,
        transaction_id=transaction.id,
        decision_id=decision_id,
        event_type=event_type,
        actor_type=actor_type,
        message=explanation,
        payload={
            "recommended_action": recommended_action,
            "database_recommended_action": (
                _map_business_action_to_db_action(
                    recommended_action
                )
            ),
            "business_policy_result": policy_result,
            "database_policy_result": (
                _map_business_action_to_policy_result(
                    recommended_action
                )
            ),
            "risk_score": risk_score,
            "recovery_probability": recovery_probability,
            "payment_failed": payment_failed,
            "payment_attempt_id": payment_attempt_id,
        },
    )

    return DecisionLogRecord(
        transaction_id=transaction.id,
        payment_attempt_id=payment_attempt_id,
        decision_id=decision_id,
        audit_event_id=audit_event_id,
    )


# ============================================================================
# LOCAL IMPORT / VALIDATION TEST
# ============================================================================


if __name__ == "__main__":
    print("=" * 70)
    print("DECISION LOGGING SERVICE - IMPORT / VALIDATION TEST")
    print("=" * 70)

    sample = DecisionLogRecord(
        transaction_id="test-transaction-id",
        payment_attempt_id="test-attempt-id",
        decision_id="test-decision-id",
        audit_event_id="test-audit-id",
    )

    print("Service imported successfully.")
    print(f"Sample record: {sample}")
    print("No database write performed by this local test.")