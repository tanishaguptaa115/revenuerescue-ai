"""
RevenueRescue AI - FastAPI Backend

HTTP API layer over:
- Supabase merchant policy service
- RevenueRescue AI Decision Engine

This module:
- exposes health and root endpoints
- accepts merchant_id + already-computed risk/recovery scores
- loads the merchant's policy from Supabase
- converts merchant risk tolerance (low/medium/high) to a numeric value
- delegates business decisions to ml.decision_engine.decide()
- performs request validation through Pydantic
- does not train models
- does not perform feature engineering
"""

from pathlib import Path
from typing import Dict, Optional

import csv
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ml.decision_engine import DecisionInput, decide
from services.merchant_policy_service import get_merchant_policy
from ml.inference import score_transaction
from services.decision_logging_service import persist_decision_flow


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ALLOWED_CORS_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:5173",
]
# ---------------------------------------------------------------------------
# Customer profile lookup
# ---------------------------------------------------------------------------

CUSTOMER_PROFILES_PATH = (
    Path(__file__).resolve().parent
    / "output"
    / "customer_profiles.csv"
)


def load_customer_profiles() -> Dict[str, dict]:
    """Load synthetic customer profiles for real feature enrichment."""
    if not CUSTOMER_PROFILES_PATH.exists():
        raise FileNotFoundError(
            f"Customer profiles file not found: {CUSTOMER_PROFILES_PATH}"
        )

    with CUSTOMER_PROFILES_PATH.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        profiles = {}

        for row in reader:
            customer_id = row.get("customer_id")

            if not customer_id:
                continue

            profiles[customer_id] = row

    return profiles


CUSTOMER_PROFILES = load_customer_profiles()

# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="RevenueRescue AI API",
    description="API layer for RevenueRescue AI decisioning.",
    version="1.1.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class DecisionRequest(BaseModel):
    """Validated API request for a business decision."""

    merchant_id: str = Field(..., min_length=1)

    risk_score: float = Field(..., ge=0.0, le=1.0)
    recovery_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    payment_failed: bool

    is_soft_failure: Optional[bool] = None

    retry_count_so_far: int = Field(
        default=0,
        ge=0,
    )

    amount: float = Field(
        default=0.0,
        ge=0.0,
    )

    payment_method: str = "unknown"


class DecisionResponse(BaseModel):
    """Clean JSON response returned by the Decision API."""

    merchant_id: str

    action: str

    risk_score: float
    recovery_score: Optional[float]

    merchant_risk_tolerance: float

    reason_code: str
    human_readable_reason: str

    priority: str

    metadata: dict


# ---------------------------------------------------------------------------
# Decision helper
# ---------------------------------------------------------------------------

def _make_decision(request: DecisionRequest) -> DecisionResponse:
    """
    Load merchant policy from Supabase and execute the deterministic
    Decision Engine using the merchant-specific policy.
    """

    # ---------------------------------------------------------------
    # 1. Load merchant policy
    # ---------------------------------------------------------------

    try:
        policy = get_merchant_policy(request.merchant_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=502,
            detail="Unable to read merchant policy from Supabase.",
        ) from exc

    # ---------------------------------------------------------------
    # 2. Convert policy to Decision Engine input
    # ---------------------------------------------------------------

    decision_input = DecisionInput(
        risk_score=request.risk_score,
        recovery_score=request.recovery_score,
        payment_failed=request.payment_failed,
        merchant_risk_tolerance=policy.risk_tolerance_score,
        is_soft_failure=request.is_soft_failure,
        retry_count_so_far=request.retry_count_so_far,
        amount=request.amount,
        payment_method=request.payment_method,
    )

    # ---------------------------------------------------------------
    # 3. Execute Decision Engine
    # ---------------------------------------------------------------

    try:
        result = decide(
            decision_input,
            recovery_score_threshold=(
                policy.min_recovery_probability_for_auto_action
            ),
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Decision Engine failed unexpectedly.",
        ) from exc

    # ---------------------------------------------------------------
    # 4. Add merchant-policy context to audit metadata
    # ---------------------------------------------------------------

    metadata = dict(result.metadata)

    metadata.update(
        {
            "merchant_policy": {
                "risk_tolerance_label": policy.merchant_risk_tolerance,
                "max_auto_retries": policy.max_auto_retries,
                "max_auto_action_amount": policy.max_auto_action_amount,
                "max_risk_score_for_auto_action": (
                    policy.max_risk_score_for_auto_action
                ),
                "min_recovery_probability_for_auto_action": (
                    policy.min_recovery_probability_for_auto_action
                ),
                "max_fatigue_score_for_auto_action": (
                    policy.max_fatigue_score_for_auto_action
                ),
                "cooldown_minutes": policy.cooldown_minutes,
                "auto_nudge_enabled": policy.auto_nudge_enabled,
                "auto_retry_enabled": policy.auto_retry_enabled,
            }
        }
    )

    return DecisionResponse(
        merchant_id=policy.merchant_id,
        action=result.action,
        risk_score=result.risk_score,
        recovery_score=result.recovery_score,
        merchant_risk_tolerance=result.merchant_risk_tolerance,
        reason_code=result.reason_code,
        human_readable_reason=result.human_readable_reason,
        priority=result.priority,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def root() -> dict:
    """Basic API availability endpoint."""
    return {
        "message": "RevenueRescue AI API is running"
    }


@app.get("/health")
def health() -> dict:
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "RevenueRescue AI API",
    }


@app.post("/decision", response_model=DecisionResponse)
def create_decision(request: DecisionRequest) -> DecisionResponse:
    """
    Generate a RevenueRescue AI business decision.

    merchant_id is used to load the merchant's policy from Supabase.
    Risk and recovery scores are supplied by the upstream ML layer.
    """
    return _make_decision(request)


@app.post("/demo/decision", response_model=DecisionResponse)
def demo_decision() -> DecisionResponse:
    """
    Run one safe successful demo transaction.

    The demo uses the first available merchant policy from Supabase.
    """
    from services.merchant_policy_service import list_merchant_policies

    try:
        policies = list_merchant_policies(limit=1)
    except RuntimeError as exc:
        raise HTTPException(
            status_code=502,
            detail="Unable to read merchant policies from Supabase.",
        ) from exc

    if not policies:
        raise HTTPException(
            status_code=404,
            detail="No merchant policy is available for the demo.",
        )

    policy = policies[0]

    demo_request = DecisionRequest(
        merchant_id=policy.merchant_id,
        risk_score=0.03,
        recovery_score=None,
        payment_failed=False,
        is_soft_failure=None,
        retry_count_so_far=0,
        amount=1200.0,
        payment_method="UPI",
    )

    return _make_decision(demo_request)




# ---------------------------------------------------------------------------
# ML scoring + decision endpoint
# ---------------------------------------------------------------------------

class ScoreAndDecisionRequest(BaseModel):
    """Complete feature payload for ML scoring followed by decisioning."""

    merchant_id: str = Field(..., min_length=1)

    customer_id: str = Field(..., min_length=1)

    amount: float = Field(..., gt=0.0)

    currency: str = Field(default="INR", min_length=3, max_length=3)

    payment_method: str = "unknown"

    payment_failed: bool

    failure_reason: Optional[str] = None

    is_soft_failure: Optional[bool] = None

    retry_count_so_far: int = Field(
        default=0,
        ge=0,
    )

    risk_features: dict

    recovery_features: Optional[dict] = None


@app.post("/score-and-decide", response_model=DecisionResponse)
def score_and_decide(
    request: ScoreAndDecisionRequest,
) -> DecisionResponse:
    """
    Score a transaction with the trained ML models, run the merchant
    Decision Engine, and persist the complete decision flow to Supabase.
    """

       # ---------------------------------------------------------------
    # 1. Customer profile enrichment + ML scoring
    # ---------------------------------------------------------------

    customer_profile = CUSTOMER_PROFILES.get(request.customer_id)

    if customer_profile is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown customer_id: {request.customer_id}",
        )

    try:
        enriched_risk_features = dict(request.risk_features)

        enriched_risk_features.update(
            {
                "amount_to_customer_avg_ratio": (
                    request.amount
                    / float(
                        customer_profile[
                            "avg_transaction_amount_customer"
                        ]
                    )
                ),
                "is_new_customer": (
                    customer_profile["archetype"] == "new_customer"
                ),
                "customer_past_success_rate": float(
                    customer_profile["customer_past_success_rate"]
                ),
                "customer_past_recovery_rate": float(
                    customer_profile["customer_past_recovery_rate"]
                ),
                "nudge_ignore_tendency": float(
                    customer_profile["nudge_ignore_tendency"]
                ),
                "chargeback_history_count": int(
                    float(
                        customer_profile[
                            "chargeback_history_count"
                        ]
                    )
                ),
            }
        )

        enriched_recovery_features = None

        if request.payment_failed:
            enriched_recovery_features = dict(
                request.recovery_features or {}
            )

            enriched_recovery_features.update(
                {
                    "amount_to_customer_avg_ratio": (
                        request.amount
                        / float(
                            customer_profile[
                                "avg_transaction_amount_customer"
                            ]
                        )
                    ),
                    "is_new_customer": (
                        customer_profile["archetype"] == "new_customer"
                    ),
                    "customer_past_success_rate": float(
                        customer_profile["customer_past_success_rate"]
                    ),
                    "customer_past_recovery_rate": float(
                        customer_profile["customer_past_recovery_rate"]
                    ),
                    "nudge_ignore_tendency": float(
                        customer_profile["nudge_ignore_tendency"]
                    ),
                }
            )

        scores = score_transaction(
            risk_features=enriched_risk_features,
            recovery_features=enriched_recovery_features,
            payment_failed=request.payment_failed,
        )

    except (ValueError, FileNotFoundError, KeyError) as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="ML scoring failed unexpectedly.",
        ) from exc
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="ML scoring failed unexpectedly.",
        ) from exc

    # ---------------------------------------------------------------
    # 2. Decision Engine
    # ---------------------------------------------------------------

    decision_request = DecisionRequest(
        merchant_id=request.merchant_id,
        risk_score=scores["risk_score"],
        recovery_score=scores["recovery_score"],
        payment_failed=request.payment_failed,
        is_soft_failure=request.is_soft_failure,
        retry_count_so_far=request.retry_count_so_far,
        amount=request.amount,
        payment_method=request.payment_method,
    )

    result = _make_decision(decision_request)

    # ---------------------------------------------------------------
    # 3. Persist transaction + attempt + decision + audit
    # ---------------------------------------------------------------

    try:
        persisted = persist_decision_flow(
            merchant_id=request.merchant_id,
            amount=request.amount,
            currency=request.currency,
            payment_method=request.payment_method,
            payment_failed=request.payment_failed,
            failure_reason=request.failure_reason,
            recovery_probability=scores["recovery_score"],
            risk_score=scores["risk_score"],
            fatigue_score=None,
            opportunity_score=None,
            recommended_action=result.action,
            policy_result=(
                "blocked"
                if result.action == "BLOCK"
                else "requires_review"
                if result.action == "REVIEW"
                else "approved"
            ),
            reasoning={
                "reason_code": result.reason_code,
                "human_readable_reason": result.human_readable_reason,
                "priority": result.priority,
                "model_scores": scores,
                "decision_metadata": result.metadata,
            },
            explanation=result.human_readable_reason,
            retry_count_so_far=request.retry_count_so_far,
        )
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Decision persistence failed: {exc}",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Decision persistence failed unexpectedly.",
        ) from exc

    # ---------------------------------------------------------------
    # 4. Add persistence IDs to response metadata
    # ---------------------------------------------------------------

    metadata = dict(result.metadata)

    metadata["persistence"] = {
        "transaction_id": persisted.transaction_id,
        "payment_attempt_id": persisted.payment_attempt_id,
        "decision_id": persisted.decision_id,
        "audit_event_id": persisted.audit_event_id,
    }

    return DecisionResponse(
        merchant_id=result.metadata.get(
            "merchant_id",
            request.merchant_id,
        ),
        action=result.action,
        risk_score=result.risk_score,
        recovery_score=result.recovery_score,
        merchant_risk_tolerance=result.merchant_risk_tolerance,
        reason_code=result.reason_code,
        human_readable_reason=result.human_readable_reason,
        priority=result.priority,
        metadata=metadata,
    )



# ---------------------------------------------------------------------------
# Local development entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
    )
