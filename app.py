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

from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ml.decision_engine import DecisionInput, decide
from services.merchant_policy_service import get_merchant_policy


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ALLOWED_CORS_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:5173",
]


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
        action=result.action.value,
        risk_score=result.risk_score,
        recovery_score=result.recovery_score,
        merchant_risk_tolerance=result.merchant_risk_tolerance,
        reason_code=result.reason_code.value,
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
# Local development entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
    )