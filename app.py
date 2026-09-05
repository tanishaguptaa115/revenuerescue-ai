"""
RevenueRescue AI - FastAPI Backend

Thin HTTP API layer over the existing Decision Engine.

This module:
- exposes health and root endpoints
- accepts already-computed risk/recovery scores
- delegates business decisions to ml.decision_engine.decide()
- performs request validation through Pydantic
- does not train models
- does not connect to a database
- does not perform feature engineering
"""

from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ml.decision_engine import DecisionInput, decide


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
    version="1.0.0",
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

    risk_score: float = Field(..., ge=0.0, le=1.0)
    recovery_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    payment_failed: bool
    merchant_risk_tolerance: float = Field(..., ge=0.0, le=1.0)
    is_soft_failure: Optional[bool] = None
    retry_count_so_far: int = Field(default=0, ge=0)
    amount: float = Field(default=0.0, ge=0.0)
    payment_method: str = "unknown"


class DecisionResponse(BaseModel):
    """Clean JSON response returned by the Decision API."""

    action: str
    risk_score: float
    recovery_score: Optional[float]
    merchant_risk_tolerance: float
    reason_code: str
    human_readable_reason: str
    priority: str
    metadata: dict


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_decision(request: DecisionRequest) -> DecisionResponse:
    """
    Convert an API request into DecisionInput and execute the
    deterministic business decision layer.
    """
    decision_input = DecisionInput(
        risk_score=request.risk_score,
        recovery_score=request.recovery_score,
        payment_failed=request.payment_failed,
        merchant_risk_tolerance=request.merchant_risk_tolerance,
        is_soft_failure=request.is_soft_failure,
        retry_count_so_far=request.retry_count_so_far,
        amount=request.amount,
        payment_method=request.payment_method,
    )

    try:
        result = decide(decision_input)
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

    return DecisionResponse(
        action=result.action.value,
        risk_score=result.risk_score,
        recovery_score=result.recovery_score,
        merchant_risk_tolerance=result.merchant_risk_tolerance,
        reason_code=result.reason_code.value,
        human_readable_reason=result.human_readable_reason,
        priority=result.priority,
        metadata=result.metadata,
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
    Generate a RevenueRescue AI business decision from supplied
    risk/recovery scores and transaction context.
    """
    return _make_decision(request)


@app.post("/demo/decision", response_model=DecisionResponse)
def demo_decision() -> DecisionResponse:
    """
    Run one safe, successful demo transaction through the
    Decision Engine.
    """
    demo_request = DecisionRequest(
        risk_score=0.03,
        recovery_score=None,
        payment_failed=False,
        merchant_risk_tolerance=0.50,
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