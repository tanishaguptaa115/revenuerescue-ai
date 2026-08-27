from fastapi import FastAPI

app = FastAPI(
    title="RevenueRescue AI",
    description="Risk-aware autonomous revenue recovery platform.",
    version="0.1.0",
)


@app.get("/health")
def health_check() -> dict[str, str]:
    """Basic service health check."""
    return {
        "status": "healthy",
        "service": "RevenueRescue AI",
    }