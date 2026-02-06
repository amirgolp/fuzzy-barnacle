"""Health check routes."""

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health_check():
    """Check API health status."""
    return {"status": "healthy", "service": "quantdash-api"}


@router.get("/version")
async def get_version():
    """Get API version."""
    from quantdash import __version__
    return {"version": __version__}
