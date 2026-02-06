"""FastAPI application main entry point."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from quantdash.config import get_settings, setup_logging

from .routes import data, derivatives, features, health, news, strategies, symbols

settings = get_settings()
logger = setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    logger.info("Starting QuantDash API...")
    yield
    logger.info("Shutting down QuantDash API...")


app = FastAPI(
    title="QuantDash API",
    description="Quantitative trading dashboard API",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, tags=["Health"])
app.include_router(data.router, prefix="/data", tags=["Data"])
app.include_router(features.router, prefix="/features", tags=["Features"])
app.include_router(symbols.router, prefix="/symbols", tags=["Symbols"])
app.include_router(strategies.router, prefix="/strategies", tags=["Strategies"])
app.include_router(news.router, prefix="/news", tags=["News"])
app.include_router(derivatives.router, tags=["Derivatives"])
