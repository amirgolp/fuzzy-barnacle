"""Pytest configuration and fixtures."""

import pytest


@pytest.fixture(scope="session")
def vcr_config():
    """VCR configuration for recording HTTP requests."""
    return {
        "filter_headers": ["authorization"],
        "record_mode": "once",
    }
