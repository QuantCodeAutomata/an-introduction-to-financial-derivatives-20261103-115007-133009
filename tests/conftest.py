"""Pytest configuration: custom markers and environment-based skip logic."""

import os
import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Register the market_data marker so pytest recognises it."""
    config.addinivalue_line(
        "markers",
        "market_data: marks tests that require a live DATALOADER_API_KEY",
    )


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    """Skip market_data tests when DATALOADER_API_KEY is absent."""
    if os.environ.get("DATALOADER_API_KEY"):
        return
    skip = pytest.mark.skip(reason="DATALOADER_API_KEY not set; skipping market data tests")
    for item in items:
        if item.get_closest_marker("market_data"):
            item.add_marker(skip)
