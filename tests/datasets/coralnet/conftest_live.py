"""Playwright fixtures for live CoralNet validation tests (not run in CI)."""

from __future__ import annotations

import os

import pytest

from tests.datasets.coralnet.coralnet_live import resolve_public_source_id


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--refresh-fixtures",
        action="store_true",
        default=False,
        help="Save live CoralNet HTML snapshots into fixtures/coralnet_html/",
    )


@pytest.fixture(scope="session")
def refresh_fixtures(request: pytest.FixtureRequest) -> bool:
    return bool(request.config.getoption("--refresh-fixtures"))


@pytest.fixture(scope="session")
def public_source_id() -> int:
    return resolve_public_source_id()


@pytest.fixture(scope="session")
def playwright_browser():
    pytest.importorskip("playwright")
    from playwright.sync_api import sync_playwright

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        yield browser
        browser.close()


@pytest.fixture
def page(playwright_browser):
    context = playwright_browser.new_context(
        user_agent=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
    )
    pg = context.new_page()
    yield pg
    context.close()


@pytest.fixture
def coralnet_credentials() -> tuple[str, str] | None:
    user = os.environ.get("CORALNET_USERNAME", "").strip()
    pw = os.environ.get("CORALNET_PASSWORD", "")
    if user and pw:
        return user, pw
    return None
