"""Shared constants and helpers for live CoralNet Playwright tests."""

from __future__ import annotations

import os
from pathlib import Path

CORALNET_BASE = "https://coralnet.ucsd.edu"
DEFAULT_PUBLIC_SOURCE_ID = 23
NOT_FOUND_SOURCE_ID = 999_999_999


def resolve_public_source_id() -> int:
    """Public source for live tests; override with CORALNET_LIVE_SOURCE_ID."""
    raw = os.environ.get("CORALNET_LIVE_SOURCE_ID", "").strip()
    if raw:
        return int(raw)
    return DEFAULT_PUBLIC_SOURCE_ID


FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures" / "coralnet_html"


def save_fixture(refresh: bool, name: str, html: str) -> None:
    if not refresh:
        return
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    (FIXTURES_DIR / name).write_text(html, encoding="utf-8")


def login_coralnet(page, username: str, password: str) -> bool:
    page.goto(f"{CORALNET_BASE}/accounts/login/", wait_until="domcontentloaded")
    page.fill('input[name="username"]', username)
    page.fill('input[name="password"]', password)
    page.click('input[type="submit"]')
    page.wait_for_load_state("networkidle")
    html = page.content()
    return "Sign out" in html or page.url.rstrip("/") != f"{CORALNET_BASE}/accounts/login"


def skip_reason_for_source_response(response, source_id: int) -> str | None:
    if response is None:
        return None
    if response.status >= 400:
        return f"source {source_id} unavailable (HTTP {response.status})"
    return None
