"""Offline tests for the CoralNet HTML parsers."""

from __future__ import annotations

from pathlib import Path

import pytest
from coralnet_scraper.parsers import (
    parse_export_annotations_prep_form,
    parse_image_status_counts,
    parse_source_access,
    parse_total_images_from_source_html,
)

_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "coralnet_html"


def _fixture(name: str) -> str:
    return (_FIXTURES / name).read_text(encoding="utf-8")


def test_parse_total_images_from_fixture():
    html = _fixture("source_ok.html")
    assert parse_total_images_from_source_html(html) == 1234


def test_parse_image_status_fixture_only_total_missing_confirmed():
    html = _fixture("source_ok.html")
    c = parse_image_status_counts(html)
    assert c.table_found is True
    assert c.total_images == 1234
    assert c.confirmed is None
    assert c.unclassified is None


def test_parse_image_status_zero_confirmed_like_public_source():
    html = _fixture("source_zero_confirmed.html")
    c = parse_image_status_counts(html)
    assert c.table_found is True
    assert c.unclassified == 40
    assert c.unconfirmed == 0
    assert c.confirmed == 0
    assert c.total_images == 40


@pytest.mark.parametrize(
    ("fixture", "exp_ok"),
    [
        ("source_ok.html", True),
        ("source_not_found.html", False),
        ("source_denied.html", False),
    ],
)
def test_parse_source_access_fixtures(fixture, exp_ok):
    html = _fixture(fixture)
    ok, err = parse_source_access(html)
    assert ok is exp_ok
    if exp_ok:
        assert err is None
    else:
        assert err is not None


def test_parse_export_annotations_prep_form():
    html = (
        '<form id="export-annotations-prep-form">'
        '<input type="hidden" name="csrfmiddlewaretoken" value="tok123"/></form>'
    )
    assert parse_export_annotations_prep_form(html) == "tok123"
