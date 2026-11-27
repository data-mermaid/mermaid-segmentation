"""Offline tests for the CoralNet HTML parsers."""

from __future__ import annotations

from pathlib import Path

import pytest

from coralnet_scraper.parsers import (
    parse_export_annotations_prep_form,
    parse_image_status_counts,
    parse_points_per_image,
    parse_source_access,
    parse_total_images_from_source_html,
)

_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "coralnet_html"


def _fixture(name: str) -> str:
    return (_FIXTURES / name).read_text(encoding="utf-8")


def test_parse_total_images_is_zero_when_page_has_no_image_status_table():
    assert parse_total_images_from_source_html("<html><body><h1>Source</h1></body></html>") == 0


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


def _details_html(method: str) -> str:
    return (
        '<ul class="flexible-details"><li><span>Visibility:</span><span>Public</span></li>'
        f"<li><span>Point generation method:</span><span>{method}</span></li></ul>"
    )


def test_parse_points_per_image_from_fixture():
    assert parse_points_per_image(_fixture("source_ok.html")) == 10


@pytest.mark.parametrize(
    ("method", "points"),
    [
        ("Simple random, 200 points", 200),
        (
            "Stratified random, 2 rows x 5 columns of cells, 1 points per cell (total of 10 points)",
            10,
        ),
        ("Uniform grid, 10 rows x 10 columns (total of 100 points)", 100),
        ("Simple random, 1,500 points", 1500),
        ("Imported, 50 points", 50),
    ],
)
def test_parse_points_per_image_reads_each_method(method, points):
    assert parse_points_per_image(_details_html(method)) == points


@pytest.mark.parametrize(
    "html",
    [
        "<html><body><h1>Source</h1></body></html>",
        _details_html("Unknown method"),
        _details_html("2 rows x 5 columns of cells, 1 points per cell"),
    ],
)
def test_parse_points_per_image_is_none_when_unknown(html):
    assert parse_points_per_image(html) is None
