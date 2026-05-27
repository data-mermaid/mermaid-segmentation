"""Live Playwright smoke tests against coralnet.ucsd.edu (excluded from default pytest)."""

from __future__ import annotations

import io
import re
import urllib.parse

import pandas as pd
import pytest
from bs4 import BeautifulSoup

from mermaidseg.datasets.coralnet.scraper.downloader import _extract_page_image_ids
from mermaidseg.datasets.coralnet.scraper.parsers import (
    classifier_plot_data_from_source_html,
    parse_export_annotations_prep_form,
    parse_source_access,
    parse_total_images_from_source_html,
)
from tests.datasets.coralnet.coralnet_live import (
    CORALNET_BASE,
    NOT_FOUND_SOURCE_ID,
    login_coralnet,
    save_fixture,
    skip_reason_for_source_response,
)

PlaywrightError = pytest.importorskip("playwright.sync_api").Error

pytestmark = pytest.mark.live


def test_public_source_overview_structure(page, public_source_id, refresh_fixtures):
    url = f"{CORALNET_BASE}/source/{public_source_id}/"
    response = page.goto(url, wait_until="domcontentloaded")
    if reason := skip_reason_for_source_response(response, public_source_id):
        pytest.skip(reason)

    html = page.content()
    save_fixture(refresh_fixtures, "source_ok.html", html)

    header = page.locator("h4", has_text="Image Status")
    assert header.count() >= 1

    table = header.first.locator(
        "xpath=following-sibling::table[contains(@class, 'detail_box_table')]"
    )
    assert table.count() >= 1

    total_row = table.first.locator("tr", has_text="Total images:")
    assert total_row.count() >= 1
    link_text = total_row.first.locator("a").inner_text().strip().replace(",", "")
    assert link_text.isdigit()
    dom_count = int(link_text)

    parser_count = parse_total_images_from_source_html(html)
    assert parser_count == dom_count
    assert parser_count > 0

    classifier_data = classifier_plot_data_from_source_html(html)
    if classifier_data is not None:
        assert len(classifier_data) > 0


def test_source_access_denied_page(page, refresh_fixtures):
    url = f"{CORALNET_BASE}/source/{NOT_FOUND_SOURCE_ID}/"
    page.goto(url, wait_until="domcontentloaded")
    html = page.content()

    accessible, err = parse_source_access(html)
    assert accessible is False
    assert err is not None

    if err == "Permission denied":
        save_fixture(refresh_fixtures, "source_denied.html", html)
    else:
        save_fixture(refresh_fixtures, "source_not_found.html", html)


def test_browse_images_pagination(page, public_source_id):
    url = f"{CORALNET_BASE}/source/{public_source_id}/browse/images"
    response = page.goto(url, wait_until="domcontentloaded")
    if reason := skip_reason_for_source_response(response, public_source_id):
        pytest.skip(reason)

    wrappers = page.locator("span.thumb_wrapper")
    assert wrappers.count() >= 1

    first = wrappers.first
    assert first.locator("a").count() >= 1
    assert first.locator("img").count() >= 1
    assert first.locator("img").first.get_attribute("title")

    if wrappers.count() > 20:
        assert page.locator('a[title="Next page"]').count() >= 1


def test_export_form_csrf(page, public_source_id, coralnet_credentials):
    if coralnet_credentials is None:
        pytest.skip("CORALNET_USERNAME / CORALNET_PASSWORD not set")

    username, password = coralnet_credentials
    if not login_coralnet(page, username, password):
        pytest.skip("CoralNet login failed — check CORALNET_USERNAME / CORALNET_PASSWORD")

    url = f"{CORALNET_BASE}/source/{public_source_id}/browse/images/"
    response = page.goto(url, wait_until="domcontentloaded")
    if reason := skip_reason_for_source_response(response, public_source_id):
        pytest.skip(reason)

    form = page.locator("form#export-annotations-prep-form")
    if form.count() == 0:
        pytest.skip(
            f"export form missing for source {public_source_id} "
            "(source may be private or have no exportable annotations)"
        )

    csrf_input = form.first.locator('input[name="csrfmiddlewaretoken"]')
    assert csrf_input.count() >= 1
    dom_token = csrf_input.first.get_attribute("value")
    assert dom_token

    parser_token = parse_export_annotations_prep_form(page.content())
    assert parser_token == dom_token


def test_labelset_table(page, public_source_id):
    url = f"{CORALNET_BASE}/source/{public_source_id}/labelset/"
    response = page.goto(url, wait_until="domcontentloaded")
    if reason := skip_reason_for_source_response(response, public_source_id):
        pytest.skip(reason)

    table = page.locator("table#label-table")
    assert table.count() >= 1
    assert table.first.locator("tr td a").count() >= 1


def test_image_view_page(page, public_source_id):
    browse_url = f"{CORALNET_BASE}/source/{public_source_id}/browse/images"
    response = page.goto(browse_url, wait_until="domcontentloaded")
    if reason := skip_reason_for_source_response(response, public_source_id):
        pytest.skip(reason)

    first_link = page.locator("span.thumb_wrapper a").first
    if first_link.count() == 0:
        pytest.skip("no image links on browse page")

    href = first_link.get_attribute("href")
    assert href

    image_url = href if href.startswith("http") else f"{CORALNET_BASE}{href}"
    page.goto(image_url, wait_until="domcontentloaded")

    img = page.locator("div#original_image_container > img")
    assert img.count() >= 1
    assert img.first.get_attribute("src")


def test_pagination_links_well_formed(page, public_source_id):
    """Walk to page 2 and verify the resolved URL is well-formed.

    Exercises the same ``urljoin`` flow used by ``download_annotations_paginated`` and
    ``get_images``. Regression test for the bug where path/query concatenation produced URLs like
    ``.../images/?...?page=2``.
    """
    browse_url = f"{CORALNET_BASE}/source/{public_source_id}/browse/images/"
    response = page.goto(browse_url, wait_until="domcontentloaded")
    if reason := skip_reason_for_source_response(response, public_source_id):
        pytest.skip(reason)

    next_link = page.locator('a[title="Next page"]')
    if next_link.count() == 0:
        pytest.skip(f"source {public_source_id} has <= 1 browse page")

    next_href = next_link.first.get_attribute("href")
    assert next_href
    page_two_url = urllib.parse.urljoin(browse_url, next_href)

    parsed = urllib.parse.urlparse(page_two_url)
    assert page_two_url.count("?") <= 1, f"malformed URL with multiple '?': {page_two_url}"
    assert parsed.scheme in ("http", "https")
    assert parsed.netloc == "coralnet.ucsd.edu"
    assert "?" not in parsed.path, f"query embedded in path: {page_two_url}"

    response = page.goto(page_two_url, wait_until="domcontentloaded")
    assert response is not None and response.status < 400
    assert page.locator("span.thumb_wrapper").count() >= 1


def test_paginated_annotation_export_produces_csv(page, public_source_id, coralnet_credentials):
    """End-to-end: log in, paginate, export annotations from 1-2 pages, validate CSV.

    Mirrors the production ``download_annotations_paginated`` flow against the
    live site: GET the browse page, POST ``export_prep`` with the page CSRF,
    GET ``export/serve``, parse CSV. Asserts the merged annotations have no
    duplicate ``(Name, Row, Column)`` tuples — the same dedup key the
    production code uses.
    """
    if coralnet_credentials is None:
        pytest.skip("CORALNET_USERNAME / CORALNET_PASSWORD not set")

    username, password = coralnet_credentials
    if not login_coralnet(page, username, password):
        pytest.skip("CoralNet login failed")

    context = page.context

    def export_current_page(browse_url: str) -> pd.DataFrame | None:
        response = page.goto(browse_url, wait_until="domcontentloaded")
        if reason := skip_reason_for_source_response(response, public_source_id):
            pytest.skip(reason)
        form = page.locator("form#export-annotations-prep-form")
        if form.count() == 0:
            return None
        page_html = page.content()
        csrf_token = parse_export_annotations_prep_form(page_html)
        assert csrf_token

        page_image_ids = _extract_page_image_ids(BeautifulSoup(page_html, "html.parser"))
        if not page_image_ids:
            return None

        prep_resp = context.request.post(
            f"{CORALNET_BASE}/source/{public_source_id}/annotation/export_prep/",
            form={
                "csrfmiddlewaretoken": csrf_token,
                "browse_action": "export_annotations",
                "image_form_type": "search",
                "image_select_type": "selected",
                "image_id_list": "_".join(str(pk) for pk in page_image_ids),
                "label_format": "both",
            },
            headers={"Referer": browse_url},
            timeout=180_000,
        )
        if not prep_resp.ok:
            pytest.skip(f"export_prep returned HTTP {prep_resp.status}")
        ts = prep_resp.json()["session_data_timestamp"]
        serve_resp = context.request.get(
            f"{CORALNET_BASE}/source/{public_source_id}/export/serve/?session_data_timestamp={ts}",
            timeout=180_000,
        )
        if not serve_resp.ok:
            pytest.skip(f"export/serve returned HTTP {serve_resp.status}")
        return pd.read_csv(io.StringIO(serve_resp.text()))

    browse_url = f"{CORALNET_BASE}/source/{public_source_id}/browse/images/"
    df_page1 = export_current_page(browse_url)
    if df_page1 is None:
        pytest.skip(f"source {public_source_id} has no exportable annotations")
    for col in ("Name", "Row", "Column"):
        assert col in df_page1.columns, f"missing column {col} in export"

    next_link = page.locator('a[title="Next page"]')
    if next_link.count() == 0:
        return  # single-page source, page-1 assertions are enough

    next_href = next_link.first.get_attribute("href")
    assert next_href
    page_two_url = urllib.parse.urljoin(browse_url, next_href)
    df_page2 = export_current_page(page_two_url)
    if df_page2 is None or df_page2.empty:
        return

    combined = pd.concat([df_page1, df_page2], ignore_index=True)
    dedup = combined.drop_duplicates(subset=["Name", "Row", "Column"])
    assert len(dedup) == len(combined), (
        f"page 1 and page 2 exports share {len(combined) - len(dedup)} duplicate annotations"
    )


def _parse_post_body(body: str, content_type: str | None) -> dict[str, str]:
    """Parse a Playwright-captured POST body into ``{field_name: value}``.

    Handles both ``application/x-www-form-urlencoded`` (what our scraper sends) and
    ``multipart/form-data`` (what CoralNet's UI sends). The multipart path uses a hand-rolled
    boundary split rather than the ``email`` module so it doesn't choke on body-only content (no
    headers).

    Fails the calling test if the multipart boundary can't be located; a silent empty dict would
    masquerade as a body-shape mismatch and hide a parser regression.
    """
    ct = content_type or ""
    looks_multipart = "multipart/form-data" in ct.lower() or body.lstrip().startswith("--")
    if looks_multipart:
        boundary_m = re.search(r"boundary=([^;\s]+)", ct, re.IGNORECASE)
        if boundary_m:
            boundary = boundary_m.group(1)
        else:
            first_line_m = re.match(r"--+([^\r\n-]+)", body)
            if not first_line_m:
                pytest.fail(
                    f"could not extract multipart boundary; content_type={ct!r}, "
                    f"body_prefix={body[:120]!r}"
                )
            boundary = first_line_m.group(1)
        delim_re = re.compile(r"--+" + re.escape(boundary) + r"(?:--)?")
        out: dict[str, str] = {}
        for part in delim_re.split(body):
            part = part.strip("\r\n")
            if not part or part == "--":
                continue
            header_m = re.match(
                r'\s*Content-Disposition:\s*form-data;\s*name="([^"]+)"[^\r\n]*\r?\n\s*\r?\n(.*)\Z',
                part,
                re.DOTALL,
            )
            if header_m:
                out.setdefault(header_m.group(1), header_m.group(2).rstrip("\r\n"))
        return out
    parsed = urllib.parse.parse_qs(body, keep_blank_values=True)
    return {k: v[0] for k, v in parsed.items()}


def test_ui_export_post_body_matches_production_payload(
    page, public_source_id, coralnet_credentials
):
    """Drive the real CoralNet UI and assert its export POST body matches our scraper's.

    Catches CoralNet UI/server changes that would silently break paginated
    export (e.g. a renamed ``image_id_list`` field, a different image-id
    encoding, or an added required field). The scraper builds its POST body
    in :meth:`CoralNetDownloader._export_one_page`; if CoralNet starts sending
    a different shape, this test fails and forces us to update both.

    Note on payload shape: the CoralNet UI sends ``multipart/form-data`` with
    a minimal set of fields (``csrfmiddlewaretoken``, ``label_format``,
    ``image_id_list``); the scraper sends ``application/x-www-form-urlencoded``
    with a defensive superset (also ``browse_action``, ``image_select_type``,
    ``image_form_type``, ``optional_columns``). CoralNet's server accepts both
    encodings and ignores fields it doesn't expect, so the only field we
    strictly need to match is ``image_id_list``.
    """
    if coralnet_credentials is None:
        pytest.skip("CORALNET_USERNAME / CORALNET_PASSWORD not set")
    username, password = coralnet_credentials
    if not login_coralnet(page, username, password):
        pytest.skip("CoralNet login failed")

    browse_url = f"{CORALNET_BASE}/source/{public_source_id}/browse/images/"
    response = page.goto(browse_url, wait_until="domcontentloaded")
    if reason := skip_reason_for_source_response(response, public_source_id):
        pytest.skip(reason)
    if page.locator("span.thumb_wrapper").count() == 0:
        pytest.skip(f"source {public_source_id} has no images on browse page")

    image_select = page.locator('select[name="image_select_type"]')
    browse_action = page.locator('select[name="browse_action"]')
    if image_select.count() == 0 or browse_action.count() == 0:
        pytest.skip("Image Actions dropdowns not present (anonymous user?)")

    image_select.select_option("selected")
    browse_action.select_option("export_annotations")
    page.wait_for_timeout(300)

    go_candidates = page.locator(
        'input[type="submit"][value="Go"], button:has-text("Go"), input[type="button"][value="Go"]'
    ).all()
    visible_go = next((g for g in go_candidates if g.is_visible()), None)
    if visible_go is None:
        pytest.skip(
            f'No visible "Go" button after selecting export_annotations '
            f"(found {len(go_candidates)} candidates, all hidden)"
        )
    go_locator = visible_go

    try:
        with page.expect_request(
            lambda r: "/annotation/export_prep/" in r.url and r.method == "POST",
            timeout=15_000,
        ) as req_info:
            go_locator.click()
        request = req_info.value
        post_body = request.post_data or ""
        content_type = request.headers.get("content-type")
    except PlaywrightError as e:
        pytest.skip(
            f"Did not capture export_prep POST (UI may have changed): {type(e).__name__}: {e}"
        )

    fields = _parse_post_body(post_body, content_type)
    expected_image_ids = _extract_page_image_ids(BeautifulSoup(page.content(), "html.parser"))

    assert "csrfmiddlewaretoken" in fields, fields
    assert "image_id_list" in fields, (
        f"CoralNet UI POST did not include image_id_list; our scraper relies on it. "
        f"Captured fields: {sorted(fields.keys())}"
    )

    ui_ids = tuple(int(x) for x in fields["image_id_list"].split("_") if x)
    assert set(ui_ids) == set(expected_image_ids), (
        f"image_id_list mismatch: UI sent {ui_ids}, page DOM has {expected_image_ids}"
    )
    assert len(ui_ids) <= 100, (
        f"CoralNet validator caps image_id_list at 100, UI sent {len(ui_ids)}"
    )
