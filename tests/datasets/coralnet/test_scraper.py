"""Offline tests for CoralNet scraper parsers, classification, and batch slicing."""

from __future__ import annotations

import io
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import requests
from bs4 import BeautifulSoup

from mermaidseg.datasets.coralnet.scraper import batch_download as bd
from mermaidseg.datasets.coralnet.scraper import downloader as downloader_mod
from mermaidseg.datasets.coralnet.scraper import s3_io
from mermaidseg.datasets.coralnet.scraper.classify import classify_remediation_action
from mermaidseg.datasets.coralnet.scraper.downloader import (
    _PAGINATED_EXPORT_THRESHOLD,
    CoralNetDownloader,
    _extract_page_image_ids,
    export_prep_timeout_seconds,
    export_serve_timeout_seconds,
    half_open_batches,
)
from mermaidseg.datasets.coralnet.scraper.models import RemediationAction, SourceProbe
from mermaidseg.datasets.coralnet.scraper.parsers import (
    parse_export_annotations_prep_form,
    parse_image_status_counts,
    parse_source_access,
    parse_total_images_from_source_html,
)

_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "coralnet_html"


def _fixture(name: str) -> str:
    return (_FIXTURES / name).read_text(encoding="utf-8")


def test_half_open_batches_covers_disjoint_ranges():
    n, bs = 1001, 1000
    ranges = half_open_batches(n, bs)
    assert ranges == [(0, 1000), (1000, 1001)]
    visited = []
    for a, b in ranges:
        visited.extend(range(a, b))
    assert visited == list(range(n))


def test_export_timeouts_scale_with_source_size():
    assert export_prep_timeout_seconds(None) == 300
    assert export_prep_timeout_seconds(100) == 300
    assert export_prep_timeout_seconds(32176) == 1728
    assert export_prep_timeout_seconds(100_000) == 5120
    assert export_serve_timeout_seconds(32176) == 703


def test_download_annotations_retries_export_prep_on_timeout():
    dl = CoralNetDownloader("u", "p")
    dl.logged_in = True
    dl.s3 = MagicMock()
    browse_html = (
        '<form id="export-annotations-prep-form">'
        '<input type="hidden" name="csrfmiddlewaretoken" value="tok"/></form>'
    )
    browse_resp = MagicMock()
    browse_resp.text = browse_html
    browse_resp.raise_for_status = MagicMock()
    post_resp = MagicMock()
    post_resp.json.return_value = {"session_data_timestamp": "123"}
    post_resp.raise_for_status = MagicMock()
    get_serve = MagicMock()
    get_serve.text = "Name,Row,Column,Label ID\na.jpg,1,2,3\n"
    get_serve.raise_for_status = MagicMock()

    mock_post = MagicMock(side_effect=[requests.exceptions.ReadTimeout("t"), post_resp])

    with (
        patch.object(dl.session, "get", side_effect=[browse_resp, get_serve]),
        patch.object(dl.session, "post", mock_post),
        patch("mermaidseg.datasets.coralnet.scraper.downloader.time.sleep"),
    ):
        ok = dl.download_annotations(1, "b", "p", total_images_hint=500)

    assert ok is True
    assert mock_post.call_count == 2


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


def test_classify_skip_no_confirmed_when_site_reports_zero_confirmed():
    probe = SourceProbe(
        5961,
        "url",
        True,
        None,
        total_images_website=40,
        n_confirmed_website=0,
        n_unconfirmed_website=0,
        n_unclassified_website=40,
    )
    act = classify_remediation_action(
        probe=probe,
        has_images_folder=False,
        has_annotations_csv=False,
        has_image_list_csv=False,
        n_images_s3=0,
        n_images_csv=0,
        n_annotations=0,
        annotations_csv_read_failed=False,
        annotations_empty=True,
        image_count_match=False,
    )
    assert act == RemediationAction.SKIP_NO_CONFIRMED_ANNOTATIONS


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


def test_classify_manual_when_website_fewer_than_s3_images():
    probe = SourceProbe(1, "https://example/s1/", True, None, total_images_website=5)
    act = classify_remediation_action(
        probe=probe,
        has_images_folder=True,
        has_annotations_csv=True,
        has_image_list_csv=True,
        n_images_s3=50,
        n_images_csv=50,
        n_annotations=100,
        annotations_csv_read_failed=False,
        annotations_empty=False,
        image_count_match=False,
    )
    assert act == RemediationAction.MANUAL_REVIEW


def test_classify_redownload_csv_when_annotation_read_failed_and_site_has_images():
    probe = SourceProbe(1, "url", True, None, total_images_website=10)
    act = classify_remediation_action(
        probe=probe,
        has_images_folder=True,
        has_annotations_csv=True,
        has_image_list_csv=True,
        n_images_s3=5,
        n_images_csv=5,
        n_annotations=0,
        annotations_csv_read_failed=True,
        annotations_empty=False,
        image_count_match=False,
    )
    assert act == RemediationAction.REDOWNLOAD_CSV


def test_classify_skip_not_accessible():
    probe = SourceProbe(1, "url", False, "Permission denied", total_images_website=0)
    act = classify_remediation_action(
        probe=probe,
        has_images_folder=True,
        has_annotations_csv=True,
        has_image_list_csv=True,
        n_images_s3=10,
        n_images_csv=10,
        n_annotations=50,
        annotations_csv_read_failed=False,
        annotations_empty=False,
        image_count_match=True,
    )
    assert act == RemediationAction.SKIP_NOT_ACCESSIBLE


def test_classify_skip_empty_website():
    probe = SourceProbe(2, "url", True, None, total_images_website=0)
    act = classify_remediation_action(
        probe=probe,
        has_images_folder=True,
        has_annotations_csv=True,
        has_image_list_csv=True,
        n_images_s3=0,
        n_images_csv=0,
        n_annotations=0,
        annotations_csv_read_failed=False,
        annotations_empty=True,
        image_count_match=True,
    )
    assert act == RemediationAction.SKIP_EMPTY_WEBSITE


def test_classify_full_redownload_when_missing_structure():
    probe = SourceProbe(3, "url", True, None, total_images_website=20)
    act = classify_remediation_action(
        probe=probe,
        has_images_folder=False,
        has_annotations_csv=False,
        has_image_list_csv=False,
        n_images_s3=0,
        n_images_csv=0,
        n_annotations=0,
        annotations_csv_read_failed=False,
        annotations_empty=True,
        image_count_match=False,
    )
    assert act == RemediationAction.FULL_REDOWNLOAD


def test_classify_redownload_images_when_csvs_ok_but_gap():
    probe = SourceProbe(4, "url", True, None, total_images_website=100)
    act = classify_remediation_action(
        probe=probe,
        has_images_folder=True,
        has_annotations_csv=True,
        has_image_list_csv=True,
        n_images_s3=80,
        n_images_csv=80,
        n_annotations=200,
        annotations_csv_read_failed=False,
        annotations_empty=False,
        image_count_match=False,
    )
    assert act == RemediationAction.REDOWNLOAD_IMAGES


def test_download_source_skips_zero_confirmed_on_website():
    dl = CoralNetDownloader("u", "p")
    dl.logged_in = True
    dl.s3 = MagicMock()
    zero_html = _fixture("source_zero_confirmed.html")
    with (
        patch.object(dl, "check_permissions", return_value=True),
        patch.object(dl, "_get_source_overview_html", return_value=zero_html),
        patch.object(dl, "download_metadata") as mock_meta,
        patch.object(dl, "download_labelset") as mock_label,
        patch.object(dl, "download_annotations") as mock_ann,
    ):
        res = dl.download_source(5961, bucket_name="b", s3_prefix="p")
    assert res.ok is True
    assert res.skipped_no_confirmed is True
    mock_meta.assert_not_called()
    mock_label.assert_not_called()
    mock_ann.assert_not_called()


def test_should_skip_force_never_when_enabled(fake_s3):
    assert (
        bd.should_skip_download(
            source_id=1,
            bucket="b",
            s3_prefix="p",
            s3_client=fake_s3,
            force=True,
            legacy_skip_annotations=True,
            complete_source_ids=None,
        )
        is False
    )


def test_should_skip_when_complete_set_contains_id(fake_s3):
    fake_s3.put_object(Bucket="b", Key=s3_io.object_key_annotation_csv("p", 1), Body=b"a,b\n")
    assert (
        bd.should_skip_download(
            source_id=1,
            bucket="b",
            s3_prefix="p",
            s3_client=fake_s3,
            force=False,
            legacy_skip_annotations=False,
            complete_source_ids={1},
        )
        is True
    )


def test_should_legacy_skip_when_annotations_exist(fake_s3):
    fake_s3.put_object(Bucket="b", Key=s3_io.object_key_annotation_csv("p", 99), Body=b"a\n")
    assert bd.should_skip_download(
        source_id=99,
        bucket="b",
        s3_prefix="p",
        s3_client=fake_s3,
        force=False,
        legacy_skip_annotations=True,
        complete_source_ids=None,
    )


# ---------------------------------------------------------------------------
# Paginated export tests
# ---------------------------------------------------------------------------

_BROWSE_HTML_PAGE = """\
<html>
<form id="export-annotations-prep-form">
  <input type="hidden" name="csrfmiddlewaretoken" value="{csrf}"/>
</form>
{thumbs}
{next_link}
</html>"""


def _thumb(image_id: int) -> str:
    return (
        f'<span class="thumb_wrapper">'
        f'<a href="/image/{image_id}/view/"><img title="img{image_id}.jpg"/></a>'
        "</span>"
    )


def _browse_page(
    csrf: str, n_thumbs: int = 1, next_href: str | None = None, start_id: int = 1
) -> str:
    """Build a fake browse page with ``n_thumbs`` thumbs whose image IDs start at ``start_id``.

    Each page in a multi-page test should use a disjoint ``start_id`` range so
    ``_extract_page_image_ids`` produces distinct ID tuples across pages.
    """
    thumbs = "".join(_thumb(start_id + i) for i in range(n_thumbs))
    next_link = f'<a title="Next page" href="{next_href}">Next</a>' if next_href else ""
    return _BROWSE_HTML_PAGE.format(csrf=csrf, thumbs=thumbs, next_link=next_link)


def test_get_export_session_creates_independent_logged_in_session_per_thread():
    """Each phase-2 worker thread gets its own ``requests.Session`` from a fresh login.

    Regression guard for the CoralNet server-side session race: if two threads
    share one ``sessionid`` cookie, ``export_prep`` writes from one thread can
    evict another thread's pending export state.
    """
    dl = CoralNetDownloader("u", "p")
    dl.session = MagicMock()
    new_session_count = [0]

    def fake_new_session() -> requests.Session:
        new_session_count[0] += 1
        return requests.Session()

    dl._new_logged_in_session = fake_new_session  # type: ignore[method-assign]
    dl._export_thread_local = threading.local()

    # Each worker calls _get_export_session twice; both calls must return the
    # same Session (cached per-thread), but distinct threads must get distinct
    # Sessions (independent login).
    seen: dict[int, tuple[requests.Session, requests.Session]] = {}

    def worker(thread_id: int) -> None:
        seen[thread_id] = (dl._get_export_session(), dl._get_export_session())

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert new_session_count[0] == 3
    for first, second in seen.values():
        assert first is second
    assert len({id(first) for first, _ in seen.values()}) == 3


def test_extract_page_image_ids_parses_thumb_anchors():
    html = (
        "<html><body>"
        + _thumb(11)
        + _thumb(22)
        + _thumb(33)
        + '<a href="/image/999/view/">elsewhere</a>'
        + "</body></html>"
    )
    soup = BeautifulSoup(html, "html.parser")
    assert _extract_page_image_ids(soup) == (11, 22, 33)


def test_extract_page_image_ids_deduplicates_preserving_order():
    html = "<html><body>" + _thumb(5) + _thumb(7) + _thumb(5) + "</body></html>"
    soup = BeautifulSoup(html, "html.parser")
    assert _extract_page_image_ids(soup) == (5, 7)


def _ann_csv(rows: list[tuple]) -> str:
    lines = ["Name,Row,Column,Label ID"]
    for name, row, col, label in rows:
        lines.append(f"{name},{row},{col},{label}")
    return "\n".join(lines) + "\n"


def _mock_response(text: str | None = None, json_data: dict | None = None) -> MagicMock:
    r = MagicMock()
    r.raise_for_status = MagicMock()
    if text is not None:
        r.text = text
    if json_data is not None:
        r.json.return_value = json_data
    return r


@contextmanager
def _mock_paginated_session(
    dl: CoralNetDownloader,
    *,
    get: list | Callable,
    post: list | Callable,
) -> Iterator[MagicMock]:
    """Patch ``dl.session.get``/``post`` and ``time.sleep`` for a paginated-export test.

    Yields the ``post`` mock so tests can introspect ``call_args_list`` to assert the export_prep
    POST body. ``get`` and ``post`` accept either a list (consumed in order as ``side_effect``) or a
    callable (for URL-based dispatch in the parallel test).
    """
    with (
        patch.object(dl.session, "get", side_effect=get),
        patch.object(dl.session, "post", side_effect=post) as post_mock,
        patch("mermaidseg.datasets.coralnet.scraper.downloader.time.sleep"),
    ):
        yield post_mock


def test_download_annotations_dispatches_to_paginated_for_large_source(fake_s3):
    """Sources above _PAGINATED_EXPORT_THRESHOLD must use the paginated path."""
    dl = CoralNetDownloader("u", "p")
    dl.logged_in = True
    dl.s3 = fake_s3

    with patch.object(dl, "download_annotations_paginated", return_value=True) as mock_pag:
        ok = dl.download_annotations(1, "b", "p", total_images_hint=_PAGINATED_EXPORT_THRESHOLD + 1)

    assert ok is True
    mock_pag.assert_called_once_with(
        1, "b", "p", total_images_website=_PAGINATED_EXPORT_THRESHOLD + 1
    )


def test_download_annotations_bulk_for_small_source():
    """Sources at or below threshold must use the existing bulk path."""
    dl = CoralNetDownloader("u", "p")
    dl.logged_in = True
    dl.s3 = MagicMock()

    browse_html = (
        '<form id="export-annotations-prep-form">'
        '<input type="hidden" name="csrfmiddlewaretoken" value="tok"/></form>'
    )
    browse_resp = _mock_response(text=browse_html)
    post_resp = _mock_response(json_data={"session_data_timestamp": "t1"})
    serve_resp = _mock_response(text="Name,Row,Column,Label ID\na.jpg,1,2,3\n")

    with (
        patch.object(dl.session, "get", side_effect=[browse_resp, serve_resp]),
        patch.object(dl.session, "post", return_value=post_resp),
        patch("mermaidseg.datasets.coralnet.scraper.downloader.time.sleep"),
    ):
        ok = dl.download_annotations(1, "b", "p", total_images_hint=_PAGINATED_EXPORT_THRESHOLD)

    assert ok is True


def test_paginated_export_skips_when_existing_annotations_are_complete(
    paginated_downloader, fake_s3
):
    """No pages are fetched when existing annotations.csv already covers all website images."""
    dl = paginated_downloader

    existing_csv = _ann_csv([("img1.jpg", 1, 1, 10), ("img2.jpg", 2, 2, 20)])
    key = s3_io.object_key_annotation_csv("p", 1)
    fake_s3.put_object(Bucket="b", Key=key, Body=existing_csv.encode())

    with patch.object(dl.session, "get") as mock_get:
        ok = dl.download_annotations_paginated(1, "b", "p", total_images_website=2)

    assert ok is True
    mock_get.assert_not_called()


def test_paginated_export_three_pages_concatenated(paginated_downloader, fake_s3):
    """Three browse pages produce three annotation chunks concatenated in page order.

    Also locks in the export_prep POST body shape: each per-page POST must send
    ``image_select_type=selected`` and ``image_id_list`` of underscore-joined
    image PKs from that browse page. This is the contract that prevents the
    "exports the whole source per page" regression.
    """
    dl = paginated_downloader

    page1_html = _browse_page("csrf1", n_thumbs=20, next_href="?page=2", start_id=1000)
    page2_html = _browse_page("csrf2", n_thumbs=20, next_href="?page=3", start_id=2000)
    page3_html = _browse_page("csrf3", n_thumbs=10, start_id=3000)

    ann1 = _ann_csv([("a.jpg", 1, 1, 100), ("b.jpg", 2, 2, 101)])
    ann2 = _ann_csv([("c.jpg", 3, 3, 102)])
    ann3 = _ann_csv([("d.jpg", 4, 4, 103), ("e.jpg", 5, 5, 104)])

    get_responses = [
        _mock_response(text=page1_html),
        _mock_response(text=page2_html),
        _mock_response(text=page3_html),
        _mock_response(text=page1_html),
        _mock_response(text=ann1),
        _mock_response(text=page2_html),
        _mock_response(text=ann2),
        _mock_response(text=page3_html),
        _mock_response(text=ann3),
    ]
    post_responses = [
        _mock_response(json_data={"session_data_timestamp": "t1"}),
        _mock_response(json_data={"session_data_timestamp": "t2"}),
        _mock_response(json_data={"session_data_timestamp": "t3"}),
    ]

    with _mock_paginated_session(dl, get=get_responses, post=post_responses) as mock_post:
        ok = dl.download_annotations_paginated(1, "b", "p", page_workers=1)

    assert ok is True
    key = s3_io.object_key_annotation_csv("p", 1)
    assert ("b", key) in fake_s3.objects
    df = pd.read_csv(io.BytesIO(fake_s3.objects[("b", key)]))
    assert len(df) == 5
    assert set(df["Name"]) == {"a.jpg", "b.jpg", "c.jpg", "d.jpg", "e.jpg"}

    expected_id_lists = {
        "csrf1": "_".join(str(i) for i in range(1000, 1020)),
        "csrf2": "_".join(str(i) for i in range(2000, 2020)),
        "csrf3": "_".join(str(i) for i in range(3000, 3010)),
    }
    assert mock_post.call_count == 3
    for call in mock_post.call_args_list:
        form = call.kwargs["data"]
        assert form["image_select_type"] == "selected"
        assert form["image_form_type"] == "search"
        assert form["browse_action"] == "export_annotations"
        assert form["image_id_list"] == expected_id_lists[form["csrfmiddlewaretoken"]]


def test_paginated_export_merges_with_existing_annotations(paginated_downloader, fake_s3):
    """When annotations.csv already exists on S3, new rows are merged and deduplicated."""
    dl = paginated_downloader

    existing_csv = _ann_csv(
        [
            ("old.jpg", 1, 1, 99),
            ("overlap.jpg", 5, 5, 99),  # will be overwritten by new export
        ]
    )
    key = s3_io.object_key_annotation_csv("p", 1)
    fake_s3.put_object(Bucket="b", Key=key, Body=existing_csv.encode())

    page_html = _browse_page("csrf1", n_thumbs=2)
    new_ann = _ann_csv(
        [
            ("overlap.jpg", 5, 5, 200),  # same Name/Row/Column – new label should win
            ("new.jpg", 9, 9, 300),
        ]
    )

    get_responses = [
        _mock_response(text=page_html),
        _mock_response(text=page_html),
        _mock_response(text=new_ann),
    ]
    post_responses = [_mock_response(json_data={"session_data_timestamp": "t1"})]

    with _mock_paginated_session(dl, get=get_responses, post=post_responses):
        ok = dl.download_annotations_paginated(1, "b", "p", page_workers=1)

    assert ok is True
    df = pd.read_csv(io.BytesIO(fake_s3.objects[("b", key)]))
    assert len(df) == 3
    overlap_row = df[df["Name"] == "overlap.jpg"]
    assert overlap_row["Label ID"].iloc[0] == 200


# ---------------------------------------------------------------------------
# get_images pagination tests (offline)
# ---------------------------------------------------------------------------


def test_get_images_multi_page_uses_urljoin():
    """Second-page URL must be resolved via urljoin, not f-string concatenation.

    Regression test for the bug where ``f"{base_url}/{next_page}"`` produced paths like
    ``.../images/?page=2`` instead of ``.../images?page=2``. Also pins the resulting DataFrame shape
    (``Name`` and ``Image Page`` columns).
    """
    dl = CoralNetDownloader("u", "p")
    dl.logged_in = True

    call_urls: list[str] = []

    def fake_get_images_on_page(url: str):
        call_urls.append(url)
        if len(call_urls) == 1:
            return {"a.jpg": "/image/1/view/"}, "?page=2"
        return {"b.jpg": "/image/2/view/"}, None

    with patch.object(dl, "get_images_on_page", side_effect=fake_get_images_on_page):
        df, ok = dl.get_images(7)

    assert ok is True
    assert df is not None
    assert sorted(df["Name"].tolist()) == ["a.jpg", "b.jpg"]
    assert sorted(df["Image Page"].tolist()) == ["/image/1/view/", "/image/2/view/"]
    assert call_urls[0] == "https://coralnet.ucsd.edu/source/7/browse/images"
    assert call_urls[1] == "https://coralnet.ucsd.edu/source/7/browse/images?page=2"


def test_get_images_with_annotation_status_preserves_filter():
    """Pagination with annotation_status must not produce a double '?' in the URL."""
    dl = CoralNetDownloader("u", "p")
    dl.logged_in = True

    call_urls: list[str] = []

    def fake_get_images_on_page(url: str):
        call_urls.append(url)
        if len(call_urls) == 1:
            return {"a.jpg": "/image/1/view/"}, "?annotation_status=confirmed&page=2"
        return {"b.jpg": "/image/2/view/"}, None

    with patch.object(dl, "get_images_on_page", side_effect=fake_get_images_on_page):
        df, ok = dl.get_images(7, annotation_status="confirmed")

    assert ok is True
    assert df is not None and len(df) == 2
    assert call_urls[0].endswith("/browse/images/?annotation_status=confirmed")
    second = call_urls[1]
    assert second.count("?") == 1
    assert "annotation_status=confirmed" in second
    assert "page=2" in second


# ---------------------------------------------------------------------------
# Paginated export retry / checkpoint / parallelism tests
# ---------------------------------------------------------------------------


def test_paginated_export_retries_on_page_timeout(paginated_downloader, fake_s3):
    """A ReadTimeout on attempt 1 of page 2's export_prep is retried and succeeds.

    Phase 2 ``do_round_trip`` re-fetches the browse page for a fresh CSRF before each attempt, so
    the retry triggers a second browse GET for page 2.
    """
    dl = paginated_downloader

    page1_html = _browse_page("csrf1", n_thumbs=2, next_href="?page=2", start_id=10)
    page2_html = _browse_page("csrf2", n_thumbs=1, start_id=20)
    ann1 = _ann_csv([("a.jpg", 1, 1, 10)])
    ann2 = _ann_csv([("b.jpg", 2, 2, 20)])

    get_responses = [
        _mock_response(text=page1_html),
        _mock_response(text=page2_html),
        _mock_response(text=page1_html),
        _mock_response(text=ann1),
        _mock_response(text=page2_html),
        _mock_response(text=page2_html),
        _mock_response(text=ann2),
    ]
    post_responses = [
        _mock_response(json_data={"session_data_timestamp": "t1"}),
        requests.exceptions.ReadTimeout("simulated page-2 timeout"),
        _mock_response(json_data={"session_data_timestamp": "t2"}),
    ]

    with _mock_paginated_session(dl, get=get_responses, post=post_responses):
        ok = dl.download_annotations_paginated(1, "b", "p", page_workers=1)

    assert ok is True
    key = s3_io.object_key_annotation_csv("p", 1)
    df = pd.read_csv(io.BytesIO(fake_s3.objects[("b", key)]))
    assert set(df["Name"]) == {"a.jpg", "b.jpg"}


def test_paginated_export_checkpoint_uploads_partial(paginated_downloader, fake_s3):
    """With ``checkpoint_every_pages=2`` and 3 pages, the CSV is uploaded twice.

    Once at the checkpoint after page 2, and again as the final upload after page 3. The final CSV
    must contain rows from all three pages.
    """
    dl = paginated_downloader

    page1_html = _browse_page("csrf1", n_thumbs=1, next_href="?page=2", start_id=100)
    page2_html = _browse_page("csrf2", n_thumbs=1, next_href="?page=3", start_id=200)
    page3_html = _browse_page("csrf3", n_thumbs=1, start_id=300)
    ann1 = _ann_csv([("a.jpg", 1, 1, 10)])
    ann2 = _ann_csv([("b.jpg", 2, 2, 20)])
    ann3 = _ann_csv([("c.jpg", 3, 3, 30)])

    get_responses = [
        _mock_response(text=page1_html),
        _mock_response(text=page2_html),
        _mock_response(text=page3_html),
        _mock_response(text=page1_html),
        _mock_response(text=ann1),
        _mock_response(text=page2_html),
        _mock_response(text=ann2),
        _mock_response(text=page3_html),
        _mock_response(text=ann3),
    ]
    post_responses = [
        _mock_response(json_data={"session_data_timestamp": "t1"}),
        _mock_response(json_data={"session_data_timestamp": "t2"}),
        _mock_response(json_data={"session_data_timestamp": "t3"}),
    ]

    key = s3_io.object_key_annotation_csv("p", 1)
    with (
        _mock_paginated_session(dl, get=get_responses, post=post_responses),
        patch.object(downloader_mod, "_upload_csv", wraps=downloader_mod._upload_csv) as upload_spy,
    ):
        ok = dl.download_annotations_paginated(
            1, "b", "p", page_workers=1, checkpoint_every_pages=2
        )

    assert ok is True
    assert upload_spy.call_count == 2
    df = pd.read_csv(io.BytesIO(fake_s3.objects[("b", key)]))
    assert set(df["Name"]) == {"a.jpg", "b.jpg", "c.jpg"}


def test_paginated_export_parallel_produces_correct_output(paginated_downloader, fake_s3):
    """``page_workers > 1`` produces the same final CSV regardless of completion order.

    Uses URL-based mock dispatch so the test is not dependent on call order across worker threads.
    """
    dl = paginated_downloader

    pages = {
        "https://coralnet.ucsd.edu/source/1/browse/images/": _browse_page(
            "csrf1", n_thumbs=1, next_href="?page=2", start_id=1
        ),
        "https://coralnet.ucsd.edu/source/1/browse/images/?page=2": _browse_page(
            "csrf2", n_thumbs=1, next_href="?page=3", start_id=2
        ),
        "https://coralnet.ucsd.edu/source/1/browse/images/?page=3": _browse_page(
            "csrf3", n_thumbs=1, next_href="?page=4", start_id=3
        ),
        "https://coralnet.ucsd.edu/source/1/browse/images/?page=4": _browse_page(
            "csrf4", n_thumbs=1, start_id=4
        ),
    }
    serve_payloads = {
        "ts1": _ann_csv([("a.jpg", 1, 1, 10)]),
        "ts2": _ann_csv([("b.jpg", 2, 2, 20)]),
        "ts3": _ann_csv([("c.jpg", 3, 3, 30)]),
        "ts4": _ann_csv([("d.jpg", 4, 4, 40)]),
    }
    csrf_to_ts = {"csrf1": "ts1", "csrf2": "ts2", "csrf3": "ts3", "csrf4": "ts4"}

    def fake_get(url, timeout=None, **_):
        if url in pages:
            return _mock_response(text=pages[url])
        for ts, csv in serve_payloads.items():
            if f"session_data_timestamp={ts}" in url:
                return _mock_response(text=csv)
        raise AssertionError(f"unexpected GET {url}")

    def fake_post(url, data=None, headers=None, timeout=None, **_):
        csrf = data["csrfmiddlewaretoken"]
        return _mock_response(json_data={"session_data_timestamp": csrf_to_ts[csrf]})

    with _mock_paginated_session(dl, get=fake_get, post=fake_post):
        ok = dl.download_annotations_paginated(1, "b", "p", page_workers=3)

    assert ok is True
    key = s3_io.object_key_annotation_csv("p", 1)
    df = pd.read_csv(io.BytesIO(fake_s3.objects[("b", key)]))
    assert set(df["Name"]) == {"a.jpg", "b.jpg", "c.jpg", "d.jpg"}
