"""Offline tests for CoralNetDownloader."""

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
from botocore.exceptions import ClientError
from bs4 import BeautifulSoup

from coralnet_scraper import downloader as downloader_mod
from coralnet_scraper import s3_io
from coralnet_scraper.downloader import (
    _PAGINATED_EXPORT_THRESHOLD,
    CoralNetDownloader,
    _extract_page_image_ids,
    _is_transient,
    export_prep_timeout_seconds,
    export_serve_timeout_seconds,
    half_open_batches,
)
from coralnet_scraper.models import ImageFailures

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
        patch("coralnet_scraper.downloader.time.sleep"),
    ):
        ok = dl.download_annotations(1, "b", "p", total_images_hint=500)

    assert ok is True
    assert mock_post.call_count == 2


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


def _image_source_downloader() -> CoralNetDownloader:
    dl = CoralNetDownloader("u", "p", max_workers=2)
    dl.logged_in = True
    dl.s3 = MagicMock()
    return dl


def _download_source_with_images(
    dl: CoralNetDownloader, upload_ok: bool, urls: list[object] | None = None
):
    images = pd.DataFrame({"Image Page": ["/image/1/view/", "/image/2/view/"]})
    image_urls = ["http://x/1", "http://x/2"] if urls is None else urls
    with (
        patch.object(dl, "check_permissions", return_value=True),
        patch.object(dl, "_get_source_overview_html", return_value=_fixture("source_ok.html")),
        patch.object(dl, "download_metadata", return_value=(True, 2)),
        patch.object(dl, "download_labelset", return_value=True),
        patch.object(dl, "download_annotations", return_value=True),
        patch.object(dl, "get_images", return_value=(images, True)),
        patch.object(dl, "get_image_urls", return_value=image_urls),
        patch.object(s3_io, "upload_csv_body"),
        patch.object(dl, "download_image_to_s3", side_effect=lambda u, b, k: (k, upload_ok)),
    ):
        return dl.download_source(1, bucket_name="b", s3_prefix="p")


def test_download_source_fails_when_every_image_upload_fails():
    res = _download_source_with_images(_image_source_downloader(), upload_ok=False)
    assert res.ok is False
    assert res.errors == ["images_upload_failed"]
    assert res.image_failures == ImageFailures(missing_url=0, upload_failed=2)


def test_download_source_reports_only_missing_urls():
    res = _download_source_with_images(
        _image_source_downloader(), upload_ok=True, urls=["http://x/1", pd.NA]
    )
    assert res.ok is False
    assert res.errors == ["images_missing_url"]
    assert res.image_failures == ImageFailures(missing_url=1, upload_failed=0)


def test_download_source_ok_when_every_image_uploads():
    res = _download_source_with_images(_image_source_downloader(), upload_ok=True)
    assert res.ok is True
    assert res.errors == []
    assert res.image_failures == ImageFailures()


def test_download_images_separates_missing_url_from_upload_failures():
    dl = _image_source_downloader()
    # Rows: 1 uploads, 2 fails to upload, 3 raises in the worker, 4 has no URL.
    df = pd.DataFrame(
        {
            "Image Page": [f"/image/{i}/view/" for i in (1, 2, 3, 4)],
            "Image URL": ["http://x/1", "http://x/2", "http://x/3", pd.NA],
        }
    )

    def fake_upload(url: str, bucket: str, key: str) -> tuple[str, bool]:
        if url.endswith("/3"):
            raise RuntimeError("boom")
        return key, url.endswith("/1")

    with patch.object(dl, "download_image_to_s3", side_effect=fake_upload):
        result = dl.download_images(df, 1, bucket_name="b")
    assert result == ImageFailures(missing_url=1, upload_failed=2)


def test_download_images_counts_every_row_missing_when_no_url_is_set():
    dl = _image_source_downloader()
    df = pd.DataFrame(
        {"Image Page": [f"/image/{i}/view/" for i in (1, 2, 3)], "Image URL": [pd.NA] * 3}
    )
    assert dl.download_images(df, 1, bucket_name="b") == ImageFailures(
        missing_url=3, upload_failed=0
    )


def test_download_source_sums_image_failures_across_batches():
    dl = _image_source_downloader()
    images = pd.DataFrame({"Image Page": ["/image/1/view/", "/image/2/view/"]})

    def fake_upload(url: str, bucket: str, key: str) -> tuple[str, bool]:
        return key, not url.endswith("/1")  # only the first batch's image fails

    with (
        patch.object(dl, "check_permissions", return_value=True),
        patch.object(dl, "_get_source_overview_html", return_value=_fixture("source_ok.html")),
        patch.object(dl, "download_metadata", return_value=(True, 2)),
        patch.object(dl, "download_labelset", return_value=True),
        patch.object(dl, "download_annotations", return_value=True),
        patch.object(dl, "get_images", return_value=(images, True)),
        patch.object(
            dl,
            "get_image_urls",
            side_effect=lambda pages: ["http://x/" + p.split("/")[2] for p in pages],
        ),
        patch.object(s3_io, "upload_csv_body"),
        patch.object(dl, "download_image_to_s3", side_effect=fake_upload),
        patch.object(downloader_mod, "half_open_batches", return_value=[(0, 1), (1, 2)]),
    ):
        res = dl.download_source(1, bucket_name="b", s3_prefix="p")
    assert res.errors == ["images_upload_failed"]
    assert res.image_failures == ImageFailures(missing_url=0, upload_failed=1)


def test_get_image_urls_serial_path_keeps_order_around_a_failure():
    dl = CoralNetDownloader("u", "p", max_workers=1)

    def resolve(rel: str) -> str:
        if rel == "/image/2/view/":
            raise ValueError("private")
        return "http://x" + rel

    with patch.object(dl, "_resolve_one_image_url", side_effect=resolve):
        urls = dl.get_image_urls(["/image/1/view/", "/image/2/view/", "/image/3/view/"])
    assert urls == ["http://x/image/1/view/", None, "http://x/image/3/view/"]


def test_download_source_reports_missing_url_when_single_image_lookup_raises():
    dl = _image_source_downloader()
    images = pd.DataFrame({"Image Page": ["/image/1/view/"]})
    with (
        patch.object(dl, "check_permissions", return_value=True),
        patch.object(dl, "_get_source_overview_html", return_value=_fixture("source_ok.html")),
        patch.object(dl, "download_metadata", return_value=(True, 1)),
        patch.object(dl, "download_labelset", return_value=True),
        patch.object(dl, "download_annotations", return_value=True),
        patch.object(dl, "get_images", return_value=(images, True)),
        patch.object(dl, "_resolve_one_image_url", side_effect=ValueError("private")),
        patch.object(s3_io, "upload_csv_body"),
        patch.object(dl, "download_image_to_s3", side_effect=lambda u, b, k: (k, True)),
    ):
        res = dl.download_source(1, bucket_name="b", s3_prefix="p")
    assert res.errors == ["images_missing_url"]
    assert res.image_failures == ImageFailures(missing_url=1, upload_failed=0)


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
    """Build a fake browse page with ``n_thumbs`` thumbs whose image IDs start at
    ``start_id``.

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

    Yields the ``post`` mock so tests can introspect ``call_args_list`` to assert the
    export_prep POST body. ``get`` and ``post`` accept either a list (consumed in order
    as ``side_effect``) or a callable (for URL-based dispatch in the parallel test).
    """
    with (
        patch.object(dl.session, "get", side_effect=get),
        patch.object(dl.session, "post", side_effect=post) as post_mock,
        patch("coralnet_scraper.downloader.time.sleep"),
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


_BULK_BROWSE_HTML = (
    '<form id="export-annotations-prep-form">'
    '<input type="hidden" name="csrfmiddlewaretoken" value="tok"/></form>'
)
_BULK_KEY = ("b", "p/s1/annotations.csv")


def _run_bulk_download(
    fake_s3, *, browse_html: str, serve_text: str = ""
) -> tuple[bool, MagicMock]:
    """Run the bulk annotations path on a small source; return (ok, export_prep POST mock)."""
    dl = CoralNetDownloader("u", "p")
    dl.logged_in = True
    dl.s3 = fake_s3
    browse_resp = _mock_response(text=browse_html)
    post_resp = _mock_response(json_data={"session_data_timestamp": "t1"})
    serve_resp = _mock_response(text=serve_text)
    with (
        patch.object(dl.session, "get", side_effect=[browse_resp, serve_resp]),
        patch.object(dl.session, "post", return_value=post_resp) as post_mock,
        patch("coralnet_scraper.downloader.time.sleep"),
    ):
        ok = dl.download_annotations(1, "b", "p", total_images_hint=_PAGINATED_EXPORT_THRESHOLD)
    return ok, post_mock


def test_download_annotations_bulk_for_small_source(fake_s3):
    """Sources at or below threshold must use the existing bulk path."""
    ok, _ = _run_bulk_download(
        fake_s3,
        browse_html=_BULK_BROWSE_HTML,
        serve_text="Name,Row,Column,Label ID\na.jpg,1,2,3\nb.jpg,4,5,6\n",
    )

    assert ok is True
    uploaded = pd.read_csv(io.BytesIO(fake_s3.objects[_BULK_KEY]))
    assert list(uploaded.columns) == ["Name", "Row", "Column", "Label ID"]
    assert uploaded.to_numpy().tolist() == [["a.jpg", 1, 2, 3], ["b.jpg", 4, 5, 6]]


def test_download_annotations_bulk_without_export_form_uploads_nothing(fake_s3):
    ok, post_mock = _run_bulk_download(fake_s3, browse_html="<html><body>login</body></html>")

    assert ok is False
    assert _BULK_KEY not in fake_s3.objects
    post_mock.assert_not_called()


def test_download_annotations_bulk_html_export_uploads_nothing(fake_s3):
    ok, _ = _run_bulk_download(
        fake_s3,
        browse_html=_BULK_BROWSE_HTML,
        serve_text="<html><body>Error</body></html>",
    )

    assert ok is False
    assert _BULK_KEY not in fake_s3.objects


def test_download_annotations_bulk_empty_export_uploads_nothing(fake_s3):
    ok, _ = _run_bulk_download(fake_s3, browse_html=_BULK_BROWSE_HTML, serve_text="")

    assert ok is False
    assert _BULK_KEY not in fake_s3.objects


def test_paginated_export_skips_when_existing_annotations_are_complete(
    paginated_downloader, fake_s3
):
    """No pages are fetched when existing annotations.csv already covers all website
    images.
    """
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
    """When annotations.csv already exists on S3, new rows are merged and
    deduplicated.
    """
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


_EXISTING_KEY = s3_io.object_key_annotation_csv("p", 1)


def test_paginated_export_keeps_existing_csv_when_page_export_fails_with_4xx(
    paginated_downloader, fake_s3
):
    dl = paginated_downloader
    existing = _ann_csv([("old.jpg", 1, 1, 99)]).encode()
    fake_s3.put_object(Bucket="b", Key=_EXISTING_KEY, Body=existing)
    page_html = _browse_page("csrf1", n_thumbs=2)
    forbidden = requests.HTTPError("403", response=MagicMock(status_code=403))

    with _mock_paginated_session(
        dl,
        get=[_mock_response(text=page_html), _mock_response(text=page_html)],
        post=[forbidden],
    ) as post_mock:
        ok = dl.download_annotations_paginated(1, "b", "p", page_workers=1)

    assert ok is False
    assert post_mock.call_count == 1  # a 4xx is not retried
    assert fake_s3.objects[("b", _EXISTING_KEY)] == existing


def test_paginated_export_returns_false_when_existing_read_is_denied(paginated_downloader, fake_s3):
    denied = ClientError({"Error": {"Code": "AccessDenied", "Message": "no"}}, "GetObject")
    with (
        patch.object(fake_s3, "get_object", side_effect=denied),
        patch.object(paginated_downloader, "_walk_browse_pages") as walk,
    ):
        ok = paginated_downloader.download_annotations_paginated(1, "b", "p")

    assert ok is False
    walk.assert_not_called()
    assert fake_s3.objects == {}


def test_paginated_export_runs_when_existing_csv_covers_fewer_images_than_website(
    paginated_downloader, fake_s3
):
    dl = paginated_downloader
    existing = _ann_csv([("old.jpg", 1, 1, 99), ("old.jpg", 2, 2, 98)])
    fake_s3.put_object(Bucket="b", Key=_EXISTING_KEY, Body=existing.encode())
    page_html = _browse_page("csrf1", n_thumbs=1)
    new_ann = _ann_csv([("new.jpg", 9, 9, 300), ("old.jpg", 1, 1, 99)])

    with _mock_paginated_session(
        dl,
        get=[
            _mock_response(text=page_html),
            _mock_response(text=page_html),
            _mock_response(text=new_ann),
        ],
        post=[_mock_response(json_data={"session_data_timestamp": "t1"})],
    ) as post_mock:
        ok = dl.download_annotations_paginated(1, "b", "p", total_images_website=2, page_workers=1)

    assert ok is True
    assert post_mock.call_count == 1
    df = pd.read_csv(io.BytesIO(fake_s3.objects[("b", _EXISTING_KEY)]))
    assert sorted(zip(df["Name"], df["Row"], df["Column"], strict=True)) == [
        ("new.jpg", 9, 9),
        ("old.jpg", 1, 1),
        ("old.jpg", 2, 2),
    ]


# ---------------------------------------------------------------------------
# get_images pagination tests (offline)
# ---------------------------------------------------------------------------


def test_get_images_multi_page_uses_urljoin():
    """Second-page URL must be resolved via urljoin, not f-string concatenation.

    Regression test for the bug where ``f"{base_url}/{next_page}"`` produced paths like
    ``.../images/?page=2`` instead of ``.../images?page=2``. Also pins the resulting
    DataFrame shape (``Name`` and ``Image Page`` columns).
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


# ---------------------------------------------------------------------------
# Paginated export retry / checkpoint / parallelism tests
# ---------------------------------------------------------------------------


def test_paginated_export_retries_on_page_timeout(paginated_downloader, fake_s3):
    """A ReadTimeout on attempt 1 of page 2's export_prep is retried and succeeds.

    Phase 2 ``do_round_trip`` re-fetches the browse page for a fresh CSRF before each
    attempt, so the retry triggers a second browse GET for page 2.
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

    Once at the checkpoint after page 2, and again as the final upload after page 3. The
    final CSV must contain rows from all three pages.
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

    Uses URL-based mock dispatch so the test is not dependent on call order across
    worker threads.
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


# ---------------------------------------------------------------------------
# Retry tests
# ---------------------------------------------------------------------------


def test_get_images_retries_on_page1_timeout():
    """A ReadTimeout on page 1 is retried; scrape succeeds after the retry."""
    dl = CoralNetDownloader("u", "p")
    dl.logged_in = True

    call_count = [0]

    def flaky_get_images_on_page(url: str):
        call_count[0] += 1
        if call_count[0] == 1:
            raise requests.ReadTimeout("simulated timeout")
        return {"img1.jpg": "/image/1/view/", "img2.jpg": "/image/2/view/"}, None

    with (
        patch.object(dl, "get_images_on_page", side_effect=flaky_get_images_on_page),
        patch("coralnet_scraper.downloader.time.sleep"),
    ):
        df, ok = dl.get_images(1)

    assert ok is True
    assert call_count[0] == 2  # timed out once, then succeeded
    assert df is not None and len(df) == 2


def test_get_images_page1_gives_up_after_max_retries():
    """After _PAGE_RETRIES failed attempts on page 1, get_images returns (None,
    False).
    """
    dl = CoralNetDownloader("u", "p")
    dl.logged_in = True

    def always_timeout(url: str):
        raise requests.ReadTimeout("always")

    with (
        patch.object(dl, "get_images_on_page", side_effect=always_timeout),
        patch("coralnet_scraper.downloader.time.sleep"),
    ):
        df, ok = dl.get_images(1)

    assert ok is False
    assert df is None


def _http_error(status: int) -> requests.HTTPError:
    resp = requests.Response()
    resp.status_code = status
    return requests.HTTPError(response=resp)


def test_is_transient_covers_timeouts_drops_and_5xx():
    """Transient set spans connect/read timeouts, dropped connections, and 5xx."""
    assert _is_transient(requests.ReadTimeout()) is True
    assert _is_transient(requests.ConnectTimeout()) is True  # sibling of ReadTimeout
    assert _is_transient(requests.ConnectionError()) is True
    assert _is_transient(requests.exceptions.ChunkedEncodingError()) is True
    assert _is_transient(_http_error(503)) is True
    # 4xx won't fix itself on retry.
    assert _is_transient(_http_error(404)) is False


def test_get_images_retries_on_5xx_then_succeeds():
    """A 503 on a browse page is retried, not fatal."""
    dl = CoralNetDownloader("u", "p")
    dl.logged_in = True
    call_count = [0]

    def flaky(url: str):
        call_count[0] += 1
        if call_count[0] == 1:
            raise _http_error(503)
        return {"img1.jpg": "/image/1/view/"}, None

    with (
        patch.object(dl, "get_images_on_page", side_effect=flaky),
        patch("coralnet_scraper.downloader.time.sleep"),
    ):
        df, ok = dl.get_images(1)

    assert ok is True
    assert call_count[0] == 2
    assert df is not None and len(df) == 1


def test_get_images_4xx_is_not_retried():
    """A 404 propagates immediately (no retry) and fails the source."""
    dl = CoralNetDownloader("u", "p")
    dl.logged_in = True
    call_count = [0]

    def always_404(url: str):
        call_count[0] += 1
        raise _http_error(404)

    with (
        patch.object(dl, "get_images_on_page", side_effect=always_404),
        patch("coralnet_scraper.downloader.time.sleep"),
    ):
        df, ok = dl.get_images(1)

    assert ok is False
    assert df is None
    assert call_count[0] == 1  # not retried


def test_get_images_sequential_retries_on_inner_page_timeout():
    """A ReadTimeout on page 2 (sequential mode) is retried transparently."""
    dl = CoralNetDownloader("u", "p")
    dl.logged_in = True

    call_count = [0]

    def flaky_get_images_on_page(url: str):
        call_count[0] += 1
        if call_count[0] == 2:
            raise requests.ReadTimeout("page 2 timeout")
        if call_count[0] in (1, 3):
            return {f"img{call_count[0]}.jpg": f"/image/{call_count[0]}/view/"}, (
                "?page=2" if call_count[0] == 1 else None
            )
        return {f"img{call_count[0]}.jpg": f"/image/{call_count[0]}/view/"}, None

    with (
        patch.object(dl, "get_images_on_page", side_effect=flaky_get_images_on_page),
        patch("coralnet_scraper.downloader.time.sleep"),
    ):
        df, ok = dl.get_images(1)  # no hint → sequential

    assert ok is True
    assert df is not None and len(df) == 2  # page 1 + page 2 (after retry)


# ---------------------------------------------------------------------------
# Login retry + jittered backoff
# ---------------------------------------------------------------------------


def _login_get_html() -> str:
    return '<input type="hidden" name="csrfmiddlewaretoken" value="logintok"/>'


def test_login_retries_transient_timeout_then_succeeds():
    """A ReadTimeout on the login GET is retried; login then succeeds.

    The original bug: a single login timeout killed the whole job before any
    source was touched, because _perform_login's GET was not retry-wrapped.
    """
    dl = CoralNetDownloader("u", "p")
    get_calls = [0]

    def flaky_get(url, timeout=None, **_):
        get_calls[0] += 1
        if get_calls[0] == 1:
            raise requests.ReadTimeout("login page slow")
        return _mock_response(text=_login_get_html())

    post_resp = _mock_response(text="<html>Sign out</html>")

    with (
        patch.object(dl.session, "get", side_effect=flaky_get),
        patch.object(dl.session, "post", return_value=post_resp),
        patch("coralnet_scraper.downloader.time.sleep"),
    ):
        ok = dl.login()

    assert ok is True
    assert dl.logged_in is True
    assert get_calls[0] == 2  # timed out once, retried, then succeeded


def test_login_gives_up_after_repeated_timeouts():
    """If the login GET keeps timing out, login() returns False after retries (so the
    shard exits cleanly instead of hammering CoralNet per-source).
    """
    dl = CoralNetDownloader("u", "p")
    get_calls = [0]

    def always_timeout(url, timeout=None, **_):
        get_calls[0] += 1
        raise requests.ReadTimeout("coralnet down")

    with (
        patch.object(dl.session, "get", side_effect=always_timeout),
        patch("coralnet_scraper.downloader.time.sleep"),
    ):
        ok = dl.login()

    assert ok is False
    assert dl.logged_in is False
    assert get_calls[0] == 3  # exhausted the retry budget


def test_login_does_not_retry_invalid_credentials():
    """A genuine auth failure (not a transient error) fails fast — no retry."""
    dl = CoralNetDownloader("u", "p")
    post_calls = [0]
    bad_post = _mock_response(text="<html>Please enter a correct username and password</html>")
    bad_post.url = dl.LOGIN_URL  # stayed on the login page => failed auth

    def fake_post(*_a, **_k):
        post_calls[0] += 1
        return bad_post

    with (
        patch.object(dl.session, "get", return_value=_mock_response(text=_login_get_html())),
        patch.object(dl.session, "post", side_effect=fake_post),
        patch("coralnet_scraper.downloader.time.sleep"),
    ):
        ok = dl.login()

    assert ok is False
    assert post_calls[0] == 1  # not retried — invalid creds won't fix themselves


def test_with_retry_backoff_is_jittered():
    """Backoff is randomized within [0.5x, 1.5x] of the exponential base so that
    concurrent shards desync instead of retrying in lockstep (thundering herd).
    """
    slept: list[float] = []

    def always_timeout():
        raise requests.ReadTimeout("x")

    with (
        patch.object(downloader_mod.time, "sleep", side_effect=lambda s: slept.append(s)),
        patch.object(downloader_mod.random, "uniform", side_effect=lambda lo, _hi: lo) as uni,
        pytest.raises(requests.ReadTimeout),
    ):
        downloader_mod._with_retry(always_timeout, source_id=1, page_num=1, kind="test")

    # 3 attempts => 2 backoff sleeps; exponential bases are 30 and 60, jittered
    # over [0.5*base, 1.5*base].
    assert [c.args for c in uni.call_args_list] == [(0.5 * 30, 1.5 * 30), (0.5 * 60, 1.5 * 60)]
    assert slept == [0.5 * 30, 0.5 * 60]  # stub returns the low bound


@pytest.mark.parametrize("workers", [0, 2.5, True, "4"])
def test_constructor_rejects_invalid_max_workers(workers):
    with pytest.raises(ValueError, match="workers"):
        CoralNetDownloader("u", "p", max_workers=workers)


def _failing_step_downloader(**overrides) -> tuple[CoralNetDownloader, dict[str, MagicMock]]:
    """A downloader whose steps succeed except those overridden; returns it with its mocks."""
    dl = CoralNetDownloader("u", "p", max_workers=2)
    dl.logged_in = True
    dl.s3 = MagicMock()
    images = pd.DataFrame({"Image Page": ["/image/1/view/"]})
    defaults = {
        "check_permissions": True,
        "download_metadata": (True, 2),
        "download_labelset": True,
        "download_annotations": True,
        "get_images": (images, True),
    }
    mocks = {
        name: MagicMock(return_value=overrides.get(name, default))
        for name, default in defaults.items()
    }
    for name, mock in mocks.items():
        setattr(dl, name, mock)
    dl._get_source_overview_html = MagicMock(  # type: ignore[method-assign]
        return_value=_fixture("source_ok.html"),
        **({"side_effect": overrides["overview"]} if "overview" in overrides else {}),
    )
    dl.get_image_urls = MagicMock(return_value=["http://x/1"])  # type: ignore[method-assign]
    dl.download_image_to_s3 = MagicMock(  # type: ignore[method-assign]
        side_effect=lambda u, b, k: (k, True)
    )
    return dl, mocks


def test_download_source_returns_login_failed_when_login_fails():
    dl, mocks = _failing_step_downloader()
    dl.logged_in = False
    with patch.object(dl, "login", return_value=False):
        res = dl.download_source(1, bucket_name="b", s3_prefix="p")

    assert res.ok is False
    assert res.errors == ["login failed"]
    mocks["check_permissions"].assert_not_called()


def test_download_source_returns_cannot_access_when_permissions_denied():
    dl, mocks = _failing_step_downloader(check_permissions=False)
    res = dl.download_source(7, bucket_name="b", s3_prefix="p")

    assert res.ok is False
    assert res.errors == ["cannot access source 7"]
    mocks["download_metadata"].assert_not_called()


def test_download_source_reports_overview_fetch_failure():
    dl, mocks = _failing_step_downloader(overview=requests.ConnectionError("down"))
    res = dl.download_source(1, bucket_name="b", s3_prefix="p")

    assert res.ok is False
    assert len(res.errors) == 1
    assert res.errors[0].startswith("overview fetch failed:")
    mocks["download_metadata"].assert_not_called()


@pytest.mark.parametrize(
    ("step", "failure", "code"),
    [
        ("download_metadata", (False, 2), "metadata_download_failed"),
        ("download_labelset", False, "labelset_failed"),
        ("get_images", (None, False), "image_list_failed"),
    ],
)
def test_download_source_records_a_failed_step_and_still_runs_the_rest(step, failure, code):
    dl, mocks = _failing_step_downloader(**{step: failure})
    res = dl.download_source(1, bucket_name="b", s3_prefix="p")

    assert res.ok is False
    assert code in res.errors
    mocks["download_annotations"].assert_called_once()
    mocks["get_images"].assert_called_once()


def test_download_source_stops_after_annotations_fail():
    dl, mocks = _failing_step_downloader(download_annotations=False)
    res = dl.download_source(1, bucket_name="b", s3_prefix="p")

    assert res.ok is False
    assert res.errors[-1] == "annotations_failed"
    mocks["get_images"].assert_not_called()
