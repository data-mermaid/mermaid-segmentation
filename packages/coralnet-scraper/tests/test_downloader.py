"""Offline tests for CoralNetDownloader."""

from __future__ import annotations

import io
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pandas as pd
import pytest
import requests
from botocore.exceptions import ClientError
from bs4 import BeautifulSoup
from conftest import Reply, ScriptedServer

from coralnet_scraper import downloader as downloader_mod
from coralnet_scraper import s3_io
from coralnet_scraper.downloader import (
    _PAGINATED_EXPORT_THRESHOLD,
    CoralNetDownloader,
    _extract_page_image_ids,
    _PageWorkItem,
    export_prep_timeout_seconds,
    export_serve_timeout_seconds,
    half_open_batches,
)
from coralnet_scraper.models import DownloadResult, ImageFailures

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


def _download_source_with_images(dl: CoralNetDownloader) -> tuple[DownloadResult, MagicMock]:
    """Run ``download_source`` on a two-image source; return the result and the upload mock."""
    images = pd.DataFrame(
        {"Name": ["a.jpg", "b.jpg"], "Image Page": ["/image/1/view/", "/image/2/view/"]}
    )
    with (
        patch.object(dl, "check_permissions", return_value=True),
        patch.object(dl, "_get_source_overview_html", return_value=_fixture("source_ok.html")),
        patch.object(dl, "download_metadata", return_value=(True, 2)),
        patch.object(dl, "download_labelset", return_value=True),
        patch.object(dl, "download_annotations", return_value=True),
        patch.object(dl, "get_images", return_value=(images, True)),
        patch.object(dl, "get_image_urls", return_value=["http://x/1", "http://x/2"]),
        patch.object(dl, "download_image_to_s3", side_effect=lambda u, b, k: (k, True)) as upload,
    ):
        return dl.download_source(1, bucket_name="b", s3_prefix="p"), upload


def test_download_source_ok_when_every_image_uploads(fake_s3):
    dl = _image_source_downloader()
    dl.s3 = fake_s3

    res, upload = _download_source_with_images(dl)

    assert res.ok is True
    assert res.errors == []
    assert res.image_failures == ImageFailures()
    # The image list goes up before any URL is resolved, so its Image URL column is empty.
    assert fake_s3.objects[("b", "p/s1/image_list.csv")].decode().splitlines() == [
        "Name,Image Page,Image URL",
        "a.jpg,/image/1/view/,",
        "b.jpg,/image/2/view/,",
    ]
    assert sorted(c.args for c in upload.call_args_list) == [
        ("http://x/1", "b", "p/s1/images/1.jpg"),
        ("http://x/2", "b", "p/s1/images/2.jpg"),
    ]


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


def test_thread_sessions_are_not_shared_between_downloaders():
    """Each downloader's per-thread session carries its own downloader's cookies."""
    first = CoralNetDownloader("u", "p")
    second = CoralNetDownloader("u", "p")
    first.session.cookies.set("sessionid", "first-id")
    second.session.cookies.set("sessionid", "second-id")

    first_session = first._get_thread_session()
    second_session = second._get_thread_session()

    assert first_session.cookies.get("sessionid") == "first-id"
    assert second_session.cookies.get("sessionid") == "second-id"
    assert first_session is not second_session
    assert first._get_thread_session() is first_session


def test_export_sessions_are_not_shared_between_downloaders():
    """Each downloader's export session comes from its own login, not another's cache."""
    first = CoralNetDownloader("u", "p")
    second = CoralNetDownloader("u", "p")
    first_sentinel = requests.Session()
    second_sentinel = requests.Session()
    first._new_logged_in_session = lambda: first_sentinel  # type: ignore[method-assign]
    second._new_logged_in_session = lambda: second_sentinel  # type: ignore[method-assign]

    assert first._get_export_session() is first_sentinel
    assert second._get_export_session() is second_sentinel


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


@dataclass
class _BulkRun:
    ok: bool
    get: MagicMock
    post: MagicMock
    sleep: MagicMock

    @property
    def prep_timeouts(self) -> list[int]:
        return [c.kwargs["timeout"] for c in self.post.call_args_list]


def _overview_html(total: int, confirmed: int | None = None) -> str:
    """A source overview page whose Image Status table reports ``total`` images."""
    confirmed_row = (
        f'<tr><td>Confirmed:</td><td><a href="#">{confirmed}</a></td></tr>'
        if confirmed is not None
        else ""
    )
    return (
        "<html><body><h4>Image Status</h4>"
        '<table class="detail_box_table">'
        '<tr><td>Unclassified:</td><td><a href="#">1</a></td></tr>'
        f"{confirmed_row}"
        f'<tr><td>Total images:</td><td><a href="/browse/images/">{total}</a></td></tr>'
        "</table></body></html>"
    )


def _run_bulk_download(
    fake_s3,
    *,
    browse_html: str,
    serve_text: str = "",
    export_prep: list | None = None,
    hint: int | None = _PAGINATED_EXPORT_THRESHOLD,
    overview_html: str | None = None,
) -> _BulkRun:
    """Run the bulk annotations path on a small source.

    ``export_prep`` lists the export_prep POST outcomes in order (responses or exceptions);
    ``overview_html`` is served first, for a run that must fetch the overview itself.
    """
    dl = CoralNetDownloader("u", "p")
    dl.logged_in = True
    dl.s3 = fake_s3
    gets = [_mock_response(text=browse_html), _mock_response(text=serve_text)]
    if overview_html is not None:
        gets.insert(0, _mock_response(text=overview_html))
    if export_prep is None:
        export_prep = [_mock_response(json_data={"session_data_timestamp": "t1"})]
    with (
        patch.object(dl.session, "get", side_effect=gets) as get_mock,
        patch.object(dl.session, "post", side_effect=export_prep) as post_mock,
        patch("coralnet_scraper.downloader.time.sleep") as sleep_mock,
    ):
        ok = dl.download_annotations(1, "b", "p", total_images_hint=hint)
    return _BulkRun(ok=ok, get=get_mock, post=post_mock, sleep=sleep_mock)


_SERVE_CSV = "Name,Row,Column,Label ID\na.jpg,1,2,3\n"
_BROWSE_URL = "https://coralnet.ucsd.edu/source/1/browse/images/"


def test_download_annotations_retries_export_prep_with_growing_timeout(fake_s3):
    export_prep = [
        requests.exceptions.ReadTimeout("t"),
        requests.exceptions.ReadTimeout("t"),
        _mock_response(json_data={"session_data_timestamp": "t1"}),
    ]
    run = _run_bulk_download(
        fake_s3,
        browse_html=_BULK_BROWSE_HTML,
        serve_text=_SERVE_CSV,
        export_prep=export_prep,
        hint=4000,
    )

    assert run.ok is True
    assert run.prep_timeouts == [320, 620, 920]  # 120 + 4000 // 20, then +300 per attempt
    assert run.sleep.call_args_list == [call(60), call(120)]
    assert _BULK_KEY in fake_s3.objects


def test_download_annotations_fails_after_five_export_prep_timeouts(fake_s3):
    run = _run_bulk_download(
        fake_s3,
        browse_html=_BULK_BROWSE_HTML,
        serve_text=_SERVE_CSV,
        export_prep=[requests.exceptions.ReadTimeout("t")] * 5,
        hint=4000,
    )

    assert run.ok is False
    assert run.prep_timeouts == [320, 620, 920, 1220, 1520]
    assert run.sleep.call_args_list == [call(60), call(120), call(180), call(240)]
    assert _BULK_KEY not in fake_s3.objects


@pytest.mark.parametrize(
    ("hint", "overview_html", "prep_timeout", "serve_timeout"),
    [
        (4000, None, 320, 140),  # 120 + 4000 // 20 and 60 + 4000 // 50
        (None, _overview_html(4000), 320, 140),
        (-1, _overview_html(4000), 320, 140),
        (0, None, 300, 120),  # zero images: no overview fetch, minimum timeouts
    ],
)
def test_download_annotations_sizes_timeouts_from_hint_or_overview(
    fake_s3, hint, overview_html, prep_timeout, serve_timeout
):
    run = _run_bulk_download(
        fake_s3,
        browse_html=_BULK_BROWSE_HTML,
        serve_text=_SERVE_CSV,
        hint=hint,
        overview_html=overview_html,
    )

    assert run.ok is True
    first_get = "https://coralnet.ucsd.edu/source/1/" if overview_html else _BROWSE_URL
    assert run.get.call_args_list[0].args == (first_get,)
    assert run.prep_timeouts == [prep_timeout]
    serve_get = run.get.call_args_list[-1]
    assert serve_get.args == (
        "https://coralnet.ucsd.edu/source/1/export/serve/?session_data_timestamp=t1",
    )
    assert serve_get.kwargs["timeout"] == serve_timeout


def test_download_annotations_bulk_for_small_source(fake_s3):
    """Sources at or below threshold must use the existing bulk path."""
    run = _run_bulk_download(
        fake_s3,
        browse_html=_BULK_BROWSE_HTML,
        serve_text="Name,Row,Column,Label ID\na.jpg,1,2,3\nb.jpg,4,5,6\n",
    )

    assert run.ok is True
    uploaded = pd.read_csv(io.BytesIO(fake_s3.objects[_BULK_KEY]))
    assert list(uploaded.columns) == ["Name", "Row", "Column", "Label ID"]
    assert uploaded.to_numpy().tolist() == [["a.jpg", 1, 2, 3], ["b.jpg", 4, 5, 6]]


def test_download_annotations_bulk_without_export_form_uploads_nothing(fake_s3):
    run = _run_bulk_download(fake_s3, browse_html="<html><body>login</body></html>")

    assert run.ok is False
    assert _BULK_KEY not in fake_s3.objects
    run.post.assert_not_called()


def test_download_annotations_bulk_html_export_uploads_nothing(fake_s3):
    run = _run_bulk_download(
        fake_s3,
        browse_html=_BULK_BROWSE_HTML,
        serve_text="<html><body>Error</body></html>",
    )

    assert run.ok is False
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
    assert df["Name"].tolist() == ["a.jpg", "b.jpg", "c.jpg", "d.jpg", "e.jpg"]

    expected_id_lists = {
        "csrf1": "_".join(str(i) for i in range(1000, 1020)),
        "csrf2": "_".join(str(i) for i in range(2000, 2020)),
        "csrf3": "_".join(str(i) for i in range(3000, 3010)),
    }
    assert mock_post.call_count == 3
    for post_call in mock_post.call_args_list:
        form = post_call.kwargs["data"]
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


def test_get_images_follows_next_page_relative_to_the_trailing_slash_url(
    http_server: ScriptedServer,
):
    dl = CoralNetDownloader("u", "p")
    dl.logged_in = True
    dl.CORALNET_URL = http_server.url
    first = "/source/7/browse/images/"
    http_server.script(
        first, Reply(200, _browse_page("c", n_thumbs=1, next_href="?page=2").encode())
    )
    http_server.script(f"{first}?page=2", Reply(200, _browse_page("c", n_thumbs=1).encode()))

    _, ok = dl.get_images(7)

    assert ok is True
    assert [r.path for r in http_server.received] == [first, f"{first}?page=2"]


# ---------------------------------------------------------------------------
# Paginated export retry / checkpoint / parallelism tests
# ---------------------------------------------------------------------------


def _page_script(page_htmls: list[str], ann_csvs: list[str]) -> tuple[list, list]:
    """GET and POST responses for a sequential (one-worker) paginated export.

    The GET sequence is the phase-1 walk over every page, then per page a browse refresh and
    the serve response; the POST sequence is one export_prep response per page.
    """
    gets = [_mock_response(text=html) for html in page_htmls]
    for html, ann in zip(page_htmls, ann_csvs, strict=True):
        gets += [_mock_response(text=html), _mock_response(text=ann)]
    posts = [
        _mock_response(json_data={"session_data_timestamp": f"t{i}"})
        for i in range(1, len(page_htmls) + 1)
    ]
    return gets, posts


def test_paginated_export_checkpoints_accumulate_earlier_batches_and_existing_rows(
    paginated_downloader, fake_s3
):
    """Four pages with a checkpoint every two upload the CSV three times (two checkpoints
    plus the final upload). Each upload holds the existing row and every page finished so far.
    """
    dl = paginated_downloader
    key = s3_io.object_key_annotation_csv("p", 1)
    fake_s3.put_object(Bucket="b", Key=key, Body=_ann_csv([("old.jpg", 9, 9, 99)]).encode())
    page_htmls = [
        _browse_page(
            f"csrf{i}", n_thumbs=1, next_href=f"?page={i + 1}" if i < 4 else None, start_id=i * 100
        )
        for i in range(1, 5)
    ]
    ann_csvs = [_ann_csv([(f"{n}.jpg", i, i, i * 10)]) for i, n in enumerate("abcd", 1)]
    gets, posts = _page_script(page_htmls, ann_csvs)

    with (
        _mock_paginated_session(dl, get=gets, post=posts),
        patch.object(fake_s3, "put_object", wraps=fake_s3.put_object) as put_spy,
    ):
        ok = dl.download_annotations_paginated(
            1, "b", "p", page_workers=1, checkpoint_every_pages=2
        )

    assert ok is True
    uploads = [pd.read_csv(io.StringIO(c.kwargs["Body"])) for c in put_spy.call_args_list]
    assert [u["Name"].tolist() for u in uploads] == [
        ["old.jpg", "a.jpg", "b.jpg"],
        ["old.jpg", "a.jpg", "b.jpg", "c.jpg", "d.jpg"],
        ["old.jpg", "a.jpg", "b.jpg", "c.jpg", "d.jpg"],
    ]
    assert uploads[1]["Label ID"].tolist() == [99, 10, 20, 30, 40]


def test_paginated_export_with_parallel_workers_keeps_page_order_when_pages_finish_out_of_order(
    paginated_downloader, fake_s3
):
    """Page 1's serve response is held until page 3's has been served, so the pages complete
    as 3 then 1 (or 2 in between) yet the uploaded rows follow page order.
    """
    dl = paginated_downloader
    base = "https://coralnet.ucsd.edu/source/1/browse/images/"
    pages = {
        base: _browse_page("csrf1", n_thumbs=1, next_href="?page=2", start_id=100),
        f"{base}?page=2": _browse_page("csrf2", n_thumbs=1, next_href="?page=3", start_id=200),
        f"{base}?page=3": _browse_page("csrf3", n_thumbs=1, start_id=300),
    }
    annotations = {
        1: _ann_csv([("a.jpg", 1, 1, 10)]),
        2: _ann_csv([("b.jpg", 2, 2, 20)]),
        3: _ann_csv([("c.jpg", 3, 3, 30)]),
    }
    page3_served = threading.Event()

    def get(url: str, timeout: int | None = None, **_: object) -> MagicMock:
        if "/export/serve/" not in url:
            return _mock_response(text=pages[url])
        page = int(url.rsplit("=t", 1)[1])
        if page == 1:
            assert page3_served.wait(timeout=10), "page 3 never finished"
        if page == 3:
            page3_served.set()
        return _mock_response(text=annotations[page])

    def post(url: str, *, data: dict, **_: object) -> MagicMock:
        page = data["csrfmiddlewaretoken"].removeprefix("csrf")
        return _mock_response(json_data={"session_data_timestamp": f"t{page}"})

    with _mock_paginated_session(dl, get=get, post=post):
        ok = dl.download_annotations_paginated(1, "b", "p", page_workers=3)

    assert ok is True
    key = s3_io.object_key_annotation_csv("p", 1)
    df = pd.read_csv(io.BytesIO(fake_s3.objects[("b", key)]))
    assert df["Name"].tolist() == ["a.jpg", "b.jpg", "c.jpg"]


def test_paginated_export_runs_export_workers_page_round_trips_at_once(fake_s3):
    dl = CoralNetDownloader("u", "p", export_workers=2)
    dl.logged_in = True
    dl.s3 = fake_s3
    dl._new_logged_in_session = lambda: dl.session  # type: ignore[method-assign]
    base = "https://coralnet.ucsd.edu/source/1/browse/images/"
    urls = [base] + [f"{base}?page={i}" for i in range(2, 7)]
    pages = {
        url: _browse_page(
            f"csrf{i}",
            n_thumbs=1,
            next_href=f"?page={i + 1}" if i < 6 else None,
            start_id=i * 100,
        )
        for i, url in enumerate(urls, 1)
    }
    lock = threading.Lock()
    in_flight = [0]
    peak = [0]

    def get(url: str, timeout: int | None = None, **_: object) -> MagicMock:
        if "/export/serve/" not in url:
            return _mock_response(text=pages[url])
        page = int(url.rsplit("=t", 1)[1])
        return _mock_response(text=_ann_csv([(f"{page}.jpg", page, page, page)]))

    def post(url: str, *, data: dict, **_: object) -> MagicMock:
        with lock:
            in_flight[0] += 1
            peak[0] = max(peak[0], in_flight[0])
        threading.Event().wait(0.05)  # hold the round trip open so workers overlap
        with lock:
            in_flight[0] -= 1
        page = data["csrfmiddlewaretoken"].removeprefix("csrf")
        return _mock_response(json_data={"session_data_timestamp": f"t{page}"})

    with _mock_paginated_session(dl, get=get, post=post):
        ok = dl.download_annotations_paginated(1, "b", "p")

    assert ok is True
    assert peak[0] == 2


def test_get_images_fails_the_walk_when_a_page_fails():
    dl = CoralNetDownloader("u", "p")
    dl.logged_in = True
    resp = requests.Response()
    resp.status_code = 404

    with patch.object(dl, "get_images_on_page", side_effect=requests.HTTPError(response=resp)):
        df, ok = dl.get_images(1)

    assert ok is False
    assert df is None


def _login_get_html() -> str:
    return '<input type="hidden" name="csrfmiddlewaretoken" value="logintok"/>'


def test_login_fails_on_invalid_credentials():
    dl = CoralNetDownloader("u", "p")
    bad_post = _mock_response(text="<html>Please enter a correct username and password</html>")
    bad_post.url = dl.LOGIN_URL  # stayed on the login page => failed auth

    with (
        patch.object(dl.session, "get", return_value=_mock_response(text=_login_get_html())),
        patch.object(dl.session, "post", return_value=bad_post),
    ):
        ok = dl.login()

    assert ok is False
    assert dl.logged_in is False


def test_constructor_rejects_invalid_max_workers():
    with pytest.raises(ValueError, match="workers"):
        CoralNetDownloader("u", "p", max_workers=0)


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


# ---------------------------------------------------------------------------
# Image upload, metadata, labelset, permissions, login
# ---------------------------------------------------------------------------


def _http_response(text: str = "", status: int = 200, url: str = "") -> requests.Response:
    """A real ``requests.Response`` so ``raise_for_status`` behaves as in production."""
    resp = requests.Response()
    resp.status_code = status
    resp._content = text.encode()
    resp.url = url
    return resp


def _downloader_on(fake_s3) -> CoralNetDownloader:
    dl = CoralNetDownloader("u", "p", max_workers=3)
    dl.logged_in = True
    dl.s3 = fake_s3
    return dl


def test_download_image_to_s3_skips_an_object_already_on_s3(fake_s3, http_server):
    dl = _downloader_on(fake_s3)
    fake_s3.put("b", "k.jpg", b"existing")

    result = dl.download_image_to_s3(f"{http_server.url}/1.jpg", "b", "k.jpg")

    assert result == ("k.jpg", True)
    assert http_server.received == []
    assert fake_s3.objects[("b", "k.jpg")] == b"existing"


def test_download_image_to_s3_retries_a_503_then_uploads_the_bytes(fake_s3, http_server):
    dl = _downloader_on(fake_s3)
    http_server.script("/1.jpg", Reply(503), Reply(200, b"JPEGBYTES"))

    result = dl.download_image_to_s3(f"{http_server.url}/1.jpg", "b", "k.jpg")

    assert result == ("k.jpg", True)
    assert fake_s3.objects[("b", "k.jpg")] == b"JPEGBYTES"


def test_download_image_to_s3_reports_failure_and_uploads_nothing(fake_s3, http_server):
    dl = _downloader_on(fake_s3)
    http_server.script("/1.jpg", Reply(404, b"<html>Not found</html>"))

    result = dl.download_image_to_s3(f"{http_server.url}/1.jpg", "b", "k.jpg")

    assert result == ("k.jpg", False)
    assert fake_s3.objects == {}


_CLASSIFIER_SCRIPT = """\
<script>
// Classifier overview
let classifierPlotData = [{'x': 1, 'y': 0.5, 'nimages': 100, 'traintime': 12, \
'date': '2020-01-01', 'pk': 77}, {'x': 2, 'y': 0.6, 'nimages': 200, 'traintime': 15, \
'date': '2020-02-01', 'pk': 88}];
</script>"""


def _uploaded_metadata(dl: CoralNetDownloader, fake_s3) -> pd.DataFrame:
    result = dl.download_metadata(
        9, "b", "p", overview_html=_overview_html(1234) + _CLASSIFIER_SCRIPT
    )
    assert result == (True, 1234)
    return pd.read_csv(io.BytesIO(fake_s3.objects[("b", "p/s9/metadata.csv")]))


def test_download_metadata_uploads_classifier_history_and_returns_total(fake_s3):
    df = _uploaded_metadata(_downloader_on(fake_s3), fake_s3)

    assert df.columns.tolist() == [
        "Classifier nbr",
        "Accuracy",
        "Trained on",
        "Date",
        "Traintime",
        "Global id",
    ]
    columns = ["Classifier nbr", "Accuracy", "Trained on", "Global id"]
    assert df[columns].to_numpy().tolist() == [[1, 0.5, 100, 77], [2, 0.6, 200, 88]]


def test_download_metadata_puts_the_date_and_train_time_in_their_own_columns(fake_s3):
    df = _uploaded_metadata(_downloader_on(fake_s3), fake_s3)

    assert df["Date"].tolist() == ["2020-01-01", "2020-02-01"]
    assert df["Traintime"].tolist() == [12, 15]


def test_download_metadata_without_classifier_data_returns_total_and_uploads_nothing(fake_s3):
    dl = _downloader_on(fake_s3)

    result = dl.download_metadata(9, "b", "p", overview_html=_overview_html(1234))

    assert result == (True, 1234)
    assert fake_s3.objects == {}


def test_download_metadata_reports_failure_when_the_upload_raises(fake_s3):
    dl = _downloader_on(fake_s3)
    denied = ClientError({"Error": {"Code": "AccessDenied", "Message": "no"}}, "PutObject")

    with patch.object(fake_s3, "put_object", side_effect=denied):
        result = dl.download_metadata(
            9, "b", "p", overview_html=_overview_html(1234) + _CLASSIFIER_SCRIPT
        )

    assert result == (False, 0)


_LABEL_TABLE_HTML = """\
<html><body>
<table id="label-table">
<tr><th>Name</th><th>Short Code</th></tr>
<tr><td><a href="/label/101/">Acropora</a></td><td>ACR</td></tr>
<tr><td><a href="/label/42/">Turf algae</a></td><td>TA</td></tr>
<tr><td>Not a label row</td><td>XX</td></tr>
</table>
</body></html>"""


def test_download_labelset_uploads_label_ids_names_and_short_codes(fake_s3):
    dl = _downloader_on(fake_s3)

    with patch.object(dl.session, "get", return_value=_http_response(_LABEL_TABLE_HTML)) as get:
        ok = dl.download_labelset(9, "b", "p")

    assert ok is True
    assert get.call_args.args == ("https://coralnet.ucsd.edu/source/9/labelset/",)
    assert fake_s3.objects[("b", "p/s9/labelset.csv")].decode().splitlines() == [
        "Label ID,Name,Short Code",
        "101,Acropora,ACR",
        "42,Turf algae,TA",
    ]


@pytest.mark.parametrize(
    ("html", "status", "expected_ok"),
    [
        ("<html><body>No table</body></html>", 200, False),
        ('<table id="label-table"><tr><th>Name</th></tr></table>', 200, True),
        (
            '<table id="label-table"><tr><th>Name</th></tr><tr><td>No link</td></tr></table>',
            200,
            True,
        ),
        (_LABEL_TABLE_HTML, 500, False),
    ],
    ids=["no-table", "header-only", "no-linked-labels", "http-500"],
)
def test_download_labelset_uploads_nothing_without_labels(fake_s3, html, status, expected_ok):
    dl = _downloader_on(fake_s3)

    with patch.object(dl.session, "get", return_value=_http_response(html, status)):
        ok = dl.download_labelset(9, "b", "p")

    assert ok is expected_ok
    assert fake_s3.objects == {}


@pytest.mark.parametrize(
    ("page", "status", "expected"),
    [
        (_fixture("source_ok.html"), 200, True),
        (_fixture("source_not_found.html"), 200, False),
        (_fixture("source_denied.html"), 200, False),
        ("<html>Server error</html>", 500, False),
    ],
    ids=["accessible", "not-found", "denied", "http-500"],
)
def test_check_permissions(page, status, expected):
    dl = CoralNetDownloader("u", "p")

    with patch.object(dl.session, "get", return_value=_http_response(page, status)):
        assert dl.check_permissions(3) is expected


_LOGIN_URL = CoralNetDownloader.LOGIN_URL


@pytest.mark.parametrize(
    ("post_text", "post_url"),
    [
        ("<html>Sign out</html>", _LOGIN_URL),  # signed in, no redirect
        ("<html>Welcome</html>", "https://coralnet.ucsd.edu/"),  # redirected away from login
    ],
    ids=["sign-out-link-only", "redirect-only"],
)
def test_login_succeeds_when_either_signal_of_success_holds(post_text, post_url):
    dl = CoralNetDownloader("alice", "secret")

    with (
        patch.object(dl.session, "get", return_value=_http_response(_login_get_html())),
        patch.object(
            dl.session, "post", return_value=_http_response(post_text, url=post_url)
        ) as post,
    ):
        ok = dl.login()

    assert ok is True
    assert post.call_args.args == (_LOGIN_URL,)
    assert post.call_args.kwargs["data"] == {
        "username": "alice",
        "password": "secret",
        "csrfmiddlewaretoken": "logintok",
    }
    assert post.call_args.kwargs["headers"] == {"Referer": _LOGIN_URL}


# ---------------------------------------------------------------------------
# Browse pages, image URLs, source orchestration, guards
# ---------------------------------------------------------------------------


def test_get_images_on_page_returns_named_thumbs_and_the_next_page_link():
    dl = CoralNetDownloader("u", "p")
    untitled_thumb = '<span class="thumb_wrapper"><a href="/image/9/view/"><img/></a></span>'
    html = _browse_page("csrf", n_thumbs=2, next_href="?page=2", start_id=1) + untitled_thumb

    with patch.object(dl.session, "get", return_value=_http_response(html)):
        images, next_rel = dl.get_images_on_page("https://coralnet.ucsd.edu/source/1/browse/images")

    assert images == {"img1.jpg": "/image/1/view/", "img2.jpg": "/image/2/view/"}
    assert next_rel == "?page=2"


def test_get_images_on_page_reports_no_next_link_on_the_last_page():
    dl = CoralNetDownloader("u", "p")

    with patch.object(dl.session, "get", return_value=_http_response(_browse_page("c", 1))):
        _, next_rel = dl.get_images_on_page("https://coralnet.ucsd.edu/source/1/browse/images")

    assert next_rel is None


def test_get_images_keeps_the_first_href_for_a_duplicate_name():
    dl = CoralNetDownloader("u", "p")
    pages = iter(
        [
            ({"a.jpg": "/image/1/view/"}, "?page=2"),
            ({"a.jpg": "/image/99/view/", "b.jpg": "/image/2/view/"}, None),
        ]
    )

    with patch.object(dl, "get_images_on_page", side_effect=lambda url: next(pages)):
        df, ok = dl.get_images(7)

    assert ok is True
    assert df is not None
    assert df["Name"].tolist() == ["a.jpg", "b.jpg"]
    assert df["Image Page"].tolist() == ["/image/1/view/", "/image/2/view/"]


_ORIGINAL_IMAGE_PAGE = (
    '<html><body><div id="original_image_container">'
    '<img src="https://bucket.s3.amazonaws.com/original/abc.jpg"/></div></body></html>'
)


def test_resolve_one_image_url_returns_the_original_image_src():
    dl = CoralNetDownloader("u", "p")

    with patch.object(
        requests.Session, "get", return_value=_http_response(_ORIGINAL_IMAGE_PAGE)
    ) as get:
        url = dl._resolve_one_image_url("/image/5/view/")

    assert url == "https://bucket.s3.amazonaws.com/original/abc.jpg"
    assert get.call_args.args == ("https://coralnet.ucsd.edu/image/5/view/",)


def test_resolve_one_image_url_raises_when_the_page_has_no_original_image():
    dl = CoralNetDownloader("u", "p")
    private_page = "<html><body>You don't have permission to view this image.</body></html>"

    with (
        patch.object(requests.Session, "get", return_value=_http_response(private_page)),
        pytest.raises(ValueError, match="original image container"),
    ):
        dl._resolve_one_image_url("/image/5/view/")


def test_get_image_urls_with_several_workers_maps_results_to_input_order():
    """Image 1 resolves last, image 2 fails, image 3 resolves first; results keep input order."""
    dl = CoralNetDownloader("u", "p", max_workers=3)
    image3_resolved = threading.Event()

    def resolve(rel: str) -> str:
        if rel == "/image/1/view/":
            assert image3_resolved.wait(timeout=10), "image 3 never resolved"
        if rel == "/image/2/view/":
            raise ValueError("private")
        if rel == "/image/3/view/":
            image3_resolved.set()
        return "http://x" + rel

    with patch.object(dl, "_resolve_one_image_url", side_effect=resolve):
        urls = dl.get_image_urls(["/image/1/view/", "/image/2/view/", "/image/3/view/"])

    assert urls == ["http://x/image/1/view/", None, "http://x/image/3/view/"]


def test_download_source_stops_when_the_website_reports_zero_images():
    dl, mocks = _failing_step_downloader()
    dl._get_source_overview_html.return_value = _overview_html(0)  # type: ignore[attr-defined]

    res = dl.download_source(1, bucket_name="b", s3_prefix="p")

    assert res.ok is True
    assert res.skipped_no_confirmed is False
    assert res.errors == []
    for step in ("download_metadata", "download_labelset", "download_annotations", "get_images"):
        mocks[step].assert_not_called()


@pytest.mark.parametrize(
    ("metadata_total", "expected_hint"),
    [(2500, 2500), (0, None)],
    ids=["positive-total", "zero-total-becomes-none"],
)
def test_download_source_hints_annotations_with_the_metadata_total_when_positive(
    metadata_total, expected_hint
):
    dl, mocks = _failing_step_downloader(download_metadata=(True, metadata_total))

    dl.download_source(1, bucket_name="b", s3_prefix="p")

    assert mocks["download_annotations"].call_args.kwargs["total_images_hint"] == expected_hint


def _walk(dl: CoralNetDownloader, page_html: str) -> None:
    with patch.object(dl.session, "get", return_value=_http_response(page_html)):
        dl._walk_browse_pages(1)


@pytest.mark.parametrize(
    ("page_html", "message"),
    [
        ("<html><body>Please log in</body></html>", "Could not find export form / CSRF on page 1"),
        (_browse_page("csrf", n_thumbs=0), "No image IDs found on page 1 of source 1"),
    ],
    ids=["no-csrf", "no-image-ids"],
)
def test_walk_browse_pages_raises_on_a_page_it_cannot_use(page_html, message):
    dl = CoralNetDownloader("u", "p")

    with pytest.raises(RuntimeError, match=message):
        _walk(dl, page_html)


def test_export_one_page_raises_when_the_worker_session_gets_no_csrf(paginated_downloader):
    dl = paginated_downloader
    item = _PageWorkItem(
        page_num=4,
        page_url=f"{_BROWSE_URL}?page=4",
        csrf="stale",
        images_on_page=1,
        image_ids=(5,),
    )

    with (
        patch.object(dl.session, "get", return_value=_http_response("<html>Please log in</html>")),
        pytest.raises(RuntimeError, match="Worker session: no CSRF on browse page 4 of source 1"),
    ):
        dl._export_one_page(1, item)


@pytest.mark.parametrize(
    ("total_images", "prep", "serve"),
    [
        (0, 300, 120),
        (None, 300, 120),
        (141_600, 7200, 1800),  # 120 + n // 20 == 7200 exactly; 60 + n // 50 is 2892
        (200_000, 7200, 1800),  # both capped
        (88_000, 4520, 1800),  # serve 60 + 1760 = 1820 is capped
    ],
)
def test_export_timeouts_floor_at_zero_and_cap_for_huge_sources(total_images, prep, serve):
    assert export_prep_timeout_seconds(total_images) == prep
    assert export_serve_timeout_seconds(total_images) == serve


@pytest.mark.parametrize(
    ("n_rows", "batch_size"),
    [(-1, 5), (5, 0), (5, -2)],
)
def test_half_open_batches_rejects_a_negative_row_count_or_a_batch_size_below_one(
    n_rows, batch_size
):
    with pytest.raises(ValueError, match="n_rows must be >= 0 and batch_size >= 1"):
        half_open_batches(n_rows, batch_size)


@pytest.mark.parametrize(
    ("n_rows", "batch_size", "expected"),
    [(0, 5, []), (3, 1, [(0, 1), (1, 2), (2, 3)]), (4, 4, [(0, 4)])],
)
def test_half_open_batches_edges(n_rows, batch_size, expected):
    assert half_open_batches(n_rows, batch_size) == expected
