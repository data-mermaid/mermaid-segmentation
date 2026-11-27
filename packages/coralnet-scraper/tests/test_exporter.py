"""Offline tests for AnnotationExporter and the export helpers."""

from __future__ import annotations

import io
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import requests
import stamina
import tenacity
from conftest import (
    _BASE,
    _CSV_HEADERS,
    _LOGIN_PATH,
    _PARTIAL,
    _PREP,
    _SERVE_T1,
    Reply,
    _ann_csv,
    _browse_html,
    _http_response,
    _login_get_html,
    _page_path,
    _page_reply,
    _posts_to,
    _refs,
    _script_login,
    _script_pages,
    _script_walk,
    _site_downloader,
)

from coralnet_scraper import exporter as exporter_mod
from coralnet_scraper.downloader import CoralNetDownloader
from coralnet_scraper.errors import ContentError, ExportEmpty, SessionExpired
from coralnet_scraper.exporter import (
    _PAGINATED_EXPORT_THRESHOLD,
    decode_export_csv,
    export_prep_timeout_seconds,
    export_serve_timeout_seconds,
)
from coralnet_scraper.walker import ImageRef


@pytest.mark.parametrize(
    ("total_images", "points", "prep"),
    [
        (1681, None, 792),  # the default of 100 points
        (1681, 10, 300),  # 120 + 67 is under the floor
        (4000, 37, 712),
        (5000, 10_000, 7200),  # capped
    ],
)
def test_export_prep_timeout_scales_with_points_per_image(total_images, points, prep):
    assert export_prep_timeout_seconds(total_images, points) == prep


def _browse_page(
    csrf: str, n_thumbs: int = 1, next_href: str | None = None, start_id: int = 1
) -> str:
    """A browse page with ``n_thumbs`` thumbnails ``img<id>.jpg`` whose ids start at ``start_id``."""
    thumbs = [(start_id + i, f"img{start_id + i}.jpg") for i in range(n_thumbs)]
    return _browse_html(thumbs, next_href, csrf)


def _mock_response(text: str | None = None, json_data: dict | None = None) -> MagicMock:
    r = MagicMock()
    r.raise_for_status = MagicMock()
    if text is not None:
        r.text = text
    if json_data is not None:
        r.json.return_value = json_data
    return r


def _csv_response(text: str) -> requests.Response:
    """An ``export/serve`` answer as CoralNet sends it: ``text/csv`` with no charset."""
    return _http_response(
        text,
        url="https://coralnet.ucsd.edu/source/1/export/serve/",
        headers={"Content-Type": "text/csv"},
    )


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
        patch("coralnet_scraper.exporter.time.sleep"),
    ):
        yield post_mock


def test_download_annotations_dispatches_to_paginated_for_large_source(fake_s3):
    """Sources above _PAGINATED_EXPORT_THRESHOLD must use the paginated path."""
    dl = CoralNetDownloader("u", "p", s3=fake_s3)
    dl.auth.logged_in = True
    refs = _refs({1: [1]})

    with patch.object(dl.exporter, "download_annotations_paginated", return_value=True) as mock_pag:
        ok = dl.exporter.download_annotations(
            1, "b", "p", refs=refs, total_images=_PAGINATED_EXPORT_THRESHOLD + 1
        )

    assert ok is True
    mock_pag.assert_called_once_with(1, "b", "p", refs=refs)


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


def _run_bulk_download(
    fake_s3,
    *,
    serve_text: str = "",
    export_prep: list | None = None,
    total: int = _PAGINATED_EXPORT_THRESHOLD,
    points: int | None = None,
) -> _BulkRun:
    """Run the bulk annotations path on a small source whose walk found images 1 and 2.

    ``export_prep`` lists the export_prep POST outcomes in order (responses or exceptions).
    """
    dl = CoralNetDownloader("u", "p", s3=fake_s3)
    dl.auth.logged_in = True

    def get(url: str, timeout: int | None = None, **_: object) -> requests.Response:
        if "/export/serve/" in url:
            return _csv_response(serve_text)
        return _http_response(_BULK_BROWSE_HTML)

    if export_prep is None:
        export_prep = [_mock_response(json_data={"session_data_timestamp": "t1"})]
    with (
        patch.object(dl.session, "get", side_effect=get) as get_mock,
        patch.object(dl.session, "post", side_effect=export_prep) as post_mock,
        patch("coralnet_scraper.exporter.time.sleep") as sleep_mock,
    ):
        ok = dl.exporter.download_annotations(
            1, "b", "p", refs=_refs({1: [1, 2]}), total_images=total, points_per_image=points
        )
    return _BulkRun(ok=ok, get=get_mock, post=post_mock, sleep=sleep_mock)


_SERVE_CSV = "Name,Row,Column,Label ID\nimg1.jpg,1,2,3\n"
_BROWSE_URL = "https://coralnet.ucsd.edu/source/1/browse/images/"


def test_download_annotations_fails_after_four_export_prep_timeouts(fake_s3):
    run = _run_bulk_download(
        fake_s3,
        serve_text=_SERVE_CSV,
        export_prep=[requests.exceptions.ReadTimeout("t")] * 5,
        total=4000,
    )

    assert run.ok is False
    assert run.prep_timeouts == [1720, 2020, 2320, 2620]  # 120 + 4000 * 100 // 250, +300 each
    assert _BULK_KEY not in fake_s3.objects


def test_a_failed_page_stops_the_export_retries_of_the_pages_still_running(
    http_server, fake_s3, monkeypatch
):
    monkeypatch.setattr(exporter_mod, "_PAGE_DELAY_SECONDS", 0.0)
    _script_pages(http_server, [[(1, "img1.jpg")], [(2, "img2.jpg")]])
    dl = _site_downloader(http_server, fake_s3)
    page2_started = threading.Event()
    page1_failed = threading.Event()
    page2_attempts: list[int] = []
    stops: list[threading.Event] = []
    real_validated = dl.exporter._export_validated

    def capture_stop(*args: Any, **kwargs: Any) -> pd.DataFrame:
        stops.append(kwargs["stop"])
        return real_validated(*args, **kwargs)

    monkeypatch.setattr(dl.exporter, "_export_validated", capture_stop)

    def round_trip(session, source_id, page_url, form, *_: object) -> pd.DataFrame:
        if form["image_id_list"] == "1":
            assert page2_started.wait(5)
            page1_failed.set()
            raise requests.HTTPError(response=_http_response(status=403))
        page2_attempts.append(1)
        page2_started.set()
        assert page1_failed.wait(5)
        assert stops[0].wait(5)
        raise ContentError("page 2")

    monkeypatch.setattr(dl.exporter, "_export_round_trip", round_trip)

    ok = dl.exporter.download_annotations_paginated(
        1, "b", "p", refs=_refs({1: [1], 2: [2]}), page_workers=2
    )

    assert ok is False
    assert len(page2_attempts) == 1


def test_an_export_prep_read_timeout_delays_the_next_attempt_by_a_minute(fake_s3):
    run = _run_bulk_download(
        fake_s3,
        serve_text=_SERVE_CSV,
        export_prep=[
            requests.exceptions.ReadTimeout("t"),
            _mock_response(json_data={"session_data_timestamp": "t1"}),
        ],
    )

    assert run.ok is True
    assert [c.args[0] for c in run.sleep.call_args_list if c.args[0] >= 60] == [60]


def test_an_export_failure_other_than_a_prep_read_timeout_adds_no_delay(fake_s3):
    run = _run_bulk_download(
        fake_s3,
        serve_text=_SERVE_CSV,
        export_prep=[
            _mock_response(json_data={}),
            _mock_response(json_data={"session_data_timestamp": "t1"}),
        ],
    )

    assert run.ok is True
    assert [c.args[0] for c in run.sleep.call_args_list if c.args[0] >= 60] == []


def test_download_annotations_sizes_timeouts_from_the_total(fake_s3):
    run = _run_bulk_download(fake_s3, serve_text=_SERVE_CSV, total=4000, points=37)

    assert run.ok is True
    assert run.get.call_args_list[0].args == (_BROWSE_URL,)
    assert run.prep_timeouts == [712]  # 120 + 4000 * 37 // 250
    serve_get = run.get.call_args_list[-1]
    assert serve_get.args == (
        "https://coralnet.ucsd.edu/source/1/export/serve/?session_data_timestamp=t1",
    )
    assert serve_get.kwargs["timeout"] == 140  # 60 + 4000 // 50


def test_download_annotations_bulk_for_small_source(fake_s3):
    """Sources at or below threshold must use the existing bulk path."""
    run = _run_bulk_download(
        fake_s3, serve_text="Name,Row,Column,Label ID\nimg1.jpg,1,2,3\nimg2.jpg,4,5,6\n"
    )

    assert run.ok is True
    uploaded = pd.read_csv(io.BytesIO(fake_s3.objects[_BULK_KEY]))
    assert list(uploaded.columns) == ["Name", "Row", "Column", "Label ID"]
    assert uploaded.to_numpy().tolist() == [["img1.jpg", 1, 2, 3], ["img2.jpg", 4, 5, 6]]


def _page_script(page_htmls: list[str], ann_csvs: list[str]) -> tuple[list, list]:
    """GET and POST responses for a sequential (one-worker) paginated export.

    The GET sequence is, per page, a browse refresh and the serve response; the POST
    sequence is one export_prep response per page.
    """
    gets = []
    for html, ann in zip(page_htmls, ann_csvs, strict=True):
        gets += [_mock_response(text=html), _csv_response(ann)]
    posts = [
        _mock_response(json_data={"session_data_timestamp": f"t{i}"})
        for i in range(1, len(page_htmls) + 1)
    ]
    return gets, posts


def test_paginated_export_sends_each_pages_image_ids_and_concatenates_in_page_order(
    paginated_downloader, fake_s3
):
    """Each per-page POST sends ``image_select_type=selected`` and that page's image ids.

    This is the contract that keeps a page export from exporting the whole source.
    """
    dl = paginated_downloader
    refs = _refs({1: list(range(1000, 1020)), 2: list(range(2000, 2020)), 3: [3000, 3001]})
    page_htmls = [_browse_page(f"csrf{i}") for i in (1, 2, 3)]
    ann_csvs = [
        _ann_csv([("img1000.jpg", 1, 1, 100), ("img1001.jpg", 2, 2, 101)]),
        _ann_csv([("img2000.jpg", 3, 3, 102)]),
        _ann_csv([("img3000.jpg", 4, 4, 103), ("img3001.jpg", 5, 5, 104)]),
    ]
    gets, posts = _page_script(page_htmls, ann_csvs)

    with _mock_paginated_session(dl, get=gets, post=posts) as mock_post:
        ok = dl.exporter.download_annotations_paginated(1, "b", "p", refs=refs, page_workers=1)

    assert ok is True
    df = pd.read_csv(io.BytesIO(fake_s3.objects[_BULK_KEY]))
    assert df["Name"].tolist() == [
        "img1000.jpg",
        "img1001.jpg",
        "img2000.jpg",
        "img3000.jpg",
        "img3001.jpg",
    ]
    expected_id_lists = {
        "csrf1": "_".join(str(i) for i in range(1000, 1020)),
        "csrf2": "_".join(str(i) for i in range(2000, 2020)),
        "csrf3": "3000_3001",
    }
    assert mock_post.call_count == 3
    for post_call in mock_post.call_args_list:
        form = post_call.kwargs["data"]
        assert form["image_select_type"] == "selected"
        assert form["image_form_type"] == "search"
        assert form["browse_action"] == "export_annotations"
        assert form["image_id_list"] == expected_id_lists[form["csrfmiddlewaretoken"]]


def test_paginated_export_writes_no_annotations_when_a_page_export_fails_with_4xx(
    paginated_downloader, fake_s3
):
    dl = paginated_downloader
    forbidden = requests.HTTPError("403", response=MagicMock(status_code=403))

    with _mock_paginated_session(
        dl, get=[_mock_response(text=_browse_page("csrf1"))], post=[forbidden]
    ) as post_mock:
        ok = dl.exporter.download_annotations_paginated(
            1, "b", "p", refs=_refs({1: [1]}), page_workers=1
        )

    assert ok is False
    assert post_mock.call_count == 1  # a 4xx is not retried
    assert _BULK_KEY not in fake_s3.objects


def test_paginated_export_with_parallel_workers_keeps_page_order_when_pages_finish_out_of_order(
    paginated_downloader, fake_s3
):
    """Page 1's serve response is held until page 3's has been served, so the pages complete
    as 3 then 1 (or 2 in between) yet the uploaded rows follow page order.
    """
    dl = paginated_downloader
    pages = {
        _BROWSE_URL: _browse_page("csrf1"),
        f"{_BROWSE_URL}?page=2": _browse_page("csrf2"),
        f"{_BROWSE_URL}?page=3": _browse_page("csrf3"),
    }
    annotations = {
        1: _ann_csv([("img100.jpg", 1, 1, 10)]),
        2: _ann_csv([("img200.jpg", 2, 2, 20)]),
        3: _ann_csv([("img300.jpg", 3, 3, 30)]),
    }
    page3_served = threading.Event()

    def get(url: str, timeout: int | None = None, **_: object) -> MagicMock | requests.Response:
        if "/export/serve/" not in url:
            return _mock_response(text=pages[url])
        page = int(url.rsplit("=t", 1)[1])
        if page == 1:
            assert page3_served.wait(timeout=10), "page 3 never finished"
        if page == 3:
            page3_served.set()
        return _csv_response(annotations[page])

    def post(url: str, *, data: dict, **_: object) -> MagicMock:
        page = data["csrfmiddlewaretoken"].removeprefix("csrf")
        return _mock_response(json_data={"session_data_timestamp": f"t{page}"})

    with _mock_paginated_session(dl, get=get, post=post):
        ok = dl.exporter.download_annotations_paginated(
            1, "b", "p", refs=_refs({1: [100], 2: [200], 3: [300]}), page_workers=3
        )

    assert ok is True
    df = pd.read_csv(io.BytesIO(fake_s3.objects[_BULK_KEY]))
    assert df["Name"].tolist() == ["img100.jpg", "img200.jpg", "img300.jpg"]


def test_paginated_export_runs_export_workers_page_round_trips_at_once(fake_s3):
    dl = CoralNetDownloader("u", "p", export_workers=2, s3=fake_s3)
    dl.auth.logged_in = True
    dl.auth._new_logged_in_session = lambda: dl.session  # type: ignore[method-assign]
    lock = threading.Lock()
    in_flight = [0]
    peak = [0]

    def get(url: str, timeout: int | None = None, **_: object) -> MagicMock | requests.Response:
        if "/export/serve/" not in url:
            page = int(url.rsplit("=", 1)[1]) if "?page=" in url else 1
            return _mock_response(text=_browse_page(f"csrf{page}"))
        page = int(url.rsplit("=t", 1)[1])
        return _csv_response(_ann_csv([(f"img{page * 100}.jpg", page, page, page)]))

    def post(url: str, *, data: dict, **_: object) -> MagicMock:
        with lock:
            in_flight[0] += 1
            peak[0] = max(peak[0], in_flight[0])
        threading.Event().wait(0.05)  # hold the round trip open so workers overlap
        with lock:
            in_flight[0] -= 1
        page = data["csrfmiddlewaretoken"].removeprefix("csrf")
        return _mock_response(json_data={"session_data_timestamp": f"t{page}"})

    refs = _refs({page: [page * 100] for page in range(1, 7)})
    with _mock_paginated_session(dl, get=get, post=post):
        ok = dl.exporter.download_annotations_paginated(1, "b", "p", refs=refs)

    assert ok is True
    assert peak[0] == 2


def test_the_first_failed_page_stops_the_pages_not_yet_started(fake_s3):
    dl = CoralNetDownloader("u", "p", export_workers=3, s3=fake_s3)
    attempted: list[int] = []

    def export_one_page(
        source_id: int, page: int, page_refs: list[ImageRef], stop: threading.Event | None = None
    ) -> pd.DataFrame:
        attempted.append(page)
        if page == 1:
            threading.Event().wait(0.3)  # still running when pages 2 and 3 fail
            return pd.DataFrame({"Name": ["img100.jpg"]})
        raise RuntimeError(f"page {page}")

    dl.exporter._export_one_page = export_one_page  # type: ignore[method-assign]
    pages = {page: [ImageRef(page * 100, f"img{page * 100}.jpg", 1)] for page in range(1, 201)}

    with pytest.raises(RuntimeError, match="page"):
        dl.exporter._export_pages_parallel(1, "b", "p", pages, 3)

    assert len(attempted) <= 3
    assert ("b", f"{_PARTIAL}page-1.csv") in fake_s3.objects


@pytest.mark.parametrize(
    ("total_images", "prep", "serve"),
    [
        (0, 300, 120),
        (None, 300, 120),
        (17_000, 6920, 400),  # 120 + 1,700,000 // 250 is under the cap
        (17_700, 7200, 414),  # 120 + 1,770,000 // 250 == 7200 exactly
        (200_000, 7200, 1800),  # both capped
        (88_000, 7200, 1800),  # serve 60 + 1760 = 1820 is capped
    ],
)
def test_export_timeouts_floor_at_zero_and_cap_for_huge_sources(total_images, prep, serve):
    assert export_prep_timeout_seconds(total_images) == prep
    assert export_serve_timeout_seconds(total_images) == serve


def test_a_paginated_export_skips_pages_with_a_saved_chunk(http_server, fake_s3, monkeypatch):
    monkeypatch.setattr(exporter_mod, "_PAGE_DELAY_SECONDS", 0.0)
    for page in (1, 2, 3):
        chunk = _ann_csv([(f"img{page}.jpg", 1, 1, 7)])
        fake_s3.put("b", f"{_PARTIAL}page-{page}.csv", chunk.encode())
    _script_pages(http_server, [[(i, f"img{i}.jpg")] for i in range(1, 7)])
    chunk_reads: list[str] = []
    real_get_object = fake_s3.get_object

    def counting_get_object(*, Bucket, Key, Range=None):
        if "/_partial/page-" in Key:
            chunk_reads.append(Key)
        return real_get_object(Bucket=Bucket, Key=Key, Range=Range)

    monkeypatch.setattr(fake_s3, "get_object", counting_get_object)
    http_server.script(
        _PREP, *[Reply(200, f'{{"session_data_timestamp": "t{p}"}}'.encode()) for p in (4, 5, 6)]
    )
    for page in (4, 5, 6):
        http_server.script(
            f"/source/1/export/serve/?session_data_timestamp=t{page}",
            Reply(200, _ann_csv([(f"img{page}.jpg", 1, 1, 7)]).encode(), _CSV_HEADERS),
        )
    dl = _site_downloader(http_server, fake_s3)

    ok = dl.exporter.download_annotations_paginated(
        1, "b", "p", refs=_refs({p: [p] for p in range(1, 7)}), page_workers=1
    )

    assert ok is True
    browsed = [r.path for r in http_server.received if r.path.startswith(_BASE)]
    assert browsed == [_page_path(n) for n in (4, 5, 6)]
    df = pd.read_csv(io.BytesIO(fake_s3.objects[_BULK_KEY]))
    assert df["Name"].tolist() == [f"img{p}.jpg" for p in range(1, 7)]
    assert sorted(chunk_reads) == sorted(f"{_PARTIAL}page-{p}.csv" for p in range(1, 7))


def _csv_reply(rows: list[tuple]) -> Reply:
    return Reply(200, _ann_csv(rows).encode(), _CSV_HEADERS)


def test_decode_export_csv_decodes_utf8_served_without_a_charset(http_server):
    body = "Name,Row\nIrmãos 1.jpg,1\nצילום 2.jpg,2\n".encode()
    http_server.script("/serve", Reply(200, body, _CSV_HEADERS))
    resp = requests.get(f"{http_server.url}/serve", timeout=5)

    assert decode_export_csv(resp)["Name"].tolist() == ["Irmãos 1.jpg", "צילום 2.jpg"]


@pytest.mark.parametrize(
    ("body", "content_type"),
    [
        ("Name,Row\nimg1.jpg,1\n", "text/html; charset=utf-8"),
        ("<html><body>Server Error</body></html>", "text/csv"),
        ("Row,Column\n", "text/csv"),
        ("Name,Row\n,1\n", "text/csv"),
        ("Name,<td>Row</td>\nimg1.jpg,1\n", "text/csv"),
        ("Name,Unnamed: 5\nimg1.jpg,1\n", "text/csv"),
    ],
    ids=[
        "html-content-type",
        "html-body",
        "no-name-column",
        "null-name",
        "html-column",
        "unnamed-column",
    ],
)
def test_decode_export_csv_rejects_a_body_that_is_not_an_annotations_csv(body, content_type):
    resp = _http_response(body, headers={"Content-Type": content_type})

    with pytest.raises(ContentError):
        decode_export_csv(resp)


def test_decode_export_csv_reports_an_expired_session_when_served_the_login_page():
    resp = _http_response(
        "<html>Sign in</html>",
        url="https://coralnet.ucsd.edu/accounts/login/?next=/source/1/export/serve/",
        headers={"Content-Type": "text/html; charset=utf-8"},
    )

    with pytest.raises(SessionExpired):
        decode_export_csv(resp)


def _export_page_one(dl: CoralNetDownloader) -> bool:
    return dl.exporter.download_annotations_paginated(
        1, "b", "p", refs=[ImageRef(1, "img1.jpg - Confirmed", 1)], page_workers=1
    )


def _chunk_names(fake_s3) -> list[str]:
    chunk = pd.read_csv(io.BytesIO(fake_s3.objects[("b", f"{_PARTIAL}page-1.csv")]))
    return chunk["Name"].tolist()


def test_a_page_export_with_rows_for_other_images_is_exported_again(
    http_server, fake_s3, monkeypatch
):
    monkeypatch.setattr(exporter_mod, "_PAGE_DELAY_SECONDS", 0.0)
    _script_walk(http_server, [[1]])
    http_server.script(_PREP, Reply(200, b'{"session_data_timestamp": "t1"}'))
    http_server.script(
        _SERVE_T1,
        _csv_reply([("img1.jpg", 1, 1, 7), ("img9.jpg", 1, 1, 7)]),
        _csv_reply([("img1.jpg", 1, 1, 7)]),
    )

    ok = _export_page_one(_site_downloader(http_server, fake_s3))

    assert ok is True
    assert _chunk_names(fake_s3) == ["img1.jpg"]
    assert len(http_server.requests_to(_PREP)) == 2


def test_an_export_prep_signed_out_error_logs_the_export_session_in_again(
    http_server, fake_s3, monkeypatch
):
    monkeypatch.setattr(exporter_mod, "_PAGE_DELAY_SECONDS", 0.0)
    _script_walk(http_server, [[1]])
    _script_login(http_server)
    http_server.script(
        _PREP,
        Reply(200, b'{"error": "You must be signed in to access this function."}'),
        Reply(200, b'{"session_data_timestamp": "t1"}'),
    )
    http_server.script(_SERVE_T1, _csv_reply([("img1.jpg", 1, 1, 7)]))

    ok = _export_page_one(_site_downloader(http_server, fake_s3))

    assert ok is True
    assert len(_posts_to(http_server, _LOGIN_PATH)) == 1
    assert _chunk_names(fake_s3) == ["img1.jpg"]


def test_a_page_export_browse_page_without_the_export_form_logs_in_again(
    http_server, fake_s3, monkeypatch
):
    monkeypatch.setattr(exporter_mod, "_PAGE_DELAY_SECONDS", 0.0)
    _script_login(http_server)
    http_server.script(_page_path(1), _page_reply([1], form=False), _page_reply([1]))
    http_server.script(_PREP, Reply(200, b'{"session_data_timestamp": "t1"}'))
    http_server.script(_SERVE_T1, _csv_reply([("img1.jpg", 1, 1, 7)]))

    ok = _export_page_one(_site_downloader(http_server, fake_s3))

    assert ok is True
    assert len(_posts_to(http_server, _LOGIN_PATH)) == 1
    assert len(http_server.requests_to(_PREP)) == 1
    assert _chunk_names(fake_s3) == ["img1.jpg"]


class _SlowClock:
    """Stands in for ``time`` inside tenacity: every clock read is 30 s after the last."""

    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        self.now += 30.0
        return self.now

    def sleep(self, seconds: float) -> None:
        pass


def test_content_retries_continue_past_stamina_default_total_timeout(
    http_server, fake_s3, monkeypatch
):
    """Four export attempts span more than stamina's default 45 s budget and all still run."""
    stamina.set_testing(False)
    clock = _SlowClock()
    monkeypatch.setattr(tenacity, "time", clock)
    monkeypatch.setattr(tenacity.nap, "time", clock)
    monkeypatch.setattr(exporter_mod, "_PAGE_DELAY_SECONDS", 0.0)
    _script_walk(http_server, [[1]])
    http_server.script(
        _PREP, Reply(500), Reply(500), Reply(500), Reply(200, b'{"session_data_timestamp": "t1"}')
    )
    http_server.script(_SERVE_T1, _csv_reply([("img1.jpg", 1, 1, 7)]))

    ok = _export_page_one(_site_downloader(http_server, fake_s3))

    assert ok is True
    assert len(http_server.requests_to(_PREP)) == 4


def test_an_export_fails_when_logging_in_again_fails(http_server, fake_s3, monkeypatch):
    monkeypatch.setattr(exporter_mod, "_PAGE_DELAY_SECONDS", 0.0)
    _script_walk(http_server, [[1]])
    http_server.script(_LOGIN_PATH, Reply(200, _login_get_html().encode()))
    http_server.script(
        _PREP, Reply(200, b'{"error": "You must be signed in to access this function."}')
    )

    ok = _export_page_one(_site_downloader(http_server, fake_s3))

    assert ok is False
    assert len(_posts_to(http_server, _LOGIN_PATH)) == 1
    assert len(http_server.requests_to(_PREP)) == 1
    assert ("b", f"{_PARTIAL}page-1.csv") not in fake_s3.objects


def test_an_export_whose_session_stays_expired_fails_after_four_attempts(
    http_server, fake_s3, monkeypatch
):
    monkeypatch.setattr(exporter_mod, "_PAGE_DELAY_SECONDS", 0.0)
    _script_walk(http_server, [[1]])
    _script_login(http_server)
    http_server.script(
        _PREP, Reply(200, b'{"error": "You must be signed in to access this function."}')
    )

    ok = _export_page_one(_site_downloader(http_server, fake_s3))

    assert ok is False
    assert len(http_server.requests_to(_PREP)) == 4
    assert ("b", f"{_PARTIAL}page-1.csv") not in fake_s3.objects


# ---------------------------------------------------------------------------
# Header-only exports and unusable saved chunks
# ---------------------------------------------------------------------------

_HEADER_ONLY = "Name,Row,Column,Label ID\n"


def _status_refs(statuses: dict[int, str]) -> list[ImageRef]:
    """One image ``<n>`` per page ``n``, titled ``img<n>.jpg - <status>``."""
    return [ImageRef(n, f"img{n}.jpg - {status}", n) for n, status in statuses.items()]


def _serve_by_page(answers: dict[int, list[str]]) -> tuple[dict[int, int], Callable, Callable]:
    """GET/POST handlers for the paginated export; each page serves its answers in order.

    The last answer of a page repeats. Returns the per-page serve counts with the handlers.
    """
    served = dict.fromkeys(answers, 0)

    def get(url: str, timeout: int | None = None, **_: object) -> MagicMock | requests.Response:
        if "/export/serve/" not in url:
            page = int(url.rsplit("=", 1)[1]) if "?page=" in url else 1
            return _mock_response(text=_browse_page(f"csrf{page}"))
        page = int(url.rsplit("=t", 1)[1])
        queue = answers[page]
        served[page] += 1
        return _csv_response(queue[min(served[page], len(queue)) - 1])

    def post(url: str, *, data: dict, **_: object) -> MagicMock:
        page = data["csrfmiddlewaretoken"].removeprefix("csrf")
        return _mock_response(json_data={"session_data_timestamp": f"t{page}"})

    return served, get, post


def _rows(page: int) -> str:
    return _ann_csv([(f"img{page}.jpg", page, page, 7)])


def _saved_names(fake_s3, key: tuple[str, str]) -> list[str]:
    return pd.read_csv(io.BytesIO(fake_s3.objects[key]))["Name"].tolist()


def test_a_header_only_page_export_with_a_confirmed_image_is_exported_again(
    paginated_downloader, fake_s3
):
    dl = paginated_downloader
    refs = _status_refs({1: "Confirmed", 2: "Confirmed"})
    served, get, post = _serve_by_page({1: [_rows(1)], 2: [_HEADER_ONLY, _rows(2)]})

    with _mock_paginated_session(dl, get=get, post=post):
        ok = dl.exporter.download_annotations_paginated(1, "b", "p", refs=refs, page_workers=1)

    assert ok is True
    assert served[2] == 2
    assert _saved_names(fake_s3, ("b", f"{_PARTIAL}page-2.csv")) == ["img2.jpg"]
    assert _saved_names(fake_s3, _BULK_KEY) == ["img1.jpg", "img2.jpg"]


def test_a_header_only_page_export_without_a_confirmed_image_is_accepted(
    paginated_downloader, fake_s3
):
    dl = paginated_downloader
    refs = _status_refs({1: "Confirmed", 2: "Unconfirmed"})
    served, get, post = _serve_by_page({1: [_rows(1)], 2: [_HEADER_ONLY, _rows(2)]})

    with _mock_paginated_session(dl, get=get, post=post):
        ok = dl.exporter.download_annotations_paginated(1, "b", "p", refs=refs, page_workers=1)

    assert ok is True
    assert served[2] == 1
    assert _saved_names(fake_s3, _BULK_KEY) == ["img1.jpg"]


def test_a_confirmed_page_that_stays_header_only_raises_export_empty(paginated_downloader, fake_s3):
    dl = paginated_downloader
    refs = _status_refs({1: "Confirmed", 2: "Confirmed"})
    served, get, post = _serve_by_page({1: [_rows(1)], 2: [_HEADER_ONLY]})

    with _mock_paginated_session(dl, get=get, post=post), pytest.raises(ExportEmpty):
        dl.exporter.download_annotations_paginated(1, "b", "p", refs=refs, page_workers=1)

    assert served[2] > 1
    assert _BULK_KEY not in fake_s3.objects


def test_a_confirmed_bulk_export_that_stays_header_only_fails_the_export(
    paginated_downloader, fake_s3
):
    dl = paginated_downloader
    refs = _status_refs({1: "Confirmed"})
    served, get, post = _serve_by_page({1: [_HEADER_ONLY]})

    with _mock_paginated_session(dl, get=get, post=post), pytest.raises(ExportEmpty):
        dl.exporter.download_annotations(1, "b", "p", refs=refs, total_images=1, points_per_image=1)

    assert served[1] > 1
    assert _BULK_KEY not in fake_s3.objects


def test_a_header_only_bulk_export_with_a_confirmed_image_is_exported_again(
    paginated_downloader, fake_s3
):
    dl = paginated_downloader
    refs = _status_refs({1: "Confirmed"})
    served, get, post = _serve_by_page({1: [_HEADER_ONLY, _rows(1)]})

    with _mock_paginated_session(dl, get=get, post=post):
        ok = dl.exporter.download_annotations(
            1, "b", "p", refs=refs, total_images=1, points_per_image=1
        )

    assert ok is True
    assert served[1] == 2
    assert _saved_names(fake_s3, _BULK_KEY) == ["img1.jpg"]


@pytest.mark.parametrize(
    "saved",
    [b"", _HEADER_ONLY.encode(), b"Row,Column,Label ID\n1,1,7\n"],
    ids=["zero-bytes", "header-only", "no-name-column"],
)
def test_an_unusable_saved_chunk_is_exported_again(paginated_downloader, fake_s3, saved):
    dl = paginated_downloader
    refs = _status_refs({1: "Confirmed", 2: "Confirmed"})
    fake_s3.put("b", f"{_PARTIAL}page-1.csv", saved)
    fake_s3.put("b", f"{_PARTIAL}page-2.csv", _rows(2).encode())
    served, get, post = _serve_by_page({1: [_rows(1)], 2: [_rows(2)]})

    with _mock_paginated_session(dl, get=get, post=post):
        ok = dl.exporter.download_annotations_paginated(1, "b", "p", refs=refs, page_workers=1)

    assert ok is True
    assert served == {1: 1, 2: 0}
    assert _saved_names(fake_s3, _BULK_KEY) == ["img1.jpg", "img2.jpg"]


def test_a_header_only_saved_chunk_of_an_unconfirmed_page_is_kept(paginated_downloader, fake_s3):
    dl = paginated_downloader
    refs = _status_refs({1: "Confirmed", 2: "Unconfirmed"})
    fake_s3.put("b", f"{_PARTIAL}page-1.csv", _rows(1).encode())
    fake_s3.put("b", f"{_PARTIAL}page-2.csv", _HEADER_ONLY.encode())
    served, get, post = _serve_by_page({1: [_rows(1)], 2: [_rows(2)]})

    with _mock_paginated_session(dl, get=get, post=post):
        ok = dl.exporter.download_annotations_paginated(1, "b", "p", refs=refs, page_workers=1)

    assert ok is True
    assert served == {1: 0, 2: 0}
    assert _saved_names(fake_s3, ("b", f"{_PARTIAL}page-2.csv")) == []
    assert _saved_names(fake_s3, _BULK_KEY) == ["img1.jpg"]
