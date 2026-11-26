"""Offline tests for CoralNetDownloader."""

from __future__ import annotations

import io
import json
import logging
import random
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import requests
import stamina
import tenacity
from botocore.exceptions import ClientError
from conftest import Reply, ScriptedServer
from PIL import Image

from coralnet_scraper import downloader as downloader_mod
from coralnet_scraper.downloader import (
    _PAGINATED_EXPORT_THRESHOLD,
    ContentError,
    CoralNetDownloader,
    ImageInvalid,
    ImageRef,
    SessionExpired,
    _parse_browse_thumbs,
    _read_image_body,
    decode_export_csv,
    export_prep_timeout_seconds,
    export_serve_timeout_seconds,
    validate_image_bytes,
)
from coralnet_scraper.models import ImageFailures
from coralnet_scraper.verify import Verification

_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "coralnet_html"


def _fixture(name: str) -> str:
    return (_FIXTURES / name).read_text(encoding="utf-8")


def _jpeg_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (4, 4), (10, 20, 30)).save(buf, "JPEG")
    return buf.getvalue()


_JPEG = _jpeg_bytes()
_JPEG_HEADERS = {"Content-Type": "image/jpeg"}
_CSV_HEADERS = {"Content-Type": "text/csv"}


def test_export_timeouts_scale_with_source_size():
    assert export_prep_timeout_seconds(None) == 300
    assert export_prep_timeout_seconds(100) == 300
    assert export_prep_timeout_seconds(1681, 100) == 792  # s57: 120 + 168,100 // 250
    assert export_prep_timeout_seconds(5000, 200) == 4120
    assert export_serve_timeout_seconds(32176) == 703


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


@pytest.mark.parametrize(
    ("overview", "no_confirmed", "no_images"),
    [
        (_fixture("source_zero_confirmed.html"), True, False),
        (_overview_html(0), False, True),
    ],
    ids=["zero-confirmed", "zero-images"],
)
def test_download_source_skips_a_source_without_confirmed_images_or_images(
    fake_s3, overview, no_confirmed, no_images
):
    dl = CoralNetDownloader("u", "p")
    dl.logged_in = True
    dl.s3 = fake_s3
    with (
        patch.object(dl, "_get_source_overview_html", return_value=overview),
        patch.object(dl, "download_metadata") as mock_meta,
        patch.object(dl, "walk_source") as mock_walk,
    ):
        res = dl.download_source(5961, bucket_name="b", s3_prefix="p")
    assert res.ok is True
    assert res.errors == []
    assert res.skipped_no_confirmed is no_confirmed
    assert res.skipped_no_images is no_images
    mock_meta.assert_not_called()
    mock_walk.assert_not_called()
    assert fake_s3.objects == {}


def test_download_images_separates_missing_url_from_upload_failures():
    dl = CoralNetDownloader("u", "p", max_workers=2)
    # Image 1 uploads, 2 has no original on its view page, 3 fails to upload.

    def resolve(rel: str) -> str:
        if rel == "/image/2/view/":
            raise ValueError("private")
        return "http://x" + rel

    def upload(url: str, bucket: str, key: str) -> int:
        if url.endswith("/3/view/"):
            raise RuntimeError("boom")
        return 1

    with (
        patch.object(dl, "_resolve_one_image_url", side_effect=resolve),
        patch.object(dl, "download_image_to_s3", side_effect=upload),
    ):
        result = dl.download_images(_refs({1: [1, 2, 3]}), 1, "b", "p")
    assert result == ImageFailures(missing_url=1, upload_failed=1)


def _outage_run(
    failing: Callable[[int], bool], count: int, *, await_stop: bool = False
) -> tuple[ImageFailures, list[str]]:
    """Run ``count`` single-worker image downloads whose view page fails when ``failing(id)``.

    ``await_stop`` makes image 21 wait for the collecting thread to set ``stop`` first.
    """
    dl = CoralNetDownloader("u", "p", max_workers=1)
    resolved: list[str] = []

    def resolve(rel: str) -> str:
        resolved.append(rel)
        if failing(int(rel.split("/")[2])):
            raise requests.HTTPError("503")
        return "http://x" + rel

    real = dl._download_image

    def paced(ref: ImageRef, *args: Any) -> str:
        if await_stop and ref.id == downloader_mod._IMAGE_OUTAGE_STREAK + 1:
            assert args[-1].wait(5)  # stop is the last positional argument
        return real(ref, *args)

    with (
        patch.object(dl, "_resolve_one_image_url", side_effect=resolve),
        patch.object(dl, "download_image_to_s3", return_value=1),
        patch.object(dl, "_download_image", side_effect=paced),
    ):
        failures = dl.download_images(_refs({1: list(range(1, count + 1))}), 1, "b", "p")
    return failures, resolved


def test_a_run_of_failed_images_stops_the_stage_and_skips_the_rest():
    failures, resolved = _outage_run(lambda _i: True, 25, await_stop=True)

    assert failures == ImageFailures(missing_url=20, skipped=5)
    assert len(resolved) < 25 * downloader_mod._IMAGE_ATTEMPTS
    assert downloader_mod._image_failure_codes(failures)[-1] == "images_outage"


def test_an_image_that_fails_as_stop_is_set_reports_its_failure_and_uploads_nothing():
    dl = CoralNetDownloader("u", "p")
    stop = threading.Event()

    def resolve(_rel: str) -> str:
        stop.set()
        raise requests.HTTPError("503")

    with (
        patch.object(dl, "_resolve_one_image_url", side_effect=resolve),
        patch.object(dl, "download_image_to_s3", return_value=1) as upload,
    ):
        result = dl._download_image(ImageRef(1, 1, "/image/1/view/"), "b", "k", stop)

    assert result == "missing_url"
    upload.assert_not_called()


def test_a_success_resets_the_image_failure_streak():
    # Images 1-19 fail, 20 succeeds, 21-39 fail: 19 consecutive failures twice.
    failures, _ = _outage_run(lambda i: i != 20, 39)

    assert failures.skipped == 0
    assert "images_outage" not in downloader_mod._image_failure_codes(failures)


_EXPORT_FORM = """\
<form id="export-annotations-prep-form">
  <input type="hidden" name="csrfmiddlewaretoken" value="{csrf}"/>
</form>"""


def _browse_html(
    thumbs: list[tuple[int, str]],
    next_href: str | None = None,
    csrf: str = "tok",
    *,
    form: bool = True,
) -> str:
    """A browse page with one thumbnail per ``(image id, title)``.

    ``form=False`` leaves out the export form, as for a logged-out visitor.
    """
    spans = "".join(
        f'<span class="thumb_wrapper"><a href="/image/{i}/view/"><img title="{t}"/></a></span>'
        for i, t in thumbs
    )
    next_link = f'<a title="Next page" href="{next_href}">Next</a>' if next_href else ""
    export_form = _EXPORT_FORM.format(csrf=csrf) if form else ""
    return f"<html>{export_form}{spans}{next_link}</html>"


def _browse_page(
    csrf: str, n_thumbs: int = 1, next_href: str | None = None, start_id: int = 1
) -> str:
    """A browse page with ``n_thumbs`` thumbnails ``img<id>.jpg`` whose ids start at ``start_id``."""
    thumbs = [(start_id + i, f"img{start_id + i}.jpg") for i in range(n_thumbs)]
    return _browse_html(thumbs, next_href, csrf)


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
    """Each export worker thread gets its own ``requests.Session`` from a fresh login.

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
        patch("coralnet_scraper.downloader.time.sleep"),
    ):
        yield post_mock


def _refs(pages: dict[int, list[int]]) -> list[ImageRef]:
    """Walk results: image ``img<id>.jpg`` for each id, on the page it is listed under."""
    return [ImageRef(i, f"img{i}.jpg", page) for page, ids in pages.items() for i in ids]


def test_download_annotations_dispatches_to_paginated_for_large_source(fake_s3):
    """Sources above _PAGINATED_EXPORT_THRESHOLD must use the paginated path."""
    dl = CoralNetDownloader("u", "p")
    dl.logged_in = True
    dl.s3 = fake_s3
    refs = _refs({1: [1]})

    with patch.object(dl, "download_annotations_paginated", return_value=True) as mock_pag:
        ok = dl.download_annotations(
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
    dl = CoralNetDownloader("u", "p")
    dl.logged_in = True
    dl.s3 = fake_s3

    def get(url: str, timeout: int | None = None, **_: object) -> requests.Response:
        if "/export/serve/" in url:
            return _csv_response(serve_text)
        return _http_response(_BULK_BROWSE_HTML)

    if export_prep is None:
        export_prep = [_mock_response(json_data={"session_data_timestamp": "t1"})]
    with (
        patch.object(dl.session, "get", side_effect=get) as get_mock,
        patch.object(dl.session, "post", side_effect=export_prep) as post_mock,
        patch("coralnet_scraper.downloader.time.sleep") as sleep_mock,
    ):
        ok = dl.download_annotations(
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


def _export_validated_on(dl: CoralNetDownloader, stop: threading.Event | None) -> pd.DataFrame:
    return dl._export_validated(
        dl.session,
        1,
        _BROWSE_URL,
        {},
        {"img1.jpg"},
        prep_timeout=300,
        serve_timeout=120,
        stop=stop,
    )


def test_export_validated_makes_no_further_attempt_once_stop_is_set(fake_s3):
    dl = CoralNetDownloader("u", "p")
    stop = threading.Event()
    attempts: list[int] = []

    def failing_round_trip(*_: object) -> pd.DataFrame:
        attempts.append(1)
        stop.set()
        raise ContentError(f"attempt {len(attempts)}")

    with (
        patch.object(dl, "_export_round_trip", side_effect=failing_round_trip),
        pytest.raises(ContentError, match="attempt 1"),
    ):
        _export_validated_on(dl, stop)

    assert len(attempts) == 1


def test_a_failed_page_stops_the_export_retries_of_the_pages_still_running(
    http_server, fake_s3, monkeypatch
):
    monkeypatch.setattr(downloader_mod, "_PAGE_DELAY_SECONDS", 0.0)
    _script_pages(http_server, [[(1, "img1.jpg")], [(2, "img2.jpg")]])
    dl = _site_downloader(http_server, fake_s3)
    page2_started = threading.Event()
    page1_failed = threading.Event()
    page2_attempts: list[int] = []
    stops: list[threading.Event] = []
    real_validated = dl._export_validated

    def capture_stop(*args: Any, **kwargs: Any) -> pd.DataFrame:
        stops.append(kwargs["stop"])
        return real_validated(*args, **kwargs)

    monkeypatch.setattr(dl, "_export_validated", capture_stop)

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

    monkeypatch.setattr(dl, "_export_round_trip", round_trip)

    ok = dl.download_annotations_paginated(
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
        ok = dl.download_annotations_paginated(1, "b", "p", refs=refs, page_workers=1)

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
        ok = dl.download_annotations_paginated(1, "b", "p", refs=_refs({1: [1]}), page_workers=1)

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
        ok = dl.download_annotations_paginated(
            1, "b", "p", refs=_refs({1: [100], 2: [200], 3: [300]}), page_workers=3
        )

    assert ok is True
    df = pd.read_csv(io.BytesIO(fake_s3.objects[_BULK_KEY]))
    assert df["Name"].tolist() == ["img100.jpg", "img200.jpg", "img300.jpg"]


def test_paginated_export_runs_export_workers_page_round_trips_at_once(fake_s3):
    dl = CoralNetDownloader("u", "p", export_workers=2)
    dl.logged_in = True
    dl.s3 = fake_s3
    dl._new_logged_in_session = lambda: dl.session  # type: ignore[method-assign]
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
        ok = dl.download_annotations_paginated(1, "b", "p", refs=refs)

    assert ok is True
    assert peak[0] == 2


def test_the_first_failed_page_stops_the_pages_not_yet_started(fake_s3):
    dl = CoralNetDownloader("u", "p", export_workers=3)
    dl.s3 = fake_s3
    attempted: list[int] = []

    def export_one_page(
        source_id: int, page: int, page_refs: list[ImageRef], stop: threading.Event | None = None
    ) -> pd.DataFrame:
        attempted.append(page)
        if page == 1:
            threading.Event().wait(0.3)  # still running when pages 2 and 3 fail
            return pd.DataFrame({"Name": ["img100.jpg"]})
        raise RuntimeError(f"page {page}")

    dl._export_one_page = export_one_page  # type: ignore[method-assign]
    pages = {page: [ImageRef(page * 100, f"img{page * 100}.jpg", 1)] for page in range(1, 201)}

    with pytest.raises(RuntimeError, match="page"):
        dl._export_pages_parallel(1, "b", "p", pages, 3)

    assert len(attempted) <= 3
    assert ("b", f"{_PARTIAL}page-1.csv") in fake_s3.objects


def _login_get_html() -> str:
    return '<input type="hidden" name="csrfmiddlewaretoken" value="logintok"/>'


def test_constructor_rejects_invalid_max_workers():
    with pytest.raises(ValueError, match="workers"):
        CoralNetDownloader("u", "p", max_workers=0)


def _failing_step_downloader(
    fake_s3, **overrides
) -> tuple[CoralNetDownloader, dict[str, MagicMock]]:
    """A downloader whose steps succeed except those overridden; returns it with its mocks."""
    dl = CoralNetDownloader("u", "p", max_workers=2)
    dl.logged_in = True
    dl.s3 = fake_s3
    defaults = {
        "download_metadata": (True, 1234),
        "download_labelset": True,
        "download_annotations": True,
    }
    mocks = {
        name: MagicMock(return_value=overrides.get(name, default))
        for name, default in defaults.items()
    }
    mocks["walk_source"] = MagicMock(
        side_effect=lambda *args, total_images: (_refs({1: [1]}), total_images)
    )
    if "walk_source" in overrides and isinstance(overrides["walk_source"], Exception):
        mocks["walk_source"].side_effect = overrides["walk_source"]
    for name, mock in mocks.items():
        setattr(dl, name, mock)
    dl._get_source_overview_html = MagicMock(  # type: ignore[method-assign]
        return_value=_fixture("source_ok.html"),
        **({"side_effect": overrides["overview"]} if "overview" in overrides else {}),
    )
    dl._resolve_one_image_url = MagicMock(return_value="http://x/1")  # type: ignore[method-assign]
    dl.download_image_to_s3 = MagicMock(return_value=1)  # type: ignore[method-assign]
    return dl, mocks


def test_download_source_returns_login_failed_when_login_fails(fake_s3):
    dl, mocks = _failing_step_downloader(fake_s3)
    dl.logged_in = False
    with patch.object(dl, "login", return_value=False):
        res = dl.download_source(1, bucket_name="b", s3_prefix="p")

    assert res.ok is False
    assert res.errors == ["login failed"]
    dl._get_source_overview_html.assert_not_called()  # type: ignore[attr-defined]


def test_download_source_reports_overview_fetch_failure(fake_s3):
    dl, mocks = _failing_step_downloader(fake_s3, overview=requests.ConnectionError("down"))
    res = dl.download_source(1, bucket_name="b", s3_prefix="p")

    assert res.ok is False
    assert len(res.errors) == 1
    assert res.errors[0].startswith("overview fetch failed:")
    mocks["download_metadata"].assert_not_called()


def test_download_source_fails_when_the_overview_has_no_image_count(fake_s3):
    dl, mocks = _failing_step_downloader(fake_s3)
    dl._get_source_overview_html.return_value = "<html><body>No table</body></html>"  # type: ignore[attr-defined]

    res = dl.download_source(1, bucket_name="b", s3_prefix="p")

    assert res.ok is False
    assert res.errors == ["overview_total_missing"]
    mocks["download_metadata"].assert_not_called()
    assert fake_s3.objects == {}


@pytest.mark.parametrize(
    ("step", "failure", "code"),
    [
        ("download_metadata", (False, 0), "metadata_download_failed"),
    ],
)
def test_download_source_records_a_failed_step_and_still_runs_the_rest(
    fake_s3, step, failure, code
):
    dl, mocks = _failing_step_downloader(fake_s3, **{step: failure})
    res = dl.download_source(1, bucket_name="b", s3_prefix="p")

    assert res.ok is False
    assert code in res.errors
    mocks["download_annotations"].assert_called_once()
    dl.download_image_to_s3.assert_called_once()  # type: ignore[attr-defined]


def test_download_source_stops_when_the_walk_fails(fake_s3):
    dl, mocks = _failing_step_downloader(fake_s3, walk_source=requests.ConnectionError("down"))
    res = dl.download_source(1, bucket_name="b", s3_prefix="p")

    assert res.ok is False
    assert res.errors == ["image_list_failed"]
    mocks["download_annotations"].assert_not_called()


def test_download_source_passes_the_overview_points_per_image_to_the_export(fake_s3):
    dl, mocks = _failing_step_downloader(fake_s3)
    dl.download_source(1, bucket_name="b", s3_prefix="p")

    assert mocks["download_annotations"].call_args.kwargs["points_per_image"] == 10


def test_download_source_stops_after_annotations_fail(fake_s3):
    dl, mocks = _failing_step_downloader(fake_s3, download_annotations=False)
    res = dl.download_source(1, bucket_name="b", s3_prefix="p")

    assert res.ok is False
    assert res.errors == ["annotations_failed"]
    dl.download_image_to_s3.assert_not_called()  # type: ignore[attr-defined]


def test_download_source_reports_missing_url_when_single_image_lookup_raises(fake_s3):
    dl, _ = _failing_step_downloader(fake_s3)

    with patch.object(dl, "_resolve_one_image_url", side_effect=ValueError("private")):
        res = dl.download_source(1, bucket_name="b", s3_prefix="p")

    assert "images_missing_url" in res.errors
    assert res.image_failures == ImageFailures(missing_url=1, upload_failed=0)


# ---------------------------------------------------------------------------
# Image upload, metadata, labelset, permissions, login
# ---------------------------------------------------------------------------


def _http_response(
    text: str = "", status: int = 200, url: str = "", headers: dict[str, str] | None = None
) -> requests.Response:
    """A real ``requests.Response`` so ``raise_for_status`` behaves as in production."""
    resp = requests.Response()
    resp.status_code = status
    resp._content = text.encode()
    resp.url = url
    resp.headers.update(headers or {})
    return resp


def _downloader_on(fake_s3) -> CoralNetDownloader:
    dl = CoralNetDownloader("u", "p", max_workers=3)
    dl.logged_in = True
    dl.s3 = fake_s3
    return dl


def test_download_image_to_s3_retries_a_503_then_uploads_the_bytes(fake_s3, http_server):
    dl = _downloader_on(fake_s3)
    http_server.script("/1.jpg", Reply(503), Reply(200, _JPEG, _JPEG_HEADERS))

    size = dl.download_image_to_s3(f"{http_server.url}/1.jpg", "b", "k.jpg")

    assert size == len(_JPEG)
    assert fake_s3.objects[("b", "k.jpg")] == _JPEG


def test_download_image_to_s3_raises_on_a_404_and_uploads_nothing(fake_s3, http_server):
    dl = _downloader_on(fake_s3)
    http_server.script("/1.jpg", Reply(404, b"<html>Not found</html>"))

    with pytest.raises(requests.HTTPError):
        dl.download_image_to_s3(f"{http_server.url}/1.jpg", "b", "k.jpg")

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


def test_an_overview_outage_fails_with_one_overview_fetch_failure(http_server, fake_s3):
    http_server.script("/source/1/", Reply(503))
    dl = _site_downloader(http_server, fake_s3)

    res = dl.download_source(1, bucket_name="b", s3_prefix="p")

    assert res.ok is False
    assert len(res.errors) == 1
    assert res.errors[0].startswith("overview fetch failed")
    assert len(http_server.requests_to("/source/1/")) == 6  # one fetch: 1 try + 5 retries
    assert fake_s3.objects == {}


@pytest.mark.parametrize(
    ("status", "errors"),
    [(403, ["overview_forbidden"]), (404, ["cannot access source 1: HTTP 404"])],
)
def test_an_overview_answering_403_or_404_fails_with_its_code_and_is_not_retried(
    http_server, fake_s3, status, errors
):
    http_server.script("/source/1/", Reply(status))
    dl = _site_downloader(http_server, fake_s3)

    res = dl.download_source(1, bucket_name="b", s3_prefix="p")

    assert res.ok is False
    assert res.errors == errors
    assert len(http_server.requests_to("/source/1/")) == 1
    assert fake_s3.objects == {}


def test_a_permission_denied_overview_fails_without_writing(http_server, fake_s3):
    http_server.script("/source/1/", Reply(200, _fixture("source_denied.html").encode()))
    dl = _site_downloader(http_server, fake_s3)

    res = dl.download_source(1, bucket_name="b", s3_prefix="p")

    assert res.ok is False
    assert res.errors == ["cannot access source 1: Permission denied"]
    assert fake_s3.objects == {}


_LOGIN_URL = CoralNetDownloader.LOGIN_URL


def test_login_posts_the_credentials_with_the_login_pages_csrf_token():
    dl = CoralNetDownloader("alice", "secret")

    with (
        patch.object(dl.session, "get", return_value=_http_response(_login_get_html())),
        patch.object(
            dl.session, "post", return_value=_http_response("<html>Sign out</html>")
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


# ---------------------------------------------------------------------------
# Browse walk, staged download and restart, against a scripted CoralNet
# ---------------------------------------------------------------------------

_BASE = "/source/1/browse/images/"
_PREP = "/source/1/annotation/export_prep/"
_SERVE_T1 = "/source/1/export/serve/?session_data_timestamp=t1"
_LOGIN_PATH = "/accounts/login/"
_MARKER = "p/s1/_complete.json"
_PARTIAL = "p/s1/_partial/"


def _page_path(n: int) -> str:
    return _BASE if n == 1 else f"{_BASE}?page={n}"


def _script_pages(server: ScriptedServer, pages: list[list[tuple[int, str]]]) -> None:
    """Script the browse pages, each linking to the next by ``?page=<n>``."""
    for n, thumbs in enumerate(pages, 1):
        next_href = f"?page={n + 1}" if n < len(pages) else None
        server.script(_page_path(n), Reply(200, _browse_html(thumbs, next_href).encode()))


def _script_source(
    server: ScriptedServer, pages: list[list[int]], *, titles: dict[int, str] | None = None
) -> None:
    """Script a whole source: overview, labelset, browse pages, a bulk export, every image.

    Image ``<id>`` is titled ``img<id>.jpg`` unless ``titles`` names it, and has one annotation.
    """
    titles = {i: (titles or {}).get(i, f"img{i}.jpg") for page in pages for i in page}
    server.script("/source/1/", Reply(200, _overview_html(len(titles)).encode()))
    server.script("/source/1/labelset/", Reply(200, _LABEL_TABLE_HTML.encode()))
    _script_pages(server, [[(i, titles[i]) for i in page] for page in pages])
    server.script(_PREP, Reply(200, b'{"session_data_timestamp": "t1"}'))
    rows = sorted({(t, 1, 1, 7) for t in titles.values()})
    server.script(
        "/source/1/export/serve/?session_data_timestamp=t1",
        Reply(200, _ann_csv(rows).encode(), _CSV_HEADERS),
    )
    for i in titles:
        server.script(f"/image/{i}/view/", Reply(200, _view_page(server, f"/media/{i}.jpg")))
        server.script(f"/media/{i}.jpg", Reply(200, _JPEG, _JPEG_HEADERS))


def _view_page(server: ScriptedServer, media_path: str) -> bytes:
    """An image view page whose original image is at ``media_path`` on ``server``."""
    return f'<div id="original_image_container"><img src="{server.url}{media_path}"/>'.encode()


def _site_downloader(server: ScriptedServer, fake_s3) -> CoralNetDownloader:
    dl = CoralNetDownloader("u", "p", max_workers=2)
    dl.logged_in = True
    dl.s3 = fake_s3
    dl.CORALNET_URL = server.url
    dl.LOGIN_URL = f"{server.url}{_LOGIN_PATH}"
    dl._new_logged_in_session = lambda: dl.session  # type: ignore[method-assign]
    return dl


def _keys(fake_s3) -> list[str]:
    return sorted(k for (_b, k) in fake_s3.objects)


def _image_list_ids(fake_s3) -> list[int]:
    df = pd.read_csv(io.BytesIO(fake_s3.objects[("b", "p/s1/image_list.csv")]))
    return [int(page.split("/")[2]) for page in df["Image Page"]]


def test_a_clean_source_writes_its_marker_last_and_leaves_no_partial_state(http_server, fake_s3):
    _script_source(http_server, [[1, 2], [3]])
    dl = _site_downloader(http_server, fake_s3)

    res = dl.download_source(1, bucket_name="b", s3_prefix="p")

    assert (res.ok, res.errors) == (True, [])
    puts = [key for op, key in fake_s3.events if op == "put"]
    assert puts[-1] == _MARKER
    for key in ["p/s1/annotations.csv", "p/s1/image_list.csv"] + [
        f"p/s1/images/{i}.jpg" for i in (1, 2, 3)
    ]:
        assert key in puts[:-1]
    assert not [k for k in _keys(fake_s3) if k.startswith(_PARTIAL)]
    marker = json.loads(fake_s3.objects[("b", _MARKER)])
    assert (marker["total_images"], marker["warnings"]) == (3, [])


def _first_run_failing_the_export(server: ScriptedServer, fake_s3, pages: list[list[int]]):
    """Run a source whose export_prep answers 500, so the stages up to the walk are saved."""
    _script_source(server, pages)
    server.script(_PREP, Reply(500))
    res = _site_downloader(server, fake_s3).download_source(1, bucket_name="b", s3_prefix="p")
    assert res.errors == ["annotations_failed"]
    server.script(_PREP, Reply(200, b'{"session_data_timestamp": "t1"}'))
    server.received.clear()


def test_a_restart_after_the_walk_skips_the_walk_and_finishes_the_source(http_server, fake_s3):
    _first_run_failing_the_export(http_server, fake_s3, [[1, 2], [3]])

    res = _site_downloader(http_server, fake_s3).download_source(1, "b", "p")

    assert res.ok is True
    assert http_server.requests_to(_page_path(2)) == []
    assert len(http_server.requests_to(_BASE)) == 1  # the export's own form fetch
    assert http_server.requests_to("/source/1/labelset/") == []
    assert len(http_server.requests_to(_PREP)) == 1
    assert ("b", _MARKER) in fake_s3.objects


def test_a_restart_after_the_total_changed_discards_partial_state_and_walks_again(
    http_server, fake_s3
):
    _first_run_failing_the_export(http_server, fake_s3, [[1, 2], [3]])
    _script_source(http_server, [[1, 2], [3, 4]])

    res = _site_downloader(http_server, fake_s3).download_source(1, "b", "p")

    assert res.ok is True
    assert len(http_server.requests_to(_BASE)) == 2  # the walk's page 1, then the export
    assert len(http_server.requests_to(_page_path(2))) == 1
    assert len(http_server.requests_to("/source/1/labelset/")) == 1
    assert _image_list_ids(fake_s3) == [1, 2, 3, 4]
    assert json.loads(fake_s3.objects[("b", _MARKER)])["total_images"] == 4


def test_a_failed_verification_writes_no_marker_and_discards_partial_state(
    http_server, fake_s3, monkeypatch
):
    _script_source(http_server, [[1, 2]])
    failed = Verification(failures=["image_missing:1"])
    monkeypatch.setattr(downloader_mod, "verify_source", lambda *a, **k: failed)

    res = _site_downloader(http_server, fake_s3).download_source(1, "b", "p")

    assert res.ok is False
    assert res.errors == ["verification_failed"]
    assert ("b", _MARKER) not in fake_s3.objects
    assert not [k for k in _keys(fake_s3) if k.startswith(_PARTIAL)]


def test_a_failed_image_keeps_partial_state_and_a_rerun_fetches_only_that_image(
    http_server, fake_s3
):
    _script_source(http_server, [[1, 2]])
    http_server.script("/media/2.jpg", Reply(404))

    res = _site_downloader(http_server, fake_s3).download_source(1, "b", "p")

    assert res.ok is False
    assert "images_upload_failed" in res.errors
    assert "verification_failed" not in res.errors
    assert ("b", _MARKER) not in fake_s3.objects
    assert f"{_PARTIAL}state.json" in _keys(fake_s3)

    http_server.script("/media/2.jpg", Reply(200, _JPEG, _JPEG_HEADERS))
    http_server.received.clear()
    res = _site_downloader(http_server, fake_s3).download_source(1, "b", "p")

    assert res.ok is True
    assert http_server.requests_to(_page_path(1)) == []
    assert http_server.requests_to(_PREP) == []
    assert len(http_server.requests_to("/media/2.jpg")) == 1
    assert http_server.requests_to("/media/1.jpg") == []
    assert ("b", _MARKER) in fake_s3.objects


def test_a_failed_labelset_writes_no_marker_even_when_the_source_verifies(http_server, fake_s3):
    _script_source(http_server, [[1]])
    http_server.script("/source/1/labelset/", Reply(500))

    res = _site_downloader(http_server, fake_s3).download_source(1, "b", "p")

    assert res.errors == ["labelset_failed"]
    assert ("b", _MARKER) not in fake_s3.objects
    state = json.loads(fake_s3.objects[("b", f"{_PARTIAL}state.json")])
    assert state["stages"] == ["metadata", "walk", "export", "images"]


def test_an_interrupted_walk_resumes_after_its_last_checkpoint(http_server, fake_s3):
    _script_pages(http_server, [[(i, f"img{i}.jpg")] for i in range(1, 121)])
    http_server.script("/source/1/", Reply(200, _overview_html(120).encode()))
    http_server.script(_page_path(111), Reply(404))
    dl = _site_downloader(http_server, fake_s3)

    with pytest.raises(requests.HTTPError):
        dl.walk_source(1, "b", "p", total_images=120)
    saved = json.loads(fake_s3.objects[("b", f"{_PARTIAL}walk.json")])
    assert saved["last_page"] == 100

    _script_pages(http_server, [[(i, f"img{i}.jpg")] for i in range(1, 121)])
    http_server.received.clear()
    refs, total = dl.walk_source(1, "b", "p", total_images=120)

    walked = [r.path for r in http_server.received if r.path.startswith(_BASE)]
    assert walked == [_page_path(n) for n in range(101, 121)]
    assert [r.id for r in refs] == list(range(1, 121))
    assert total == 120


def test_two_images_with_the_same_title_are_both_listed(http_server, fake_s3):
    _script_source(http_server, [[1, 2]], titles={1: "same.jpg", 2: "same.jpg"})

    res = _site_downloader(http_server, fake_s3).download_source(1, "b", "p")

    assert res.ok is True
    assert _image_list_ids(fake_s3) == [1, 2]
    assert {f"p/s1/images/{i}.jpg" for i in (1, 2)} <= set(_keys(fake_s3))


def test_a_paginated_export_skips_pages_with_a_saved_chunk(http_server, fake_s3, monkeypatch):
    monkeypatch.setattr(downloader_mod, "_PAGE_DELAY_SECONDS", 0.0)
    for page in (1, 2, 3):
        chunk = _ann_csv([(f"img{page}.jpg", 1, 1, 7)])
        fake_s3.put("b", f"{_PARTIAL}page-{page}.csv", chunk.encode())
    _script_pages(http_server, [[(i, f"img{i}.jpg")] for i in range(1, 7)])
    http_server.script(
        _PREP, *[Reply(200, f'{{"session_data_timestamp": "t{p}"}}'.encode()) for p in (4, 5, 6)]
    )
    for page in (4, 5, 6):
        http_server.script(
            f"/source/1/export/serve/?session_data_timestamp=t{page}",
            Reply(200, _ann_csv([(f"img{page}.jpg", 1, 1, 7)]).encode(), _CSV_HEADERS),
        )
    dl = _site_downloader(http_server, fake_s3)

    ok = dl.download_annotations_paginated(
        1, "b", "p", refs=_refs({p: [p] for p in range(1, 7)}), page_workers=1
    )

    assert ok is True
    browsed = [r.path for r in http_server.received if r.path.startswith(_BASE)]
    assert browsed == [_page_path(n) for n in (4, 5, 6)]
    df = pd.read_csv(io.BytesIO(fake_s3.objects[_BULK_KEY]))
    assert df["Name"].tolist() == [f"img{p}.jpg" for p in range(1, 7)]


@pytest.mark.parametrize(
    ("prune", "dry_run", "orphan_kept", "warnings"),
    [
        (True, False, False, []),
        (False, False, True, ["orphan_image_objects:1"]),
        (True, True, True, ["orphan_image_objects:1"]),
    ],
    ids=["prune", "keep", "dry-run"],
)
def test_prune_deletes_orphan_objects_before_the_marker(
    http_server, fake_s3, caplog, prune, dry_run, orphan_kept, warnings
):
    caplog.set_level(logging.INFO)
    _script_source(http_server, [[1]])
    fake_s3.put("b", "p/s1/images/99.jpg", b"OLD")

    res = _site_downloader(http_server, fake_s3).download_source(
        1, "b", "p", prune=prune, dry_run=dry_run
    )

    assert res.ok is True
    assert (("b", "p/s1/images/99.jpg") in fake_s3.objects) is orphan_kept
    assert json.loads(fake_s3.objects[("b", _MARKER)])["warnings"] == warnings
    assert any("p/s1/images/99.jpg" in r.getMessage() for r in caplog.records) is dry_run
    if prune and not dry_run:
        assert fake_s3.events.index(("delete", "p/s1/images/99.jpg")) > fake_s3.events.index(
            ("put", "p/s1/images/1.jpg")
        )
        assert fake_s3.events.index(("delete", "p/s1/images/99.jpg")) < fake_s3.events.index(
            ("put", _MARKER)
        )


def test_verify_existing_marks_a_source_that_passes(http_server, fake_s3):
    _script_source(http_server, [[1]])
    fake_s3.put(
        "b", "p/s1/image_list.csv", b"Name,Image Page,Image URL\nimg1.jpg,/image/1/view/,\n"
    )
    fake_s3.put("b", "p/s1/annotations.csv", _ann_csv([("img1.jpg", 1, 1, 7)]).encode())
    fake_s3.put("b", "p/s1/images/1.jpg", b"JPEG1")

    res = _site_downloader(http_server, fake_s3).verify_existing(1, "b", "p")

    assert res.ok is True
    assert res.errors == []
    assert json.loads(fake_s3.objects[("b", _MARKER)])["total_images"] == 1


def test_verify_existing_writes_no_marker_for_a_source_that_fails(http_server, fake_s3):
    _script_source(http_server, [[1]])

    res = _site_downloader(http_server, fake_s3).verify_existing(1, "b", "p")

    assert res.ok is False
    assert res.errors == ["annotations_missing", "image_list_missing"]
    assert ("b", _MARKER) not in fake_s3.objects


@pytest.mark.parametrize(
    ("overview", "no_confirmed", "no_images"),
    [
        (_fixture("source_zero_confirmed.html"), True, False),
        (_overview_html(0), False, True),
    ],
    ids=["zero-confirmed", "zero-images"],
)
def test_verify_existing_skips_a_source_without_confirmed_images_or_images(
    http_server, fake_s3, overview, no_confirmed, no_images
):
    http_server.script("/source/1/", Reply(200, overview.encode()))

    res = _site_downloader(http_server, fake_s3).verify_existing(1, "b", "p")

    assert res.ok is True
    assert res.errors == []
    assert res.skipped_no_confirmed is no_confirmed
    assert res.skipped_no_images is no_images
    assert ("b", _MARKER) not in fake_s3.objects


def test_verify_existing_fails_a_source_whose_overview_lists_no_total(http_server, fake_s3):
    http_server.script("/source/1/", Reply(200, b"<html><body>No table</body></html>"))

    res = _site_downloader(http_server, fake_s3).verify_existing(1, "b", "p")

    assert res.ok is False
    assert res.errors == ["overview_total_missing"]


# ---------------------------------------------------------------------------
# Response validation: exports, walk rules, session health, login, images
# ---------------------------------------------------------------------------


def _posts_to(server: ScriptedServer, path: str) -> list:
    return [r for r in server.requests_to(path) if r.method == "POST"]


def _script_login(server: ScriptedServer) -> None:
    """Script a login page whose POST result shows a signed-in user."""
    page = _login_get_html() + "<a>Sign out</a>"
    server.script(_LOGIN_PATH, Reply(200, page.encode()))


def _csv_reply(rows: list[tuple]) -> Reply:
    return Reply(200, _ann_csv(rows).encode(), _CSV_HEADERS)


def _page_reply(ids: list[int], next_page: int | None = None, *, form: bool = True) -> Reply:
    """A browse page listing images ``ids`` titled ``img<id>.jpg``, linking to ``next_page``."""
    thumbs = [(i, f"img{i}.jpg") for i in ids]
    next_href = f"?page={next_page}" if next_page else None
    return Reply(200, _browse_html(thumbs, next_href, form=form).encode())


def _script_walk(server: ScriptedServer, pages: list[list[int]]) -> None:
    """Script the overview and browse pages of a source listing ``pages`` of image ids."""
    total = sum(len(ids) for ids in pages)
    server.script("/source/1/", Reply(200, _overview_html(total).encode()))
    for n, ids in enumerate(pages, 1):
        server.script(_page_path(n), _page_reply(ids, n + 1 if n < len(pages) else None))


def _fetches(server: ScriptedServer, pages: range) -> list[int]:
    return [len(server.requests_to(_page_path(n))) for n in pages]


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
    return dl.download_annotations_paginated(
        1, "b", "p", refs=[ImageRef(1, "img1.jpg - Confirmed", 1)], page_workers=1
    )


def _chunk_names(fake_s3) -> list[str]:
    chunk = pd.read_csv(io.BytesIO(fake_s3.objects[("b", f"{_PARTIAL}page-1.csv")]))
    return chunk["Name"].tolist()


def test_a_page_export_with_rows_for_other_images_is_exported_again(
    http_server, fake_s3, monkeypatch
):
    monkeypatch.setattr(downloader_mod, "_PAGE_DELAY_SECONDS", 0.0)
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
    monkeypatch.setattr(downloader_mod, "_PAGE_DELAY_SECONDS", 0.0)
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
    monkeypatch.setattr(downloader_mod, "_PAGE_DELAY_SECONDS", 0.0)
    _script_login(http_server)
    http_server.script(_page_path(1), _page_reply([1], form=False), _page_reply([1]))
    http_server.script(_PREP, Reply(200, b'{"session_data_timestamp": "t1"}'))
    http_server.script(_SERVE_T1, _csv_reply([("img1.jpg", 1, 1, 7)]))

    ok = _export_page_one(_site_downloader(http_server, fake_s3))

    assert ok is True
    assert len(_posts_to(http_server, _LOGIN_PATH)) == 1
    assert len(http_server.requests_to(_PREP)) == 1
    assert _chunk_names(fake_s3) == ["img1.jpg"]


def test_a_bulk_export_with_names_outside_the_walk_is_exported_again(http_server, fake_s3):
    _script_walk(http_server, [[1]])
    http_server.script(_PREP, Reply(200, b'{"session_data_timestamp": "t1"}'))
    http_server.script(
        _SERVE_T1,
        _csv_reply([("img1.jpg", 1, 1, 7), ("img9.jpg", 1, 1, 7)]),
        _csv_reply([("img1.jpg", 1, 1, 7)]),
    )

    ok = _site_downloader(http_server, fake_s3).download_annotations(
        1, "b", "p", refs=[ImageRef(1, "img1.jpg - Confirmed", 1)], total_images=1
    )

    assert ok is True
    annotations = pd.read_csv(io.BytesIO(fake_s3.objects[_BULK_KEY]))
    assert annotations["Name"].tolist() == ["img1.jpg"]
    assert len(http_server.requests_to(_SERVE_T1)) == 2


def test_a_page_rule_failure_resumes_the_walk_from_the_last_checkpoint(
    http_server, fake_s3, monkeypatch
):
    monkeypatch.setattr(downloader_mod, "_WALK_CHECKPOINT_EVERY_PAGES", 3)
    pages = [[2 * n - 1, 2 * n] for n in range(1, 9)] + [[17]]
    _script_walk(http_server, pages)
    http_server.script(_page_path(8), _page_reply([15], 9), _page_reply([15, 16], 9))

    refs, _ = _site_downloader(http_server, fake_s3).walk_source(1, "b", "p", total_images=17)

    assert [r.id for r in refs] == list(range(1, 18))
    assert _fetches(http_server, range(1, 10)) == [1, 1, 1, 1, 1, 1, 2, 2, 1]


def test_a_walk_ending_short_of_the_total_is_redone_from_page_one(
    http_server, fake_s3, monkeypatch
):
    monkeypatch.setattr(downloader_mod, "_WALK_CHECKPOINT_EVERY_PAGES", 1)
    _script_walk(http_server, [[1, 2], [3, 4], [5]])
    http_server.script(_page_path(2), _page_reply([5]), _page_reply([3, 4], 3))

    refs, _ = _site_downloader(http_server, fake_s3).walk_source(1, "b", "p", total_images=5)

    assert [r.id for r in refs] == [1, 2, 3, 4, 5]
    assert _fetches(http_server, range(1, 4)) == [2, 2, 1]


def test_an_image_listed_on_two_pages_makes_the_walk_retry(http_server, fake_s3, monkeypatch):
    monkeypatch.setattr(downloader_mod, "_WALK_CHECKPOINT_EVERY_PAGES", 1)
    _script_walk(http_server, [[1, 2], [3, 4], [5]])
    http_server.script(_page_path(2), _page_reply([2, 3], 3), _page_reply([3, 4], 3))

    refs, _ = _site_downloader(http_server, fake_s3).walk_source(1, "b", "p", total_images=5)

    assert [r.id for r in refs] == [1, 2, 3, 4, 5]
    # Caught on page 2 itself: the retry resumes after the page-1 checkpoint.
    assert _fetches(http_server, range(1, 4)) == [1, 2, 1]


def test_an_overview_total_changed_during_the_walk_redoes_it_against_the_new_total(
    http_server, fake_s3
):
    _script_source(http_server, [[1, 2, 3]])
    http_server.script(
        "/source/1/",
        Reply(200, _overview_html(2).encode()),
        Reply(200, _overview_html(3).encode()),
    )
    http_server.script(_page_path(1), _page_reply([1, 2]), _page_reply([1, 2, 3]))

    res = _site_downloader(http_server, fake_s3).download_source(1, "b", "p")

    assert (res.ok, res.errors) == (True, [])
    assert _image_list_ids(fake_s3) == [1, 2, 3]
    assert json.loads(fake_s3.objects[("b", _MARKER)])["total_images"] == 3


def test_a_kill_after_a_changed_total_walk_leaves_state_for_the_new_total(http_server, fake_s3):
    _script_source(http_server, [[1, 2, 3]])
    http_server.script(
        "/source/1/",
        Reply(200, _overview_html(2).encode()),
        Reply(200, _overview_html(3).encode()),
    )
    http_server.script(_page_path(1), _page_reply([1, 2]), _page_reply([1, 2, 3]))
    dl = _site_downloader(http_server, fake_s3)
    walk_source = dl.walk_source

    def killed_after_the_walk(*args: object, **kwargs: object) -> None:
        walk_source(*args, **kwargs)
        raise KeyboardInterrupt

    dl.walk_source = killed_after_the_walk  # type: ignore[method-assign]
    with pytest.raises(KeyboardInterrupt):
        dl.download_source(1, "b", "p")

    state = json.loads(fake_s3.objects[("b", f"{_PARTIAL}state.json")])
    assert state["total_images"] == 3

    http_server.received.clear()
    res = _site_downloader(http_server, fake_s3).download_source(1, "b", "p")

    assert res.ok is True
    assert len(http_server.requests_to(_BASE)) == 1  # the export's form page, not a walk fetch


@pytest.mark.parametrize(
    ("html", "expected"),
    [
        (
            '<span class="thumb_wrapper"><a href="/image/5/view/"><img/></a></span>',
            ([(5, "")], None, False),
        ),
        (_browse_html([(5, "a.jpg")]), ([(5, "a.jpg")], None, True)),
        (_browse_html([(5, "a.jpg")], "?page=2"), ([(5, "a.jpg")], "?page=2", True)),
    ],
    ids=["no-title", "no-next-anchor", "next-anchor"],
)
def test_parse_browse_thumbs_reads_titles_the_next_page_link_and_the_form(html, expected):
    assert _parse_browse_thumbs(html) == expected


def test_a_failed_page_export_starts_no_queued_page(http_server, fake_s3, monkeypatch):
    monkeypatch.setattr(downloader_mod, "_PAGE_DELAY_SECONDS", 0.0)
    _script_pages(http_server, [[(i, f"img{i}.jpg")] for i in (1, 2, 3)])
    http_server.script(_PREP, Reply(403))
    dl = _site_downloader(http_server, fake_s3)

    ok = dl.download_annotations_paginated(
        1, "b", "p", refs=_refs({p: [p] for p in (1, 2, 3)}), page_workers=1
    )

    assert ok is False
    # The lone worker may dequeue page 2 before the failure is seen; page 3 stays queued.
    assert len(http_server.requests_to(_PREP)) <= 2
    assert http_server.requests_to(_page_path(3)) == []


def test_a_walk_short_of_the_overview_total_on_every_attempt_is_walked_again_on_the_next_run(
    http_server, fake_s3
):
    _script_source(http_server, [[1, 2]])
    http_server.script("/source/1/", Reply(200, _overview_html(3).encode()))

    res = _site_downloader(http_server, fake_s3).download_source(1, "b", "p")

    assert (res.ok, res.errors) == (False, ["walk_count_mismatch:2!=3"])
    assert http_server.requests_to(_PREP) == []
    assert len(http_server.requests_to(_page_path(1))) == downloader_mod._WALK_ATTEMPTS
    assert ("b", "p/s1/image_list.csv") not in fake_s3.objects
    assert ("b", f"{_PARTIAL}walk.json") not in fake_s3.objects
    state = json.loads(fake_s3.objects[("b", f"{_PARTIAL}state.json")])
    assert "walk" not in state["stages"]

    _script_source(http_server, [[1, 2], [3]])
    http_server.received.clear()
    res = _site_downloader(http_server, fake_s3).download_source(1, "b", "p")

    assert res.ok is True
    assert len(http_server.requests_to(_page_path(1))) == 2  # the walk's page 1, then the export
    assert len(http_server.requests_to(_page_path(2))) == 1
    assert ("b", _MARKER) in fake_s3.objects


def test_a_browse_page_without_the_export_form_logs_in_again(http_server, fake_s3):
    _script_walk(http_server, [[1]])
    _script_login(http_server)
    http_server.script(_page_path(1), _page_reply([1], form=False), _page_reply([1]))

    refs, _ = _site_downloader(http_server, fake_s3).walk_source(1, "b", "p", total_images=1)

    assert [r.id for r in refs] == [1]
    assert len(_posts_to(http_server, _LOGIN_PATH)) == 1


def _login_downloader(server: ScriptedServer) -> CoralNetDownloader:
    dl = CoralNetDownloader("u", "p")
    dl.LOGIN_URL = f"{server.url}{_LOGIN_PATH}"
    return dl


def test_login_retries_a_post_answered_503(http_server):
    login_page = Reply(200, _login_get_html().encode())
    http_server.script(_LOGIN_PATH, login_page, Reply(503), Reply(200, b"<a>Sign out</a>"))
    dl = _login_downloader(http_server)

    assert dl.login() is True
    assert len(_posts_to(http_server, _LOGIN_PATH)) == 2


def test_login_fails_when_the_result_page_shows_no_sign_out_link(http_server):
    login_page = Reply(200, _login_get_html().encode())
    http_server.script(_LOGIN_PATH, login_page, Reply(200, b"<html>Welcome</html>"))
    dl = _login_downloader(http_server)

    assert dl.login() is False
    assert dl.logged_in is False
    assert len(_posts_to(http_server, _LOGIN_PATH)) == 1


def _download_image_one(server: ScriptedServer, fake_s3) -> ImageFailures:
    return _site_downloader(server, fake_s3).download_images(
        [ImageRef(1, "img1.jpg", 1)], 1, "b", "p"
    )


_IMAGE_KEY = ("b", "p/s1/images/1.jpg")


def test_an_image_body_cut_short_is_downloaded_again(http_server, fake_s3):
    http_server.script("/image/1/view/", Reply(200, _view_page(http_server, "/media/1.jpg")))
    short = {**_JPEG_HEADERS, "Content-Length": str(len(_JPEG))}
    http_server.script(
        "/media/1.jpg", Reply(200, _JPEG[:100], short), Reply(200, _JPEG, _JPEG_HEADERS)
    )

    assert _download_image_one(http_server, fake_s3) == ImageFailures()
    assert fake_s3.objects[_IMAGE_KEY] == _JPEG
    assert len(http_server.requests_to("/media/1.jpg")) == 2


@pytest.mark.parametrize(
    ("body", "content_type"),
    [
        (b"<html><body>Service unavailable</body></html>", "image/jpeg"),
        (b"", "image/jpeg"),
        (_JPEG, "text/html; charset=utf-8"),
    ],
    ids=["html", "empty", "html-content-type"],
)
def test_an_invalid_image_response_is_retried_and_never_uploaded(
    http_server, fake_s3, body, content_type
):
    http_server.script("/image/1/view/", Reply(200, _view_page(http_server, "/media/1.jpg")))
    http_server.script("/media/1.jpg", Reply(200, body, {"Content-Type": content_type}))

    assert _download_image_one(http_server, fake_s3) == ImageFailures(upload_failed=1)
    assert _IMAGE_KEY not in fake_s3.objects
    assert len(http_server.requests_to("/media/1.jpg")) == 4


def test_a_failed_image_is_logged_without_its_media_url_query(
    http_server, fake_s3, caplog: pytest.LogCaptureFixture
):
    signed = "/media/1.jpg?AWSAccessKeyId=A&Signature=S"
    http_server.script("/image/1/view/", Reply(200, _view_page(http_server, signed)))
    caplog.set_level(logging.DEBUG)

    assert _download_image_one(http_server, fake_s3) == ImageFailures(upload_failed=1)

    failed = [
        r.getMessage() for r in caplog.records if "Image 1 failed (upload_failed)" in r.getMessage()
    ]
    assert failed
    assert all("?<redacted>" in m for m in failed)
    assert not any("Signature" in r.getMessage() for r in caplog.records)


def test_an_expired_image_url_is_resolved_again_from_the_view_page(http_server, fake_s3):
    http_server.script(
        "/image/1/view/",
        Reply(200, _view_page(http_server, "/media/old.jpg")),
        Reply(200, _view_page(http_server, "/media/new.jpg")),
    )
    expired = b"<Error><Code>AccessDenied</Code><Message>Request has expired</Message></Error>"
    http_server.script("/media/old.jpg", Reply(403, expired, {"Content-Type": "application/xml"}))
    http_server.script("/media/new.jpg", Reply(200, _JPEG, _JPEG_HEADERS))

    assert _download_image_one(http_server, fake_s3) == ImageFailures()
    assert fake_s3.objects[_IMAGE_KEY] == _JPEG
    assert len(http_server.requests_to("/image/1/view/")) == 2


def test_a_jpeg_with_bytes_after_its_end_marker_is_uploaded_with_its_format(http_server, fake_s3):
    body = _JPEG + b"\x00trailing"
    http_server.script("/image/1/view/", Reply(200, _view_page(http_server, "/media/1.jpg")))
    http_server.script("/media/1.jpg", Reply(200, body, _JPEG_HEADERS))

    assert _download_image_one(http_server, fake_s3) == ImageFailures()
    assert fake_s3.objects[_IMAGE_KEY] == body
    assert fake_s3.headers[_IMAGE_KEY] == {
        "ContentType": "image/jpeg",
        "Metadata": {"source-format": "JPEG", "source-bytes": str(len(body))},
    }


def test_an_image_trickling_past_the_deadline_is_downloaded_again(
    http_server, fake_s3, monkeypatch
):
    monkeypatch.setattr(downloader_mod, "_IMAGE_DEADLINE_S", 0.2)
    http_server.script("/image/1/view/", Reply(200, _view_page(http_server, "/media/1.jpg")))
    http_server.script(
        "/media/1.jpg",
        Reply(200, _JPEG, _JPEG_HEADERS, trickle_s=0.001),
        Reply(200, _JPEG, _JPEG_HEADERS),
    )

    assert _download_image_one(http_server, fake_s3) == ImageFailures()
    assert fake_s3.objects[_IMAGE_KEY] == _JPEG
    assert len(http_server.requests_to("/media/1.jpg")) == 2


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
    monkeypatch.setattr(downloader_mod, "_PAGE_DELAY_SECONDS", 0.0)
    _script_walk(http_server, [[1]])
    http_server.script(
        _PREP, Reply(500), Reply(500), Reply(500), Reply(200, b'{"session_data_timestamp": "t1"}')
    )
    http_server.script(_SERVE_T1, _csv_reply([("img1.jpg", 1, 1, 7)]))

    ok = _export_page_one(_site_downloader(http_server, fake_s3))

    assert ok is True
    assert len(http_server.requests_to(_PREP)) == 4


def test_prune_deletes_nothing_when_verification_fails(http_server, fake_s3, monkeypatch):
    _script_source(http_server, [[1]])
    failed = Verification(failures=["image_missing:1"])
    monkeypatch.setattr(downloader_mod, "verify_source", lambda *a, **k: failed)
    fake_s3.put("b", "p/s1/images/99.jpg", b"OLD")

    res = _site_downloader(http_server, fake_s3).download_source(1, "b", "p", prune=True)

    assert res.errors == ["verification_failed"]
    assert ("delete", "p/s1/images/99.jpg") not in fake_s3.events
    assert ("b", "p/s1/images/99.jpg") in fake_s3.objects


def _overview_counting_two_images(server: ScriptedServer) -> None:
    server.script("/source/1/", Reply(200, _overview_html(2).encode()))


def _image_answering_404(server: ScriptedServer) -> None:
    server.script("/media/1.jpg", Reply(404))


@pytest.mark.parametrize(
    "break_source", [_image_answering_404, _overview_counting_two_images], ids=["404", "walk"]
)
def test_prune_deletes_nothing_when_a_stage_before_verification_fails(
    http_server, fake_s3, break_source
):
    _script_source(http_server, [[1]])
    break_source(http_server)
    fake_s3.put("b", "p/s1/images/99.jpg", b"OLD")

    res = _site_downloader(http_server, fake_s3).download_source(1, "b", "p", prune=True)

    assert res.ok is False
    assert ("b", "p/s1/images/99.jpg") in fake_s3.objects
    assert not [k for op, k in fake_s3.events if op == "delete" and "/images/" in k]
    assert ("b", _MARKER) not in fake_s3.objects


def test_an_export_fails_when_logging_in_again_fails(http_server, fake_s3, monkeypatch):
    monkeypatch.setattr(downloader_mod, "_PAGE_DELAY_SECONDS", 0.0)
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
    monkeypatch.setattr(downloader_mod, "_PAGE_DELAY_SECONDS", 0.0)
    _script_walk(http_server, [[1]])
    _script_login(http_server)
    http_server.script(
        _PREP, Reply(200, b'{"error": "You must be signed in to access this function."}')
    )

    ok = _export_page_one(_site_downloader(http_server, fake_s3))

    assert ok is False
    assert len(http_server.requests_to(_PREP)) == 4
    assert ("b", f"{_PARTIAL}page-1.csv") not in fake_s3.objects


def test_login_fails_after_four_posts_that_all_fail(http_server):
    http_server.script(_LOGIN_PATH, Reply(200, _login_get_html().encode()), Reply(503))
    dl = _login_downloader(http_server)

    assert dl.login() is False
    assert dl.logged_in is False
    assert len(_posts_to(http_server, _LOGIN_PATH)) == 4


def _noise_image_bytes(fmt: str) -> bytes:
    """A 256x256 random-pixel image, large enough that cutting it in half drops scan data."""
    pixels = random.Random(0).randbytes(256 * 256 * 3)
    image = Image.frombytes("RGB", (256, 256), pixels)
    buf = io.BytesIO()
    if fmt == "MPO":  # a single-frame MPO save is written as a plain JPEG
        image.save(buf, fmt, save_all=True, append_images=[image])
    else:
        image.save(buf, fmt)
    return buf.getvalue()


_NOISE_JPEG = _noise_image_bytes("JPEG")
_NOISE_PNG = _noise_image_bytes("PNG")
_NOISE_GIF = _noise_image_bytes("GIF")


def _serve_image(server: ScriptedServer, body: bytes, headers: dict[str, str]) -> None:
    server.script("/image/1/view/", Reply(200, _view_page(server, "/media/1.jpg")))
    server.script("/media/1.jpg", Reply(200, body, headers))


def test_an_octet_stream_jpeg_is_stored_as_image_jpeg(http_server, fake_s3):
    _serve_image(http_server, _NOISE_JPEG, {"Content-Type": "binary/octet-stream"})

    assert _download_image_one(http_server, fake_s3) == ImageFailures()
    assert fake_s3.headers[_IMAGE_KEY]["ContentType"] == "image/jpeg"


def test_a_png_served_as_image_jpeg_is_stored_with_the_detected_format(http_server, fake_s3):
    _serve_image(http_server, _NOISE_PNG, _JPEG_HEADERS)

    assert _download_image_one(http_server, fake_s3) == ImageFailures()
    assert fake_s3.objects[_IMAGE_KEY] == _NOISE_PNG
    assert fake_s3.headers[_IMAGE_KEY] == {
        "ContentType": "image/png",
        "Metadata": {"source-format": "PNG", "source-bytes": str(len(_NOISE_PNG))},
    }


# Content-Encoding makes the downloader skip its Content-Length check, so the body's own
# decodability is the only thing that can reject the cut image.
@pytest.mark.parametrize(
    ("body", "headers"),
    [
        (
            _NOISE_JPEG[: len(_NOISE_JPEG) // 2],
            {**_JPEG_HEADERS, "Content-Encoding": "identity"},
        ),
        (_NOISE_GIF, {"Content-Type": "image/gif"}),
    ],
    ids=["jpeg-cut-no-length-check", "gif"],
)
def test_an_undecodable_or_unsupported_image_is_retried_and_never_uploaded(
    http_server, fake_s3, body, headers
):
    _serve_image(http_server, body, headers)

    assert _download_image_one(http_server, fake_s3) == ImageFailures(upload_failed=1)
    assert _IMAGE_KEY not in fake_s3.objects
    assert len(http_server.requests_to("/media/1.jpg")) == 4


@pytest.mark.parametrize(
    ("body", "headers", "accepted"),
    [
        (_NOISE_JPEG, _JPEG_HEADERS, True),
        (_NOISE_JPEG + b"\x00", {**_JPEG_HEADERS, "Content-Encoding": "identity"}, False),
    ],
    ids=["exactly-the-cap", "streamed-over-the-cap"],
)
def test_an_image_body_over_the_size_cap_is_rejected(
    http_server, fake_s3, monkeypatch, body, headers, accepted
):
    monkeypatch.setattr(downloader_mod, "_IMAGE_MAX_BYTES", len(_NOISE_JPEG))
    _serve_image(http_server, body, headers)

    failures = _download_image_one(http_server, fake_s3)

    assert failures == ImageFailures(upload_failed=0 if accepted else 1)
    assert (_IMAGE_KEY in fake_s3.objects) is accepted


class _NeverStreamed:
    """A response declaring a body of 11 bytes whose body must not be read."""

    headers = {"Content-Length": "11"}  # noqa: RUF012

    def iter_content(self, chunk_size: int) -> Iterator[bytes]:
        raise AssertionError("body read despite a declared length over the cap")


def test_a_declared_length_over_the_cap_is_rejected_before_any_body_is_read(monkeypatch):
    monkeypatch.setattr(downloader_mod, "_IMAGE_MAX_BYTES", 10)

    with pytest.raises(ImageInvalid, match="11"):
        _read_image_body(_NeverStreamed(), deadline=float("inf"))  # type: ignore[arg-type]


@pytest.mark.parametrize("fmt", ["JPEG", "PNG", "MPO"])
def test_validate_image_bytes_returns_the_format_and_leaves_the_buffer_at_the_start(fmt):
    data = io.BytesIO(_noise_image_bytes(fmt))

    assert validate_image_bytes(data) == fmt
    assert data.tell() == 0


def test_validate_image_bytes_rejects_an_empty_buffer():
    with pytest.raises(ImageInvalid, match="empty body"):
        validate_image_bytes(io.BytesIO())


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
        ok = dl.download_annotations_paginated(1, "b", "p", refs=refs, page_workers=1)

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
        ok = dl.download_annotations_paginated(1, "b", "p", refs=refs, page_workers=1)

    assert ok is True
    assert served[2] == 1
    assert _saved_names(fake_s3, _BULK_KEY) == ["img1.jpg"]


def test_a_confirmed_page_that_stays_header_only_fails_the_export(paginated_downloader, fake_s3):
    dl = paginated_downloader
    refs = _status_refs({1: "Confirmed", 2: "Confirmed"})
    served, get, post = _serve_by_page({1: [_rows(1)], 2: [_HEADER_ONLY]})

    with _mock_paginated_session(dl, get=get, post=post):
        ok = dl.download_annotations_paginated(1, "b", "p", refs=refs, page_workers=1)

    assert ok is False
    assert served[2] > 1
    assert _BULK_KEY not in fake_s3.objects


def test_a_header_only_bulk_export_with_a_confirmed_image_is_exported_again(
    paginated_downloader, fake_s3
):
    dl = paginated_downloader
    refs = _status_refs({1: "Confirmed"})
    served, get, post = _serve_by_page({1: [_HEADER_ONLY, _rows(1)]})

    with _mock_paginated_session(dl, get=get, post=post):
        ok = dl.download_annotations(1, "b", "p", refs=refs, total_images=1, points_per_image=1)

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
        ok = dl.download_annotations_paginated(1, "b", "p", refs=refs, page_workers=1)

    assert ok is True
    assert served == {1: 1, 2: 0}
    assert _saved_names(fake_s3, _BULK_KEY) == ["img1.jpg", "img2.jpg"]
