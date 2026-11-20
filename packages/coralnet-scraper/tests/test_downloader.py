"""Offline tests for CoralNetDownloader."""

from __future__ import annotations

import io
import json
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
from conftest import Reply, ScriptedServer

from coralnet_scraper import downloader as downloader_mod
from coralnet_scraper.downloader import (
    _PAGINATED_EXPORT_THRESHOLD,
    CoralNetDownloader,
    ImageRef,
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
    dl = CoralNetDownloader("u", "p", max_workers=2)
    df = pd.DataFrame(
        {"Image Page": [f"/image/{i}/view/" for i in (1, 2, 3)], "Image URL": [pd.NA] * 3}
    )
    assert dl.download_images(df, 1, bucket_name="b") == ImageFailures(
        missing_url=3, upload_failed=0
    )


def test_get_image_urls_serial_path_keeps_order_around_a_failure():
    dl = CoralNetDownloader("u", "p", max_workers=1)

    def resolve(rel: str) -> str:
        if rel == "/image/2/view/":
            raise ValueError("private")
        return "http://x" + rel

    with patch.object(dl, "_resolve_one_image_url", side_effect=resolve):
        urls = dl.get_image_urls(["/image/1/view/", "/image/2/view/", "/image/3/view/"])
    assert urls == ["http://x/image/1/view/", None, "http://x/image/3/view/"]


_BROWSE_HTML_PAGE = """\
<html>
<form id="export-annotations-prep-form">
  <input type="hidden" name="csrfmiddlewaretoken" value="{csrf}"/>
</form>
{thumbs}
{next_link}
</html>"""


def _browse_html(
    thumbs: list[tuple[int, str]], next_href: str | None = None, csrf: str = "tok"
) -> str:
    """A browse page with one thumbnail per ``(image id, title)``."""
    spans = "".join(
        f'<span class="thumb_wrapper"><a href="/image/{i}/view/"><img title="{t}"/></a></span>'
        for i, t in thumbs
    )
    next_link = f'<a title="Next page" href="{next_href}">Next</a>' if next_href else ""
    return _BROWSE_HTML_PAGE.format(csrf=csrf, thumbs=spans, next_link=next_link)


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
    browse_html: str,
    serve_text: str = "",
    export_prep: list | None = None,
    total: int = _PAGINATED_EXPORT_THRESHOLD,
) -> _BulkRun:
    """Run the bulk annotations path on a small source.

    ``export_prep`` lists the export_prep POST outcomes in order (responses or exceptions).
    """
    dl = CoralNetDownloader("u", "p")
    dl.logged_in = True
    dl.s3 = fake_s3
    gets = [_mock_response(text=browse_html), _mock_response(text=serve_text)]
    if export_prep is None:
        export_prep = [_mock_response(json_data={"session_data_timestamp": "t1"})]
    with (
        patch.object(dl.session, "get", side_effect=gets) as get_mock,
        patch.object(dl.session, "post", side_effect=export_prep) as post_mock,
        patch("coralnet_scraper.downloader.time.sleep") as sleep_mock,
    ):
        ok = dl.download_annotations(1, "b", "p", refs=[], total_images=total)
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
        total=4000,
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
        total=4000,
    )

    assert run.ok is False
    assert run.prep_timeouts == [320, 620, 920, 1220, 1520]
    assert run.sleep.call_args_list == [call(60), call(120), call(180), call(240)]
    assert _BULK_KEY not in fake_s3.objects


def test_download_annotations_sizes_timeouts_from_the_total(fake_s3):
    run = _run_bulk_download(
        fake_s3, browse_html=_BULK_BROWSE_HTML, serve_text=_SERVE_CSV, total=4000
    )

    assert run.ok is True
    assert run.get.call_args_list[0].args == (_BROWSE_URL,)
    assert run.prep_timeouts == [320]  # 120 + 4000 // 20
    serve_get = run.get.call_args_list[-1]
    assert serve_get.args == (
        "https://coralnet.ucsd.edu/source/1/export/serve/?session_data_timestamp=t1",
    )
    assert serve_get.kwargs["timeout"] == 140  # 60 + 4000 // 50


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


def _page_script(page_htmls: list[str], ann_csvs: list[str]) -> tuple[list, list]:
    """GET and POST responses for a sequential (one-worker) paginated export.

    The GET sequence is, per page, a browse refresh and the serve response; the POST
    sequence is one export_prep response per page.
    """
    gets = []
    for html, ann in zip(page_htmls, ann_csvs, strict=True):
        gets += [_mock_response(text=html), _mock_response(text=ann)]
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
        _ann_csv([("a.jpg", 1, 1, 100), ("b.jpg", 2, 2, 101)]),
        _ann_csv([("c.jpg", 3, 3, 102)]),
        _ann_csv([("d.jpg", 4, 4, 103), ("e.jpg", 5, 5, 104)]),
    ]
    gets, posts = _page_script(page_htmls, ann_csvs)

    with _mock_paginated_session(dl, get=gets, post=posts) as mock_post:
        ok = dl.download_annotations_paginated(1, "b", "p", refs=refs, page_workers=1)

    assert ok is True
    df = pd.read_csv(io.BytesIO(fake_s3.objects[_BULK_KEY]))
    assert df["Name"].tolist() == ["a.jpg", "b.jpg", "c.jpg", "d.jpg", "e.jpg"]
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
        ok = dl.download_annotations_paginated(
            1, "b", "p", refs=_refs({1: [100], 2: [200], 3: [300]}), page_workers=3
        )

    assert ok is True
    df = pd.read_csv(io.BytesIO(fake_s3.objects[_BULK_KEY]))
    assert df["Name"].tolist() == ["a.jpg", "b.jpg", "c.jpg"]


def test_paginated_export_runs_export_workers_page_round_trips_at_once(fake_s3):
    dl = CoralNetDownloader("u", "p", export_workers=2)
    dl.logged_in = True
    dl.s3 = fake_s3
    dl._new_logged_in_session = lambda: dl.session  # type: ignore[method-assign]
    lock = threading.Lock()
    in_flight = [0]
    peak = [0]

    def get(url: str, timeout: int | None = None, **_: object) -> MagicMock:
        if "/export/serve/" not in url:
            page = int(url.rsplit("=", 1)[1]) if "?page=" in url else 1
            return _mock_response(text=_browse_page(f"csrf{page}"))
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

    refs = _refs({page: [page * 100] for page in range(1, 7)})
    with _mock_paginated_session(dl, get=get, post=post):
        ok = dl.download_annotations_paginated(1, "b", "p", refs=refs)

    assert ok is True
    assert peak[0] == 2


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
        "walk_source": _refs({1: [1]}),
    }
    mocks = {
        name: MagicMock(return_value=overrides.get(name, default))
        for name, default in defaults.items()
    }
    if "walk_source" in overrides and isinstance(overrides["walk_source"], Exception):
        mocks["walk_source"].side_effect = overrides["walk_source"]
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
    dl.get_image_urls.assert_called_once()  # type: ignore[attr-defined]


def test_download_source_stops_when_the_walk_fails(fake_s3):
    dl, mocks = _failing_step_downloader(fake_s3, walk_source=requests.ConnectionError("down"))
    res = dl.download_source(1, bucket_name="b", s3_prefix="p")

    assert res.ok is False
    assert res.errors == ["image_list_failed"]
    mocks["download_annotations"].assert_not_called()


def test_download_source_stops_after_annotations_fail(fake_s3):
    dl, mocks = _failing_step_downloader(fake_s3, download_annotations=False)
    res = dl.download_source(1, bucket_name="b", s3_prefix="p")

    assert res.ok is False
    assert res.errors == ["annotations_failed"]
    dl.get_image_urls.assert_not_called()  # type: ignore[attr-defined]


def test_download_source_sums_image_failures_across_batches(fake_s3):
    dl, _ = _failing_step_downloader(fake_s3, walk_source=_refs({1: [1, 2]}))
    dl.get_image_urls = MagicMock(  # type: ignore[method-assign]
        side_effect=lambda pages: ["http://x/" + p.split("/")[2] for p in pages]
    )
    dl.download_image_to_s3 = MagicMock(  # type: ignore[method-assign]
        side_effect=lambda u, b, k: (k, not u.endswith("/1"))  # only the first batch's fails
    )

    with patch.object(downloader_mod, "half_open_batches", return_value=[(0, 1), (1, 2)]):
        res = dl.download_source(1, bucket_name="b", s3_prefix="p")

    assert "images_upload_failed" in res.errors
    assert res.image_failures == ImageFailures(missing_url=0, upload_failed=1)


def test_download_source_reports_missing_url_when_single_image_lookup_raises(fake_s3):
    dl, _ = _failing_step_downloader(fake_s3)
    del dl.get_image_urls

    with patch.object(dl, "_resolve_one_image_url", side_effect=ValueError("private")):
        res = dl.download_source(1, bucket_name="b", s3_prefix="p")

    assert "images_missing_url" in res.errors
    assert res.image_failures == ImageFailures(missing_url=1, upload_failed=0)


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


def test_download_image_to_s3_replaces_a_zero_byte_object(fake_s3, http_server):
    dl = _downloader_on(fake_s3)
    fake_s3.put("b", "k.jpg", b"")
    http_server.script("/1.jpg", Reply(200, b"JPEGBYTES"))

    result = dl.download_image_to_s3(f"{http_server.url}/1.jpg", "b", "k.jpg")

    assert result == ("k.jpg", True)
    assert fake_s3.objects[("b", "k.jpg")] == b"JPEGBYTES"


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


def test_an_overview_outage_fails_with_one_overview_fetch_failure(http_server, fake_s3):
    http_server.script("/source/1/", Reply(503))
    dl = _site_downloader(http_server, fake_s3)

    res = dl.download_source(1, bucket_name="b", s3_prefix="p")

    assert res.ok is False
    assert len(res.errors) == 1
    assert res.errors[0].startswith("overview fetch failed")
    assert len(http_server.requests_to("/source/1/")) == 6  # one fetch: 1 try + 5 retries
    assert fake_s3.objects == {}


@pytest.mark.parametrize("status", [403, 404])
def test_an_overview_answering_403_or_404_is_inaccessible_and_not_retried(
    http_server, fake_s3, status
):
    http_server.script("/source/1/", Reply(status))
    dl = _site_downloader(http_server, fake_s3)

    res = dl.download_source(1, bucket_name="b", s3_prefix="p")

    assert res.ok is False
    assert res.errors == [f"cannot access source 1: HTTP {status}"]
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


def test_export_one_page_raises_when_the_worker_session_gets_no_csrf(paginated_downloader):
    dl = paginated_downloader

    with (
        patch.object(dl.session, "get", return_value=_http_response("<html>Please log in</html>")),
        pytest.raises(RuntimeError, match="Worker session: no CSRF on browse page 4 of source 1"),
    ):
        dl._export_one_page(1, 4, (5,))


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


# ---------------------------------------------------------------------------
# Browse walk, staged download and restart, against a scripted CoralNet
# ---------------------------------------------------------------------------

_BASE = "/source/1/browse/images/"
_PREP = "/source/1/annotation/export_prep/"
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
        "/source/1/export/serve/?session_data_timestamp=t1", Reply(200, _ann_csv(rows).encode())
    )
    for i in titles:
        original = f'<div id="original_image_container"><img src="{server.url}/media/{i}.jpg"/>'
        server.script(f"/image/{i}/view/", Reply(200, original.encode()))
        server.script(f"/media/{i}.jpg", Reply(200, f"JPEG{i}".encode()))


def _site_downloader(server: ScriptedServer, fake_s3) -> CoralNetDownloader:
    dl = CoralNetDownloader("u", "p", max_workers=2)
    dl.logged_in = True
    dl.s3 = fake_s3
    dl.CORALNET_URL = server.url
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


def test_a_failed_verification_writes_no_marker_and_discards_partial_state(http_server, fake_s3):
    _script_source(http_server, [[1, 2]])
    http_server.script("/source/1/", Reply(200, _overview_html(3).encode()))

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

    http_server.script("/media/2.jpg", Reply(200, b"JPEG2"))
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
    http_server.script(_page_path(111), Reply(500))
    dl = _site_downloader(http_server, fake_s3)

    with pytest.raises(requests.HTTPError):
        dl.walk_source(1, "b", "p")
    saved = json.loads(fake_s3.objects[("b", f"{_PARTIAL}walk.json")])
    assert saved["last_page"] == 100

    _script_pages(http_server, [[(i, f"img{i}.jpg")] for i in range(1, 121)])
    http_server.received.clear()
    refs = dl.walk_source(1, "b", "p")

    assert [r.path for r in http_server.received] == [_page_path(n) for n in range(101, 121)]
    assert [r.id for r in refs] == list(range(1, 121))


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
            Reply(200, _ann_csv([(f"img{page}.jpg", 1, 1, 7)]).encode()),
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
    ("prune", "orphan_kept", "warnings"),
    [(True, False, []), (False, True, ["orphan_image_objects:1"])],
    ids=["prune", "keep"],
)
def test_prune_deletes_orphan_objects_before_the_marker(
    http_server, fake_s3, prune, orphan_kept, warnings
):
    _script_source(http_server, [[1]])
    fake_s3.put("b", "p/s1/images/99.jpg", b"OLD")

    res = _site_downloader(http_server, fake_s3).download_source(1, "b", "p", prune=prune)

    assert res.ok is True
    assert (("b", "p/s1/images/99.jpg") in fake_s3.objects) is orphan_kept
    assert json.loads(fake_s3.objects[("b", _MARKER)])["warnings"] == warnings
    if prune:
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

    failures = _site_downloader(http_server, fake_s3).verify_existing(1, "b", "p")

    assert failures == []
    assert json.loads(fake_s3.objects[("b", _MARKER)])["total_images"] == 1


def test_verify_existing_writes_no_marker_for_a_source_that_fails(http_server, fake_s3):
    _script_source(http_server, [[1]])

    failures = _site_downloader(http_server, fake_s3).verify_existing(1, "b", "p")

    assert failures == ["annotations_missing", "image_list_missing"]
    assert ("b", _MARKER) not in fake_s3.objects
