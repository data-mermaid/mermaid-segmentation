"""Offline tests for the CoralNetDownloader stage driver."""

from __future__ import annotations

import io
import json
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import requests
from botocore.exceptions import ClientError
from conftest import (
    _BASE,
    _CSV_HEADERS,
    _JPEG,
    _JPEG_HEADERS,
    _PARTIAL,
    _PREP,
    Reply,
    ScriptedServer,
    _ann_csv,
    _downloader_on,
    _http_response,
    _overview_html,
    _page_path,
    _page_reply,
    _refs,
    _script_pages,
    _site_downloader,
    _view_page,
)

from coralnet_scraper import downloader as downloader_mod
from coralnet_scraper import walker as walker_mod
from coralnet_scraper.downloader import CoralNetDownloader
from coralnet_scraper.models import ImageFailures
from coralnet_scraper.verify import Verification

_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "coralnet_html"


def _fixture(name: str) -> str:
    return (_FIXTURES / name).read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("failures", "codes"),
    [
        (ImageFailures(missing_url=20, skipped=5), ["images_missing_url", "images_outage"]),
        (ImageFailures(upload_failed=1), ["images_upload_failed"]),
        (ImageFailures(), []),
    ],
    ids=["missing-url-and-outage", "upload-failed", "none"],
)
def test_image_failure_codes_name_each_kind_of_failure(failures, codes):
    assert downloader_mod._image_failure_codes(failures) == codes


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
    dl = CoralNetDownloader("u", "p", s3=fake_s3)
    dl.auth.logged_in = True
    with (
        patch.object(dl.auth, "_get_source_overview_html", return_value=overview),
        patch.object(dl, "download_metadata") as mock_meta,
        patch.object(dl.walker, "walk_source") as mock_walk,
    ):
        res = dl.download_source(5961, bucket_name="b", s3_prefix="p")
    assert res.ok is True
    assert res.errors == []
    assert res.skipped_no_confirmed is no_confirmed
    assert res.skipped_no_images is no_images
    mock_meta.assert_not_called()
    mock_walk.assert_not_called()
    assert fake_s3.objects == {}


def test_constructor_rejects_invalid_max_workers():
    with pytest.raises(ValueError, match="workers"):
        CoralNetDownloader("u", "p", max_workers=0)


def _failing_step_downloader(
    fake_s3, **overrides
) -> tuple[CoralNetDownloader, dict[str, MagicMock]]:
    """A downloader whose steps succeed except those overridden; returns it with its mocks."""
    dl = CoralNetDownloader("u", "p", max_workers=2, s3=fake_s3)
    dl.auth.logged_in = True
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
    owners = {
        "download_metadata": dl,
        "download_labelset": dl,
        "download_annotations": dl.exporter,
        "walk_source": dl.walker,
    }
    for name, mock in mocks.items():
        setattr(owners[name], name, mock)
    dl.auth._get_source_overview_html = MagicMock(  # type: ignore[method-assign]
        return_value=_fixture("source_ok.html"),
        **({"side_effect": overrides["overview"]} if "overview" in overrides else {}),
    )
    dl.images._resolve_one_image_url = MagicMock(return_value="http://x/1")  # type: ignore[method-assign]
    dl.images.download_image_to_s3 = MagicMock(return_value=1)  # type: ignore[method-assign]
    return dl, mocks


def test_download_source_returns_login_failed_when_login_fails(fake_s3):
    dl, mocks = _failing_step_downloader(fake_s3)
    dl.auth.logged_in = False
    with patch.object(dl, "login", return_value=False):
        res = dl.download_source(1, bucket_name="b", s3_prefix="p")

    assert res.ok is False
    assert res.errors == ["login failed"]
    dl.auth._get_source_overview_html.assert_not_called()  # type: ignore[attr-defined]


def test_download_source_reports_overview_fetch_failure(fake_s3):
    dl, mocks = _failing_step_downloader(fake_s3, overview=requests.ConnectionError("down"))
    res = dl.download_source(1, bucket_name="b", s3_prefix="p")

    assert res.ok is False
    assert len(res.errors) == 1
    assert res.errors[0].startswith("overview fetch failed:")
    mocks["download_metadata"].assert_not_called()


def test_download_source_fails_when_the_overview_has_no_image_count(fake_s3):
    dl, mocks = _failing_step_downloader(fake_s3)
    dl.auth._get_source_overview_html.return_value = "<html><body>No table</body></html>"  # type: ignore[attr-defined]

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
    dl.images.download_image_to_s3.assert_called_once()  # type: ignore[attr-defined]


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
    dl.images.download_image_to_s3.assert_not_called()  # type: ignore[attr-defined]


def test_download_source_reports_missing_url_when_single_image_lookup_raises(fake_s3):
    dl, _ = _failing_step_downloader(fake_s3)

    with patch.object(dl.images, "_resolve_one_image_url", side_effect=ValueError("private")):
        res = dl.download_source(1, bucket_name="b", s3_prefix="p")

    assert "images_missing_url" in res.errors
    assert res.image_failures == ImageFailures(missing_url=1, upload_failed=0)


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


_MARKER = "p/s1/_complete.json"


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


def test_two_images_with_the_same_title_are_both_listed(http_server, fake_s3):
    _script_source(http_server, [[1, 2]], titles={1: "same.jpg", 2: "same.jpg"})

    res = _site_downloader(http_server, fake_s3).download_source(1, "b", "p")

    assert res.ok is True
    assert _image_list_ids(fake_s3) == [1, 2]
    assert {f"p/s1/images/{i}.jpg" for i in (1, 2)} <= set(_keys(fake_s3))


@pytest.mark.parametrize(
    ("prune", "dry_run", "orphan_kept", "warnings"),
    [
        (True, False, False, []),
        (False, False, True, ["orphan_image_objects:1"]),
        (True, True, True, None),
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
    if dry_run:
        assert ("b", _MARKER) not in fake_s3.objects
        assert any(k.startswith("p/s1/_partial/") for _, k in fake_s3.objects)
    else:
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
    walk_source = dl.walker.walk_source

    def killed_after_the_walk(*args: object, **kwargs: object) -> None:
        walk_source(*args, **kwargs)
        raise KeyboardInterrupt

    dl.walker.walk_source = killed_after_the_walk  # type: ignore[method-assign]
    with pytest.raises(KeyboardInterrupt):
        dl.download_source(1, "b", "p")

    state = json.loads(fake_s3.objects[("b", f"{_PARTIAL}state.json")])
    assert state["total_images"] == 3

    http_server.received.clear()
    res = _site_downloader(http_server, fake_s3).download_source(1, "b", "p")

    assert res.ok is True
    assert len(http_server.requests_to(_BASE)) == 1  # the export's form page, not a walk fetch


def test_a_walk_short_of_the_overview_total_on_every_attempt_is_walked_again_on_the_next_run(
    http_server, fake_s3
):
    _script_source(http_server, [[1, 2]])
    http_server.script("/source/1/", Reply(200, _overview_html(3).encode()))

    res = _site_downloader(http_server, fake_s3).download_source(1, "b", "p")

    assert (res.ok, res.errors) == (False, ["walk_count_mismatch:2!=3"])
    assert http_server.requests_to(_PREP) == []
    assert len(http_server.requests_to(_page_path(1))) == walker_mod._WALK_ATTEMPTS
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
