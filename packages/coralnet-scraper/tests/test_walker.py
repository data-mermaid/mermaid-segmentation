"""Offline tests for SourceWalker and the browse-page helpers."""

from __future__ import annotations

import json

import pytest
import requests
from conftest import (
    _BASE,
    _LOGIN_PATH,
    _PARTIAL,
    Reply,
    ScriptedServer,
    _browse_html,
    _overview_html,
    _page_path,
    _page_reply,
    _posts_to,
    _script_login,
    _script_pages,
    _script_walk,
    _site_downloader,
)

from coralnet_scraper import walker as walker_mod
from coralnet_scraper.walker import _parse_browse_thumbs


def test_an_interrupted_walk_resumes_after_its_last_checkpoint(http_server, fake_s3):
    _script_pages(http_server, [[(i, f"img{i}.jpg")] for i in range(1, 121)])
    http_server.script("/source/1/", Reply(200, _overview_html(120).encode()))
    http_server.script(_page_path(111), Reply(404))
    dl = _site_downloader(http_server, fake_s3)

    with pytest.raises(requests.HTTPError):
        dl.walker.walk_source(1, "b", "p", total_images=120)
    saved = json.loads(fake_s3.objects[("b", f"{_PARTIAL}walk.json")])
    assert saved["last_page"] == 100

    _script_pages(http_server, [[(i, f"img{i}.jpg")] for i in range(1, 121)])
    http_server.received.clear()
    refs, total = dl.walker.walk_source(1, "b", "p", total_images=120)

    walked = [r.path for r in http_server.received if r.path.startswith(_BASE)]
    assert walked == [_page_path(n) for n in range(101, 121)]
    assert [r.id for r in refs] == list(range(1, 121))
    assert total == 120


def _fetches(server: ScriptedServer, pages: range) -> list[int]:
    return [len(server.requests_to(_page_path(n))) for n in pages]


def test_a_page_rule_failure_resumes_the_walk_from_the_last_checkpoint(
    http_server, fake_s3, monkeypatch
):
    monkeypatch.setattr(walker_mod, "_WALK_CHECKPOINT_EVERY_PAGES", 3)
    pages = [[2 * n - 1, 2 * n] for n in range(1, 9)] + [[17]]
    _script_walk(http_server, pages)
    http_server.script(_page_path(8), _page_reply([15], 9), _page_reply([15, 16], 9))

    refs, _ = _site_downloader(http_server, fake_s3).walker.walk_source(
        1, "b", "p", total_images=17
    )

    assert [r.id for r in refs] == list(range(1, 18))
    assert _fetches(http_server, range(1, 10)) == [1, 1, 1, 1, 1, 1, 2, 2, 1]


def test_a_walk_ending_short_of_the_total_is_redone_from_page_one(
    http_server, fake_s3, monkeypatch
):
    monkeypatch.setattr(walker_mod, "_WALK_CHECKPOINT_EVERY_PAGES", 1)
    _script_walk(http_server, [[1, 2], [3, 4], [5]])
    http_server.script(_page_path(2), _page_reply([5]), _page_reply([3, 4], 3))

    refs, _ = _site_downloader(http_server, fake_s3).walker.walk_source(1, "b", "p", total_images=5)

    assert [r.id for r in refs] == [1, 2, 3, 4, 5]
    assert _fetches(http_server, range(1, 4)) == [2, 2, 1]


def test_an_image_listed_on_two_pages_makes_the_walk_retry(http_server, fake_s3, monkeypatch):
    monkeypatch.setattr(walker_mod, "_WALK_CHECKPOINT_EVERY_PAGES", 1)
    _script_walk(http_server, [[1, 2], [3, 4], [5]])
    http_server.script(_page_path(2), _page_reply([2, 3], 3), _page_reply([3, 4], 3))

    refs, _ = _site_downloader(http_server, fake_s3).walker.walk_source(1, "b", "p", total_images=5)

    assert [r.id for r in refs] == [1, 2, 3, 4, 5]
    # Caught on page 2 itself: the retry resumes after the page-1 checkpoint.
    assert _fetches(http_server, range(1, 4)) == [1, 2, 1]


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


def test_a_browse_page_without_the_export_form_logs_in_again(http_server, fake_s3):
    _script_walk(http_server, [[1]])
    _script_login(http_server)
    http_server.script(_page_path(1), _page_reply([1], form=False), _page_reply([1]))

    refs, _ = _site_downloader(http_server, fake_s3).walker.walk_source(1, "b", "p", total_images=1)

    assert [r.id for r in refs] == [1]
    assert len(_posts_to(http_server, _LOGIN_PATH)) == 1
