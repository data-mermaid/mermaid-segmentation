"""Offline tests for CoralNetSession."""

from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

import requests
from conftest import (
    _LOGIN_PATH,
    Reply,
    ScriptedServer,
    _http_response,
    _login_get_html,
    _posts_to,
)

from coralnet_scraper.downloader import CoralNetDownloader
from coralnet_scraper.session import CoralNetSession


def test_thread_sessions_are_not_shared_between_downloaders():
    """Each downloader's per-thread session carries its own downloader's cookies."""
    first = CoralNetDownloader("u", "p")
    second = CoralNetDownloader("u", "p")
    first.session.cookies.set("sessionid", "first-id")
    second.session.cookies.set("sessionid", "second-id")

    first_session = first.auth._get_thread_session()
    second_session = second.auth._get_thread_session()

    assert first_session.cookies.get("sessionid") == "first-id"
    assert second_session.cookies.get("sessionid") == "second-id"
    assert first_session is not second_session
    assert first.auth._get_thread_session() is first_session


def test_export_sessions_are_not_shared_between_downloaders():
    """Each downloader's export session comes from its own login, not another's cache."""
    first = CoralNetDownloader("u", "p")
    second = CoralNetDownloader("u", "p")
    first_sentinel = requests.Session()
    second_sentinel = requests.Session()
    first.auth._new_logged_in_session = lambda: first_sentinel  # type: ignore[method-assign]
    second.auth._new_logged_in_session = lambda: second_sentinel  # type: ignore[method-assign]

    assert first.auth._get_export_session() is first_sentinel
    assert second.auth._get_export_session() is second_sentinel


def test_get_export_session_creates_independent_logged_in_session_per_thread():
    """Each export worker thread gets its own ``requests.Session`` from a fresh login.

    Regression guard for the CoralNet server-side session race: if two threads
    share one ``sessionid`` cookie, ``export_prep`` writes from one thread can
    evict another thread's pending export state.
    """
    dl = CoralNetDownloader("u", "p")
    dl.auth.session = MagicMock()
    new_session_count = [0]

    def fake_new_session() -> requests.Session:
        new_session_count[0] += 1
        return requests.Session()

    dl.auth._new_logged_in_session = fake_new_session  # type: ignore[method-assign]

    # Each worker calls _get_export_session twice; both calls must return the
    # same Session (cached per-thread), but distinct threads must get distinct
    # Sessions (independent login).
    seen: dict[int, tuple[requests.Session, requests.Session]] = {}

    def worker(thread_id: int) -> None:
        seen[thread_id] = (dl.auth._get_export_session(), dl.auth._get_export_session())

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert new_session_count[0] == 3
    for first, second in seen.values():
        assert first is second
    assert len({id(first) for first, _ in seen.values()}) == 3


_LOGIN_URL = CoralNetSession.LOGIN_URL


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


def _login_downloader(server: ScriptedServer) -> CoralNetDownloader:
    dl = CoralNetDownloader("u", "p")
    dl.auth.LOGIN_URL = f"{server.url}{_LOGIN_PATH}"
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


def test_login_fails_after_four_posts_that_all_fail(http_server):
    http_server.script(_LOGIN_PATH, Reply(200, _login_get_html().encode()), Reply(503))
    dl = _login_downloader(http_server)

    assert dl.login() is False
    assert dl.logged_in is False
    assert len(_posts_to(http_server, _LOGIN_PATH)) == 4
