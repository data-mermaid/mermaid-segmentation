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


def testexport_session_creates_independent_logged_in_session_per_thread():
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

    # Each worker calls export_session twice; both calls must return the
    # same Session (cached per-thread), but distinct threads must get distinct
    # Sessions (independent login).
    seen: dict[int, tuple[requests.Session, requests.Session]] = {}

    def worker(thread_id: int) -> None:
        seen[thread_id] = (dl.auth.export_session(), dl.auth.export_session())

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


def test_login_fails_when_the_result_page_shows_no_sign_out_link(http_server):
    login_page = Reply(200, _login_get_html().encode())
    http_server.script(_LOGIN_PATH, login_page, Reply(200, b"<html>Welcome</html>"))
    dl = _login_downloader(http_server)

    assert dl.login() is False
    assert dl.logged_in is False
    assert len(_posts_to(http_server, _LOGIN_PATH)) == 1


def _session_with_cookie_login(value_after_login: str) -> CoralNetSession:
    """Return a session whose stubbed login sets ``sessionid`` to ``value_after_login``."""
    auth = CoralNetSession("u", "p", max_workers=1, requests_per_second=None)

    def fake_login(session: requests.Session) -> None:
        session.cookies.set("sessionid", value_after_login)

    auth._perform_login = fake_login  # type: ignore[method-assign]
    return auth


def test_thread_session_picks_up_cookies_after_main_relogin_and_keeps_its_session():
    auth = _session_with_cookie_login("b")
    auth.session.cookies.set("sessionid", "a")

    first = auth.thread_session()
    assert first.cookies.get("sessionid") == "a"
    assert auth.thread_session() is first
    assert first.cookies.get("sessionid") == "a"

    auth.relogin(auth.session)

    second = auth.thread_session()
    assert second is first
    assert second.cookies.get("sessionid") == "b"


def test_relogin_of_another_session_does_not_refresh_thread_sessions():
    auth = _session_with_cookie_login("b")
    auth.session.cookies.set("sessionid", "a")
    thread_session = auth.thread_session()

    auth.relogin(requests.Session())
    auth.session.cookies.set("sessionid", "c")

    assert auth.thread_session() is thread_session
    assert thread_session.cookies.get("sessionid") == "a"
