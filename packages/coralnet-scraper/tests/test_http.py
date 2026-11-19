"""Transport tests for the session factory, against a real local HTTP server."""

from __future__ import annotations

import time

import pytest
from conftest import Reply, ScriptedServer

from coralnet_scraper.http import SessionFactory


def _factory(requests_per_second: float | None = None) -> SessionFactory:
    return SessionFactory(pool_size=4, requests_per_second=requests_per_second)


def test_429_with_retry_after_is_retried_and_tallied(http_server: ScriptedServer):
    http_server.script("/p", Reply(429, headers={"Retry-After": "0"}), Reply(200, b"ok"))
    factory = _factory()

    resp = factory.coralnet().get(f"{http_server.url}/p", timeout=5)

    assert resp.status_code == 200
    stats = factory.stats()
    assert stats["retries"] == {"429": 1}
    assert stats["final_status"] == {"200": 1}
    assert stats["requests"] == 2


def test_429_without_retry_after_is_retried_and_tallied(http_server: ScriptedServer):
    http_server.script("/p", Reply(429), Reply(200, b"ok"))
    factory = _factory()

    resp = factory.coralnet().get(f"{http_server.url}/p", timeout=5)

    assert resp.status_code == 200
    assert factory.stats()["retries"] == {"429": 1}


def test_every_retry_of_one_request_is_tallied(http_server: ScriptedServer):
    http_server.script("/p", Reply(503), Reply(503), Reply(200))
    factory = _factory()

    resp = factory.coralnet().get(f"{http_server.url}/p", timeout=5)

    assert resp.status_code == 200
    stats = factory.stats()
    assert stats["retries"] == {"503": 2}
    assert stats["final_status"] == {"200": 1}
    assert stats["requests"] == 3


def test_read_timeout_is_retried_and_tallied(http_server: ScriptedServer):
    http_server.script("/p", Reply(200, delay_s=1.0), Reply(200))
    factory = _factory()

    resp = factory.coralnet().get(f"{http_server.url}/p", timeout=0.3)

    assert resp.status_code == 200
    assert factory.stats()["retries"] == {"timeout": 1}


def test_persistent_503_returns_the_last_response_after_five_retries(
    http_server: ScriptedServer,
):
    http_server.script("/p", Reply(503))

    resp = _factory().coralnet().get(f"{http_server.url}/p", timeout=5)

    assert resp.status_code == 503
    assert len(http_server.requests_to("/p")) == 6


def test_404_is_not_retried(http_server: ScriptedServer):
    http_server.script("/p", Reply(404))

    _factory().coralnet().get(f"{http_server.url}/p", timeout=5)

    assert len(http_server.requests_to("/p")) == 1


def test_post_is_not_retried(http_server: ScriptedServer):
    http_server.script("/p", Reply(503))

    _factory().coralnet().post(f"{http_server.url}/p", data={"a": "b"}, timeout=5)

    assert len(http_server.requests_to("/p")) == 1


def _eleven_gets_over_two_sessions(factory: SessionFactory, url: str) -> float:
    first, second = factory.coralnet(), factory.coralnet()
    start = time.monotonic()
    for i in range(11):
        (first if i % 2 else second).get(url, timeout=5)
    return time.monotonic() - start


def test_rate_limit_is_shared_by_every_coralnet_session(http_server: ScriptedServer):
    http_server.script("/p", Reply(200))

    elapsed = _eleven_gets_over_two_sessions(_factory(10), f"{http_server.url}/p")

    assert elapsed >= 0.9


def test_no_rate_limit_when_requests_per_second_is_unset(http_server: ScriptedServer):
    http_server.script("/p", Reply(200))

    elapsed = _eleven_gets_over_two_sessions(_factory(None), f"{http_server.url}/p")

    assert elapsed < 0.9


def test_media_session_is_never_rate_limited(http_server: ScriptedServer):
    http_server.script("/img", Reply(200, b"x"))
    media = _factory(1).media()

    start = time.monotonic()
    for _ in range(3):
        media.get(f"{http_server.url}/img", timeout=5)

    assert time.monotonic() - start < 0.9


@pytest.mark.parametrize(
    ("env_value", "expected"),
    [
        (None, "mermaid-coralnet-scraper (+https://github.com/data-mermaid/mermaid-segmentation)"),
        ("ops-team-scraper/2", "ops-team-scraper/2"),
    ],
)
def test_sessions_send_the_user_agent(
    http_server: ScriptedServer, monkeypatch: pytest.MonkeyPatch, env_value, expected
):
    if env_value is None:
        monkeypatch.delenv("MERMAID_CORALNET_USER_AGENT", raising=False)
    else:
        monkeypatch.setenv("MERMAID_CORALNET_USER_AGENT", env_value)
    http_server.script("/p", Reply(200))

    _factory().coralnet().get(f"{http_server.url}/p", timeout=5)

    assert http_server.requests_to("/p")[0].headers["User-Agent"] == expected


def test_mean_rate_counts_retries_over_the_elapsed_time(http_server: ScriptedServer, monkeypatch):
    http_server.script("/p", Reply(503), Reply(503), Reply(200))
    monkeypatch.setattr("coralnet_scraper.http.time.monotonic", lambda: 100.0)
    factory = _factory()
    monkeypatch.setattr("coralnet_scraper.http.time.monotonic", lambda: 102.0)

    factory.coralnet().get(f"{http_server.url}/p", timeout=5)

    stats = factory.stats()
    assert stats["requests"] == 3
    assert stats["mean_rate_per_s"] == 1.5
