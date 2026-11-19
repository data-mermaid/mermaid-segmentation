"""HTTP sessions for the CoralNet scraper: transport retries, rate limiting and accounting."""

from __future__ import annotations

import os
import threading
import time
from collections import Counter
from typing import Any, Self

import requests
from pyrate_limiter import Limiter
from requests.adapters import HTTPAdapter
from requests_ratelimiter import LimiterAdapter
from urllib3.exceptions import NewConnectionError
from urllib3.exceptions import TimeoutError as Urllib3TimeoutError
from urllib3.util.retry import Retry

ENV_USER_AGENT = "MERMAID_CORALNET_USER_AGENT"
DEFAULT_USER_AGENT = (
    "mermaid-coralnet-scraper (+https://github.com/data-mermaid/mermaid-segmentation)"
)

RETRY_METHODS = frozenset({"GET", "HEAD"})
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
HTTP_RETRIES = 5
HTTP_BACKOFF_FACTOR = 2.0  # 2, 4, 8, 16, 32 s plus jitter, unless Retry-After says otherwise
HTTP_BACKOFF_JITTER = 2.0
HTTP_BACKOFF_MAX = 120.0
_POOL_CONNECTIONS = 10


def user_agent() -> str:
    """Return the User-Agent sent on every request; ``MERMAID_CORALNET_USER_AGENT`` overrides it."""
    return os.getenv(ENV_USER_AGENT) or DEFAULT_USER_AGENT


class _Tally:
    """A thread-safe counter of string keys."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counts: Counter[str] = Counter()

    def add(self, key: str) -> None:
        with self._lock:
            self._counts[key] += 1

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return dict(self._counts)


def _retry_cause(response: Any, error: Exception | None) -> str:
    """Return the tally key of one retry: the status code, ``timeout`` or ``connection``."""
    if error is None and response is not None:
        return str(response.status)
    # NewConnectionError subclasses ConnectTimeoutError, yet is a refused or failed connect.
    if isinstance(error, Urllib3TimeoutError) and not isinstance(error, NewConnectionError):
        return "timeout"
    return "connection"


class CountingRetry(Retry):
    """urllib3 ``Retry`` that tallies each retry it allows, by cause, in a shared counter.

    urllib3 replaces the ``Retry`` object on every attempt through :meth:`new`; the tally
    is carried over so every copy counts into the same place.
    """

    def __init__(self, *args: Any, tally: _Tally | None = None, **kwargs: Any) -> None:
        """Create the policy; ``tally`` is the shared counter, a fresh one when ``None``."""
        super().__init__(*args, **kwargs)
        self.tally = tally if tally is not None else _Tally()

    def new(self, **kw: Any) -> Self:
        """Return urllib3's copy of this policy, counting into the same tally."""
        retry = super().new(**kw)
        retry.tally = self.tally
        return retry

    def increment(
        self,
        method: str | None = None,
        url: str | None = None,
        response: Any = None,
        error: Exception | None = None,
        _pool: Any = None,
        _stacktrace: Any = None,
    ) -> Self:
        """Return the policy for the next attempt and tally this retry by cause.

        A retry urllib3 refuses (exhausted) raises before it is tallied.
        """
        retry = super().increment(method, url, response, error, _pool, _stacktrace)
        self.tally.add(_retry_cause(response, error))
        return retry


def _retry_policy(tally: _Tally) -> CountingRetry:
    """Return the transport retry for idempotent requests, counting into ``tally``.

    A response still failing after the last attempt is returned, not raised, so callers
    see its status through ``raise_for_status``.
    """
    return CountingRetry(
        total=HTTP_RETRIES,
        connect=HTTP_RETRIES,
        read=HTTP_RETRIES,
        status=HTTP_RETRIES,
        other=0,
        status_forcelist=RETRY_STATUSES,
        allowed_methods=RETRY_METHODS,
        backoff_factor=HTTP_BACKOFF_FACTOR,
        backoff_jitter=HTTP_BACKOFF_JITTER,
        backoff_max=HTTP_BACKOFF_MAX,
        respect_retry_after_header=True,
        raise_on_status=False,
        tally=tally,
    )


class SessionFactory:
    """Builds every HTTP session of a run, with shared rate control and request accounting."""

    def __init__(self, *, pool_size: int, requests_per_second: float | None) -> None:
        """Create the factory; no session or network call is made.

        Args:
            pool_size: Connections kept per host by each session.
            requests_per_second: Request rate shared by all :meth:`coralnet` sessions;
                ``None`` sets no limit.
        """
        self.pool_size = pool_size
        self.requests_per_second = requests_per_second
        self._limiter: Limiter | None = None
        self._limiter_lock = threading.Lock()
        self._retries = _Tally()
        self._final_status = _Tally()
        self._started = time.monotonic()

    def coralnet(self) -> requests.Session:
        """Return a session for CoralNet pages: transport retries and the shared rate limit."""
        if self.requests_per_second is None:
            return self._session(self._plain_adapter())
        with self._limiter_lock:
            adapter = LimiterAdapter(
                per_second=self.requests_per_second,
                limiter=self._limiter,
                max_retries=_retry_policy(self._retries),
                pool_connections=_POOL_CONNECTIONS,
                pool_maxsize=self.pool_size,
            )
            if self._limiter is None:
                self._limiter = adapter.limiter
        return self._session(adapter)

    def media(self) -> requests.Session:
        """Return a session for original-image downloads: transport retries, never rate-limited."""
        return self._session(self._plain_adapter())

    def stats(self) -> dict[str, Any]:
        """Return the request accounting of every session built by this factory.

        Returns:
            ``{"requests", "mean_rate_per_s", "retries", "final_status"}``: requests sent
            (final responses plus retries), their mean rate since the factory was created,
            retries by cause, and final responses by status code. Retries are included in
            ``requests`` and are not counted against the rate limit.
        """
        final_status = self._final_status.snapshot()
        retries = self._retries.snapshot()
        n_requests = sum(final_status.values()) + sum(retries.values())
        elapsed = time.monotonic() - self._started
        return {
            "requests": n_requests,
            "mean_rate_per_s": round(n_requests / elapsed, 3) if elapsed > 0 else 0.0,
            "retries": retries,
            "final_status": final_status,
        }

    def _plain_adapter(self) -> HTTPAdapter:
        return HTTPAdapter(
            max_retries=_retry_policy(self._retries),
            pool_connections=_POOL_CONNECTIONS,
            pool_maxsize=self.pool_size,
        )

    def _count_response(self, response: requests.Response, *args: Any, **kwargs: Any) -> None:
        self._final_status.add(str(response.status_code))

    def _session(self, adapter: HTTPAdapter) -> requests.Session:
        session = requests.Session()
        session.headers["User-Agent"] = user_agent()
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        session.hooks["response"].append(self._count_response)
        return session
