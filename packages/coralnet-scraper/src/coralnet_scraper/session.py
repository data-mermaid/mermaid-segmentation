"""CoralNet login and the HTTP sessions the scraper components share."""

from __future__ import annotations

import logging
import threading
from typing import Any

import requests

from coralnet_scraper.errors import _retrying
from coralnet_scraper.http import SessionFactory
from coralnet_scraper.parsers import extract_csrf_token, parse_image_status_counts

logger = logging.getLogger(__name__)

_PAGE_GET_TIMEOUT = 90  # simple page GETs; CoralNet can be slow under load
_PAGE_EXPORT_TIMEOUT = 180  # CoralNet session setup can take >60s even for 20 images
_LOGIN_ATTEMPTS = 4


class CoralNetSession:
    """Log in to CoralNet and hand out the sessions its requests run on."""

    CORALNET_URL = "https://coralnet.ucsd.edu"
    LOGIN_URL = "https://coralnet.ucsd.edu/accounts/login/"

    def __init__(
        self,
        username: str,
        password: str,
        *,
        max_workers: int,
        requests_per_second: float | None,
    ) -> None:
        """Create the HTTP sessions; no network call is made.

        Args:
            username: CoralNet account name.
            password: CoralNet account password.
            max_workers: Concurrent image download workers; the pool is sized for them.
            requests_per_second: Rate limit shared by every CoralNet session; ``None`` means
                no limit.
        """
        self.username = username
        self.password = password
        # Pool sized for concurrent image URL resolution
        self.sessions = SessionFactory(
            pool_size=max_workers + 4, requests_per_second=requests_per_second
        )
        self.session = self.sessions.coralnet()
        self.media = self.sessions.media()
        self.logged_in = False
        self._login_lock = threading.Lock()
        self._thread_local = threading.local()
        self._export_thread_local = threading.local()

    def transport_stats(self) -> dict[str, Any]:
        """Return the request accounting of every session of this client."""
        return self.sessions.stats()

    def _source_main_url(self, source_id: int) -> str:
        """Return the URL of a source's overview page."""
        return f"{self.CORALNET_URL}/source/{source_id}/"

    def source_overview_html(self, source_id: int) -> str:
        """Return the HTML of a source's overview page, fetched with the main session.

        Raises:
            requests.HTTPError: On any non-2xx response, including 403 and 404.
        """
        url = self._source_main_url(source_id)
        response = self.session.get(url, timeout=_PAGE_GET_TIMEOUT)
        response.raise_for_status()
        return response.text

    def _perform_login(self, session: requests.Session) -> None:
        """Execute the CoralNet login flow on ``session`` (independent cookie jar).

        Pure side-effect helper: pops a fresh login CSRF, POSTs credentials, and
        leaves ``session`` with a logged-in ``sessionid`` cookie, mutating no other
        state. Callers compose this into :meth:`login` (the main session) and
        :meth:`_new_logged_in_session` (per-worker independent sessions used by
        parallel export).

        Raises:
            RuntimeError: If the login page has no CSRF token or the login fails.
        """
        response = session.get(self.LOGIN_URL, timeout=_PAGE_GET_TIMEOUT)
        response.raise_for_status()
        csrf_token = extract_csrf_token(response.text)
        if not csrf_token:
            raise RuntimeError("Could not find CSRF token")
        for attempt in _retrying(_LOGIN_ATTEMPTS):
            with attempt:
                login_response = session.post(
                    self.LOGIN_URL,
                    data={
                        "username": self.username,
                        "password": self.password,
                        "csrfmiddlewaretoken": csrf_token,
                    },
                    headers={"Referer": self.LOGIN_URL},
                    timeout=_PAGE_GET_TIMEOUT,
                    allow_redirects=True,
                )
                login_response.raise_for_status()
        if "Sign out" not in login_response.text:
            raise RuntimeError("Login failed - the result page shows no signed-in user")

    def relogin(self, session: requests.Session) -> None:
        """Log ``session`` in again after CoralNet treated it as logged out.

        Clears the session's cookies and runs the login flow, which fetches a new
        CSRF token. Holds the login lock, so concurrent calls log in one at a time.

        Raises:
            RuntimeError: If the login page has no CSRF token or the login fails.
        """
        with self._login_lock:
            logger.warning("CoralNet session expired; logging in again")
            session.cookies.clear()
            self._perform_login(session)

    def login(self) -> bool:
        """Log the main session in to CoralNet.

        Returns:
            True on success; False when the login fails, after logging the error.
        """
        try:
            with self._login_lock:
                self._perform_login(self.session)
            self.logged_in = True
            logger.info("CoralNet login successful for %s", self.username)
            return True
        except Exception:
            logger.exception("CoralNet login failed for %s", self.username)
            return False

    def thread_session(self) -> requests.Session:
        """Return this thread's own session, with cookies copied from the main session.

        The session is created on first call and reused for later calls from the same
        thread; the cookies are copied once, at creation.

        Safe for read-only requests (e.g. image URL resolution) where threads only
        consume server state. NOT safe for ``export_prep``-style endpoints that write to
        the Django server-side session keyed on ``sessionid`` -- use
        :meth:`export_session` for those.
        """
        s = getattr(self._thread_local, "session", None)
        if s is None:
            s = self.sessions.coralnet()
            s.cookies.update(self.session.cookies)
            self._thread_local.session = s
        return s

    def _new_logged_in_session(self) -> requests.Session:
        """Create a fresh ``requests.Session`` and log it in independently.

        Each call produces an isolated cookie jar (own ``sessionid``), so the returned
        session corresponds to its own server-side Django session. Used by
        :meth:`export_session` to give each phase-2 worker its own session and
        avoid the ``export_prep`` / ``serve`` race that happens when multiple threads
        share one Django sessionid.
        """
        session = self.sessions.coralnet()
        self._perform_login(session)
        return session

    def export_session(self) -> requests.Session:
        """Return this worker thread's dedicated, independently-logged-in session.

        Lazy: the first call from a thread does one login round-trip (~1s),
        then caches the session for all subsequent ``_export_one_page`` calls
        from the same worker. Because CoralNet's ``/annotation/export_prep/``
        writes to ``request.session[f"export_{timestamp}"]`` and
        ``/export/serve/`` reads it back, sharing a sessionid across threads
        causes one thread's prep to evict another's data and the serve to
        return an HTML error page (which then fails ``pd.read_csv``).
        """
        s = getattr(self._export_thread_local, "session", None)
        if s is None:
            s = self._new_logged_in_session()
            self._export_thread_local.session = s
        return s

    def fetch_total_images(self, source_id: int) -> int | None:
        """Return the image count on the source's overview page, or None when it lists none."""
        return parse_image_status_counts(self.source_overview_html(source_id)).total_images


def browse_page_url(coralnet_url: str, source_id: int, page_num: int) -> str:
    """Return the URL of a source's browse page ``page_num``, counted from 1."""
    base = f"{coralnet_url}/source/{source_id}/browse/images/"
    return base if page_num == 1 else f"{base}?page={page_num}"
