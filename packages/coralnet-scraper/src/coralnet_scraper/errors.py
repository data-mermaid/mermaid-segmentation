"""Exceptions shared by the CoralNet scraper components, and their retry policy."""

from __future__ import annotations

from typing import Any

import requests
import stamina

# stamina's default 45 s total timeout would end the retries of an export running for minutes.
_RETRY_WAITS: dict[str, Any] = {
    "timeout": None,
    "wait_initial": 5.0,
    "wait_max": 120.0,
    "wait_jitter": 5.0,
}


class ContentError(Exception):
    """A successful response whose content is not what the request should return."""


class WalkIncomplete(ContentError):
    """The browse walk breaks a page rule or does not match the overview's image count."""


class SessionExpired(Exception):
    """CoralNet treated a logged-in session's request as logged out."""


class ImageInvalid(Exception):
    """A downloaded image body failed validation."""


class UrlExpired(Exception):
    """The signed original-image URL has expired."""


class _ExportPrepTimeout(requests.ReadTimeout):
    """The ``export_prep`` POST, not a GET, timed out reading its response."""


class _TotalChanged(WalkIncomplete):
    """The overview's image count changed while the walk ran."""

    def __init__(self, total: int) -> None:
        """Record the overview's new image count ``total``."""
        super().__init__(f"overview image count changed to {total} during the walk")
        self.total = total


class WalkCountMismatch(WalkIncomplete):
    """The browse walk found a different number of images than the overview lists."""

    def __init__(self, found: int, expected: int) -> None:
        """Record the walked ``found`` count and the overview's ``expected`` count."""
        super().__init__(f"walk_count_mismatch:{found}!={expected}")


def _is_retryable(exc: BaseException) -> bool:
    """Return whether an attempt that raised ``exc`` is worth repeating.

    Content, session and image failures, transport errors and 5xx statuses are; a 4xx
    status and any other error are not.
    """
    if isinstance(exc, requests.HTTPError):
        return exc.response is None or exc.response.status_code >= 500
    return isinstance(
        exc,
        ContentError | SessionExpired | ImageInvalid | UrlExpired | requests.RequestException,
    )


def _retrying(attempts: int) -> Any:
    """Return a stamina retry context of ``attempts`` attempts with the scraper's waits."""
    return stamina.retry_context(on=_is_retryable, attempts=attempts, **_RETRY_WAITS)
