"""Shared fixtures for the CoralNet scraper tests.

Provides a hand-rolled in-memory ``FakeS3`` that implements the boto3 client surface the scraper
touches. Keeps the test suite free of network and AWS deps.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

import pytest
from botocore.exceptions import ClientError
from coralnet_scraper.downloader import CoralNetDownloader


@dataclass
class _StubResponse:
    body: bytes
    content_length: int
    content_range: str | None = None

    def __getitem__(self, key):
        if key == "Body":
            return _Body(self.body)
        if key == "ContentLength":
            return self.content_length
        if key == "ContentRange" and self.content_range is not None:
            return self.content_range
        raise KeyError(key)

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default


class _Body:
    def __init__(self, data: bytes):
        self._data = data

    def read(self) -> bytes:
        return self._data


def _client_error(code: str, op: str) -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": code}}, op)


class _FakeS3ExceptionsShim:
    ClientError = ClientError


@dataclass
class FakeS3:
    """In-memory S3 client surface used by the scraper.

    Populate ``objects`` with ``{(bucket, key): bytes}`` to control what gets returned. ``Range``
    semantics follow ``bytes=start-end`` (inclusive).

    Implements the subset of boto3 used by CoralNet scraping (``put_object``, ``upload_fileobj``,
    ``head_object``, ``get_object``, ``list_objects_v2``, ``exceptions.ClientError``).
    """

    objects: dict[tuple[str, str], bytes] = field(default_factory=dict)

    def put(self, bucket: str, key: str, data: bytes) -> None:
        self.objects[(bucket, key)] = data

    def put_object(self, *, Bucket: str, Key: str, Body, ContentType: str | None = None):
        """Scraper uploads CSV bodies."""
        del ContentType  # parity with boto
        if isinstance(Body, str):
            raw: bytes = Body.encode("utf-8")
        elif isinstance(Body, bytes | memoryview):
            raw = bytes(Body)
        else:
            raw = Body.read()
            if isinstance(raw, str):
                raw = raw.encode("utf-8")
        self.objects[(Bucket, Key)] = raw

    def upload_fileobj(self, fileobj, bucket, key, ExtraArgs=None):
        """Scraper uploads image streams."""
        data = fileobj.read()
        self.objects[(bucket, key)] = data

    def head_object(self, *, Bucket: str, Key: str):
        if (Bucket, Key) not in self.objects:
            raise _client_error("404", "HeadObject")
        return {"ContentLength": len(self.objects[(Bucket, Key)])}

    def get_object(self, *, Bucket: str, Key: str, Range: str | None = None):
        if (Bucket, Key) not in self.objects:
            raise _client_error("404", "GetObject")
        full = self.objects[(Bucket, Key)]
        content_range: str | None = None
        data = full
        if Range:
            assert Range.startswith("bytes=")
            start_str, end_str = Range[len("bytes=") :].split("-")
            start = int(start_str)
            end = int(end_str) if end_str else len(full) - 1
            data = full[start : end + 1]
            content_range = f"bytes {start}-{end}/{len(full)}"
        return _StubResponse(body=data, content_length=len(data), content_range=content_range)

    def list_objects_v2(self, *, Bucket: str, Prefix: str = "", MaxKeys: int | None = None):
        keys = [k for (b, k) in self.objects if b == Bucket and k.startswith(Prefix)]
        if MaxKeys is not None:
            keys = keys[:MaxKeys]
        # Real S3 omits "Contents" when nothing matches.
        if not keys:
            return {"KeyCount": 0}
        return {"Contents": [{"Key": k} for k in keys], "KeyCount": len(keys)}


FakeS3.exceptions = _FakeS3ExceptionsShim


@pytest.fixture
def fake_s3() -> FakeS3:
    return FakeS3()


@pytest.fixture
def paginated_downloader(fake_s3) -> CoralNetDownloader:
    """A :class:`CoralNetDownloader` wired for offline paginated-export tests.

    - ``logged_in = True`` (skip real login)
    - ``s3 = fake_s3`` (in-memory S3)
    - Phase-2 worker session creation short-circuited to return ``dl.session``
      so a single ``patch.object(dl.session, ...)`` intercepts both phases.

    Mock-sequence consequence of the worker short-circuit: each browse page is
    fetched TWICE in the mocked ``dl.session.get`` sequence — once during the
    phase-1 walk (to discover ``Next page`` links) and once again inside the
    phase-2 worker (to refresh the CSRF token bound to the worker's session).
    Tests must size ``get_responses`` as ``len(pages) * 2 + len(pages)``
    (page walk + per-page CSRF refresh + per-page serve response).

    Production gives each worker its own logged-in ``requests.Session`` (see
    :meth:`CoralNetDownloader._new_logged_in_session`) to avoid the CoralNet
    server-side session race; that behaviour is exercised by
    ``test_get_export_session_creates_independent_logged_in_session_per_thread``.
    """
    dl = CoralNetDownloader("u", "p")
    dl.logged_in = True
    dl.s3 = fake_s3
    dl._new_logged_in_session = lambda: dl.session  # type: ignore[method-assign]
    dl._export_thread_local = threading.local()
    return dl
