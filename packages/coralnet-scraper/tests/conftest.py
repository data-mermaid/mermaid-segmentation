"""Shared fixtures for the CoralNet scraper tests.

Provides a hand-rolled in-memory ``FakeS3`` that implements the boto3 client surface the scraper
touches. Keeps the test suite free of network and AWS deps.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import stamina
from botocore.exceptions import ClientError

from coralnet_scraper import http
from coralnet_scraper.downloader import CoralNetDownloader


@pytest.fixture(autouse=True)
def _fake_aws_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give boto3 static credentials so it never queries the EC2 metadata endpoint."""
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
    monkeypatch.delenv("AWS_PROFILE", raising=False)


@pytest.fixture(autouse=True)
def _no_retry_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    """Retry immediately so transport tests do not sleep through urllib3 backoff."""
    monkeypatch.setattr(http, "HTTP_BACKOFF_FACTOR", 0.0)
    monkeypatch.setattr(http, "HTTP_BACKOFF_JITTER", 0.0)


@pytest.fixture(autouse=True)
def _no_stamina_backoff() -> Iterator[None]:
    """Retry content failures immediately, keeping each call site's own attempt count."""
    with stamina.set_testing(True, attempts=100, cap=True):
        yield


@dataclass(frozen=True)
class Reply:
    """One scripted HTTP answer; ``delay_s`` stalls before the status line is sent.

    A ``Content-Length`` in ``headers`` replaces the true body length.
    """

    status: int = 200
    body: bytes = b""
    headers: dict[str, str] = field(default_factory=dict)
    delay_s: float = 0.0


@dataclass(frozen=True)
class ReceivedRequest:
    method: str
    path: str
    headers: dict[str, str]


class ScriptedServer:
    """A real HTTP server on 127.0.0.1 answering each path from a scripted reply queue.

    The last reply of a path's script repeats once the queue is down to it; an unscripted
    path answers 404. Every received request is recorded in arrival order.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._scripts: dict[str, list[Reply]] = {}
        self.received: list[ReceivedRequest] = []
        server = self

        class Handler(BaseHTTPRequestHandler):
            def _answer(self) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                if length:
                    self.rfile.read(length)
                reply = server._record_and_pop(self.command, self.path, dict(self.headers))
                time.sleep(reply.delay_s)
                try:
                    self.send_response(reply.status)
                    for name, value in reply.headers.items():
                        self.send_header(name, value)
                    if "Content-Length" not in reply.headers:
                        self.send_header("Content-Length", str(len(reply.body)))
                    self.end_headers()
                    self.wfile.write(reply.body)
                except (BrokenPipeError, ConnectionResetError):
                    pass  # the client gave up waiting (read-timeout scripts)

            do_GET = _answer
            do_HEAD = _answer
            do_POST = _answer

            def log_message(self, format: str, *args: object) -> None:
                pass

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._httpd.daemon_threads = True
        self.url = f"http://127.0.0.1:{self._httpd.server_address[1]}"
        self._thread = threading.Thread(
            target=self._httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
        )

    def script(self, path: str, *replies: Reply) -> None:
        with self._lock:
            self._scripts[path] = list(replies)

    def requests_to(self, path: str) -> list[ReceivedRequest]:
        with self._lock:
            return [r for r in self.received if r.path == path]

    def _record_and_pop(self, method: str, path: str, headers: dict[str, str]) -> Reply:
        with self._lock:
            self.received.append(ReceivedRequest(method, path, headers))
            queue = self._scripts.get(path)
            if not queue:
                return Reply(status=404)
            return queue.pop(0) if len(queue) > 1 else queue[0]

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()


@pytest.fixture
def http_server() -> Iterator[ScriptedServer]:
    server = ScriptedServer()
    server.start()
    yield server
    server.stop()


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
    semantics follow ``bytes=start-end`` (inclusive). Listings are sorted by key like S3, and the
    paginator returns ``list_page_size`` keys per page. ``events`` records every write and
    delete in order as ``("put" | "delete", key)``; ``headers`` keeps the ``ContentType`` and
    ``Metadata`` of each ``put_object``. ``delete_errors`` maps a key to the error code
    ``delete_objects`` reports for it instead of deleting it; ``delete_objects_calls`` records the
    key count of each ``delete_objects`` call.
    """

    objects: dict[tuple[str, str], bytes] = field(default_factory=dict)
    events: list[tuple[str, str]] = field(default_factory=list)
    headers: dict[tuple[str, str], dict] = field(default_factory=dict)
    list_page_size: int = 1000
    delete_errors: dict[str, str] = field(default_factory=dict)
    delete_objects_calls: list[int] = field(default_factory=list)

    def put(self, bucket: str, key: str, data: bytes) -> None:
        self.objects[(bucket, key)] = data

    def put_object(
        self, *, Bucket: str, Key: str, Body, ContentType: str | None = None, Metadata=None
    ):
        """Scraper uploads CSV, JSON and image bodies."""
        self.headers[(Bucket, Key)] = {"ContentType": ContentType, "Metadata": Metadata}
        if isinstance(Body, str):
            raw: bytes = Body.encode("utf-8")
        elif isinstance(Body, bytes | bytearray | memoryview):
            raw = bytes(Body)
        else:
            raw = Body.read()
            if isinstance(raw, str):
                raw = raw.encode("utf-8")
        self.objects[(Bucket, Key)] = raw
        self.events.append(("put", Key))

    def upload_fileobj(self, fileobj, bucket, key, ExtraArgs=None):
        """Scraper uploads image streams."""
        data = fileobj.read()
        self.objects[(bucket, key)] = data
        self.events.append(("put", key))

    def delete_object(self, *, Bucket: str, Key: str):
        self.objects.pop((Bucket, Key), None)
        self.events.append(("delete", Key))
        return {}

    def delete_objects(self, *, Bucket: str, Delete: dict):
        keys = [o["Key"] for o in Delete["Objects"]]
        self.delete_objects_calls.append(len(keys))
        deleted, errors = [], []
        for key in keys:
            if key in self.delete_errors:
                errors.append({"Key": key, "Code": self.delete_errors[key], "Message": "m"})
                continue
            self.objects.pop((Bucket, key), None)
            self.events.append(("delete", key))
            deleted.append({"Key": key})
        response: dict = {"Deleted": deleted}
        if errors:
            response["Errors"] = errors
        return response

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

    def _sorted_keys(self, bucket: str, prefix: str) -> list[str]:
        return sorted(k for (b, k) in self.objects if b == bucket and k.startswith(prefix))

    def _contents(self, bucket: str, keys: list[str]) -> list[dict]:
        return [{"Key": k, "Size": len(self.objects[(bucket, k)])} for k in keys]

    def list_objects_v2(self, *, Bucket: str, Prefix: str = "", MaxKeys: int | None = None):
        keys = self._sorted_keys(Bucket, Prefix)
        if MaxKeys is not None:
            keys = keys[:MaxKeys]
        # Real S3 omits "Contents" when nothing matches.
        if not keys:
            return {"KeyCount": 0}
        return {"Contents": self._contents(Bucket, keys), "KeyCount": len(keys)}

    def get_paginator(self, operation: str) -> _FakeListPaginator:
        assert operation == "list_objects_v2"
        return _FakeListPaginator(self)


class _FakeListPaginator:
    """``list_objects_v2`` paginator: ``list_page_size`` entries per page, ``Delimiter`` aware."""

    def __init__(self, s3: FakeS3) -> None:
        self._s3 = s3

    def paginate(self, *, Bucket: str, Prefix: str = "", Delimiter: str | None = None):
        keys = self._s3._sorted_keys(Bucket, Prefix)
        common: list[str] = []
        if Delimiter:
            direct = []
            for k in keys:
                rest = k[len(Prefix) :]
                if Delimiter in rest:
                    cp = Prefix + rest.split(Delimiter, 1)[0] + Delimiter
                    if cp not in common:
                        common.append(cp)
                else:
                    direct.append(k)
            keys = direct
        size = self._s3.list_page_size
        n_pages = max(1, -(-max(len(keys), len(common)) // size))
        for i in range(n_pages):
            page: dict = {}
            chunk = keys[i * size : (i + 1) * size]
            if chunk:
                page["Contents"] = self._s3._contents(Bucket, chunk)
            prefixes = common[i * size : (i + 1) * size]
            if prefixes:
                page["CommonPrefixes"] = [{"Prefix": p} for p in prefixes]
            yield page


FakeS3.exceptions = _FakeS3ExceptionsShim


@pytest.fixture
def fake_s3() -> FakeS3:
    return FakeS3()


@pytest.fixture
def paginated_downloader(fake_s3) -> CoralNetDownloader:
    """A :class:`CoralNetDownloader` wired for offline paginated-export tests.

    - ``logged_in = True`` (skip real login)
    - ``s3 = fake_s3`` (in-memory S3)
    - Export worker session creation short-circuited to return ``dl.session``
      so a single ``patch.object(dl.session, ...)`` intercepts every request.

    Each exported page costs one browse GET (the worker's own CSRF token), one
    ``export_prep`` POST and one serve GET.

    Production gives each worker its own logged-in ``requests.Session`` (see
    :meth:`CoralNetDownloader._new_logged_in_session`) to avoid the CoralNet
    server-side session race; that behaviour is exercised by
    ``test_get_export_session_creates_independent_logged_in_session_per_thread``.
    """
    dl = CoralNetDownloader("u", "p")
    dl.logged_in = True
    dl.s3 = fake_s3
    dl._new_logged_in_session = lambda: dl.session  # type: ignore[method-assign]
    return dl
