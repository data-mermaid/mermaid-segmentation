"""A loopback HTTP server that serves scripted responses, so the real requests retry stack runs."""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from segdata_etl.label_maps import coralnet_api

# (status, JSON body) for one GET of a path
Response = tuple[int, dict]


class ScriptedServer:
    """Serves ``responses[path]`` in order, repeating the last one; counts hits per path."""

    def __init__(self) -> None:
        self.responses: dict[str, list[Response | Callable[[str], Response]]] = {}
        self.hits: dict[str, int] = {}
        self.base = ""

    def respond(self, path: str, *responses: Response | Callable[[str], Response]) -> None:
        self.responses[path] = list(responses)

    def next_response(self, path: str) -> Response:
        count = self.hits.get(path, 0)
        self.hits[path] = count + 1
        scripted = self.responses[path]
        item = scripted[min(count, len(scripted) - 1)]
        return item(self.base) if callable(item) else item


@pytest.fixture
def server(monkeypatch: pytest.MonkeyPatch) -> Iterator[ScriptedServer]:
    scripted = ScriptedServer()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            status, body = scripted.next_response(self.path)
            payload = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format: str, *args: object) -> None:
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    scripted.base = f"http://127.0.0.1:{httpd.server_address[1]}"
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setattr(coralnet_api, "_RETRY_BACKOFF_SEC", 0)
    yield scripted
    httpd.shutdown()
    httpd.server_close()
