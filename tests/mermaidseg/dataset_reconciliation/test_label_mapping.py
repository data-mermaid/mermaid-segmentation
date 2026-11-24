"""Source -> MERMAID label maps and the source-to-target lookup."""

from __future__ import annotations

import functools
import json
import socket
import threading
import time
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import requests
import torch

from mermaidseg.dataset_reconciliation import label_mapping as lm
from mermaidseg.datasets.coralscapes_v2.coralscapes_v2_dataset import CORALSCAPES_V2_ID2NAME
from mermaidseg.datasets.local_cache import DataLoadError, LocalS3Cache

STATIC_MAPS = [
    lm.fetch_catlin_seaview_to_mermaid,
    lm.fetch_moorea_labeled_corals_to_mermaid,
    lm.fetch_pacific_labeled_corals_to_mermaid,
    lm.fetch_ucsd_mosaics_to_mermaid,
    lm.fetch_benthos_yuval_to_mermaid,
    lm.coralscapes_v2_to_mermaid,
]


@pytest.mark.parametrize("build", STATIC_MAPS, ids=lambda f: f.__name__)
def test_static_maps_are_lowercase_nonempty_name_to_name(build):
    mapping = build()

    assert mapping
    for source, target in mapping.items():
        assert source == source.lower() and source.strip() == source
        assert isinstance(target, str) and target == target.lower() and target.strip()


@pytest.mark.parametrize(
    ("build", "id2name"),
    [
        (lm.coralscapes_v2_to_mermaid, CORALSCAPES_V2_ID2NAME),
    ],
)
def test_coralscapes_maps_cover_exactly_the_native_class_names(build, id2name):
    assert set(build()) == set(id2name.values())


@pytest.mark.parametrize(
    ("build", "source", "target"),
    [
        (lm.fetch_catlin_seaview_to_mermaid, "cca", "crustose coralline algae"),
        (lm.fetch_moorea_labeled_corals_to_mermaid, "p mass", "porites"),
        (lm.fetch_pacific_labeled_corals_to_mermaid, "unclear", "background"),
        (lm.fetch_ucsd_mosaics_to_mermaid, "favia matthai", "dipsastraea matthaii"),
        (lm.fetch_benthos_yuval_to_mermaid, "rock", "bare substrate"),
        (lm.coralscapes_v2_to_mermaid, "acropora bleached", "bleached coral"),
    ],
)
def test_static_maps_pin_a_known_entry(build, source, target):
    assert build()[source] == target


def _row(provider_id, name="Acropora", label="ACRO"):
    return {
        "id": f"uuid-{provider_id}",
        "benthic_attribute_id": "ba",
        "benthic_attribute_name": name,
        "growth_form_id": None,
        "growth_form_name": None,
        "provider": "CoralNet",
        "provider_id": provider_id,
        "provider_label": label,
    }


class _ApiServer:
    """A local HTTP server whose reply is ``respond(path, n)`` -> (status, JSON body).

    ``n`` counts every request received so far, starting at 1.
    """

    def __init__(self) -> None:
        self.paths: list[str] = []
        self.base = ""
        self.delay = 0.0
        self.respond: Callable[[str, int], tuple[int, dict]] = lambda path, n: (
            200,
            {"results": [], "next": None},
        )

    def url(self, path: str) -> str:
        return self.base + path


def _page(rows, next_url=None):
    return 200, {"results": rows, "next": next_url}


@pytest.fixture
def api_server(monkeypatch):
    """Serve scripted pages over real HTTP, so requests' retry stack runs unmocked."""
    monkeypatch.setattr(lm, "_RETRY_BACKOFF_SEC", 0)
    state = _ApiServer()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            state.paths.append(self.path)
            time.sleep(state.delay)
            status, body = state.respond(self.path, len(state.paths))
            payload = json.dumps(body).encode()
            try:
                self.send_response(status)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            except OSError:  # the client timed out and hung up
                pass

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    state.base = f"http://127.0.0.1:{server.server_port}"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield state
    server.shutdown()
    server.server_close()


@pytest.fixture
def cache(monkeypatch):
    """A fresh LocalS3Cache singleton, restored after the test."""
    for var in ("MERMAIDSEG_LOCAL_CACHE_DIR", "MERMAIDSEG_S3_OFFLINE"):
        monkeypatch.setenv(var, "")
        monkeypatch.delenv(var)
    LocalS3Cache.reset_singleton()
    yield LocalS3Cache
    LocalS3Cache.reset_singleton()


def test_fetch_follows_every_page_and_keys_rows_by_int_provider_id(api_server):
    pages = {
        "/p1": _page([_row("82")], api_server.url("/p2")),
        "/p2": _page([_row("7", "Sand", "SAND")]),
    }
    api_server.respond = lambda path, n: pages[path]

    mappings = lm.fetch_coralnet_label_mappings(api_server.url("/p1"))

    assert api_server.paths == ["/p1", "/p2"]
    assert set(mappings.records) == {82, 7}
    assert mappings.records[7]["benthic_attribute_name"] == "Sand"
    assert mappings.records[82]["provider_id"] == 82
    assert mappings.source == "api"


def test_a_transient_503_is_retried_and_the_fetch_succeeds(api_server):
    api_server.respond = lambda path, n: (503, {}) if n == 1 else _page([_row("4")])

    mappings = lm.fetch_coralnet_label_mappings(api_server.url("/p1"))

    assert list(mappings.records) == [4]
    assert len(api_server.paths) == 2


@pytest.mark.parametrize(("raw", "expected"), [("82", 82), (82.0, 82), ("82.0", 82), (82, 82)])
def test_integer_like_provider_ids_normalize_to_int(raw, expected):
    assert list(lm.parse_coralnet_label_mappings([_row(raw)])) == [expected]


@pytest.mark.parametrize(("raw", "shown"), [("abc", "'abc'"), ("8.5", "'8.5'"), (None, "None")])
def test_non_integer_provider_ids_raise_naming_the_value(raw, shown):
    with pytest.raises(ValueError, match="provider_id") as excinfo:
        lm.parse_coralnet_label_mappings([_row("3"), _row(raw)])

    assert shown in str(excinfo.value)


def test_duplicate_provider_ids_raise():
    with pytest.raises(ValueError, match="provider_id 5"):
        lm.parse_coralnet_label_mappings([_row("5"), _row("5.0")])


def test_a_persistent_503_raises_after_four_attempts_without_falling_back(api_server):
    api_server.respond = lambda path, n: (503, {})

    with pytest.raises(requests.RequestException):
        lm.fetch_coralnet_label_mappings(api_server.url("/p1"))

    assert len(api_server.paths) == 4


def test_a_404_raises_http_error_without_retrying(api_server):
    api_server.respond = lambda path, n: (404, {})

    with pytest.raises(requests.HTTPError):
        lm.fetch_coralnet_label_mappings(api_server.url("/p1"))

    assert len(api_server.paths) == 1


def test_a_next_url_that_repeats_an_earlier_page_raises(api_server):
    pages = {
        "/p1": _page([_row("1")], api_server.url("/p2")),
        "/p2": _page([_row("2")], api_server.url("/p1")),
    }
    api_server.respond = lambda path, n: pages[path]

    with pytest.raises(ValueError, match="/p1"):
        lm.fetch_coralnet_label_mappings(api_server.url("/p1"))

    assert api_server.paths == ["/p1", "/p2"]


def test_a_server_slower_than_the_timeout_raises_a_request_error(api_server, monkeypatch):
    monkeypatch.setattr(lm, "_REQUEST_TIMEOUT_SEC", 0.2)
    api_server.delay = 0.6

    with pytest.raises(requests.ConnectionError):
        lm.fetch_coralnet_label_mappings(api_server.url("/p1"))


def test_a_closed_port_raises_connection_error(api_server):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]

    with pytest.raises(requests.ConnectionError):
        lm.fetch_coralnet_label_mappings(f"http://127.0.0.1:{port}/p1")


def _mappings(rows, source="api"):
    return lm.CoralNetLabelMappings(
        records=lm.parse_coralnet_label_mappings(rows), source=source, fetched_at="t0"
    )


def test_digest_ignores_row_order():
    a = _mappings([_row("1"), _row("2", "Sand")])
    b = _mappings([_row("2", "Sand"), _row("1")])
    c = _mappings([_row("1"), _row("2", "Rubble")])

    assert a.sha256 == b.sha256
    assert a.sha256 != c.sha256
    assert a.provenance() == {
        "source": "api",
        "sha256": a.sha256,
        "n_records": 2,
        "fetched_at": "t0",
    }


def test_snapshot_round_trips_through_disk(tmp_path):
    mappings = _mappings([_row("1"), _row("2", "Sand")])
    path = tmp_path / "nested" / "coralnet.json"

    lm.write_coralnet_snapshot(mappings, path)
    loaded = lm.read_coralnet_snapshot(path)

    assert loaded.records == mappings.records
    assert loaded.sha256 == mappings.sha256
    assert loaded.source == "snapshot"
    assert loaded.fetched_at == "t0"


def test_a_snapshot_whose_records_no_longer_match_its_digest_is_rejected(tmp_path):
    path = tmp_path / "coralnet.json"
    lm.write_coralnet_snapshot(_mappings([_row("1", "Acropora")]), path)
    data = json.loads(path.read_text())
    data["records"][0]["benthic_attribute_name"] = "Sand"
    path.write_text(json.dumps(data))

    with pytest.raises(ValueError, match="sha256"):
        lm.read_coralnet_snapshot(path)


def test_offline_load_reads_only_the_cached_snapshot(cache, monkeypatch, tmp_path):
    cache.configure(tmp_path, offline=True)
    lm.write_coralnet_snapshot(_mappings([_row("9")]), lm.coralnet_snapshot_path())
    monkeypatch.setattr(
        requests.Session, "request", lambda *a, **k: pytest.fail("network used offline")
    )

    mappings = lm.load_coralnet_label_mappings()

    assert list(mappings.records) == [9]
    assert mappings.source == "snapshot"


def test_offline_load_without_a_snapshot_says_to_run_prefetch(cache, tmp_path):
    cache.configure(tmp_path, offline=True)

    with pytest.raises(DataLoadError, match="prefetch_cache.py"):
        lm.load_coralnet_label_mappings()


def test_offline_load_without_a_cache_says_to_run_prefetch(cache):
    cache.configure(None, offline=True)

    with pytest.raises(DataLoadError, match="no local cache configured"):
        lm.load_coralnet_label_mappings()


def test_online_load_returns_the_api_rows_and_writes_no_snapshot(cache, api_server, tmp_path):
    cache.configure(tmp_path, offline=False)
    api_server.respond = lambda path, n: _page([_row("4")])
    fetch = functools.partial(lm.fetch_coralnet_label_mappings, api_server.url("/p1"))

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(lm, "fetch_coralnet_label_mappings", fetch)
        mappings = lm.load_coralnet_label_mappings()

    assert mappings.source == "api"
    assert list(mappings.records) == [4]
    assert not lm.coralnet_snapshot_path().exists()


def test_join_maps_lowercased_names_through_the_label_id():
    records = lm.parse_coralnet_label_mappings([_row("1", "Acropora"), _row("2", "Sand")])

    mapping = lm.coralnet_to_mermaid(
        {"1": "ACRO renamed", 2: "Sand", 3: "Unmapped", 4: None, 5: float("nan")}, records
    )

    assert mapping == {"acro renamed": "Acropora", "sand": "Sand"}


def test_join_leaves_out_a_record_with_no_benthic_attribute_name():
    records = lm.parse_coralnet_label_mappings([_row("1", None), _row("2", "Sand")])

    assert lm.coralnet_to_mermaid({1: "Acro", 2: "Sand"}, records) == {"sand": "Sand"}


def test_join_raises_when_one_name_has_ids_with_different_targets():
    records = lm.parse_coralnet_label_mappings([_row("1", "Acropora"), _row("2", "Sand")])

    with pytest.raises(ValueError, match="'acro'.*1 and 2"):
        lm.coralnet_to_mermaid({1: "Acro", 2: "ACRO"}, records)


def test_join_accepts_one_name_with_ids_that_agree():
    records = lm.parse_coralnet_label_mappings([_row("1"), _row("2")])

    assert lm.coralnet_to_mermaid({1: "Acro", 2: "ACRO"}, records) == {"acro": "Acropora"}


def test_join_skips_unused_names_before_the_collision_check():
    records = lm.parse_coralnet_label_mappings(
        [_row("1", "sand"), _row("2", "rubble"), _row("3", "rock"), _row("4", "coral")]
    )

    mapping = lm.coralnet_to_mermaid(
        {1: "Sand", 2: "Rubble", 3: "Dup", 4: "DUP"}, records, used_names={"sand"}
    )

    assert mapping == {"sand": "sand"}


@pytest.mark.live
def test_live_api_returns_int_keyed_coralnet_rows():
    mappings = lm.fetch_coralnet_label_mappings()

    assert mappings.records
    assert all(isinstance(k, int) for k in mappings.records)


def test_source_labels_to_target_labels_indexes_the_lookup_for_a_non_square_batch():
    lookup = torch.tensor([0, 5, 7, 5])
    labels = torch.tensor([[[1, 0, 2], [3, 2, 0]]])  # (B=1, H=2, W=3)

    out = lm.source_labels_to_target_labels(labels, lookup)

    assert out.tolist() == [[[5, 0, 7], [5, 7, 0]]]
    assert out.dtype == torch.long


def test_source_labels_to_target_labels_accepts_non_long_integer_labels():
    lookup = torch.tensor([0, 5, 7])
    labels = torch.tensor([[[2, 1]]], dtype=torch.uint8)

    assert lm.source_labels_to_target_labels(labels, lookup).tolist() == [[[7, 5]]]


# --- Golden digest ---
# Pins the parse and digest of the legacy runtime mapping to ``MAIN_DIGEST`` for fixed rows.

MAIN_DIGEST = "562cc2d876b238eda53a7274f7fc3e4191f1f84c8c9d382f936d37e050985b63"

_ACROPORA: lm.CoralNetLabelMapping = {
    "id": "a",
    "benthic_attribute_id": "b1",
    "benthic_attribute_name": "Acropora",
    "growth_form_id": "g1",
    "growth_form_name": "Branching",
    "provider": "CoralNet",
    "provider_id": 82,
    "provider_label": "Acropora (branching)",
}
_UNKNOWN: lm.CoralNetLabelMapping = {
    "id": "b",
    "benthic_attribute_id": None,
    "benthic_attribute_name": None,
    "growth_form_id": None,
    "growth_form_name": None,
    "provider": "CoralNet",
    "provider_id": 7,
    "provider_label": "Unknown",
}
_DIGEST_RECORDS = {82: _ACROPORA, 7: _UNKNOWN}
_FETCHED_AT = "2026-11-23T10:00:00+00:00"


def test_digest_matches_segdata_etl():
    assert lm.coralnet_mapping_digest(_DIGEST_RECORDS) == MAIN_DIGEST


def test_parse_of_the_digest_rows_gives_the_same_digest():
    parsed = lm.parse_coralnet_label_mappings([dict(_ACROPORA), dict(_UNKNOWN)])
    assert lm.coralnet_mapping_digest(parsed) == MAIN_DIGEST


def test_snapshot_json_has_the_snapshot_shape(tmp_path):
    mappings = lm.CoralNetLabelMappings(
        records=_DIGEST_RECORDS, source="api", fetched_at=_FETCHED_AT
    )
    path = tmp_path / "snapshot.json"
    lm.write_coralnet_snapshot(mappings, path)
    data = json.loads(path.read_text())
    assert data["fetched_at"] == _FETCHED_AT
    assert data["sha256"] == MAIN_DIGEST
    assert [r["provider_id"] for r in data["records"]] == [7, 82]
