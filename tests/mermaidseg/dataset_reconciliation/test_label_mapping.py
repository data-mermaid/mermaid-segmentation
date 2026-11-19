"""Source -> MERMAID label maps and the source-to-target lookup."""

from __future__ import annotations

import logging

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


class _Response:
    def __init__(self, payload, status=200):
        self._payload = payload
        self._status = status

    def raise_for_status(self):
        if self._status >= 400:
            raise requests.HTTPError(f"{self._status} error")

    def json(self):
        return self._payload


def _patch_api(monkeypatch, pages):
    """Serve ``pages`` (url -> payload) to ``requests.get``; record the urls asked for."""
    seen = []

    def fake_get(url, timeout):
        seen.append(url)
        return pages[url]

    monkeypatch.setattr(lm.requests, "get", fake_get)
    return seen


@pytest.fixture
def cache(monkeypatch):
    """A fresh LocalS3Cache singleton, restored after the test."""
    for var in ("MERMAIDSEG_LOCAL_CACHE_DIR", "MERMAIDSEG_S3_OFFLINE"):
        monkeypatch.setenv(var, "")
        monkeypatch.delenv(var)
    LocalS3Cache.reset_singleton()
    yield LocalS3Cache
    LocalS3Cache.reset_singleton()


def test_fetch_follows_every_page_and_keys_rows_by_int_provider_id(monkeypatch):
    seen = _patch_api(
        monkeypatch,
        {
            "u1": _Response({"results": [_row("82")], "next": "u2"}),
            "u2": _Response({"results": [_row("7", "Sand", "SAND")], "next": None}),
        },
    )

    mappings = lm.fetch_coralnet_label_mappings("u1")

    assert seen == ["u1", "u2"]
    assert set(mappings.records) == {82, 7}
    assert mappings.records[7]["benthic_attribute_name"] == "Sand"
    assert mappings.records[82]["provider_id"] == 82
    assert mappings.source == "api"


@pytest.mark.parametrize(("raw", "expected"), [("82", 82), (82.0, 82), ("82.0", 82), (82, 82)])
def test_integer_like_provider_ids_normalize_to_int(raw, expected):
    assert list(lm.parse_coralnet_label_mappings([_row(raw)])) == [expected]


@pytest.mark.parametrize("raw", ["abc", "8.5", None])
def test_non_integer_provider_ids_are_skipped_with_a_warning(raw, caplog):
    with caplog.at_level(logging.WARNING):
        records = lm.parse_coralnet_label_mappings([_row(raw), _row("3")])

    assert list(records) == [3]
    assert "non-integer provider_id" in caplog.text


def test_duplicate_provider_ids_raise():
    with pytest.raises(ValueError, match="provider_id 5"):
        lm.parse_coralnet_label_mappings([_row("5"), _row("5.0")])


def test_an_api_error_propagates_instead_of_falling_back(monkeypatch):
    _patch_api(monkeypatch, {"u": _Response({}, status=503)})

    with pytest.raises(requests.HTTPError):
        lm.fetch_coralnet_label_mappings("u")


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


def test_offline_load_reads_only_the_cached_snapshot(cache, monkeypatch, tmp_path):
    cache.configure(tmp_path, offline=True)
    lm.write_coralnet_snapshot(_mappings([_row("9")]), lm.coralnet_snapshot_path())
    monkeypatch.setattr(lm.requests, "get", lambda *a, **k: pytest.fail("network used offline"))

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


def test_online_load_fetches_and_refreshes_the_snapshot(cache, monkeypatch, tmp_path):
    cache.configure(tmp_path, offline=False)
    _patch_api(
        monkeypatch,
        {lm.CORALNET_LABELMAPPINGS_URL: _Response({"results": [_row("4")], "next": None})},
    )

    mappings = lm.load_coralnet_label_mappings()

    assert mappings.source == "api"
    assert lm.read_coralnet_snapshot(lm.coralnet_snapshot_path()).records == mappings.records


def test_join_maps_lowercased_names_through_the_label_id():
    records = lm.parse_coralnet_label_mappings([_row("1", "Acropora"), _row("2", "Sand")])

    mapping = lm.coralnet_to_mermaid(
        {"1": "ACRO renamed", 2: "Sand", 3: "Unmapped", 4: None, 5: float("nan")}, records
    )

    assert mapping == {"acro renamed": "Acropora", "sand": "Sand"}


def test_join_raises_when_one_name_has_ids_with_different_targets():
    records = lm.parse_coralnet_label_mappings([_row("1", "Acropora"), _row("2", "Sand")])

    with pytest.raises(ValueError, match="'acro'.*1 and 2"):
        lm.coralnet_to_mermaid({1: "Acro", 2: "ACRO"}, records)


def test_join_accepts_one_name_with_ids_that_agree():
    records = lm.parse_coralnet_label_mappings([_row("1"), _row("2")])

    assert lm.coralnet_to_mermaid({1: "Acro", 2: "ACRO"}, records) == {"acro": "Acropora"}


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
