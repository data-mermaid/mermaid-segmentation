"""prefetch_cache: the cache root comes from --cache-dir, never from the committed config."""

from __future__ import annotations

import logging
import sys
import threading
import types

import pandas as pd
import pytest
import requests

import scripts.prefetch_cache as prefetch
from mermaidseg.dataset_reconciliation.label_mapping import (
    CoralNetLabelMappings,
    coralnet_snapshot_path,
    parse_coralnet_label_mappings,
    read_coralnet_snapshot,
    write_coralnet_snapshot,
)
from mermaidseg.datasets import setup_local_cache
from mermaidseg.datasets.local_cache import LocalS3Cache


@pytest.fixture(autouse=True)
def _isolated_cache(monkeypatch):
    # setup_local_cache exports the root to the environment; keep that out of other tests.
    # setenv first so monkeypatch records a value to restore even when the var is absent.
    for var in ("MERMAIDSEG_LOCAL_CACHE_DIR", "MERMAIDSEG_S3_OFFLINE"):
        monkeypatch.setenv(var, "")
        monkeypatch.delenv(var)
    LocalS3Cache.reset_singleton()
    monkeypatch.setattr(prefetch, "build_datasets", lambda *a, **k: {})
    yield
    LocalS3Cache.reset_singleton()


def test_cache_dir_flag_reaches_setup_local_cache(monkeypatch, tmp_path):
    seen = []

    def spy(data):
        seen.append(data.get("local_cache_dir"))
        return setup_local_cache(data)

    monkeypatch.setattr(prefetch, "setup_local_cache", spy)

    prefetch.main(["--cache-dir", str(tmp_path)])

    assert seen == [str(tmp_path)]


def test_missing_cache_dir_exits_with_not_configured_error(caplog):
    with caplog.at_level(logging.ERROR, logger="prefetch_cache"):
        code = prefetch.main([])

    assert code == 2
    assert "Local cache is not configured" in caplog.text


class _FakeDataset:
    """Dataset double: ``read_image`` raises for ids in ``bad_images``; reads are recorded."""

    def __init__(self, ids, bad_images=(), bad_labels=()):
        self.df_images = pd.DataFrame({"image_id": list(ids)})
        self._bad_images = set(bad_images)
        self._bad_labels = set(bad_labels)
        self._lock = threading.Lock()
        self.images_read: list[int] = []
        self.labels_read: list[int] = []

    def read_image(self, image_id):
        with self._lock:
            self.images_read.append(image_id)
        if image_id in self._bad_images:
            raise OSError(f"image {image_id} unreadable")

    def read_label(self, image_id):
        with self._lock:
            self.labels_read.append(image_id)
        if image_id in self._bad_labels:
            raise OSError(f"label {image_id} unreadable")


class _ImageOnlyDataset(_FakeDataset):
    read_label = property(lambda self: (_ for _ in ()).throw(AttributeError("read_label")))


def test_warm_dataset_reads_every_image_and_reports_no_failures():
    ds = _FakeDataset(range(5))

    attempted, failures = prefetch._warm_dataset(ds, workers=2, fail_fast=False)

    assert (attempted, failures) == (5, [])
    assert sorted(ds.images_read) == [0, 1, 2, 3, 4]


def test_warm_dataset_tallies_each_failure_with_its_row_index_and_continues():
    ds = _FakeDataset(range(5), bad_images={1, 3})

    attempted, failures = prefetch._warm_dataset(ds, workers=1, fail_fast=False)

    assert attempted == 5
    assert sorted(failures) == ["1: OSError: image 1 unreadable", "3: OSError: image 3 unreadable"]
    assert sorted(ds.images_read) == [0, 1, 2, 3, 4]


def test_warm_dataset_keeps_at_most_the_failure_cap_but_attempts_every_row():
    n_rows = prefetch._MAX_FAILURES_KEPT + 10
    ds = _FakeDataset(range(n_rows), bad_images=set(range(n_rows)))

    attempted, failures = prefetch._warm_dataset(ds, workers=1, fail_fast=False)

    assert attempted == n_rows
    assert len(failures) == prefetch._MAX_FAILURES_KEPT


def test_warm_dataset_fail_fast_reraises_the_first_failure():
    ds = _FakeDataset(range(3), bad_images={0, 1, 2})

    with pytest.raises(OSError, match="unreadable"):
        prefetch._warm_dataset(ds, workers=1, fail_fast=True)


def test_warm_dataset_without_df_images_skips_the_row_scan():
    ds = _FakeDataset(range(3))
    ds.df_images = None

    assert prefetch._warm_dataset(ds, workers=1, fail_fast=False) == (0, [])
    assert ds.images_read == []


def test_warm_dataset_reads_the_dense_label_of_every_row():
    ds = _FakeDataset(range(4))

    prefetch._warm_dataset(ds, workers=2, fail_fast=False)

    assert sorted(ds.labels_read) == [0, 1, 2, 3]


def test_warm_dataset_counts_a_label_read_failure():
    ds = _FakeDataset(range(3), bad_labels={2})

    attempted, failures = prefetch._warm_dataset(ds, workers=1, fail_fast=False)

    assert attempted == 3
    assert failures == ["2: OSError: label 2 unreadable"]


def test_warm_dataset_skips_labels_for_a_dataset_without_read_label():
    ds = _ImageOnlyDataset(range(3))

    attempted, failures = prefetch._warm_dataset(ds, workers=1, fail_fast=False)

    assert (attempted, failures) == (3, [])
    assert ds.labels_read == []


_SNAPSHOT = CoralNetLabelMappings(
    records=parse_coralnet_label_mappings([{"provider_id": 1, "benthic_attribute_name": "Sand"}]),
    source="api",
    fetched_at="t0",
)


@pytest.fixture
def _api_calls(monkeypatch):
    """Serve a fixed CoralNet label-mapping fetch; count the calls."""
    calls: list[int] = []

    def fetch():
        calls.append(1)
        return _SNAPSHOT

    monkeypatch.setattr(prefetch, "fetch_coralnet_label_mappings", fetch)
    return calls


@pytest.fixture
def _datasets(monkeypatch, _api_calls):
    """Make ``build_datasets`` return what a test puts in the returned dict."""
    built: dict = {}
    monkeypatch.setattr(prefetch, "build_datasets", lambda *a, **k: built)
    return built


def test_main_snapshots_the_coralnet_label_mappings_into_the_cache(tmp_path, _datasets, _api_calls):
    _datasets[("coralnet", "train")] = _FakeDataset(range(1))
    _datasets[("coralnet", "val")] = _FakeDataset(range(1))

    assert prefetch.main(["--cache-dir", str(tmp_path), "--workers", "1"]) == 0

    assert _api_calls == [1]
    snapshot = read_coralnet_snapshot(coralnet_snapshot_path())
    assert snapshot.records == _SNAPSHOT.records
    assert snapshot.sha256 == _SNAPSHOT.sha256


def test_main_skips_the_coralnet_snapshot_without_a_coralnet_dataset(
    tmp_path, _datasets, _api_calls
):
    _datasets[("catlin_seaview", "train")] = _FakeDataset(range(1))

    assert prefetch.main(["--cache-dir", str(tmp_path), "--workers", "1"]) == 0

    assert _api_calls == []


def test_main_keeps_the_existing_snapshot_when_the_api_fetch_fails(
    tmp_path, _datasets, monkeypatch
):
    setup_local_cache({"local_cache_dir": str(tmp_path), "local_cache_write_through": True})
    on_disk = {
        7: {
            "id": "11",
            "benthic_attribute_id": "22",
            "benthic_attribute_name": "Sand",
            "growth_form_id": None,
            "growth_form_name": None,
            "provider": "coralnet",
            "provider_id": 7,
            "provider_label": "Sand",
        }
    }
    path = coralnet_snapshot_path()
    write_coralnet_snapshot(
        CoralNetLabelMappings(records=on_disk, source="api", fetched_at="t0"), path
    )
    _datasets[("coralnet", "train")] = _FakeDataset(range(1))

    def failing_fetch():
        raise requests.HTTPError("503 Server Error")

    monkeypatch.setattr(prefetch, "fetch_coralnet_label_mappings", failing_fetch)

    with pytest.raises(requests.HTTPError):
        prefetch.main(["--cache-dir", str(tmp_path), "--workers", "1"])

    assert read_coralnet_snapshot(path).records == on_disk


def test_main_exits_2_when_the_environment_is_offline(monkeypatch, tmp_path, caplog):
    monkeypatch.setenv("MERMAIDSEG_S3_OFFLINE", "1")

    with caplog.at_level(logging.ERROR, logger="prefetch_cache"):
        code = prefetch.main(["--cache-dir", str(tmp_path)])

    assert code == 2
    assert "needs S3 access" in caplog.text


def test_main_exits_2_when_no_dataset_matches_the_filters(tmp_path, caplog, _datasets):
    with caplog.at_level(logging.ERROR, logger="prefetch_cache"):
        code = prefetch.main(["--cache-dir", str(tmp_path)])

    assert code == 2
    assert "No datasets matched" in caplog.text


def test_main_exits_0_when_every_row_loads(tmp_path, _datasets):
    _datasets[("coralnet", "train")] = _FakeDataset(range(3))

    assert prefetch.main(["--cache-dir", str(tmp_path), "--workers", "1"]) == 0


def test_main_exits_1_when_any_row_fails_after_warming_every_dataset(tmp_path, _datasets):
    bad = _FakeDataset(range(3), bad_images={1})
    good = _FakeDataset(range(3))
    _datasets[("coralnet", "train")] = bad
    _datasets[("coralnet", "val")] = good

    code = prefetch.main(["--cache-dir", str(tmp_path), "--workers", "1"])

    assert code == 1
    assert sorted(good.images_read) == [0, 1, 2]


def test_main_exits_1_on_fail_fast_without_warming_later_datasets(tmp_path, _datasets):
    later = _FakeDataset(range(3))
    _datasets[("coralnet", "train")] = _FakeDataset(range(3), bad_images={0, 1, 2})
    _datasets[("coralnet", "val")] = later

    code = prefetch.main(["--cache-dir", str(tmp_path), "--workers", "1", "--fail-fast"])

    assert code == 1
    assert later.images_read == []


def test_main_downloads_the_requested_hf_models(monkeypatch, tmp_path, _datasets):
    _datasets[("coralnet", "train")] = _FakeDataset(range(1))
    downloaded: list[list[str]] = []
    monkeypatch.setattr(prefetch, "_download_hf_models", downloaded.append)

    code = prefetch.main(["--cache-dir", str(tmp_path), "--hf-models", "org/a", "org/b"])

    assert code == 0
    assert downloaded == [["org/a", "org/b"]]


def test_main_skips_hf_downloads_when_none_are_requested(monkeypatch, tmp_path, _datasets):
    _datasets[("coralnet", "train")] = _FakeDataset(range(1))
    monkeypatch.setattr(
        prefetch, "_download_hf_models", lambda repos: pytest.fail("no HF models requested")
    )

    assert prefetch.main(["--cache-dir", str(tmp_path)]) == 0


def test_download_hf_models_snapshots_each_repo_in_order(monkeypatch):
    calls: list[str] = []
    hub = types.ModuleType("huggingface_hub")
    hub.snapshot_download = lambda repo_id: calls.append(repo_id)
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)

    prefetch._download_hf_models(["org/a", "org/b"])

    assert calls == ["org/a", "org/b"]
