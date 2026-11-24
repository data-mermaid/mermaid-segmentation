"""``segdata curate coralnet``: the v1 and v2 curated versions published from raw."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pytest

from segdata_etl import cli
from segdata_etl.common import publishing
from segdata_etl.common.local_store import LocalStore
from segdata_etl.datasets.coralnet.curate import versions
from segdata_etl.label_maps import coralnet_api
from segdata_etl.splits.builder import _require_raw_parent
from segdata_spec import DATASET_BUCKET
from segdata_spec.errors import PublishRefused
from segdata_spec.manifest import ImageStore, Manifest, ParentRef, Pin, SourceInfo
from segdata_spec.publish import GitState, publish_dataset, publish_taxonomy
from segdata_spec.schema import IMAGES_SCHEMA, POINTS_SCHEMA

CLEAN_GIT = GitState(sha="abcdef1234567", dirty=False, pushed=True)
RAW_STORE = ImageStore(bucket="img-bucket", prefix="coralnet/", owner="segdata")
RAW_SOURCE = SourceInfo(type="coralnet-scrape", uri="s3://scrape/", revision="2026-11-01")
KEPT, BLACK, OVERRIDDEN = "10", "11", "12"
API_ROWS = [
    {
        "id": "m1",
        "benthic_attribute_id": "api-acro",
        "benthic_attribute_name": "Acropora",
        "growth_form_id": "g1",
        "growth_form_name": "Branching",
        "provider": "CoralNet",
        "provider_id": 1,
        "provider_label": "Acropora (branching)",
    }
]


class _Response:
    def __init__(self, payload: dict[str, object]):
        self._payload = payload

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict[str, object]:
        return self._payload


class FakeApiSession:
    """The MERMAID LabelMapping API as one page of ``API_ROWS``."""

    def __enter__(self) -> FakeApiSession:
        return self

    def __exit__(self, *_: object) -> None:
        pass

    def get(self, url: str, timeout: float) -> _Response:
        assert url == coralnet_api.CORALNET_LABELMAPPINGS_URL
        return _Response({"results": API_ROWS, "next": None})


def _image(image_id: str, group: str) -> dict[str, object]:
    return {
        "image_id": image_id,
        "key": f"{group}/{image_id}.jpg",
        "etag": "e",
        "size": 100,
        "crc64nvme": None,
        "media_type": "image/jpeg",
        "width": 640,
        "height": 480,
        "group": group,
        "mask_key": None,
        "mask_etag": None,
        "mask_size": None,
        "mask_crc64nvme": None,
        "source_fingerprint": "f",
        "source_name": None,
        "attrs": None,
    }


def _point(image_id: str, row: int, col: int, label_id: int) -> dict[str, object]:
    return {"image_id": image_id, "row": row, "col": col, "label_id": label_id, "annotator": "a"}


def _raw_tables() -> dict[str, pa.Table]:
    images = pa.Table.from_pylist(
        [_image(KEPT, "s1"), _image(BLACK, "s1"), _image(OVERRIDDEN, "s2")], schema=IMAGES_SCHEMA
    )
    points = pa.Table.from_pylist(
        [
            _point(KEPT, 1, 1, 1),
            _point(KEPT, 2, 2, 2),
            _point(BLACK, 1, 1, 1),
            _point(OVERRIDDEN, 5, 5, 2),
        ],
        schema=POINTS_SCHEMA,
    )
    # Label 4 is used only by the override file.
    names = {1: "Acropora (branching)", 2: "Porites", 3: "Nothing Like It", 4: "Acropora"}
    labels = pa.table(
        {
            "label_id": list(names),
            "name": list(names.values()),
            "mermaid_benthic_attribute_id": pa.nulls(len(names), pa.string()),
            "mermaid_benthic_attribute": pa.nulls(len(names), pa.string()),
            "mermaid_growth_form": pa.nulls(len(names), pa.string()),
            "mapping_source": ["identity"] * len(names),
        }
    )
    counts = pa.table(
        {
            "image_id": [KEPT, KEPT, BLACK, OVERRIDDEN],
            "label_id": [1, 2, 1, 2],
            "n": [1, 1, 1, 1],
        }
    )
    return {
        "images.parquet": images,
        "points.parquet": points,
        "labels.parquet": labels,
        "stats/label_counts.parquet": counts,
    }


def _taxonomy_tables() -> dict[str, pa.Table]:
    rows = [("ba-acro", "Acropora"), ("ba-por", "Porites")]
    return {
        "benthic_attributes.parquet": pa.table(
            {
                "id": [r[0] for r in rows],
                "name": [r[1] for r in rows],
                "parent_id": pa.nulls(len(rows), pa.string()),
            }
        ),
        "growth_forms.parquet": pa.table({"id": ["g1"], "name": ["Branching"]}),
    }


@dataclass
class World:
    """A local store holding a raw and a taxonomy version, and a fake repo with the configs."""

    store: Path
    cache: Path
    out: Path
    repo: Path
    raw: Pin
    taxonomy: Pin

    def run(self, label_map: str, *pins: tuple[str, Pin], flags: Sequence[str] = ()) -> None:
        argv = [
            "curate",
            "coralnet",
            "--label-map",
            label_map,
            "--store",
            str(self.store),
            "--cache-dir",
            str(self.cache),
            "--out-dir",
            str(self.out),
        ]
        for prefix, pin in pins:
            argv += [f"--{prefix}version", pin.version]
            argv += [f"--{prefix}manifest-sha256", pin.manifest_sha256]
        assert cli.main([*argv, *flags]) == 0

    def published(self) -> dict[str, tuple[Manifest, Pin]]:
        """Every published coralnet version, by version name."""
        found: dict[str, tuple[Manifest, Pin]] = {}
        for folder in (self.store / DATASET_BUCKET / "datasets" / "coralnet").iterdir():
            data = (folder / "manifest.json").read_bytes()
            sha = hashlib.sha256(data).hexdigest()
            pin = Pin(name="coralnet", version=folder.name, manifest_sha256=sha)
            found[folder.name] = (Manifest.model_validate_json(data), pin)
        return found

    def table(self, version: str, name: str) -> pd.DataFrame:
        path = self.store / DATASET_BUCKET / "datasets" / "coralnet" / version / name
        return pd.read_parquet(path)


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> World:
    repo = tmp_path / "repo"
    (repo / "configs" / "coralnet").mkdir(parents=True)
    (repo / "uv.lock").write_text("lock")
    pd.DataFrame({"image_id": [BLACK], "reason": ["exact"]}).to_parquet(
        repo / "configs" / "coralnet" / "image_blacklist.parquet"
    )
    pd.DataFrame({"image_id": [OVERRIDDEN], "row": [7], "col": [8], "coralnet_id": [4]}).to_parquet(
        repo / "configs" / "coralnet" / "annotation_overrides.parquet"
    )
    monkeypatch.setattr(versions, "find_repo_root", lambda _start: repo)
    monkeypatch.setattr(publishing, "probe_git", lambda _dir: CLEAN_GIT)
    monkeypatch.setattr("segdata_spec.publish.probe_git", lambda _dir: CLEAN_GIT)

    store = tmp_path / "store"
    client = LocalStore(store)
    _, raw_sha = publish_dataset(
        client,
        bucket=DATASET_BUCKET,
        dataset="coralnet-raw",
        version="20261101-1111111",
        kind="points",
        tables=_raw_tables(),
        image_store=RAW_STORE,
        etl_package="segdata-etl",
        etl_command="test",
        source=RAW_SOURCE,
        repo_dir=repo,
        git=CLEAN_GIT,
    )
    _, tax_sha = publish_taxonomy(
        client,
        bucket=DATASET_BUCKET,
        name="mermaid-benthic",
        version="20261101-2222222",
        tables=_taxonomy_tables(),
        etl_package="segdata-etl",
        etl_command="test",
        source=RAW_SOURCE,
        repo_dir=repo,
        git=CLEAN_GIT,
    )
    return World(
        store=store,
        cache=tmp_path / "cache",
        out=tmp_path / "out",
        repo=repo,
        raw=Pin(name="coralnet-raw", version="20261101-1111111", manifest_sha256=raw_sha),
        taxonomy=Pin(name="mermaid-benthic", version="20261101-2222222", manifest_sha256=tax_sha),
    )


def _ref(pin: Pin) -> ParentRef:
    return ParentRef(dataset=pin.name, version=pin.version, manifest_sha256=pin.manifest_sha256)


def test_v1_curates_raw_and_maps_labels_by_the_cleaned_up_rerun_json(world: World) -> None:
    world.run("cleaned_up_rerun", ("raw-", world.raw), ("taxonomy-", world.taxonomy))

    ((version, (manifest, _pin)),) = world.published().items()
    assert manifest.parents == (_ref(world.raw),)
    assert manifest.label_mapping is not None
    assert manifest.label_mapping.source == "cleaned_up_rerun_json"
    assert manifest.stats.dropped["blacklisted_images"] == 1
    assert manifest.image_store == RAW_STORE
    assert manifest.source == RAW_SOURCE
    labels = world.table(version, "labels.parquet")
    assert sorted(labels["label_id"]) == [1, 2, 3, 4]


def test_v2_keeps_v1_tables_and_maps_labels_by_the_api(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(coralnet_api, "_retrying_session", FakeApiSession)
    world.run("cleaned_up_rerun", ("raw-", world.raw), ("taxonomy-", world.taxonomy))
    ((v1_version, (v1, v1_pin)),) = world.published().items()

    world.run("mermaid_api", ("raw-", world.raw), ("parent-", v1_pin))

    (v2_version,) = set(world.published()) - {v1_version}
    v2 = world.published()[v2_version][0]
    assert v2.parents == (_ref(v1_pin), _ref(world.raw))
    for name in ("images.parquet", "points.parquet", "stats/label_counts.parquet"):
        assert v2.files[name].sha256 == v1.files[name].sha256
    assert v2.stats.dropped == v1.stats.dropped
    snapshot = json.loads((world.out / v2_version / "coralnet_label_mappings.json").read_text())
    assert v2.label_mapping is not None
    assert v2.label_mapping.source == "mermaid_api"
    assert v2.label_mapping.sha256 == snapshot["sha256"]
    labels = world.table(v2_version, "labels.parquet")
    assert sorted(labels["label_id"]) == [1, 2, 3, 4]
    _require_raw_parent(v2, world.raw)


@pytest.mark.parametrize(
    ("label_map", "pins", "named"),
    [
        ("mermaid_api", (), "--parent-version"),
        ("cleaned_up_rerun", (), "--taxonomy-version"),
        ("cleaned_up_rerun", ("taxonomy-", "parent-"), "neither of --parent-version"),
    ],
)
def test_a_label_map_with_the_wrong_pins_is_a_usage_error(
    world: World,
    label_map: str,
    pins: tuple[str, ...],
    named: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        world.run(label_map, ("raw-", world.raw), *((prefix, world.taxonomy) for prefix in pins))
    assert exit_info.value.code == 2
    assert named in capsys.readouterr().err
    assert not (world.store / DATASET_BUCKET / "datasets" / "coralnet").exists()


def test_half_a_parent_pin_is_a_usage_error(
    world: World, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        world.run("mermaid_api", ("raw-", world.raw), flags=["--parent-version", "v1"])
    assert exit_info.value.code == 2
    assert "needs --parent-version" in capsys.readouterr().err


def test_v2_refuses_before_publishing_when_a_reused_table_differs_from_v1(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(coralnet_api, "_retrying_session", FakeApiSession)
    world.run("cleaned_up_rerun", ("raw-", world.raw), ("taxonomy-", world.taxonomy))
    ((v1_version, (_v1, v1_pin)),) = world.published().items()
    real_build_v2 = versions.build_v2

    def tampered(*args: Any, **kwargs: Any) -> versions.CuratedVersion:
        built = real_build_v2(*args, **kwargs)
        points = built.tables["points.parquet"]
        built.tables["points.parquet"] = points.slice(0, points.num_rows - 1)
        return built

    monkeypatch.setattr(versions, "build_v2", tampered)
    with pytest.raises(RuntimeError, match=r"points\.parquet"):
        world.run("mermaid_api", ("raw-", world.raw), ("parent-", v1_pin))

    folders = (world.store / DATASET_BUCKET / "datasets" / "coralnet").iterdir()
    assert [p.name for p in folders] == [v1_version]


def test_v2_publishes_nothing_when_its_snapshot_cannot_be_written(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(coralnet_api, "_retrying_session", FakeApiSession)
    world.run("cleaned_up_rerun", ("raw-", world.raw), ("taxonomy-", world.taxonomy))
    ((v1_version, (_v1, v1_pin)),) = world.published().items()
    world.out.write_text("a file where the snapshot directory goes")

    with pytest.raises(OSError):
        world.run("mermaid_api", ("raw-", world.raw), ("parent-", v1_pin))

    folders = (world.store / DATASET_BUCKET / "datasets" / "coralnet").iterdir()
    assert [p.name for p in folders] == [v1_version]


def test_a_dirty_tree_is_refused_before_any_sync(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        publishing, "probe_git", lambda _dir: GitState(sha="abcdef1234567", dirty=True, pushed=True)
    )

    def no_sync(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("the guard must fire before any sync")

    monkeypatch.setattr(versions, "sync_tables", no_sync)

    with pytest.raises(PublishRefused, match="uncommitted changes"):
        world.run("cleaned_up_rerun", ("raw-", world.raw), ("taxonomy-", world.taxonomy))
