"""``segdata curate coralnet``: the v1 and v2 curated versions staged from raw, then promoted."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pytest
from uniform.conftest import FakeS3, crc64nvme_b64, md5_etag

from segdata_etl import cli, promote
from segdata_etl.common.copy_plan import COPIES_NAME, UPLOADS_NAME
from segdata_etl.common.stage import LocalDirS3
from segdata_etl.datasets.coralnet.curate import versions
from segdata_etl.label_maps import coralnet_api
from segdata_etl.splits.builder import require_raw_parent
from segdata_spec import DATASET_BUCKET
from segdata_spec.errors import PublishRefused
from segdata_spec.manifest import ImageStore, Manifest, ParentRef, Pin, SourceInfo
from segdata_spec.publish import GitState, publish_dataset, publish_taxonomy
from segdata_spec.schema import IMAGES_SCHEMA, POINTS_SCHEMA

CLEAN_GIT = GitState(sha="abcdef1234567", dirty=False, pushed=True)
RAW_STORE = ImageStore(bucket="img-bucket", prefix="coralnet/", owner="segdata")
RAW_SOURCE = SourceInfo(type="coralnet-scrape", uri="s3://scrape/", revision="2026-11-01")
RAW_VERSION = "20261101-1111111"
STAGED_RAW_VERSION = "20261102-3333333"
KEPT, BLACK, OVERRIDDEN = "10", "11", "12"
SNAPSHOT = "coralnet_label_mappings.json"
_TABLES = ("images.parquet", "points.parquet", "labels.parquet", "stats/label_counts.parquet")
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


class NoS3:
    """An S3 client that fails the test on any call."""

    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"S3 {name} called")


def _blob(image_id: str) -> bytes:
    return f"jpeg bytes of {image_id}".encode()


def _image(image_id: str, group: str) -> dict[str, object]:
    data = _blob(image_id)
    return {
        "image_id": image_id,
        "key": f"{group}/{image_id}.jpg",
        "etag": md5_etag(data),
        "size": len(data),
        "crc64nvme": crc64nvme_b64(data),
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


def _raw_images() -> list[dict[str, object]]:
    return [_image(KEPT, "s1"), _image(BLACK, "s1"), _image(OVERRIDDEN, "s2")]


def _raw_tables() -> dict[str, pa.Table]:
    images = pa.Table.from_pylist(_raw_images(), schema=IMAGES_SCHEMA)
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


def _publish_raw(client: Any, version: str, repo: Path) -> Pin:
    _, sha = publish_dataset(
        client,
        bucket=DATASET_BUCKET,
        dataset="coralnet-raw",
        version=version,
        kind="points",
        tables=_raw_tables(),
        image_store=RAW_STORE,
        etl_package="segdata-etl",
        etl_command="test",
        source=RAW_SOURCE,
        repo_dir=repo,
        git=CLEAN_GIT,
    )
    return Pin(name="coralnet-raw", version=version, manifest_sha256=sha)


@dataclass
class World:
    """A fake S3 holding a published raw (with its images) and taxonomy, a stage and a repo."""

    s3: FakeS3
    stage: Path
    cache: Path
    repo: Path
    raw: Pin
    taxonomy: Pin

    def run(
        self,
        label_map: str,
        *pins: tuple[str, Pin],
        flags: Sequence[str] = (),
        cache: Path | None = None,
    ) -> None:
        argv = [
            "curate",
            "coralnet",
            "--label-map",
            label_map,
            "--stage-dir",
            str(self.stage),
            "--cache-dir",
            str(cache or self.cache),
        ]
        for prefix, pin in pins:
            argv += [f"--{prefix}version", pin.version]
            argv += [f"--{prefix}manifest-sha256", pin.manifest_sha256]
        assert cli.main([*argv, *flags]) == 0

    def coralnet_dir(self) -> Path:
        return self.stage / DATASET_BUCKET / "datasets" / "coralnet"

    def staged(self) -> dict[str, tuple[Manifest, Pin]]:
        """Every staged coralnet version, by version name."""
        found: dict[str, tuple[Manifest, Pin]] = {}
        for folder in self.coralnet_dir().iterdir():
            data = (folder / "manifest.json").read_bytes()
            sha = hashlib.sha256(data).hexdigest()
            pin = Pin(name="coralnet", version=folder.name, manifest_sha256=sha)
            found[folder.name] = (Manifest.model_validate_json(data), pin)
        return found

    def table(self, version: str, name: str) -> pd.DataFrame:
        return pd.read_parquet(self.coralnet_dir() / version / name)


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
    monkeypatch.setattr(versions, "probe_git", lambda _dir: CLEAN_GIT)

    s3 = FakeS3()
    for row in _raw_images():
        s3.put(RAW_STORE.bucket, RAW_STORE.prefix + str(row["key"]), _blob(str(row["image_id"])))
    raw = _publish_raw(s3, RAW_VERSION, repo)
    _, tax_sha = publish_taxonomy(
        s3,
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
    s3.calls.clear()
    monkeypatch.setattr(cli, "make_s3_client", lambda: s3)
    return World(
        s3=s3,
        stage=tmp_path / "stage",
        cache=tmp_path / "cache",
        repo=repo,
        raw=raw,
        taxonomy=Pin(name="mermaid-benthic", version="20261101-2222222", manifest_sha256=tax_sha),
    )


def _ref(pin: Pin) -> ParentRef:
    return ParentRef(dataset=pin.name, version=pin.version, manifest_sha256=pin.manifest_sha256)


def _stage_v1(world: World, cache: Path | None = None) -> tuple[str, Manifest, Pin]:
    world.run("cleaned_up_rerun", ("raw-", world.raw), ("taxonomy-", world.taxonomy), cache=cache)
    ((version, (manifest, pin)),) = world.staged().items()
    return version, manifest, pin


def _new_keys(s3: FakeS3, before: set[tuple[str, str]]) -> set[str]:
    return {key for bucket, key in set(s3.objects) - before if bucket == DATASET_BUCKET}


def test_v1_stages_curated_tables_on_raws_image_store_without_a_plan_or_s3_write(
    world: World,
) -> None:
    before = dict(world.s3.objects)

    version, manifest, _pin = _stage_v1(world)

    assert manifest.parents == (_ref(world.raw),)
    assert manifest.image_store == RAW_STORE
    assert manifest.source == RAW_SOURCE
    assert manifest.label_mapping is not None
    assert manifest.label_mapping.source == "cleaned_up_rerun_json"
    assert manifest.stats.dropped["blacklisted_images"] == 1
    assert manifest.extra_files == {}
    labels = world.table(version, "labels.parquet")
    assert sorted(labels["label_id"]) == [1, 2, 3, 4]
    acro = labels[labels["label_id"] == 1].iloc[0]
    assert acro["mermaid_benthic_attribute_id"] == "ba-acro"
    assert acro["mapping_source"] == "cleaned_up_rerun_json"
    staged = {p.name for p in (world.coralnet_dir() / version).rglob("*") if p.is_file()}
    assert staged == {
        "manifest.json",
        "images.parquet",
        "points.parquet",
        "labels.parquet",
        "label_counts.parquet",
    }
    assert not (world.coralnet_dir() / version / COPIES_NAME).exists()
    assert not (world.coralnet_dir() / version / UPLOADS_NAME).exists()
    assert world.s3.objects == before
    assert world.s3.ops("put") == []


def test_v2_stages_v1_tables_with_api_labels_and_lists_the_snapshot_as_an_extra_file(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(coralnet_api, "_retrying_session", FakeApiSession)
    v1_version, v1, v1_pin = _stage_v1(world)

    # v1 is only staged, so it is read from the stage dir as the cache.
    world.run("mermaid_api", ("raw-", world.raw), ("parent-", v1_pin), cache=world.stage)

    (v2_version,) = set(world.staged()) - {v1_version}
    v2 = world.staged()[v2_version][0]
    assert v2.parents == (_ref(v1_pin), _ref(world.raw))
    assert v2.image_store == RAW_STORE
    for name in ("images.parquet", "points.parquet", "stats/label_counts.parquet"):
        assert v2.files[name].sha256 == v1.files[name].sha256
    assert v2.stats.dropped == v1.stats.dropped
    snapshot = (world.coralnet_dir() / v2_version / SNAPSHOT).read_bytes()
    assert set(v2.extra_files) == {SNAPSHOT}
    assert v2.extra_files[SNAPSHOT].sha256 == hashlib.sha256(snapshot).hexdigest()
    assert v2.extra_files[SNAPSHOT].size == len(snapshot)
    assert v2.label_mapping is not None
    assert v2.label_mapping.source == "mermaid_api"
    labels = world.table(v2_version, "labels.parquet")
    acro = labels[labels["label_id"] == 1].iloc[0]
    assert acro["mermaid_benthic_attribute_id"] == "api-acro"
    assert acro["mapping_source"] == "mermaid_api"
    assert not (world.coralnet_dir() / v2_version / COPIES_NAME).exists()
    require_raw_parent(v2, world.raw)
    assert world.s3.ops("put") == []


def test_staged_v1_and_v2_promote_as_tables_only_versions(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(coralnet_api, "_retrying_session", FakeApiSession)
    v1_version, _v1, v1_pin = _stage_v1(world)
    before = set(world.s3.objects)

    promote.promote(
        world.s3, stage_dir=world.stage, dataset="coralnet", version=v1_version, workers=2
    )

    v1_prefix = f"datasets/coralnet/{v1_version}/"
    assert _new_keys(world.s3, before) == {v1_prefix + n for n in (*_TABLES, "manifest.json")}
    assert world.s3.ops("copy") == []

    world.run("mermaid_api", ("raw-", world.raw), ("parent-", v1_pin))
    (v2_version,) = set(world.staged()) - {v1_version}
    before = set(world.s3.objects)

    promote.promote(
        world.s3, stage_dir=world.stage, dataset="coralnet", version=v2_version, workers=2
    )

    v2_prefix = f"datasets/coralnet/{v2_version}/"
    assert _new_keys(world.s3, before) == {
        v2_prefix + n for n in (*_TABLES, SNAPSHOT, "manifest.json")
    }
    assert world.s3.ops("copy") == []


def test_v1_against_a_raw_staged_only_in_the_stage_dir_reads_no_raw_from_s3(
    world: World,
) -> None:
    staged_raw = _publish_raw(LocalDirS3(world.stage), STAGED_RAW_VERSION, world.repo)

    world.run(
        "cleaned_up_rerun", ("raw-", staged_raw), ("taxonomy-", world.taxonomy), cache=world.stage
    )

    ((_version, (manifest, _pin)),) = world.staged().items()
    assert manifest.parents == (_ref(staged_raw),)
    assert [c for c in world.s3.calls if "coralnet-raw" in c[2]] == []


@pytest.mark.parametrize(
    ("git", "message"),
    [
        (GitState(sha="abcdef1234567", dirty=True, pushed=True), "uncommitted changes"),
        (GitState(sha="abcdef1234567", dirty=False, pushed=False), "not contained"),
    ],
)
def test_a_dirty_or_unpushed_tree_is_refused_before_any_s3_call(
    world: World, monkeypatch: pytest.MonkeyPatch, git: GitState, message: str
) -> None:
    monkeypatch.setattr(versions, "probe_git", lambda _dir: git)
    monkeypatch.setattr(cli, "make_s3_client", NoS3)

    with pytest.raises(PublishRefused, match=message):
        world.run("cleaned_up_rerun", ("raw-", world.raw), ("taxonomy-", world.taxonomy))

    assert not world.stage.exists()


def test_the_recorded_command_names_the_stage_dir(world: World) -> None:
    _version, manifest, _pin = _stage_v1(world)

    command = manifest.etl.command
    assert f"--stage-dir {world.stage}" in command
    assert "--store" not in command


@pytest.mark.parametrize(
    ("argv", "named"),
    [
        (["--cache-dir", "c"], "--stage-dir"),
        (["--stage-dir", "s", "--store", "x"], "unrecognized arguments: --store"),
    ],
)
def test_stage_dir_is_required_and_store_is_gone(
    world: World, argv: list[str], named: str, capsys: pytest.CaptureFixture[str]
) -> None:
    pins = ["--raw-version", "v", "--raw-manifest-sha256", "0" * 64]
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["curate", "coralnet", "--label-map", "mermaid_api", *pins, *argv])
    assert exit_info.value.code == 2
    assert named in capsys.readouterr().err


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
    assert not world.coralnet_dir().exists()


def test_half_a_parent_pin_is_a_usage_error(
    world: World, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        world.run("mermaid_api", ("raw-", world.raw), flags=["--parent-version", "v1"])
    assert exit_info.value.code == 2
    assert "needs --parent-version" in capsys.readouterr().err


def test_v2_refuses_before_staging_when_a_reused_table_differs_from_v1(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(coralnet_api, "_retrying_session", FakeApiSession)
    v1_version, _v1, v1_pin = _stage_v1(world)
    real_build_v2 = versions.build_v2

    def tampered(*args: Any, **kwargs: Any) -> versions.CuratedVersion:
        built = real_build_v2(*args, **kwargs)
        points = built.tables["points.parquet"]
        built.tables["points.parquet"] = points.slice(0, points.num_rows - 1)
        return built

    monkeypatch.setattr(versions, "build_v2", tampered)
    with pytest.raises(RuntimeError, match=r"points\.parquet"):
        world.run("mermaid_api", ("raw-", world.raw), ("parent-", v1_pin), cache=world.stage)

    assert [p.name for p in world.coralnet_dir().iterdir()] == [v1_version]
