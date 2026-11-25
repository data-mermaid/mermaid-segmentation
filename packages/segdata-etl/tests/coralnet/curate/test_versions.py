"""``segdata curate coralnet``: the curated version staged from raw and the API mapping, then promoted."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pytest
import requests
from s3_fake import FakeS3, crc64nvme, quoted_etag

from segdata_etl import cli, promote
from segdata_etl.common.copy_plan import COPIES_NAME, UPLOADS_NAME
from segdata_etl.datasets.coralnet.curate import versions
from segdata_etl.label_maps import coralnet_api
from segdata_spec import DATASET_BUCKET
from segdata_spec.errors import PublishRefused
from segdata_spec.git import GitState
from segdata_spec.manifest import ImageStore, Manifest, ParentRef, Pin, SourceInfo
from segdata_spec.schema import IMAGES_SCHEMA, POINTS_SCHEMA
from segdata_spec.stage import stage_dataset, stage_taxonomy

CLEAN_GIT = GitState(sha="abcdef1234567", dirty=False, pushed=True)
RAW_STORE = ImageStore(bucket="img-bucket", prefix="coralnet/", owner="segdata")
RAW_SOURCE = SourceInfo(type="coralnet-scrape", uri="s3://scrape/", revision="2026-11-01")
RAW_VERSION = "20261101-1111111"
STAGED_RAW_VERSION = "20261102-3333333"
KEPT, BLACK, OVERRIDDEN = "10", "11", "12"
_TABLES = ("images.parquet", "points.parquet", "labels.parquet", "stats/label_counts.parquet")
API_ROWS = [
    {
        "id": "m1",
        "benthic_attribute_id": "ba-acro",
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
    """The MERMAID LabelMapping API as one page of ``rows``, or an HTTP error."""

    def __init__(self, rows: list[dict[str, object]], error: Exception | None = None):
        self.rows = rows
        self.error = error

    def __enter__(self) -> FakeApiSession:
        return self

    def __exit__(self, *_: object) -> None:
        pass

    def get(self, url: str, timeout: float) -> _Response:
        assert url == coralnet_api.CORALNET_LABELMAPPINGS_URL
        if self.error:
            raise self.error
        return _Response({"results": self.rows, "next": None})


class NoS3:
    """An S3 client that fails the test on any call."""

    def __init__(self, **_kw: Any) -> None:
        pass

    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"S3 {name} called")


def _blob(image_id: str) -> bytes:
    return f"jpeg bytes of {image_id}".encode()


def _image(image_id: str, group: str) -> dict[str, object]:
    data = _blob(image_id)
    return {
        "image_id": image_id,
        "key": f"{group}/{image_id}.jpg",
        "etag": quoted_etag(data),
        "size": len(data),
        "crc64nvme": crc64nvme(data),
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


def _upload_staged(client: FakeS3, stage_dir: Path) -> None:
    """Seed ``client`` with every staged object, standing in for ``segdata promote``."""
    root = stage_dir / DATASET_BUCKET
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        client.put(DATASET_BUCKET, path.relative_to(root).as_posix(), path.read_bytes())


def _stage_raw(stage_dir: Path, version: str, repo: Path) -> Pin:
    _, sha = stage_dataset(
        stage_dir,
        parent_dirs=(stage_dir,),
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
        *pins: tuple[str, Pin],
        flags: Sequence[str] = (),
        cache: Path | None = None,
    ) -> None:
        argv = [
            "curate",
            "coralnet",
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
    repo.mkdir()
    config = tmp_path / "curate_config"
    config.mkdir()
    (repo / "uv.lock").write_text("lock")
    pd.DataFrame({"image_id": [BLACK], "reason": ["exact"]}).to_parquet(
        config / "image_blacklist.parquet"
    )
    pd.DataFrame({"image_id": [OVERRIDDEN], "row": [7], "col": [8], "coralnet_id": [4]}).to_parquet(
        config / "annotation_overrides.parquet"
    )
    monkeypatch.setattr(versions, "PACKAGED_CONFIG", config)
    monkeypatch.setattr(versions, "find_repo_root", lambda _start: repo)
    monkeypatch.setattr(versions, "probe_git", lambda _dir: CLEAN_GIT)
    monkeypatch.setattr(coralnet_api, "_retrying_session", lambda: FakeApiSession(API_ROWS))

    s3 = FakeS3()
    for row in _raw_images():
        s3.put(RAW_STORE.bucket, RAW_STORE.prefix + str(row["key"]), _blob(str(row["image_id"])))
    seed = tmp_path / "seed"
    raw = _stage_raw(seed, RAW_VERSION, repo)
    _, tax_sha = stage_taxonomy(
        seed,
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
    _upload_staged(s3, seed)
    s3.calls.clear()
    monkeypatch.setattr(cli, "make_s3_client", lambda **_kw: s3)
    # The raw parent is published to S3 only; neither the stage nor the cache dir holds a copy.
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


def _use_api(monkeypatch: pytest.MonkeyPatch, session: FakeApiSession) -> None:
    monkeypatch.setattr(coralnet_api, "_retrying_session", lambda: session)


def _stage_curated(world: World, cache: Path | None = None) -> tuple[str, Manifest, Pin]:
    world.run(("raw-", world.raw), ("taxonomy-", world.taxonomy), cache=cache)
    ((version, (manifest, pin)),) = world.staged().items()
    return version, manifest, pin


def _new_keys(s3: FakeS3, before: set[tuple[str, str]]) -> set[str]:
    return {key for bucket, key in set(s3.objects) - before if bucket == DATASET_BUCKET}


def test_curated_stages_curated_tables_on_raws_image_store_without_a_plan_or_s3_write(
    world: World,
) -> None:
    before = dict(world.s3.objects)

    version, manifest, _pin = _stage_curated(world)

    assert manifest.parents == (_ref(world.raw),)
    assert manifest.image_store == RAW_STORE
    assert manifest.source == RAW_SOURCE
    assert manifest.label_mapping is not None
    assert manifest.label_mapping.source == "mermaid_api"
    assert manifest.label_mapping.fetched_at is not None
    assert manifest.label_mapping.sha256 == coralnet_api.coralnet_mapping_digest(
        coralnet_api.parse_coralnet_label_mappings(API_ROWS)
    )
    assert manifest.label_mapping.n_mapped == 1
    assert manifest.stats.dropped["blacklisted_images"] == 1
    labels = world.table(version, "labels.parquet")
    assert sorted(labels["label_id"]) == [1, 2, 3, 4]
    acro = labels[labels["label_id"] == 1].iloc[0]
    assert acro["mermaid_benthic_attribute_id"] == "ba-acro"
    assert acro["mermaid_benthic_attribute"] == "Acropora"
    assert acro["mermaid_growth_form"] == "Branching"
    assert acro["mapping_source"] == "mermaid_api"
    assert labels[labels["label_id"] != 1]["mermaid_benthic_attribute_id"].isna().all()
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


def test_staged_curated_version_promotes_as_a_tables_only_version(world: World) -> None:
    version, _manifest, _pin = _stage_curated(world)
    before = set(world.s3.objects)

    promote.promote(world.s3, stage_dir=world.stage, dataset="coralnet", version=version, workers=2)

    prefix = f"datasets/coralnet/{version}/"
    assert _new_keys(world.s3, before) == {prefix + n for n in (*_TABLES, "manifest.json")}
    assert world.s3.ops("copy") == []


def test_curated_against_a_raw_staged_only_in_the_stage_dir_reads_no_raw_from_s3(
    world: World,
) -> None:
    staged_raw = _stage_raw(world.stage, STAGED_RAW_VERSION, world.repo)

    world.run(("raw-", staged_raw), ("taxonomy-", world.taxonomy), cache=world.stage)

    ((_version, (manifest, _pin)),) = world.staged().items()
    assert manifest.parents == (_ref(staged_raw),)
    assert [c for c in world.s3.calls if "coralnet-raw" in c[2]] == []


def test_curated_refuses_a_taxonomy_cached_but_not_published(world: World) -> None:
    first, _manifest, _pin = _stage_curated(world)
    for key in [k for k in world.s3.objects if k[1].startswith("taxonomy/")]:
        del world.s3.objects[key]

    with pytest.raises(PublishRefused, match=f"{world.taxonomy.version} is not published"):
        world.run(("raw-", world.raw), ("taxonomy-", world.taxonomy))

    assert list(world.staged()) == [first]


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
        world.run(("raw-", world.raw), ("taxonomy-", world.taxonomy))

    assert not world.stage.exists()


def test_the_recorded_command_names_the_stage_dir(world: World) -> None:
    _version, manifest, _pin = _stage_curated(world)

    command = manifest.etl.command
    assert f"--stage-dir {world.stage}" in command
    assert (
        f"--raw-version {world.raw.version} --raw-manifest-sha256 {world.raw.manifest_sha256}"
        in command
    )
    assert (
        f"--taxonomy-version {world.taxonomy.version} "
        f"--taxonomy-manifest-sha256 {world.taxonomy.manifest_sha256}"
    ) in command
    assert "--label-map" not in command


def test_an_api_attribute_absent_from_the_taxonomy_is_refused_and_stages_nothing(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    _use_api(monkeypatch, FakeApiSession([{**API_ROWS[0], "benthic_attribute_id": "ba-gone"}]))

    with pytest.raises(ValueError, match="ba-gone"):
        world.run(("raw-", world.raw), ("taxonomy-", world.taxonomy))

    assert not world.coralnet_dir().exists()


def test_an_api_growth_form_absent_from_the_taxonomy_is_refused_and_stages_nothing(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    _use_api(monkeypatch, FakeApiSession([{**API_ROWS[0], "growth_form_id": "g-gone"}]))

    with pytest.raises(ValueError, match="g-gone"):
        world.run(("raw-", world.raw), ("taxonomy-", world.taxonomy))

    assert not world.coralnet_dir().exists()


@pytest.mark.parametrize(
    "error",
    [requests.HTTPError("503"), requests.ConnectionError("refused"), requests.Timeout("slow")],
    ids=["http", "connection", "timeout"],
)
def test_an_api_request_error_propagates_and_stages_nothing(
    world: World, monkeypatch: pytest.MonkeyPatch, error: requests.RequestException
) -> None:
    _use_api(monkeypatch, FakeApiSession([], error=error))

    with pytest.raises(type(error)):
        world.run(("raw-", world.raw), ("taxonomy-", world.taxonomy))

    assert not world.coralnet_dir().exists()


def test_stage_dir_is_required(world: World, capsys: pytest.CaptureFixture[str]) -> None:
    pins = ["--raw-version", "v", "--raw-manifest-sha256", "0" * 64]
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["curate", "coralnet", *pins, "--cache-dir", "c"])
    assert exit_info.value.code == 2
    assert "--stage-dir" in capsys.readouterr().err


def test_omitting_the_taxonomy_pins_is_a_usage_error(
    world: World, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        world.run(("raw-", world.raw))
    assert exit_info.value.code == 2
    assert "--taxonomy-version" in capsys.readouterr().err
    assert not world.coralnet_dir().exists()


def test_the_removed_label_map_flag_is_a_usage_error(
    world: World, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        world.run(
            ("raw-", world.raw),
            ("taxonomy-", world.taxonomy),
            flags=["--label-map", "cleaned_up_rerun"],
        )
    assert exit_info.value.code == 2
    assert "unrecognized arguments" in capsys.readouterr().err
    assert not world.coralnet_dir().exists()


def test_packaged_config_holds_both_curate_inputs() -> None:
    with resources.as_file(versions.PACKAGED_CONFIG) as config_dir:
        blacklist = pd.read_parquet(config_dir / "image_blacklist.parquet")
        overrides = pd.read_parquet(config_dir / "annotation_overrides.parquet")
    assert len(blacklist) > 0
    assert list(blacklist.columns) == [
        "source_id",
        "image_id",
        "hash",
        "kept_source_id",
        "kept_image_id",
        "reason",
    ]
    assert len(overrides) > 0
    assert list(overrides.columns) == [
        "source_id",
        "image_id",
        "row",
        "col",
        "coralnet_id",
        "category",
        "has_unmapped",
        "in_kept",
        "n_copies_at_loc",
        "hash",
    ]
