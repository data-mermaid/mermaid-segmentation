"""Shared fixtures: a tiny trained run and the bundle exported from it."""

from __future__ import annotations

import hashlib
import io
import subprocess
from collections.abc import Callable, Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
import torch
import torch.distributed as dist
from botocore.exceptions import ClientError

from mermaidseg.data import PointDataset
from mermaidseg.data.config import DataSection
from mermaidseg.dataset_reconciliation.artifact_meta import file_sha1
from mermaidseg.dataset_reconciliation.concept_schema import ConceptSchema
from mermaidseg.dataset_reconciliation.label_space import (
    LabelSpace,
    label_keys_sha1,
    read_label_space,
    resolve_label_space,
    write_label_space,
)
from mermaidseg.export import export_bundle
from mermaidseg.io import ConfigDict, resolve_repo_path
from mermaidseg.lineage import data_version as section_data_version
from segdata_spec import DATASET_BUCKET
from tests.conftest import cbm_test_config
from tests.mermaidseg.data.conftest import ImageSpec, open_written
from tests.mermaidseg.inference import _tiny

DS_A = ("datasets", "ds-a", "20261120-aaaaaaa")
DS_B = ("datasets", "ds-b", "20261121-bbbbbbb")
TAX = ("taxonomy", "mermaid-benthic", "20261119-ccccccc")
MANIFESTS = {DS_A: b'{"name": "ds-a"}', DS_B: b'{"name": "ds-b"}', TAX: b'{"name": "tax"}'}


class FakeS3:
    """The ``get_object`` surface of a boto3 S3 client over an in-memory ``(bucket, key)`` map."""

    def __init__(self, objects: dict[tuple[str, str], bytes], error: ClientError | None = None):
        self.objects = objects
        self.error = error
        self.requested: list[tuple[str, str]] = []

    def get_object(self, *, Bucket: str, Key: str, **kwargs: Any) -> dict[str, Any]:
        self.requested.append((Bucket, Key))
        if self.error is not None:
            raise self.error
        if (Bucket, Key) not in self.objects:
            raise ClientError(
                {
                    "Error": {"Code": "NoSuchKey", "Message": "missing"},
                    "ResponseMetadata": {"HTTPStatusCode": 404},
                },
                "GetObject",
            )
        data = self.objects[(Bucket, Key)]
        return {"Body": io.BytesIO(data), "ContentLength": len(data)}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def manifest_key(pin: tuple[str, str, str]) -> tuple[str, str]:
    kind, name, version = pin
    return (DATASET_BUCKET, f"{kind}/{name}/{version}/manifest.json")


def seeded_s3(*, without: tuple[str, str, str] | None = None) -> FakeS3:
    objects = {manifest_key(pin): data for pin, data in MANIFESTS.items() if pin != without}
    return FakeS3(objects)


def data_version() -> dict:
    return {
        "datasets": {
            "ds-a": {
                "version": DS_A[2],
                "manifest_sha256": sha(MANIFESTS[DS_A]),
                "resize": None,
            },
            "ds-b": {
                "version": DS_B[2],
                "manifest_sha256": sha(MANIFESTS[DS_B]),
                "resize": {"max_side": 512},
            },
        },
        "taxonomy": {"version": TAX[2], "manifest_sha256": sha(MANIFESTS[TAX])},
    }


def git(cwd: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def commit(repo: Path, name: str) -> str:
    (repo / name).write_text(name)
    git(repo, "add", name)
    git(repo, "commit", "-m", name)
    return git(repo, "rev-parse", "HEAD")


def repo_with_origin(base: Path) -> tuple[Path, str]:
    """A repo whose ``origin/main`` tracking ref holds the returned sha."""
    origin = base / "origin.git"
    origin.mkdir()
    git(origin, "init", "--bare", "--initial-branch=main")
    repo = base / "repo"
    repo.mkdir()
    git(repo, "init", "--initial-branch=main")
    git(repo, "remote", "add", "origin", str(origin))
    pushed = commit(repo, "pushed.txt")
    git(repo, "push", "origin", "main")
    return repo, pushed


def deployable_lineage(git_sha: str, **overrides: Any) -> dict:
    return {
        "git_sha": git_sha,
        "git_dirty": False,
        "resolved_config": _tiny.resolved_config(),
        "data_version": data_version(),
        **overrides,
    }


def export_ckpt(checkpoint: Path, out: Path, **overrides: Any) -> Path:
    """``export_bundle`` with a fully seeded S3 unless the caller passes its own client."""
    return export_bundle(checkpoint, out, **{"s3_client": seeded_s3(), **overrides})


@pytest.fixture(scope="module")
def deployable_run(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, Path, str, torch.nn.Module]:
    """A run with a clean, pinned, pushed lineage: (checkpoint, repo, pushed sha, model)."""
    base = tmp_path_factory.mktemp("deployable")
    repo, pushed = repo_with_origin(base)
    checkpoint, model = _tiny.write_run(base / "run", checkpoint_lineage=deployable_lineage(pushed))
    return checkpoint, repo, pushed, model


@pytest.fixture(scope="module")
def deployable(
    deployable_run: tuple[Path, Path, str, torch.nn.Module],
) -> tuple[Path, Path, str]:
    """The exportable run's (checkpoint, repo, pushed sha)."""
    return deployable_run[:3]


@pytest.fixture(scope="module")
def run(deployable_run: tuple[Path, Path, str, torch.nn.Module]) -> tuple[Path, torch.nn.Module]:
    return deployable_run[0], deployable_run[3]


@pytest.fixture(scope="module")
def bundle_dir(
    deployable: tuple[Path, Path, str], tmp_path_factory: pytest.TempPathFactory
) -> Path:
    checkpoint, repo, _ = deployable
    return export_ckpt(checkpoint, tmp_path_factory.mktemp("out") / "bundle", repo_dir=repo)


# One collective as a rank called it: its kind, op or source rank, shape, and dtype.
_Signature = tuple[str, str, tuple[int, ...], str]
_Call = tuple[_Signature, torch.Tensor | None]


def _same_calls(a: list[list[_Call]], b: list[list[_Call]]) -> bool:
    for calls_a, calls_b in zip(a, b, strict=True):
        if [sig for sig, _ in calls_a] != [sig for sig, _ in calls_b]:
            return False
        for (_, x), (_, y) in zip(calls_a, calls_b, strict=True):
            if x is not None and y is not None and not torch.equal(x.nan_to_num(), y.nan_to_num()):
                return False
    return True


class LockstepRanks:
    """Runs a function once per rank, serially, under faked ``torch.distributed`` collectives.

    No process group is started. Each pass runs every rank in turn and records each rank's
    collective calls. In the next pass, collective ``i`` returns the combination of every
    rank's input to its collective ``i`` from the previous pass, where all ranks called the
    same collective there; elsewhere it leaves the tensor unchanged. The first pass therefore
    sees only this rank's own values. Passes repeat until no rank's calls or inputs change,
    so ``fn`` runs several times per rank and must not depend on state from earlier runs.

    At that fixed point the ranks must have called the same collectives (kind, op or source,
    shape, and dtype; barriers included) in the same order, or ``run`` raises
    ``AssertionError``: a real job would hang or mix up values there. A rank that raises out
    of a collective (as ``reduce_components`` does on a key mismatch) stops there, as it
    would in a real job.
    """

    max_passes = 20

    def __init__(self, monkeypatch: pytest.MonkeyPatch, world_size: int) -> None:
        self.monkeypatch = monkeypatch
        self.world_size = world_size
        self.barriers: list[int] = []

    def run(self, fn: Callable[[int], Any]) -> list[Any]:
        """Return ``fn(rank)`` for each rank, from the pass at which the collectives settle."""
        previous: list[list[_Call]] | None = None
        for _ in range(self.max_passes):
            self.barriers = [0] * self.world_size
            outcomes = [self._pass(rank, fn, previous) for rank in range(self.world_size)]
            calls = [rank_calls for _, rank_calls in outcomes]
            if previous is not None and _same_calls(calls, previous):
                self._check_lockstep(calls)
                return [result for result, _ in outcomes]
            previous = calls
        raise AssertionError(f"collectives did not settle within {self.max_passes} passes")

    def _check_lockstep(self, calls: list[list[_Call]]) -> None:
        counts = [len(rank_calls) for rank_calls in calls]
        for index in range(max(counts)):
            signatures = [
                rank_calls[index][0] if index < len(rank_calls) else "no collective"
                for rank_calls in calls
            ]
            if len(set(signatures)) > 1:
                raise AssertionError(f"ranks disagree at collective {index}: {signatures}")

    def _pass(
        self, rank: int, fn: Callable[[int], Any], previous: list[list[_Call]] | None
    ) -> tuple[Any, list[_Call]]:
        seen: list[_Call] = []

        def collective(
            kind: str,
            detail: str,
            tensor: torch.Tensor | None,
            combine: Callable[[torch.Tensor], torch.Tensor] | None = None,
        ) -> None:
            index = len(seen)
            shape = () if tensor is None else tuple(tensor.shape)
            dtype = "" if tensor is None else str(tensor.dtype)
            signature: _Signature = (kind, detail, shape, dtype)
            seen.append((signature, None if tensor is None else tensor.detach().clone()))
            if tensor is None or combine is None or previous is None:
                return
            if all(index < len(c) and c[index][0] == signature for c in previous):
                peers = [cast(torch.Tensor, c[index][1]) for c in previous]
                tensor.copy_(combine(torch.stack(peers)))

        def all_reduce(tensor: torch.Tensor, op: Any = dist.ReduceOp.SUM, **_: Any) -> None:
            if op == dist.ReduceOp.SUM:
                collective("all_reduce", "sum", tensor, lambda peers: peers.sum(dim=0))
            elif op == dist.ReduceOp.MAX:
                collective("all_reduce", "max", tensor, lambda peers: peers.amax(dim=0))
            elif op == dist.ReduceOp.MIN:
                collective("all_reduce", "min", tensor, lambda peers: peers.amin(dim=0))
            else:
                raise NotImplementedError(f"LockstepRanks has no all_reduce op {op}")

        def broadcast(tensor: torch.Tensor, src: int, **_: Any) -> None:
            collective("broadcast", f"src={src}", tensor, lambda peers: peers[src])

        def barrier(**_: Any) -> None:
            collective("barrier", "", None)
            self.barriers[rank] += 1

        with self.monkeypatch.context() as patched:
            patched.setattr(dist, "is_available", lambda: True)
            patched.setattr(dist, "is_initialized", lambda: True)
            patched.setattr(dist, "get_rank", lambda *_, **__: rank)
            patched.setattr(dist, "get_world_size", lambda *_, **__: self.world_size)
            patched.setattr(dist, "get_backend", lambda *_, **__: "gloo")
            patched.setattr(dist, "all_reduce", all_reduce)
            patched.setattr(dist, "broadcast", broadcast)
            patched.setattr(dist, "barrier", barrier)
            return fn(rank), seen


@pytest.fixture()
def lockstep_ranks(monkeypatch: pytest.MonkeyPatch) -> Callable[[int], LockstepRanks]:
    """Factory for a ``LockstepRanks`` of the given world size."""
    return lambda world_size: LockstepRanks(monkeypatch, world_size)


def concept_table() -> pd.DataFrame:
    """A concept table with every taxonomic rank (the CBM model and the encoder need them all)."""
    rows = [
        # label, kingdom, phylum, class, order, family, genus, bleached, branching, tubular
        (
            "acropora",
            "animalia",
            "cnidaria",
            "hexacorallia",
            "scleractinia",
            "acroporidae",
            "acropora",
            "FALSE",
            "TRUE",
            "FALSE",
        ),
        (
            "porites",
            "animalia",
            "cnidaria",
            "hexacorallia",
            "scleractinia",
            "poritidae",
            "porites",
            "FALSE",
            "FALSE",
            "TRUE",
        ),
        (
            "bleached acropora",
            "animalia",
            "cnidaria",
            "hexacorallia",
            "scleractinia",
            "acroporidae",
            "acropora",
            "TRUE",
            "TRUE",
            "FALSE",
        ),
        ("sand", "none", "none", "none", "none", "none", "none", "FALSE", "FALSE", "FALSE"),
        ("unknown", *["not_given"] * 9),
    ]
    cols = [
        "kingdom",
        "phylum",
        "class",
        "order",
        "family",
        "genus",
        "bleached",
        "branching",
        "tubular",
    ]
    df = pd.DataFrame(rows, columns=["source_label_class_name", *cols])
    df.insert(1, "source_dataset_source", "coralnet")
    return df


@pytest.fixture(scope="module")
def table_csv(tmp_path_factory):
    path = tmp_path_factory.mktemp("concepts") / "class_to_concepts.csv"
    concept_table().to_csv(path, index=False)
    return path


@pytest.fixture(scope="module")
def cbm_schema(table_csv) -> ConceptSchema:
    return ConceptSchema.from_csv(table_csv, sources={"coralnet"})


def source_lookup(schema: ConceptSchema) -> torch.Tensor:
    """Global source-label id -> concept row: 0 background, 1 acropora, 2 porites."""
    rows = [np.zeros(schema.num_channels, np.float32)]
    rows += [schema.row_for("coralnet", name) for name in ("acropora", "porites")]
    return torch.from_numpy(np.stack(rows))


def cbm_mask_config() -> ConfigDict:
    cfg = cbm_test_config()
    cfg.training.loss.per_pixel_loss_weight = 0.5
    cfg.training.loss.per_image_loss_weight = 0.5
    return cfg


def cbm_meta(make_meta_model, schema: ConceptSchema):
    return make_meta_model(
        cbm_mask_config(),
        run_name="cbm-mask",
        num_classes=3,
        num_concepts=schema.num_channels,
        concept_value2id=schema.concept_value2id,
        source_to_concepts_lookup=source_lookup(schema),
    )


def concept_keep(schema: ConceptSchema) -> torch.Tensor:
    keep = torch.ones(2, schema.num_channels, dtype=torch.bool)
    keep[1, schema.channel_names.index("bleached")] = False
    return keep


# ---------- label space ----------

LABEL_SPACE_TAXONOMY = pa.table(
    {"id": ["1", "2", "3"], "name": ["Acropora", "Porites", "Sand"], "parent_id": [None] * 3}
)


def write_test_label_space(
    directory: Path,
    tables: Mapping[str, Any],
    training_cfg: Any,
    schema: ConceptSchema,
    section: DataSection,
    taxonomy: pa.Table = LABEL_SPACE_TAXONOMY,
) -> LabelSpace:
    """Resolve ``tables`` into a label space whose meta matches ``training_cfg`` and ``section``.

    Writes it to ``directory`` and returns it as read back.
    """
    space = resolve_label_space(tables, taxonomy, training_cfg, schema, train_groups={})
    space.meta.update(
        {
            "data_version": section_data_version(section),
            "label_keys_sha1": label_keys_sha1(training_cfg),
            "concepts_csv_sha1": file_sha1(resolve_repo_path(training_cfg.concept_mapping_path)),
        }
    )
    write_label_space(directory, space)
    return read_label_space(directory)


def label_space_reader(
    tmp_path: Path,
    name: str,
    labels: dict[int, tuple[str, str]],
    split: str = "train",
    global_offset: int = 0,
) -> PointDataset:
    """A points reader of source ``name`` over ``label_id -> (name, benthic attribute)``."""
    rows = [
        {"label_id": i, "name": label, "mermaid_benthic_attribute": attribute}
        for i, (label, attribute) in labels.items()
    ]
    version = open_written(
        tmp_path / name / split,
        kind="points",
        images=[ImageSpec("i1")],
        labels=rows,
        points=[("i1", 0, 0, min(labels))],
        name=name,
    )
    return PointDataset(version, split=split, global_offset=global_offset)  # type: ignore[arg-type]


CORALSCAPES_LABELS = {1: ("acropora", "Acropora"), 2: ("sand", "Sand")}
PACIFIC_LABELS = {1: ("porites", "Porites")}


def label_space_section(versions: Mapping[str, str]) -> DataSection:
    """A data section pinning each dataset ``name -> version`` with a made-up manifest sha256."""
    return DataSection.model_validate(
        {
            "datasets": {
                name: {"version": version, "manifest_sha256": f"sha-{name}"}
                for name, version in versions.items()
            },
            "taxonomy": {"version": "tax-v", "manifest_sha256": "sha-tax"},
        }
    )


@pytest.fixture()
def label_space_run(tmp_path: Path) -> SimpleNamespace:
    """A two-dataset run and the label space written for it.

    ``coralscapes_v2`` (offset 0) holds acropora and sand, ``pacific_labeled_corals`` (offset 2)
    holds porites; the class subset is Acropora and Sand. Attributes: ``readers`` (keyed by
    ``(name, split)``), ``training_cfg``, ``section``, ``concepts_csv``,
    ``space`` and ``directory``.
    """
    concepts_csv = tmp_path / "concepts.csv"
    table = concept_table()
    pd.concat(
        [
            table.assign(source_dataset_source=s)
            for s in ("coralscapes_v2", "pacific_labeled_corals")
        ]
    ).to_csv(concepts_csv, index=False)
    sources = {"coralscapes_v2", "pacific_labeled_corals"}
    schema = ConceptSchema.from_csv(concepts_csv, sources=sources)
    readers = {
        ("coralscapes_v2", "train"): label_space_reader(
            tmp_path, "coralscapes_v2", CORALSCAPES_LABELS
        ),
        ("pacific_labeled_corals", "val"): label_space_reader(
            tmp_path, "pacific_labeled_corals", PACIFIC_LABELS, split="val", global_offset=2
        ),
    }
    directory = tmp_path / "label_space"
    training_cfg = ConfigDict(
        {
            "concept_mapping_path": str(concepts_csv),
            "concept_removal": "",
            "concept_source_mask": "",
            "class_subset": ["Acropora", "Sand"],
            "label_roll_up": False,
            "label_space": str(directory),
        }
    )
    section = label_space_section(
        {name: ds.version.manifest.version for name, ds in _by_name(readers).items()}
    )
    space = write_test_label_space(
        directory,
        {name: ds.version for name, ds in _by_name(readers).items()},
        training_cfg,
        schema,
        section,
    )
    return SimpleNamespace(
        readers=readers,
        training_cfg=training_cfg,
        section=section,
        concepts_csv=concepts_csv,
        space=space,
        directory=directory,
    )


def _by_name(readers: Mapping[tuple[str, str], PointDataset]) -> dict[str, PointDataset]:
    return {name: ds for (name, _), ds in readers.items()}
