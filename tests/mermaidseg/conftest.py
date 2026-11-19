"""Shared fixtures: a tiny trained run and the bundle exported from it."""

from __future__ import annotations

import multiprocessing as mp
import shutil
import tempfile
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from queue import Empty

import numpy as np
import pandas as pd
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as torch_mp

import mermaidseg.datasets.local_cache
from mermaidseg.dataset_reconciliation.concept_schema import ConceptSchema
from mermaidseg.datasets.local_cache import CacheStatsHandles, LocalS3Cache, create_cache_stats
from mermaidseg.export import export_bundle
from mermaidseg.io import ConfigDict
from tests.conftest import cbm_test_config
from tests.mermaidseg.inference import _tiny


@pytest.fixture(scope="module")
def run(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, torch.nn.Module]:
    return _tiny.write_run(tmp_path_factory.mktemp("run"))


@pytest.fixture(scope="module")
def bundle_dir(run: tuple[Path, torch.nn.Module], tmp_path_factory: pytest.TempPathFactory) -> Path:
    return export_bundle(run[0], tmp_path_factory.mktemp("out") / "bundle")


@pytest.fixture
def reset_local_cache() -> Iterator[None]:
    """Give the test a fresh ``LocalS3Cache`` singleton and drop it afterwards."""
    LocalS3Cache.reset_singleton()
    yield
    LocalS3Cache.reset_singleton()


def init_gloo(rank: int, world_size: int, init_method: str) -> None:
    dist.init_process_group("gloo", init_method=init_method, rank=rank, world_size=world_size)


# Each spawned worker re-imports torch before it reports, which takes over a minute on a slow CI
# runner. The deadline only bounds a real hang; a crashed worker fails the wait at once.
_GLOO_RESULT_TIMEOUT_S = 300


def spawn_gloo(worker, world_size: int, extra: tuple = ()) -> list:
    ctx = torch_mp.get_context("spawn")
    queue = ctx.Queue()
    rendezvous_dir = tempfile.mkdtemp()
    init_method = f"file://{rendezvous_dir}/rendezvous"
    processes = [
        ctx.Process(target=worker, args=(rank, world_size, init_method, queue, *extra))
        for rank in range(world_size)
    ]
    try:
        for process in processes:
            process.start()
        results = []
        deadline = time.monotonic() + _GLOO_RESULT_TIMEOUT_S
        while len(results) < world_size:
            try:
                results.append(queue.get(timeout=1))
            except Empty:
                for rank, process in enumerate(processes):
                    if process.exitcode not in (None, 0):
                        raise RuntimeError(
                            f"gloo worker {rank} exited with {process.exitcode} before reporting"
                        ) from None
                if time.monotonic() > deadline:
                    raise TimeoutError(
                        f"{len(results)} of {world_size} gloo workers reported within "
                        f"{_GLOO_RESULT_TIMEOUT_S} s"
                    ) from None
        for process in processes:
            process.join(timeout=30)
        for process in processes:
            if process.exitcode != 0:
                raise RuntimeError(f"gloo worker exited with {process.exitcode}")
    finally:
        for process in processes:
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
        shutil.rmtree(rendezvous_dir, ignore_errors=True)
    results.sort(key=lambda item: item[0])
    return [payload for _, payload in results]


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


BUCKET = "bucket"
KEY = "image.jpg"


def read_cached_object_in_worker(init_fn: Callable[[int], None]) -> None:
    """A worker body: run the init, then read one object the local cache already holds."""
    init_fn(0)
    LocalS3Cache.get().read_bytes(BUCKET, KEY)


@pytest.fixture
def spawn_cache_stats(monkeypatch) -> CacheStatsHandles:
    """Cache stats whose locks a spawn-started process can inherit, whatever the default."""
    # A fork-context SemLock cannot be passed to a spawn-context process.
    monkeypatch.setattr(mermaidseg.datasets.local_cache, "mp", mp.get_context("spawn"))
    return create_cache_stats()


def run_in_spawned_worker(init_fn: Callable[[int], None], cache_root: Path, monkeypatch) -> None:
    """Run ``read_cached_object_in_worker`` in a spawn-started process over ``cache_root``."""
    (cache_root / BUCKET).mkdir(parents=True)
    (cache_root / BUCKET / KEY).write_bytes(b"pixels")
    monkeypatch.setenv("MERMAIDSEG_LOCAL_CACHE_DIR", str(cache_root))
    worker = mp.get_context("spawn").Process(target=read_cached_object_in_worker, args=(init_fn,))
    worker.start()
    try:
        worker.join(timeout=60)
        assert worker.exitcode == 0
    finally:
        if worker.is_alive():
            worker.kill()
            worker.join()
