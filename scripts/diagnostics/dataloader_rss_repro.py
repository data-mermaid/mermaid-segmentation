#!/usr/bin/env python
"""Reproduce and isolate the DataLoader host-RAM growth (Phase A / proof).

The `dinov3-lora-qv-r8` run OOM-died after host RAM climbed linearly ~4 GB/day on a 32 GiB
box while GPU stayed flat. The hypothesis: `BaseCoralDataset.__getitem__` touches large
Python-object structures (`_annotation_positions_by_image` str-dict, the object columns of the
multi-million-row `df_annotations`) every sample, so forked DataLoader workers accumulate
copy-on-write (CoW) page copies that never reclaim under `persistent_workers=True`.

This harness measures per-config RSS-over-time to CONFIRM OR REFUTE that hypothesis before any
fix is written. It runs the same variable matrix the plan calls for:

    num_workers=0                       -> no fork, no CoW      -> expect FLAT
    num_workers=N, persistent=False     -> CoW within an epoch  -> expect RISE-then-RESET each epoch
    num_workers=N, persistent=True      -> CoW never reclaimed  -> expect MONOTONIC RISE across epochs

FORK SEMANTICS MATTER. The leak is a Linux `fork` CoW effect. macOS DataLoaders default to
`spawn` (pickles the dataset per worker — a different profile). The harness forces
`multiprocessing_context="fork"` so it reproduces the SageMaker mechanism; on macOS `fork` can be
unstable, so treat a macOS run as indicative only — the authoritative run is inside the training
Docker image or a short `ml.g6.2xlarge` SageMaker job.

Modes:
  --synthetic (default): a minimal BaseCoralDataset with a large synthetic df_annotations. No S3,
      no creds, deterministic, portable — isolates the exact structure-access hot path. This is
      the mechanism proof.
  --from-run <run.yaml>: build the REAL CoralNet+MERMAID datasets from a run config (needs
      mermaid-core creds + MERMAID_CORALNET_ANNOTATIONS_PATH). Confirms magnitude on real data.

Output: a CSV (config, epoch, batch, samples_seen, main_rss_mb, workers_rss_mb, total_rss_mb) plus
a printed per-epoch summary. The `*_rss_mb` columns hold **USS** (unique/private set size), not
RSS: summed worker RSS double-counts fork-shared pages and overstates the leak, whereas per-process
USS counts only the pages a worker actually privatized — the real copy-on-write cost. See
`scripts/diagnostics/dataloader_rss_findings.md`.

Examples:
    # portable mechanism proof (run inside the Linux training image):
    python scripts/diagnostics/dataloader_rss_repro.py --rows 3000000 --images 60000 \
        --epochs 4 --batches-per-epoch 400 --out /tmp/rss.csv

    # real-data confirmation:
    AWS_PROFILE=mermaid-core MERMAID_CORALNET_ANNOTATIONS_PATH=... \
    python scripts/diagnostics/dataloader_rss_repro.py --from-run sagemaker/runs/dinov3_lora_qv_r8_spot.yaml \
        --epochs 3 --batches-per-epoch 300 --out /tmp/rss_real.csv
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import time
from typing import Any

import numpy as np
import pandas as pd
import psutil
import torch
from torch.utils.data import ConcatDataset, DataLoader

from mermaidseg.datasets import BaseCoralDataset, worker_init_fn


# --------------------------------------------------------------------------------------------
# Synthetic dataset — reproduces the hot-path structure access without S3.
# --------------------------------------------------------------------------------------------
class _SyntheticCoralDataset(BaseCoralDataset):
    """Minimal concrete BaseCoralDataset over synthetic frames.

    `read_image` returns a zero image so the harness measures ONLY the structure-access
    hot path (`df_images.iloc[...]` and the precomputed numpy annotation-index slice +
    mask creation) — not S3 latency or JPEG decode. With `padding>0` the mask builder
    clips out-of-range annotation coords, so a fixed fake image size is safe.
    """

    SOURCE_NAME = "synthetic"

    def __init__(
        self, df_annotations: pd.DataFrame, df_images: pd.DataFrame, image_size: int, **kw: Any
    ):
        self._image_size = image_size
        super().__init__(df_annotations, df_images, **kw)

    skip_dfimages = False  # ablation: bypass the per-sample df_images.iloc[idx].to_dict() touch

    def read_image(self, **row_kwargs: Any) -> np.ndarray:
        s = self._image_size
        return np.zeros((s, s, 3), dtype=np.uint8)

    def _load_item(self, idx: int):  # type: ignore[override]
        if type(self).skip_dfimages:
            # Skip df_images access entirely to isolate its CoW contribution; keep the numpy
            # annotation-slice + mask path identical to the base hot path.
            image = self.read_image()
            start = int(self._ann_offsets[idx])
            end = int(self._ann_offsets[idx + 1])
            from mermaidseg.datasets.utils import create_annotation_mask_from_arrays

            mask = create_annotation_mask_from_arrays(
                self._ann_row[start:end],
                self._ann_col[start:end],
                self._ann_label_id[start:end],
                image.shape,
                padding=self.padding,
            )
            return image, mask
        return super()._load_item(idx)


def build_synthetic(
    rows: int, images: int, classes: int, image_size: int, padding: int, seed: int
) -> BaseCoralDataset:
    rng = np.random.default_rng(seed)
    # Python-string image ids + label names are exactly the object-dtype columns that drive the
    # CoW refcount churn we are trying to reproduce.
    image_ids = np.array([f"img_{i:07d}" for i in range(images)], dtype=object)
    class_names = np.array([f"class_{c:03d}" for c in range(classes)], dtype=object)

    ann_img = image_ids[rng.integers(0, images, size=rows)]
    df_annotations = pd.DataFrame(
        {
            "image_id": ann_img,
            "source_label_name": class_names[rng.integers(0, classes, size=rows)],
            "row": rng.integers(0, image_size, size=rows),
            "col": rng.integers(0, image_size, size=rows),
        }
    )
    df_images = pd.DataFrame({"image_id": image_ids, "site": np.full(images, "s", dtype=object)})
    return _SyntheticCoralDataset(
        df_annotations,
        df_images,
        image_size=image_size,
        split="train",
        transform=None,
        padding=padding,
    )


def build_from_run(run_yaml: str) -> BaseCoralDataset:
    """Build the REAL train ConcatDataset from a run config, with read_image stubbed to
    zeros."""
    from mermaidseg.experiment import Experiment

    exp = Experiment.from_run_yaml(run_yaml)
    _ = exp.registry  # attach the SourceLabelRegistry before any __getitem__
    dataset_dict = exp.datasets()
    train = [ds for (_, split), ds in dataset_dict.items() if split == "train"]

    # Stub read_image on each underlying BaseCoralDataset instance so we isolate structure access.
    # (Instance attribute shadows the bound method; inherited by fork workers.)
    size = 512
    for ds in train:
        for leaf in getattr(ds, "datasets", [ds]):
            if isinstance(leaf, BaseCoralDataset):
                leaf.padding = max(
                    int(leaf.padding or 0), 1
                )  # ensure mask clip-path (safe fake size)
                leaf.transform = None
                leaf.read_image = lambda **kw: np.zeros((size, size, 3), dtype=np.uint8)  # type: ignore[method-assign]
    return ConcatDataset(train) if len(train) > 1 else train[0]


# --------------------------------------------------------------------------------------------
# RSS sampling (whole process tree = main + fork workers).
# --------------------------------------------------------------------------------------------
def _proc_private(proc: psutil.Process) -> float:
    """Private (copied/allocated) memory for one process, in bytes.

    USS = pages unique to this process. For a forked DataLoader worker this is exactly the
    copy-on-write pages it has dirtied plus its own allocations — the leak-relevant number.
    Summed RSS would double-count fork-shared pages (inherited df_annotations, the numpy index),
    inflating the figure with memory that is shared, not leaked. Falls back to RSS if USS is
    unavailable on the platform.
    """
    try:
        return float(proc.memory_full_info().uss)
    except (psutil.Error, AttributeError):
        return float(proc.memory_info().rss)


def sample_rss() -> tuple[float, float]:
    """Return (main_private_mb, workers_private_mb) using USS (private set size)."""
    proc = psutil.Process()
    main = _proc_private(proc)
    workers = 0.0
    for child in proc.children(recursive=True):
        with contextlib.suppress(psutil.Error):
            workers += _proc_private(child)
    return main / 1e6, workers / 1e6


def run_config(
    dataset: BaseCoralDataset,
    *,
    num_workers: int,
    persistent: bool,
    batch_size: int,
    epochs: int,
    batches_per_epoch: int,
    sample_every: int,
    writer: csv.writer,
) -> list[tuple[str, float]]:
    label = f"nw{num_workers}_persist{int(persistent)}"
    kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "shuffle": True,
        "drop_last": True,
        "collate_fn": BaseCoralDataset.collate_fn,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = persistent
        kwargs["worker_init_fn"] = worker_init_fn
        kwargs["multiprocessing_context"] = "fork"  # reproduce the Linux/SageMaker mechanism
    loader = DataLoader(dataset, **kwargs)

    epoch_end_rss: list[tuple[str, float]] = []
    samples = 0
    for epoch in range(epochs):
        it = iter(loader)
        main0, work0 = sample_rss()
        print(
            f"[{label}] epoch {epoch} start  total={main0 + work0:8.0f} MB (main={main0:.0f} workers={work0:.0f})"
        )
        for b in range(batches_per_epoch):
            try:
                next(it)
            except StopIteration:
                it = iter(loader)
                next(it)
            samples += batch_size
            if b % sample_every == 0:
                main, work = sample_rss()
                writer.writerow(
                    [label, epoch, b, samples, f"{main:.1f}", f"{work:.1f}", f"{main + work:.1f}"]
                )
        del it
        gc.collect()
        main, work = sample_rss()
        total = main + work
        epoch_end_rss.append((label, total))
        writer.writerow(
            [label, epoch, batches_per_epoch, samples, f"{main:.1f}", f"{work:.1f}", f"{total:.1f}"]
        )
        print(
            f"[{label}] epoch {epoch} END    total={total:8.0f} MB (main={main:.0f} workers={work:.0f})"
        )
    del loader
    gc.collect()
    time.sleep(1)
    return epoch_end_rss


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--from-run", default=None, help="run YAML for real CoralNet+MERMAID data (needs creds)"
    )
    p.add_argument("--rows", type=int, default=3_000_000, help="synthetic annotation rows")
    p.add_argument("--images", type=int, default=60_000, help="synthetic image count")
    p.add_argument("--classes", type=int, default=78)
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--padding", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batches-per-epoch", type=int, default=400)
    p.add_argument("--sample-every", type=int, default=25, help="record RSS every N batches")
    p.add_argument("--num-workers", type=int, default=8, help="worker count for the worker configs")
    p.add_argument("--out", default="dataloader_rss.csv")
    p.add_argument(
        "--skip-dfimages",
        action="store_true",
        help="ablation (synthetic only): bypass the per-sample df_images.iloc access",
    )
    args = p.parse_args()

    torch.manual_seed(args.seed)
    if args.skip_dfimages:
        _SyntheticCoralDataset.skip_dfimages = True
    if args.from_run:
        print(f"Building REAL datasets from {args.from_run} ...")
        dataset = build_from_run(args.from_run)
    else:
        print(
            f"Building SYNTHETIC dataset: rows={args.rows:,} images={args.images:,} classes={args.classes}"
        )
        dataset = build_synthetic(
            args.rows, args.images, args.classes, args.image_size, args.padding, args.seed
        )
    print(f"dataset len = {len(dataset):,}")

    configs = [(0, False), (args.num_workers, False), (args.num_workers, True)]
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "config",
                "epoch",
                "batch",
                "samples_seen",
                "main_rss_mb",
                "workers_rss_mb",
                "total_rss_mb",
            ]
        )
        summary: dict[str, list[float]] = {}
        for nw, persist in configs:
            rss = run_config(
                dataset,
                num_workers=nw,
                persistent=persist,
                batch_size=args.batch_size,
                epochs=args.epochs,
                batches_per_epoch=args.batches_per_epoch,
                sample_every=args.sample_every,
                writer=writer,
            )
            summary[rss[0][0]] = [t for _, t in rss]

    print("\n==== per-epoch END total RSS (MB) ====")
    for label, totals in summary.items():
        deltas = [totals[i] - totals[i - 1] for i in range(1, len(totals))]
        drift = totals[-1] - totals[0]
        print(
            f"  {label:16s} {[round(t) for t in totals]}  epoch-to-epoch Δ={[round(d) for d in deltas]}  net drift={drift:+.0f} MB"
        )
    print(f"\nCSV written to {args.out}")
    print(
        "Interpretation: nw0 flat; nw{N}_persist0 resets each epoch; nw{N}_persist1 monotonic net drift => CoW leak confirmed."
    )


if __name__ == "__main__":
    main()
