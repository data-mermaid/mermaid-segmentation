#!/usr/bin/env python3
"""Write a static copy of a dense training dataset whose masks keep only sparse points.

Every image keeps ``--n-points`` pixels drawn uniformly without replacement from its
labelled (non-zero) pixels, or all of them when it has fewer. Each point is painted as a
``(2 * padding + 1)`` square with the centre pixel's label, the footprint
``create_annotation_mask`` gives CoralNet points under ``training.padding``. Every other
pixel becomes 0 (unclassified). Images are copied byte for byte.

    coralscapes_v2  ``train`` split only, so validation and test cannot be read from the copy
    ucsd_mosaics    all site splits, both ``original_split`` values (training uses all of them)

The points of an image depend only on ``(--seed, split, row index)``, not on ``--num-proc``.
The copy is written next to ``--output-dir`` and moved into place when complete.

    <output-dir>/
      README.md                            dataset card whose ``configs`` lists every split
      data/<split>-NNNNN-of-NNNNN.parquet  same columns as the source, sparse ``label``
      classes.json                         ucsd_mosaics only, copied from the source repo
      points.parquet                       split, index, row, col, label of every point
      run.json                             arguments, source revision and per-split counts

Load it by setting ``hf_repo_id: <output-dir>`` on the dataset's train split in the data config.

Usage:
    python scripts/build_sparse_dense_datasets.py --dataset coralscapes_v2 \\
        --output-dir data_quality/coralscapes_v2_sparse25_pad3_seed0
    python scripts/build_sparse_dense_datasets.py --dataset ucsd_mosaics \\
        --output-dir data_quality/ucsd_mosaics_sparse25_pad3_seed0 --num-proc 16
"""

from __future__ import annotations

import argparse
import io
import itertools
import json
import logging
import math
import shutil
import sys
import zlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logger = logging.getLogger("build_sparse_dense_datasets")

POINT_COLUMNS: tuple[str, ...] = ("_point_rows", "_point_cols", "_point_labels")
MAX_SHARD_BYTES = 500 * 1024**2
CLASSES_FILE = "classes.json"


@dataclass(frozen=True)
class Source:
    repo: str
    splits: Callable[[str, str | None], list[str]]
    extra_files: tuple[str, ...] = ()


def _coralscapes_v2_splits(repo: str, revision: str | None) -> list[str]:
    return ["train"]


def _ucsd_mosaics_splits(repo: str, revision: str | None) -> list[str]:
    from datasets import get_dataset_split_names

    return list(get_dataset_split_names(repo, revision=revision))


def _sources() -> dict[str, Source]:
    from mermaidseg.datasets.coralscapes_v2.coralscapes_v2_dataset import CORALSCAPES_V2_HF_REPO
    from mermaidseg.datasets.ucsd_mosaics.ucsd_mosaics_dataset import UCSDMosaicsDataset

    return {
        "coralscapes_v2": Source(CORALSCAPES_V2_HF_REPO, _coralscapes_v2_splits),
        "ucsd_mosaics": Source(
            UCSDMosaicsDataset.HF_REPO_ID, _ucsd_mosaics_splits, extra_files=(CLASSES_FILE,)
        ),
    }


def point_rng(seed: int, split: str, index: int) -> np.random.Generator:
    return np.random.default_rng([seed, zlib.crc32(split.encode()), index])


def sparsify_mask(
    mask: np.ndarray, n_points: int, padding: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Keep ``n_points`` random labelled pixels of ``mask`` as padded squares.

    Returns the sparse mask (same shape and dtype) and an ``(n, 3)`` array of
    ``(row, col, label)`` per point, in painting order. Later squares overwrite
    earlier ones where they overlap.
    """
    if mask.ndim != 2:
        raise ValueError(f"expected a 2-D label mask, got shape {mask.shape}")
    rows, cols = np.nonzero(mask)
    n = min(n_points, rows.size)
    pick = rng.choice(rows.size, size=n, replace=False)
    points = np.stack([rows[pick], cols[pick], mask[rows[pick], cols[pick]]], axis=1)

    sparse = np.zeros_like(mask)
    height, width = mask.shape
    for row, col, label in points:
        sparse[
            max(row - padding, 0) : min(row + padding + 1, height),
            max(col - padding, 0) : min(col + padding + 1, width),
        ] = label
    return sparse, points.astype(np.int64)


def _encode_png(mask: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(mask).save(buffer, format="PNG")
    return buffer.getvalue()


def _sparsify_batch(
    labels: list[dict[str, Any]],
    indices: list[int],
    *,
    split: str,
    seed: int,
    n_points: int,
    padding: int,
) -> dict[str, list]:
    out: dict[str, list] = {"label": [], **{column: [] for column in POINT_COLUMNS}}
    for label, index in zip(labels, indices, strict=True):
        mask = np.asarray(Image.open(io.BytesIO(label["bytes"])))
        sparse, points = sparsify_mask(mask, n_points, padding, point_rng(seed, split, index))
        out["label"].append({"bytes": _encode_png(sparse), "path": label.get("path")})
        for column, values in zip(POINT_COLUMNS, points.T, strict=True):
            out[column].append(values.tolist())
    return out


def _flatten(lists: list[list[int]]) -> np.ndarray:
    return np.fromiter(itertools.chain.from_iterable(lists), dtype=np.int64)


def sparsify_split(
    ds: Any,
    split: str,
    *,
    seed: int,
    n_points: int,
    padding: int,
    num_proc: int | None = None,
) -> tuple[Any, pd.DataFrame]:
    """Sparse copy of one split plus a table of its points."""
    from datasets import Image as HFImage
    from datasets import List, Value

    raw = ds.cast_column("image", HFImage(decode=False)).cast_column("label", HFImage(decode=False))
    features = raw.features.copy()
    for column in POINT_COLUMNS:
        features[column] = List(Value("int64"))
    mapped = raw.map(
        _sparsify_batch,
        batched=True,
        batch_size=16,
        with_indices=True,
        input_columns=["label"],
        fn_kwargs={"split": split, "seed": seed, "n_points": n_points, "padding": padding},
        features=features,
        num_proc=num_proc if num_proc and num_proc > 1 else None,
        load_from_cache_file=False,
        desc=f"sparsify {split}",
    )

    columns = mapped.select_columns(list(POINT_COLUMNS)).to_dict()
    counts = [len(rows) for rows in columns["_point_rows"]]
    points = pd.DataFrame(
        {
            "split": split,
            "index": np.repeat(np.arange(len(counts), dtype=np.int64), counts),
            "row": _flatten(columns["_point_rows"]),
            "col": _flatten(columns["_point_cols"]),
            "label": _flatten(columns["_point_labels"]),
        }
    )
    sparse = (
        mapped.remove_columns(list(POINT_COLUMNS))
        .cast_column("image", HFImage())
        .cast_column("label", HFImage())
    )
    return sparse, points


def write_split(ds: Any, data_dir: Path, split: str, max_shard_bytes: int) -> list[str]:
    data_dir.mkdir(parents=True, exist_ok=True)
    n_shards = max(1, math.ceil(ds.data.nbytes / max_shard_bytes))
    names = []
    for i in range(n_shards):
        name = f"{split}-{i:05d}-of-{n_shards:05d}.parquet"
        ds.shard(num_shards=n_shards, index=i, contiguous=True).to_parquet(data_dir / name)
        names.append(name)
    return names


def write_card(out_dir: Path, splits: list[str], source_repo: str, run_info: dict) -> None:
    """Dataset card whose ``configs`` lets ``load_dataset(out_dir)`` find every split."""
    header = {
        "configs": [
            {
                "config_name": "default",
                "data_files": [{"split": s, "path": f"data/{s}-*"} for s in splits],
            }
        ]
    }
    args = run_info["args"]
    body = (
        f"# Sparse copy of `{source_repo}`\n\n"
        f"Every mask keeps {args['n_points']} random labelled pixels as "
        f"{2 * args['padding'] + 1}x{2 * args['padding'] + 1} squares (seed {args['seed']}); "
        "all other pixels are 0. Images are unchanged. Written by "
        "`scripts/build_sparse_dense_datasets.py`; see `run.json` and `points.parquet`.\n"
    )
    (out_dir / "README.md").write_text(f"---\n{yaml.safe_dump(header, sort_keys=False)}---\n{body}")


def _revision_sha(repo: str, revision: str | None) -> str | None:
    if Path(repo).is_dir():
        return None
    try:
        from huggingface_hub import HfApi

        return HfApi().dataset_info(repo, revision=revision).sha
    except Exception as e:  # offline or no access: the copy is still valid, just unpinned
        logger.warning("could not resolve the revision of %s: %s", repo, e)
        return None


def _split_counts(points: pd.DataFrame, n_images: int, n_points: int) -> dict[str, int]:
    per_image = points.groupby("index").size().reindex(range(n_images), fill_value=0)
    return {
        "n_images": n_images,
        "n_points": int(per_image.sum()),
        "n_images_below_n_points": int((per_image < n_points).sum()),
        "n_images_without_labels": int((per_image == 0).sum()),
    }


def build(args: argparse.Namespace) -> Path:
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download

    out_dir: Path = args.output_dir
    if out_dir.exists() and not args.overwrite:
        raise SystemExit(f"{out_dir} exists; pass --overwrite to replace it")
    partial = out_dir.with_name(f".{out_dir.name}.partial")
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir(parents=True)

    source = _sources()[args.dataset]
    repo = args.hf_repo_id or source.repo
    splits = source.splits(repo, args.revision)
    logger.info("%s: %d split(s) from %s", args.dataset, len(splits), repo)

    all_points = []
    split_info: dict[str, dict] = {}
    for split in splits:
        ds = load_dataset(repo, split=split, revision=args.revision)
        sparse, points = sparsify_split(
            ds,
            split,
            seed=args.seed,
            n_points=args.n_points,
            padding=args.padding,
            num_proc=args.num_proc,
        )
        files = write_split(sparse, partial / "data", split, args.max_shard_bytes)
        split_info[split] = {**_split_counts(points, len(sparse), args.n_points), "files": files}
        all_points.append(points)
        logger.info("%s: %s", split, split_info[split])

    for name in source.extra_files:
        local = Path(repo) / name
        if not local.is_file():
            local = hf_hub_download(
                repo_id=repo, filename=name, repo_type="dataset", revision=args.revision
            )
        shutil.copy(local, partial / name)
    pd.concat(all_points, ignore_index=True).to_parquet(partial / "points.parquet", index=False)
    run_info = {
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "source_repo": repo,
        "source_revision": args.revision,
        "source_revision_sha": _revision_sha(repo, args.revision),
        "splits": split_info,
    }
    (partial / "run.json").write_text(json.dumps(run_info, indent=2))
    write_card(partial, splits, repo, run_info)

    if out_dir.exists():
        shutil.rmtree(out_dir)
    partial.rename(out_dir)
    logger.info("wrote %s", out_dir)
    return out_dir


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", required=True, choices=["coralscapes_v2", "ucsd_mosaics"])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-points", type=int, default=25)
    parser.add_argument(
        "--padding",
        type=int,
        default=3,
        help="Half-size of the square painted per point (default: training.padding).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-proc", type=int, default=1)
    parser.add_argument(
        "--hf-repo-id", default=None, help="Source repo or local directory (default: the mirror)."
    )
    parser.add_argument("--revision", default=None, help="Source repo revision (default: main).")
    parser.add_argument("--max-shard-bytes", type=int, default=MAX_SHARD_BYTES)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.n_points < 1:
        parser.error("--n-points must be >= 1")
    if args.padding < 0:
        parser.error("--padding must be >= 0")
    return args


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    build(args)


if __name__ == "__main__":
    main()
