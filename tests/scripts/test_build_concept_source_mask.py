"""build_concept_source_mask: per-source concept counts from a tiny CoralNet version, and the mask."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

import scripts.build_concept_source_mask as script
from segdata_spec.manifest import Pin
from tests.mermaidseg.conftest import concept_table
from tests.mermaidseg.data.conftest import ImageSpec, write_version

LABELS = {1: "Acropora", 2: "Porites", 3: "Sand", 4: "Bleached Acropora", 5: "Unknown"}
# Groups s7 and s30 are train, s8 is val: its point must not be counted.
IMAGES = [
    ImageSpec("a1", group="s7"),
    ImageSpec("a2", group="s7"),
    ImageSpec("b1", group="s30"),
    ImageSpec("c1", group="s8"),
]
POINTS = [
    ("a1", 0, 0, 1),
    ("a1", 1, 1, 2),
    ("a2", 0, 0, 3),
    ("b1", 0, 0, 4),
    ("b1", 1, 1, 5),
    ("c1", 0, 0, 4),
]


@pytest.fixture
def inputs(tmp_path: Path) -> dict[str, Path]:
    """The cache, a data config pinning it with a train block over the train groups, concepts."""
    cache_dir, _, pin = write_version(
        tmp_path, kind="points", images=IMAGES, labels=LABELS, points=POINTS, name="coralnet"
    )
    split_file = tmp_path / "coralnet.csv"
    split_file.write_text("group,split\ns7,train\ns30,train\ns8,val\n")
    data_config = tmp_path / "data_config.yaml"
    data_config.write_text(
        yaml.safe_dump(
            {
                "data": {
                    "datasets": {
                        "coralnet": {
                            "version": pin.version,
                            "manifest_sha256": pin.manifest_sha256,
                            "split_file": str(split_file),
                            "train": {"use": ["train"]},
                        }
                    }
                }
            }
        )
    )
    concepts = tmp_path / "concepts.csv"
    concept_table().to_csv(concepts, index=False)
    return {"cache_dir": cache_dir, "data_config": data_config, "concepts": concepts}


def _run(inputs: dict[str, Path], out: Path, *extra: str) -> None:
    script.main(
        [
            "--cache-dir",
            str(inputs["cache_dir"]),
            "--data-config",
            str(inputs["data_config"]),
            "--schema-config",
            str(inputs["data_config"]),
            "--concepts",
            str(inputs["concepts"]),
            "--out",
            str(out),
            *extra,
        ]
    )


def test_counts_are_per_source_and_concept_over_the_train_groups(inputs, tmp_path):
    counts_out = tmp_path / "counts.csv"

    _run(inputs, tmp_path / "mask.csv", "--counts-out", str(counts_out))

    # s7: acropora (bleached F, branching T, tubular F), porites (F, F, T), sand (F, F, F).
    # s30: bleached acropora (T, T, F), unknown (not given everywhere). Concepts follow the
    # schema's channel order; sources sort numerically.
    assert pd.read_csv(counts_out).to_numpy().tolist() == [
        [7, "tubular", 1, 2, 0],
        [7, "branching", 1, 2, 0],
        [7, "bleached", 0, 3, 0],
        [30, "tubular", 0, 1, 1],
        [30, "branching", 1, 0, 1],
        [30, "bleached", 1, 0, 1],
    ]


def test_the_mask_lists_the_pairs_with_false_points_and_no_true_points(inputs, tmp_path):
    out = tmp_path / "mask.csv"

    _run(inputs, out)

    assert out.read_text().splitlines() == [
        "source_id,concept,n_true,n_false,n_not_given",
        "7,bleached,0,3,0",
        "30,tubular,0,1,1",
    ]
    meta = yaml.safe_load((tmp_path / "mask.meta.yaml").read_text())
    assert meta["source_ids"] == ["7", "30"]
    assert {"git_sha", "git_dirty"} <= set(meta)
    assert "git_commit" not in meta
    assert (meta["split"], meta["n_points"], meta["n_masked"]) == ("train", 5, 2)


def _config(tmp_path: Path, pin: Pin, train: dict[str, Any]) -> Path:
    """A data config pinning ``pin`` as ``coralnet`` with the ``train`` block."""
    data_config = tmp_path / "config.yaml"
    entry = {"version": pin.version, "manifest_sha256": pin.manifest_sha256, "train": train}
    data_config.write_text(yaml.safe_dump({"data": {"datasets": {"coralnet": entry}}}))
    return data_config


def _inputs(tmp_path: Path, cache_dir: Path, pin: Pin, train: dict[str, Any]) -> dict[str, Path]:
    concepts = tmp_path / "concepts.csv"
    concept_table().to_csv(concepts, index=False)
    return {
        "cache_dir": cache_dir,
        "data_config": _config(tmp_path, pin, train),
        "concepts": concepts,
    }


def test_the_train_blocks_annotator_priority_picks_each_points_label(tmp_path):
    cache_dir, _, pin = write_version(
        tmp_path,
        kind="points",
        images=[ImageSpec("a1", group="s7")],
        labels=LABELS,
        points=[("a1", 0, 0, 1, "a"), ("a1", 0, 0, 3, "b")],
        name="coralnet",
    )
    out = tmp_path / "mask.csv"

    _run(_inputs(tmp_path, cache_dir, pin, {"groups": "all", "annotators": ["b", "a"]}), out)

    # Annotator b's sand wins the shared position; a's acropora would make branching True.
    assert out.read_text().splitlines()[1:] == [
        "7,tubular,0,1,0",
        "7,branching,0,1,0",
        "7,bleached,0,1,0",
    ]
    assert yaml.safe_load((tmp_path / "mask.meta.yaml").read_text())["n_points"] == 1


@pytest.mark.parametrize(
    ("kind", "error", "message"),
    [
        ("points", ValueError, r"groups not in the dataset version: \['s99'\]"),
        ("dense", SystemExit, "has no points.parquet"),
    ],
    ids=["unknown-group", "no-points"],
)
def test_main_guards_raise(tmp_path, kind, error, message):
    dense = kind == "dense"
    cache_dir, _, pin = write_version(
        tmp_path,
        kind=kind,
        images=[ImageSpec("a1", group="s7", mask=np.ones((8, 8), np.uint8) if dense else None)],
        labels=LABELS,
        points=() if dense else [("a1", 0, 0, 1)],
        name="coralnet",
    )

    with pytest.raises(error, match=message):
        _run(_inputs(tmp_path, cache_dir, pin, {"groups": ["s99"]}), tmp_path / "mask.csv")
