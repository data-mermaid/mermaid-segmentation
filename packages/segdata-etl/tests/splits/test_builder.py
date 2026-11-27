from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from segdata_etl import cli
from segdata_etl.splits.anneal import anneal
from segdata_etl.splits.builder import SplitParams, _best_assignment, build_split
from segdata_spec.manifest import ParentRef
from segdata_spec.splits import check_coverage

from .conftest import (
    BUCKET,
    FixtureCache,
    benthic_attributes_table,
    images_table,
    label_counts_table,
    labels_table,
    sha256_of,
)

SOURCES = [10, 11, 12, 13, 14, 15, 16, 17, 99]


@pytest.fixture(autouse=True)
def _published_taxonomy(monkeypatch: pytest.MonkeyPatch, fixture_cache: FixtureCache) -> None:
    """Hand the CLI the fake bucket that holds the fixture taxonomy's manifest."""
    monkeypatch.setattr(cli, "_client_for", lambda _args: fixture_cache.published)


def _write_inputs(tmp_path: Path) -> dict[str, Path]:
    paths = {
        "near_dup_labels": tmp_path / "near_dup_labels.csv",
        "metadata": tmp_path / "metadata.csv",
        "test_sources": tmp_path / "test_sources.txt",
        "exclude_sources": tmp_path / "exclude_sources.txt",
        "training_config": tmp_path / "training.yaml",
    }
    paths["near_dup_labels"].write_text(
        "image_a,image_b,label,source_a,source_b\ni12a,i13a,fp,12,13\n"
    )
    paths["metadata"].write_text(
        "source_id,admins,latitude,longitude\n"
        + "".join(f"{s},admin{s},{s}.0,0.0\n" for s in SOURCES)
    )
    paths["test_sources"].write_text("10  # whole component goes to test\n")
    paths["exclude_sources"].write_text("99\n")
    paths["training_config"].write_text("training:\n  class_subset: [Hard coral, Sand]\n")
    return paths


def _publish(
    cache: FixtureCache,
    *,
    raw_extra: list[tuple[str, str, str | None, int]] = (),
    declare_parent: bool = True,
) -> dict[str, str]:
    curated_rows = [(f"i{s}{c}", f"s{s}", f"C{s}{c}=", 100) for s in SOURCES for c in "ab"]
    counts = [(image_id, 1 + 2 * (i % 2), 10) for i, (image_id, *_) in enumerate(curated_rows)]
    # In the raw parent, s10 and s11 hold a byte-identical image.
    raw_rows = [
        *curated_rows,
        ("r10", "s10", "SHARED=", 7),
        ("r11", "s11", "SHARED=", 7),
        *raw_extra,
    ]
    raw = cache.dataset("coralnet-raw", "20261119-bbbbbbb", images=images_table(raw_rows))
    parents = (
        (
            ParentRef(
                dataset="coralnet-raw", version=raw.version, manifest_sha256=raw.manifest_sha256
            ),
        )
        if declare_parent
        else ()
    )
    curated = cache.dataset(
        "coralnet",
        "20261120-aaaaaaa",
        images=images_table(curated_rows),
        labels=labels_table([(1, "Acropora"), (3, "Sand")]),
        label_counts=label_counts_table(counts),
        parents=parents,
    )
    taxonomy = cache.taxonomy(
        "20261120-1234567",
        benthic_attributes_table(
            [("1", "Hard coral", None), ("2", "Acropora", "1"), ("3", "Sand", None)]
        ),
    )
    return {
        "--version": curated.version,
        "--manifest-sha256": curated.manifest_sha256,
        "--raw-version": raw.version,
        "--raw-manifest-sha256": raw.manifest_sha256,
        "--taxonomy-version": taxonomy.version,
        "--taxonomy-manifest-sha256": taxonomy.manifest_sha256,
    }


def _argv(
    tmp_path: Path, cache: FixtureCache, pins: dict[str, str]
) -> tuple[list[str], Path, dict[str, Path]]:
    inputs = _write_inputs(tmp_path)
    out_dir = tmp_path / "out"
    argv = ["split", "build", "--cache-dir", str(cache.cache_dir), "--bucket", BUCKET]
    for flag, value in pins.items():
        argv += [flag, value]
    for name, path in inputs.items():
        argv += [f"--{name.replace('_', '-')}", str(path)]
    argv += ["--out-dir", str(out_dir), "--iters", "300", "--restarts", "2"]
    argv += ["--max-group-frac", "10", "--min-class-points", "1"]
    return argv, out_dir, inputs


@pytest.fixture
def built(tmp_path: Path, fixture_cache: FixtureCache) -> tuple[Path, dict[str, Path]]:
    argv, out_dir, inputs = _argv(tmp_path, fixture_cache, _publish(fixture_cache))
    assert cli.main(argv) == 0
    return out_dir, inputs


def _table(out_dir: Path) -> pd.DataFrame:
    return pd.read_csv(out_dir / "coralnet.csv", dtype=str, keep_default_na=False)


def test_split_file_has_exactly_the_four_columns(built):
    assert list(_table(built[0]).columns) == ["group", "source_id", "split_group", "split"]


def test_test_source_component_goes_to_test(built):
    table = _table(built[0]).set_index("source_id")
    assert table.loc[["10", "11"], "split"].tolist() == ["test", "test"]


def test_excluded_source_is_listed_with_the_excluded_split(built):
    row = _table(built[0]).set_index("source_id").loc["99"]
    assert row.to_dict() == {"group": "s99", "split_group": "", "split": "excluded"}


def test_split_file_covers_every_group_of_the_curated_version(built):
    check_coverage(built[0] / "coralnet.csv", "coralnet", {f"s{s}" for s in SOURCES})


def test_exclude_source_outside_the_version_writes_no_row(tmp_path, fixture_cache):
    argv, out_dir, inputs = _argv(tmp_path, fixture_cache, _publish(fixture_cache))
    inputs["exclude_sources"].write_text("99\n555\n")
    assert cli.main(argv) == 0
    assert "555" not in _table(out_dir)["source_id"].tolist()


def test_meta_records_each_input_sha256(built):
    out_dir, inputs = built
    meta = yaml.safe_load((out_dir / "coralnet.meta.yaml").read_text())
    recorded = {name: entry["sha256"] for name, entry in meta["inputs"].items()}
    assert recorded == {name: sha256_of(path) for name, path in inputs.items()}


def test_null_crc_in_a_universe_source_stops_the_build(tmp_path, fixture_cache):
    pins = _publish(fixture_cache, raw_extra=[("r12", "s12", None, 7), ("r14", "s14", None, 7)])
    argv, _, _ = _argv(tmp_path, fixture_cache, pins)
    with pytest.raises(ValueError, match=r"2 images.*\[12, 14\]"):
        cli.main(argv)


def test_null_crc_in_an_excluded_source_does_not_stop_the_build(tmp_path, fixture_cache):
    pins = _publish(fixture_cache, raw_extra=[("r99", "s99", None, 7)])
    argv, _, _ = _argv(tmp_path, fixture_cache, pins)
    assert cli.main(argv) == 0


def test_curated_version_without_the_raw_parent_stops_the_build(tmp_path, fixture_cache):
    pins = _publish(fixture_cache, declare_parent=False)
    argv, _, _ = _argv(tmp_path, fixture_cache, pins)
    with pytest.raises(ValueError, match="coralnet-raw"):
        cli.main(argv)


def test_raw_pin_that_is_not_the_curated_parent_stops_the_build(tmp_path, fixture_cache):
    pins = _publish(fixture_cache)
    other_raw = fixture_cache.dataset(
        "coralnet-raw", "20261118-ccccccc", images=images_table([("z1", "s10", "Z=", 1)])
    )
    pins["--raw-version"] = other_raw.version
    pins["--raw-manifest-sha256"] = other_raw.manifest_sha256
    argv, _, _ = _argv(tmp_path, fixture_cache, pins)
    with pytest.raises(ValueError, match="20261118-ccccccc"):
        cli.main(argv)


# build_split: eligibility and grouping, with no CLI and no published versions.


def _build(
    n_images: dict[int, int],
    *,
    counts: dict[int, dict[str, int]],
    keys: list[tuple[int, str, int]] = (),
    near: list[tuple[int, int]] = (),
    metadata: list[tuple[int, str, float, float]] = (),
    test_sources: set[int] = frozenset(),
    **params: float,
) -> pd.DataFrame:
    sources = sorted(n_images)
    split_table, _, _ = build_split(
        sources=sources,
        n_images=pd.Series(n_images),
        counts=pd.DataFrame.from_dict(counts, orient="index").fillna(0),
        image_keys=pd.DataFrame(keys, columns=["source_id", "crc64nvme", "size"]),
        near_labels=pd.DataFrame(
            [(a, b, "dup") for a, b in near], columns=["source_a", "source_b", "label"]
        ),
        metadata=pd.DataFrame(metadata, columns=["source_id", "admins", "latitude", "longitude"]),
        test_sources=set(test_sources),
        subset={"common", "rare"},
        params=SplitParams(
            **{
                "max_group_frac": 10,
                "max_id_gap": 0,
                "iters": 3000,
                "restarts": 2,
                "min_class_points": 1,
                **params,
            }
        ),
    )
    return split_table.set_index("source_id")


def _equal_counts(sources: list[int]) -> dict[int, dict[str, int]]:
    return {s: {"common": 10} for s in sources}


# Thirteen sources of 7 images cannot tile an eval target of about 10 images, so a group of
# exactly 10 images is the best fit for val or test whenever it is eligible.
SMALL = dict.fromkeys(range(3, 16), 7)


def test_group_sharing_a_raw_image_stays_in_train_even_when_it_would_fit_an_eval_split():
    n_images = {1: 5, 2: 5, **SMALL}
    table = _build(
        n_images,
        counts=_equal_counts(list(n_images)),
        keys=[(1, "SHARED=", 7), (2, "SHARED=", 7)],
    )
    assert table.loc[[1, 2], "split"].tolist() == ["train", "train"]


def test_group_above_the_cap_stays_in_train_even_when_it_would_fit_an_eval_split():
    # The cap is 0.75 * 10.1 = 7.6 images: source 1 (10 images) is over it, the 7-image sources not.
    n_images = {1: 10, **SMALL}
    table = _build(n_images, counts=_equal_counts(list(n_images)), max_group_frac=0.75)
    assert table.loc[1, "split"] == "train"
    assert set(table["split"]) == {"train", "val", "test"}


@pytest.mark.parametrize(
    ("near", "metadata"),
    [
        ([(2, 6)], []),
        ([], [(2, "a", np.nan, np.nan), (6, "a", np.nan, np.nan)]),
    ],
    ids=["near-duplicate link", "project link"],
)
def test_sources_linked_only_by_a_near_duplicate_or_a_project_share_a_split_group(near, metadata):
    n_images = dict.fromkeys(range(1, 11), 10)
    table = _build(
        n_images,
        counts=_equal_counts(list(n_images)),
        near=near,
        metadata=metadata,
        max_id_gap=4,
    )
    assert table.loc[2, "split_group"] == table.loc[6, "split_group"]
    assert table["split_group"].nunique() == 9


def test_build_split_without_a_movable_group_names_the_likely_causes():
    n_images = {1: 10, 2: 10, 3: 10}
    with pytest.raises(ValueError, match="test-sources"):
        _build(n_images, counts=_equal_counts(list(n_images)), test_sources={1, 2, 3})


def test_best_assignment_returns_the_restart_with_the_lowest_cost():
    groups = 12
    X = np.full((groups, 1), 10.0)
    n = np.full(groups, 10.0)
    grp = pd.DataFrame({"eligible": [True] * groups, "fixed": [-1] * groups})
    targets = (12.0, 12.0)
    params = SplitParams(iters=5, restarts=4, w_size=10.0, seed=2)

    def one(seed: int) -> tuple[np.ndarray, float]:
        return anneal(
            X,
            n,
            X.sum(0),
            grp["eligible"].to_numpy(),
            targets,
            (0.1, 0.1),
            params.w_size,
            np.array([True]),
            np.array([0.0]),
            params.w_coverage,
            params.iters,
            seed,
            grp["fixed"].to_numpy(),
        )

    runs = [one(params.seed + r) for r in range(params.restarts)]
    costs = [c for _, c in runs]
    assert costs[0] > min(costs)  # the minimum is on a later restart, so keeping the first fails
    assignment, cost = _best_assignment(
        X, n, grp, targets, (0.1, 0.1), np.array([True]), np.array([0.0]), params
    )
    assert cost == min(costs)
    assert assignment.tolist() == runs[costs.index(min(costs))][0].tolist()
