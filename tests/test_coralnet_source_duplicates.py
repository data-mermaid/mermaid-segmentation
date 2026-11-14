"""Same-count CoralNet duplicate classification.

No S3 access.
"""

from __future__ import annotations

import io

import pandas as pd
from PIL import Image

from scripts.coralnet_source_duplicates import (
    PARTIAL_COLUMNS,
    DuplicateReport,
    ImageRef,
    build_report,
    dhash_bytes,
    group_by_count,
    hash_candidate_images,
    load_hash_rows,
    pairwise_overlap,
    parse_image_key,
    write_report,
)

PREFIX = "coralnet-public-images"


def _jpeg(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=95)
    return buffer.getvalue()


def _gradient() -> Image.Image:
    image = Image.new("L", (48, 36))
    pixels = image.load()
    assert pixels is not None
    for y in range(36):
        for x in range(48):
            pixels[x, y] = (x * 5) % 256
    return image


def _checker() -> Image.Image:
    image = Image.new("L", (48, 36))
    pixels = image.load()
    assert pixels is not None
    for y in range(36):
        for x in range(48):
            pixels[x, y] = 255 if (x // 4 + y // 4) % 2 == 0 else 0
    return image


def test_dhash_is_stable_for_the_same_image_and_differs_for_another() -> None:
    gradient = _jpeg(_gradient())
    same_pixels = _jpeg(_gradient())
    checker = _jpeg(_checker())

    assert dhash_bytes(gradient) == dhash_bytes(gradient)
    assert dhash_bytes(gradient) == dhash_bytes(same_pixels)
    assert dhash_bytes(gradient) != dhash_bytes(checker)
    assert dhash_bytes(b"not a jpeg") is None


def test_parse_image_key_only_matches_direct_source_images() -> None:
    assert parse_image_key(f"{PREFIX}/s23/images/4717009.jpg", PREFIX) == 23
    assert parse_image_key(f"{PREFIX}/s23/annotations.csv", PREFIX) is None
    assert parse_image_key(f"{PREFIX}/resized/s23/images/1.jpg", PREFIX) is None
    assert parse_image_key(f"{PREFIX}/s23/images/nested/1.jpg", PREFIX) is None
    assert parse_image_key(f"{PREFIX}/s23/images/", PREFIX) is None


def test_group_by_count_sorts_source_ids() -> None:
    assert group_by_count({3: 5, 1: 5, 2: 1, 4: 0, 5: 0}) == {5: [1, 3], 1: [2], 0: [4, 5]}


def test_overlap_is_a_multiset_intersection() -> None:
    pairs = pairwise_overlap({7: [1, 1, 2], 4: [1, 2, 2], 8: [3, 3, 3]})

    assert len(pairs) == 1
    pair = pairs[0]
    assert (pair.source_a, pair.source_b) == (4, 7)
    assert pair.n_shared == 2
    assert pair.n_images == 3
    assert not pair.is_duplicate


def test_unreadable_images_do_not_count_as_shared() -> None:
    pairs = pairwise_overlap({1: [5, None], 2: [5, None]})

    assert len(pairs) == 1
    assert pairs[0].n_shared == 1
    assert pairs[0].n_unreadable_a == 1
    assert pairs[0].n_unreadable_b == 1
    assert not pairs[0].is_duplicate


def _fixture() -> dict[int, list[int | None]]:
    return {
        1: [10, 20],
        2: [10, 20],
        3: [10, 30],
        4: [1],
        5: [],
        6: [],
        7: [7, 8, 9],
        8: [1, 2, 3],
        9: [40, 50],
    }


def test_report_splits_duplicates_partials_and_unique_sources() -> None:
    report = build_report(_fixture())
    unique = report.unique.set_index("source_id")

    assert set(unique.index) == {4, 5, 6, 7, 8, 9}
    assert unique.loc[4, "reason"] == "unique_count"
    assert unique.loc[4, "n_peers"] == 0
    assert unique.loc[5, "reason"] == "empty"
    assert unique.loc[5, "n_peers"] == 1
    assert unique.loc[6, "reason"] == "empty"
    assert unique.loc[7, "reason"] == "no_shared_images"
    assert unique.loc[7, "n_peers"] == 1
    assert unique.loc[9, "reason"] == "no_shared_images"
    assert unique.loc[9, "n_peers"] == 3
    assert unique.loc[9, "n_images"] == 2

    duplicate_pairs = list(
        zip(report.duplicates["source_a"], report.duplicates["source_b"], strict=True)
    )
    assert duplicate_pairs == [(1, 2)]
    assert int(report.duplicates.iloc[0]["cluster_id"]) == 1
    assert int(report.duplicates.iloc[0]["n_shared"]) == 2
    assert float(report.duplicates.iloc[0]["fraction"]) == 1.0

    partial_pairs = list(zip(report.partial["source_a"], report.partial["source_b"], strict=True))
    assert partial_pairs == [(1, 3), (2, 3)]
    assert float(report.partial.iloc[0]["fraction"]) == 0.5
    assert int(report.partial.iloc[0]["n_shared"]) == 1

    assert "duplicate sources: 2 (1 cluster)" in report.summary
    assert "possible / partial sources: 1" in report.summary
    assert "not duplicated: 6" in report.summary
    assert "unique image count: 1" in report.summary
    assert "empty: 2" in report.summary
    assert "no shared images: 3" in report.summary
    assert "sources=1, 2" in report.summary


def test_unreadable_byte_prevents_a_full_duplicate_label() -> None:
    report = build_report({1: [1, None], 2: [1, None]})

    assert report.duplicates.empty
    assert len(report.partial) == 1
    assert int(report.partial.iloc[0]["n_shared"]) == 1
    assert report.unique.empty
    assert "possible / partial sources: 2" in report.summary


def test_separate_duplicate_pairs_get_separate_clusters() -> None:
    report = build_report({10: [1], 11: [1], 3: [2], 4: [2]})
    by_pair = {
        (int(row.source_a), int(row.source_b)): int(row.cluster_id)
        for row in report.duplicates.itertuples(index=False)
    }

    assert by_pair == {(3, 4): 1, (10, 11): 2}
    assert "sources=3, 4" in report.summary
    assert "sources=10, 11" in report.summary
    assert "duplicate sources: 4 (2 clusters)" in report.summary


def test_identical_etag_reuses_a_checkpointed_hash(tmp_path, monkeypatch) -> None:
    image_hash = (1 << 63) + 7
    key_a = "coralnet-public-images/s1/images/a.jpg"
    key_b = "coralnet-public-images/s2/images/b.jpg"
    pd.DataFrame(
        [{"source_id": 1, "key": key_a, "etag": "abc", "dhash": f"{image_hash:016x}"}]
    ).to_parquet(tmp_path / "image_hashes.parquet", index=False)

    def fail_download(*_args, **_kwargs):
        raise AssertionError("identical bytes should not be downloaded again")

    monkeypatch.setattr("scripts.coralnet_source_duplicates._download_and_hash", fail_download)
    previous_limit = Image.MAX_IMAGE_PIXELS
    try:
        hashed = hash_candidate_images(
            "bucket",
            {2: [ImageRef(source_id=2, key=key_b, etag="abc")]},
            tmp_path,
            workers=1,
            checkpoint_every=10,
        )
    finally:
        Image.MAX_IMAGE_PIXELS = previous_limit

    assert hashed[(2, key_b)] == image_hash
    saved = load_hash_rows(tmp_path)
    assert set(saved["key"]) == {key_a, key_b}
    assert set(saved["dhash"]) == {f"{image_hash:016x}"}


def test_write_report_writes_the_four_files(tmp_path) -> None:
    report: DuplicateReport = build_report({1: [1], 2: [1]})
    write_report(tmp_path, report)

    text = (tmp_path / "summary.txt").read_text()
    assert "duplicate sources: 2 (1 cluster)" in text
    duplicates = pd.read_csv(tmp_path / "duplicates.csv")
    assert int(duplicates.loc[0, "n_shared"]) == 1
    partial = pd.read_csv(tmp_path / "partial.csv")
    assert list(partial.columns) == list(PARTIAL_COLUMNS)
    assert partial.empty
    assert pd.read_csv(tmp_path / "unique.csv").empty
