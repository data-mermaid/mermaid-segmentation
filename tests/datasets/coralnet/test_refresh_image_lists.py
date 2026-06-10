"""Tests for scripts/refresh_coralnet_image_lists.py (image_list.csv refresh + truncation guard)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

# Load the script module by path (scripts/ is not an importable package).
_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "refresh_coralnet_image_lists.py"
_spec = importlib.util.spec_from_file_location("refresh_coralnet_image_lists", _SCRIPT)
refresh = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(refresh)  # type: ignore[union-attr]

BUCKET, PREFIX = "b", "coralnet-public-images"


class _StubDownloader:
    """Returns a fixed browse-image DataFrame from get_images()."""

    def __init__(self, n_rows: int, ok: bool = True):
        self._n = n_rows
        self._ok = ok

    def get_images(self, source_id):  # noqa: ARG002 - signature parity
        if not self._ok:
            return None, False
        df = pd.DataFrame(
            {
                "Name": [f"img_{i}.jpg - Confirmed" for i in range(self._n)],
                "Image Page": [f"/image/{i}/view/" for i in range(self._n)],
            }
        )
        return df, True


def _seed(fake_s3, sid, *, n_s3_images: int, existing_list_rows: int | None):
    base = f"{PREFIX}/s{sid}/"
    for i in range(n_s3_images):
        fake_s3.put(BUCKET, f"{base}images/{i}.jpg", b"\xff\xd8\xff\xd9")
    if existing_list_rows is not None:
        body = "Name,Image Page\n" + "".join(
            f"x_{i}.jpg - Confirmed,/image/{i}/view/\n" for i in range(existing_list_rows)
        )
        fake_s3.put(BUCKET, f"{base}image_list.csv", body.encode())


def _list_key(sid):
    return f"{PREFIX}/s{sid}/image_list.csv"


def test_refresh_uploads_full_list(fake_s3):
    _seed(fake_s3, 1, n_s3_images=10, existing_list_rows=2)  # truncated existing
    r = refresh.refresh_one(_StubDownloader(10), fake_s3, BUCKET, PREFIX, 1)
    assert r["uploaded"] is True
    assert r["new_rows"] == 10
    assert r["existing_rows"] == 2
    # Uploaded object has all 10 rows and preserves the status suffix in Name.
    body = fake_s3.objects[(BUCKET, _list_key(1))].decode()
    assert body.count("/image/") == 10
    assert "- Confirmed" in body


def test_refresh_guard_rejects_shorter_scrape(fake_s3):
    _seed(fake_s3, 2, n_s3_images=10, existing_list_rows=10)
    r = refresh.refresh_one(_StubDownloader(3), fake_s3, BUCKET, PREFIX, 2)
    assert r["uploaded"] is False
    assert "existing(10)" in r["skipped_reason"]
    assert (BUCKET, _list_key(2)) not in fake_s3.objects or fake_s3.objects[
        (BUCKET, _list_key(2))
    ].decode().count("/image/") == 10  # original untouched


def test_refresh_guard_rejects_low_coverage(fake_s3):
    _seed(fake_s3, 3, n_s3_images=100, existing_list_rows=None)
    r = refresh.refresh_one(_StubDownloader(10), fake_s3, BUCKET, PREFIX, 3)
    assert r["uploaded"] is False
    assert "of s3_images(100)" in r["skipped_reason"]


def test_refresh_dry_run_does_not_upload(fake_s3):
    _seed(fake_s3, 4, n_s3_images=10, existing_list_rows=2)
    r = refresh.refresh_one(_StubDownloader(10), fake_s3, BUCKET, PREFIX, 4, dry_run=True)
    assert r["uploaded"] is False
    assert r["skipped_reason"] == "dry_run"
    assert r["new_rows"] == 10  # scrape happened, reported, but not written
    # Existing list left untouched (still the seeded 2 rows).
    assert fake_s3.objects[(BUCKET, _list_key(4))].decode().count("/image/") == 2


def test_refresh_handles_failed_scrape(fake_s3):
    _seed(fake_s3, 5, n_s3_images=10, existing_list_rows=2)
    r = refresh.refresh_one(_StubDownloader(0, ok=False), fake_s3, BUCKET, PREFIX, 5)
    assert r["uploaded"] is False
    assert r["skipped_reason"] == "scrape_failed_or_empty"
