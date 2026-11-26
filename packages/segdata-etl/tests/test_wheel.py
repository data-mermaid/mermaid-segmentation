"""The built segdata-etl wheel ships the package data the curated label maps and curation read."""

from __future__ import annotations

import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
_SOURCE = _PACKAGE_ROOT / "src"
_DATA_GLOBS = (
    "segdata_etl/datasets/coralnet/curate/data/*.parquet",
    "segdata_etl/label_maps/curated/*.csv",
)


@pytest.fixture(scope="module")
def wheel_names(tmp_path_factory: pytest.TempPathFactory) -> set[str]:
    """Build the wheel once, from a throwaway copy, and return the names it holds."""
    if shutil.which("uv") is None:
        pytest.skip("uv is not installed")
    tmp = tmp_path_factory.mktemp("wheel")
    # A copy keeps build/ and egg-info out of the source tree, so concurrent runs cannot race.
    package_copy = tmp / "pkg"
    shutil.copytree(
        _PACKAGE_ROOT,
        package_copy,
        ignore=shutil.ignore_patterns("build", "*.egg-info", "tests"),
    )
    out = tmp / "dist"
    # --no-sources: the copy sits outside the uv workspace, so workspace sources do not resolve.
    proc = subprocess.run(
        ["uv", "build", "--wheel", "--no-sources", "-o", str(out)],
        cwd=package_copy,
        check=False,
        capture_output=True,
    )
    if proc.returncode:
        pytest.fail(proc.stderr.decode())
    (wheel,) = out.glob("segdata_etl-*.whl")
    with zipfile.ZipFile(wheel) as archive:
        return set(archive.namelist())


@pytest.mark.parametrize("pattern", _DATA_GLOBS)
def test_wheel_ships_every_package_data_file_of_the_source_tree(
    pattern: str, wheel_names: set[str]
) -> None:
    expected = {p.relative_to(_SOURCE).as_posix() for p in _SOURCE.glob(pattern)}
    assert expected, f"no source files match {pattern}"
    assert expected <= wheel_names
