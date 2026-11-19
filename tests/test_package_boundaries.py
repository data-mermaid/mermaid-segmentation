"""mermaidseg, the CoralNet scraper and the CoralNet dataset package must not import each other."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_SCRAPER_SRC = _ROOT / "packages" / "coralnet-scraper" / "src"
_DATASET_SRC = _ROOT / "packages" / "coralnet-dataset" / "src"


def _imported_top_levels(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    return names


def _violations(source_dirs: list[Path], forbidden: str) -> list[str]:
    found = []
    for source_dir in source_dirs:
        assert any(source_dir.rglob("*.py")), source_dir
        for path in sorted(source_dir.rglob("*.py")):
            if forbidden in _imported_top_levels(path):
                found.append(str(path.relative_to(_ROOT)))
    return found


@pytest.mark.parametrize(
    ("source_dirs", "forbidden"),
    [
        ([_SCRAPER_SRC], "mermaidseg"),
        ([_ROOT / "mermaidseg", _ROOT / "scripts"], "coralnet_scraper"),
        ([_DATASET_SRC], "mermaidseg"),
        ([_ROOT / "mermaidseg", _ROOT / "scripts"], "coralnet_dataset"),
        ([_SCRAPER_SRC], "coralnet_dataset"),
    ],
    ids=[
        "scraper-must-not-import-mermaidseg",
        "mermaidseg-must-not-import-scraper",
        "dataset-must-not-import-mermaidseg",
        "mermaidseg-must-not-import-dataset",
        "scraper-must-not-import-dataset",
    ],
)
def test_no_forbidden_imports(source_dirs, forbidden):
    assert _violations(source_dirs, forbidden) == []
