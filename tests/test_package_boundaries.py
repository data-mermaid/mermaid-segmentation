"""mermaidseg, the CoralNet scraper and segdata-etl must not import each other, except segdata_etl.datasets.coralnet -> coralnet_scraper.

segdata_spec imports none of them, and mermaidseg/inference does not import segdata_spec. The
model package reaches no dataset-reconciliation concept-source-mask code.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_SCRAPER_SRC = _ROOT / "packages" / "coralnet-scraper" / "src"
_ETL_SRC = _ROOT / "packages" / "segdata-etl" / "src"
_ETL_PKG = _ETL_SRC / "segdata_etl"
_SPEC_SRC = _ROOT / "packages" / "segdata-spec" / "src"


def _imported_top_levels(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    return names


def _violations(source_dirs: list[Path], forbidden: str, exclude: Path | None = None) -> list[str]:
    found = []
    for source_dir in source_dirs:
        assert any(source_dir.rglob("*.py")), source_dir
        for path in sorted(source_dir.rglob("*.py")):
            if exclude is not None and exclude in path.parents:
                continue
            if forbidden in _imported_top_levels(path):
                found.append(str(path.relative_to(_ROOT)))
    return found


@pytest.mark.parametrize(
    ("source_dirs", "forbidden"),
    [
        ([_SCRAPER_SRC], "mermaidseg"),
        ([_ROOT / "mermaidseg", _ROOT / "scripts"], "coralnet_scraper"),
        ([_ETL_SRC], "mermaidseg"),
        ([_ROOT / "mermaidseg", _ROOT / "scripts"], "segdata_etl"),
        ([_SCRAPER_SRC], "segdata_etl"),
        ([_SPEC_SRC], "mermaidseg"),
        ([_SPEC_SRC], "coralnet_scraper"),
        ([_SPEC_SRC], "segdata_etl"),
        ([_ROOT / "mermaidseg" / "inference"], "segdata_spec"),
    ],
    ids=[
        "scraper-must-not-import-mermaidseg",
        "mermaidseg-must-not-import-scraper",
        "etl-must-not-import-mermaidseg",
        "mermaidseg-must-not-import-etl",
        "scraper-must-not-import-etl",
        "spec-must-not-import-mermaidseg",
        "spec-must-not-import-scraper",
        "spec-must-not-import-etl",
        "inference-must-not-import-spec",
    ],
)
def test_no_forbidden_imports(source_dirs, forbidden):
    assert _violations(source_dirs, forbidden) == []


def _mermaidseg_imports(path: Path) -> set[str]:
    """Return the ``mermaidseg`` modules a file imports, including ``from pkg import module``."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names if a.name.split(".")[0] == "mermaidseg")
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            module = node.module or ""
            if module.split(".")[0] == "mermaidseg":
                names.add(module)
                names.update(f"{module}.{a.name}" for a in node.names)
    return names


def _reachable_modules(start: list[str]) -> set[str]:
    """Return the ``mermaidseg`` modules reachable from ``start`` through imports and parents."""
    seen: set[str] = set()
    todo = list(start)
    while todo:
        name = todo.pop()
        base = _ROOT.joinpath(*name.split("."))
        path = (
            base.with_suffix(".py") if base.with_suffix(".py").is_file() else base / "__init__.py"
        )
        if name in seen or not path.is_file():
            continue
        seen.add(name)
        todo.extend(_mermaidseg_imports(path))
        parent = name.rpartition(".")[0]
        if parent:
            todo.append(parent)
    return seen


def test_the_model_package_reaches_no_concept_source_mask_code():
    reached = _reachable_modules(
        ["mermaidseg.model.models", "mermaidseg.model.meta", "mermaidseg.model.data_parallel"]
    )

    assert "mermaidseg.dataset_reconciliation.registry" in reached
    assert "mermaidseg.dataset_reconciliation.concept_source_mask" not in reached


@pytest.mark.parametrize(
    ("source_dir", "exclude"),
    [(_ETL_PKG, _ETL_PKG / "datasets" / "coralnet")],
    ids=["etl-only-coralnet-may-import-scraper"],
)
def test_scraper_imports_confined_to_etl_coralnet(source_dir, exclude):
    assert _violations([source_dir], "coralnet_scraper", exclude) == []
