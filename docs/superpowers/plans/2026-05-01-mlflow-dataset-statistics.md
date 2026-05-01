# MLflow Dataset Statistics Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `Logger.log_dataset_statistics(splits)` that emits four MLflow artifacts (`class_counts.csv`, `source_stats.csv`, `class_by_source.csv`, `train_summary.yaml`) for per-run dataset distribution QA, mirroring the `mermaid-classifier` precedent.

**Architecture:** One public method on the existing `Logger` class. A private `_resolve_annotations` helper handles plain `Dataset`, PyTorch `Subset`, and `ConcatDataset`-style wrappers. Four small `_compute_*` helpers each produce one artifact frame, and the orchestrator wraps each artifact write in its own try/except so individual failures don't block the rest. Strictly read-only against datasets; never raises into training.

**Tech Stack:** Python 3.11+, pandas, numpy, mlflow, pytest (with the `tmp_mlflow_uri` and autouse `_cleanup_mlflow_run` fixtures already in `tests/conftest.py`).

**Spec:** [docs/superpowers/specs/2026-05-01-mlflow-dataset-statistics-design.md](../specs/2026-05-01-mlflow-dataset-statistics-design.md)

## Execution rules (karpathy guidelines)

Each task below traces to the spec — do not invent extra features, columns, abstractions, or "flexibility" not asked for. Specifically:

- **Surface assumptions, don't hide them.** If something in this plan is unclear or contradicts the spec, stop and ask — don't pick silently.
- **Surgical edits only.** Touch only the files this plan names. Do not "improve" adjacent code, formatting, comments, or unrelated tests. If you notice unrelated dead code or inconsistency, mention it but don't delete it.
- **Match existing style.** Follow the patterns already in `mermaidseg/logger.py` (lazy imports inside helpers if reasonable, `try/except` shape, `_ensure_active_run()` gating, `logger.warning(...)` calls).
- **Verify before claiming done.** A task is complete only when the test in its final step actually passes — run the command, check the output. No "should pass" claims without evidence.
- **Frequent commits.** Commit at every task boundary as the plan specifies; don't batch commits across tasks.

---

## File Structure

| File | Action | Purpose |
|---|---|---|
| `mermaidseg/logger.py` | Modify | Add `_resolve_annotations` + four `_compute_*` helpers + `log_dataset_statistics` method on `Logger` |
| `tests/test_logger.py` | Modify | Add unit + integration tests using existing `tmp_mlflow_uri` fixture |
| `tests/_dataset_stubs.py` | Create | Tiny pandas-only fixtures for class/source/Subset/Concat shapes (avoids spinning up real `MermaidDataset` / S3) |
| `scripts/train.py` | Modify | Add `logger.log_dataset_statistics({"train": ..., "val": ..., "test": ...})` after the `log_dataloader_params` block |
| `nbs/Base_Pipeline.ipynb` | Modify | Add the same call in the cell that already calls `logger.log_dataset(...)` |
| `nbs/Combined_Pipeline.ipynb` | Modify | Same call (combined wrappers handled by recursion) |
| `nbs/Concept_Bottleneck_Pipeline.ipynb` | Modify | Same call |

**Decomposition rationale:** Helpers are split by artifact (one frame in / one frame out, pure functions of their inputs). The orchestrator is the only piece that touches MLflow. This keeps the unit-testable surface large and the integration surface small.

**File-size discipline:** `mermaidseg/logger.py` is already ~800 lines covering many concerns (`Logger`, deprecated `WandbLogger`, top-level helpers). Adding ~200 LOC keeps it under 1000; do **not** restructure the file as part of this work — match existing pattern, follow PR #88's call-site style.

---

## Task 1: Test stubs for dataset shapes

**Files:**
- Create: `tests/_dataset_stubs.py`

- [ ] **Step 1: Create the stub module**

Create `tests/_dataset_stubs.py`:

```python
"""Minimal in-memory dataset stubs for logger statistics tests.

Avoids spinning up MermaidDataset / CoralNetDataset (which require S3 + the
MERMAID API). Each stub exposes only the attributes that ``_resolve_annotations``
and the ``_compute_*`` helpers read.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
from torch.utils.data import Subset


@dataclass
class StubDataset:
    """Stand-in for ``MermaidDataset`` / ``CoralNetDataset``.

    Only carries the four attributes the logger reads.
    """

    df_annotations: pd.DataFrame
    df_images: pd.DataFrame
    id2label: dict[int, str]
    num_classes: int

    def __len__(self) -> int:
        return len(self.df_images)


@dataclass
class ConcatStub:
    """Stand-in for combined / ``ConcatDataset``-style wrappers.

    Mirrors the duck-typing the existing ``log_datasets`` already handles:
    expose either ``_datasets`` or ``datasets``.
    """

    _datasets: list = field(default_factory=list)


def make_mermaid_stub(
    *,
    image_to_classes: dict[str, list[str]],
    image_to_region: dict[str, str],
    class_subset: list[str],
) -> StubDataset:
    """Build a Mermaid-shaped stub from a per-image class list.

    ``image_to_classes`` maps ``image_id -> [benthic_attribute_name, ...]``
    (one row per annotation). ``image_to_region`` maps ``image_id -> region_name``.
    """
    rows: list[dict] = []
    for image_id, classes in image_to_classes.items():
        region = image_to_region[image_id]
        for cls in classes:
            rows.append(
                {
                    "image_id": image_id,
                    "benthic_attribute_name": cls,
                    "region_id": region,
                    "region_name": region,
                }
            )
    df_annotations = pd.DataFrame(rows)
    df_images = (
        df_annotations[["image_id", "region_id", "region_name"]]
        .drop_duplicates(subset=["image_id"])
        .reset_index(drop=True)
    )
    id2label = dict(enumerate(class_subset, start=1))
    return StubDataset(
        df_annotations=df_annotations,
        df_images=df_images,
        id2label=id2label,
        num_classes=len(class_subset) + 1,
    )


def make_coralnet_stub(
    *,
    image_to_classes: dict[str, list[str]],
    image_to_source: dict[str, int],
    class_subset: list[str],
) -> StubDataset:
    """CoralNet-shaped stub: ``source_id`` instead of ``region_*``."""
    rows: list[dict] = []
    for image_id, classes in image_to_classes.items():
        source = image_to_source[image_id]
        for cls in classes:
            rows.append(
                {
                    "image_id": image_id,
                    "benthic_attribute_name": cls,
                    "source_id": source,
                }
            )
    df_annotations = pd.DataFrame(rows)
    df_images = (
        df_annotations[["source_id", "image_id"]]
        .drop_duplicates(subset=["image_id"])
        .reset_index(drop=True)
    )
    id2label = dict(enumerate(class_subset, start=1))
    return StubDataset(
        df_annotations=df_annotations,
        df_images=df_images,
        id2label=id2label,
        num_classes=len(class_subset) + 1,
    )


def random_split_indices(stub: StubDataset, splits: dict[str, list[int]]) -> dict[str, Subset]:
    """Wrap a stub in PyTorch ``Subset`` objects for testing the Subset path."""
    return {name: Subset(stub, indices) for name, indices in splits.items()}
```

- [ ] **Step 2: Commit**

```bash
git add tests/_dataset_stubs.py
git commit -m "test: add in-memory dataset stubs for logger statistics tests"
```

---

## Task 2: `_resolve_annotations` — plain dataset path

**Files:**
- Modify: `mermaidseg/logger.py` (add private helper above `Logger.log` definition)
- Modify: `tests/test_logger.py` (add new test class `TestResolveAnnotations`)

- [ ] **Step 1: Write the failing test**

Add to `tests/test_logger.py` (append at bottom, before the `WandbLogger` test class if any):

```python
from mermaidseg.logger import _resolve_annotations
from tests._dataset_stubs import make_mermaid_stub


class TestResolveAnnotations:
    def test_plain_dataset_returns_attributes_unchanged(self):
        stub = make_mermaid_stub(
            image_to_classes={
                "img-1": ["Acropora", "Porites"],
                "img-2": ["Macroalgae"],
            },
            image_to_region={"img-1": "Indonesia", "img-2": "Indonesia"},
            class_subset=["Acropora", "Porites", "Macroalgae"],
        )
        df_ann, df_img, id2label = _resolve_annotations(stub)
        assert df_ann is stub.df_annotations
        assert df_img is stub.df_images
        assert id2label is stub.id2label
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_logger.py::TestResolveAnnotations::test_plain_dataset_returns_attributes_unchanged -v`
Expected: FAIL with `ImportError: cannot import name '_resolve_annotations'`.

- [ ] **Step 3: Implement minimal version in `mermaidseg/logger.py`**

Add at module level, just below the `LOCAL_DEFAULT_URI` constant:

```python
def _resolve_annotations(split):
    """Resolve a split into ``(df_annotations, df_images, id2label)``.

    Returns ``None`` if the input is an unsupported shape; never raises.
    """
    if hasattr(split, "df_annotations") and hasattr(split, "df_images") and hasattr(split, "id2label"):
        return split.df_annotations, split.df_images, split.id2label
    return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_logger.py::TestResolveAnnotations::test_plain_dataset_returns_attributes_unchanged -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add mermaidseg/logger.py tests/test_logger.py
git commit -m "feat(logger): _resolve_annotations handles plain datasets"
```

---

## Task 3: `_resolve_annotations` — Subset path

**Files:**
- Modify: `mermaidseg/logger.py`
- Modify: `tests/test_logger.py`

- [ ] **Step 1: Write the failing test**

Add to `TestResolveAnnotations`:

```python
def test_subset_filters_annotations_by_subset_image_ids(self):
    stub = make_mermaid_stub(
        image_to_classes={
            "img-1": ["Acropora"],
            "img-2": ["Porites"],
            "img-3": ["Macroalgae"],
        },
        image_to_region={"img-1": "A", "img-2": "B", "img-3": "C"},
        class_subset=["Acropora", "Porites", "Macroalgae"],
    )
    from torch.utils.data import Subset

    subset = Subset(stub, [0, 2])  # img-1 and img-3
    df_ann, df_img, id2label = _resolve_annotations(subset)

    assert sorted(df_ann["image_id"].unique().tolist()) == ["img-1", "img-3"]
    assert sorted(df_img["image_id"].tolist()) == ["img-1", "img-3"]
    assert id2label == stub.id2label
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_logger.py::TestResolveAnnotations::test_subset_filters_annotations_by_subset_image_ids -v`
Expected: FAIL — current helper returns `None` for `Subset`.

- [ ] **Step 3: Extend `_resolve_annotations`**

Replace the helper with:

```python
def _resolve_annotations(split):
    """Resolve a split into ``(df_annotations, df_images, id2label)``.

    Handles three input shapes:
      * Plain dataset with ``df_annotations`` / ``df_images`` / ``id2label``
      * PyTorch ``Subset`` (filters parent annotations by subset image ids)
      * Combined / ``ConcatDataset``-style wrapper exposing ``_datasets`` or ``datasets``

    Returns ``None`` for any other shape; never raises.
    """
    from torch.utils.data import Subset

    if isinstance(split, Subset):
        parent = split.dataset
        resolved = _resolve_annotations(parent)
        if resolved is None:
            return None
        parent_ann, parent_img, id2label = resolved
        df_images = parent_img.iloc[list(split.indices)].reset_index(drop=True)
        df_annotations = parent_ann[parent_ann["image_id"].isin(df_images["image_id"])]
        return df_annotations, df_images, id2label

    if hasattr(split, "df_annotations") and hasattr(split, "df_images") and hasattr(split, "id2label"):
        return split.df_annotations, split.df_images, split.id2label

    return None
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_logger.py::TestResolveAnnotations -v`
Expected: both tests PASS.

- [ ] **Step 5: Commit**

```bash
git add mermaidseg/logger.py tests/test_logger.py
git commit -m "feat(logger): _resolve_annotations handles PyTorch Subset"
```

---

## Task 4: `_resolve_annotations` — ConcatDataset wrapper

**Files:**
- Modify: `mermaidseg/logger.py`
- Modify: `tests/test_logger.py`

- [ ] **Step 1: Write the failing test**

Add to `TestResolveAnnotations`:

```python
def test_concat_wrapper_concatenates_children(self):
    stub_a = make_mermaid_stub(
        image_to_classes={"a-1": ["Acropora"]},
        image_to_region={"a-1": "Indonesia"},
        class_subset=["Acropora", "Porites"],
    )
    stub_b = make_mermaid_stub(
        image_to_classes={"b-1": ["Porites"]},
        image_to_region={"b-1": "Caribbean"},
        class_subset=["Acropora", "Porites"],
    )
    from tests._dataset_stubs import ConcatStub
    wrapper = ConcatStub(_datasets=[stub_a, stub_b])

    df_ann, df_img, id2label = _resolve_annotations(wrapper)
    assert sorted(df_ann["image_id"].tolist()) == ["a-1", "b-1"]
    assert sorted(df_img["image_id"].tolist()) == ["a-1", "b-1"]
    assert id2label == stub_a.id2label

def test_concat_wrapper_warns_on_id2label_mismatch(self, caplog):
    import logging
    stub_a = make_mermaid_stub(
        image_to_classes={"a-1": ["Acropora"]},
        image_to_region={"a-1": "X"},
        class_subset=["Acropora"],
    )
    stub_b = make_mermaid_stub(
        image_to_classes={"b-1": ["Porites"]},
        image_to_region={"b-1": "Y"},
        class_subset=["Porites"],
    )
    from tests._dataset_stubs import ConcatStub
    wrapper = ConcatStub(_datasets=[stub_a, stub_b])

    with caplog.at_level(logging.WARNING, logger="mermaidseg.logger"):
        _, _, id2label = _resolve_annotations(wrapper)

    assert id2label == stub_a.id2label  # first child wins
    assert any("id2label mismatch" in r.message for r in caplog.records)

def test_unknown_shape_returns_none_and_warns(self, caplog):
    import logging
    with caplog.at_level(logging.WARNING, logger="mermaidseg.logger"):
        result = _resolve_annotations(object())
    assert result is None
    assert any("unsupported split shape" in r.message.lower() for r in caplog.records)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_logger.py::TestResolveAnnotations -v`
Expected: three new tests FAIL.

- [ ] **Step 3: Extend `_resolve_annotations` with concat + unknown branches**

Replace the helper body (keep the Subset branch, add concat and unknown handling):

```python
def _resolve_annotations(split):
    """Resolve a split into ``(df_annotations, df_images, id2label)``.

    Handles four input shapes:
      * Plain dataset with ``df_annotations`` / ``df_images`` / ``id2label``
      * PyTorch ``Subset`` (filters parent annotations by subset image ids)
      * Combined / ``ConcatDataset``-style wrapper exposing ``_datasets`` or ``datasets``
      * Unknown — returns ``None`` and logs a warning

    Never raises.
    """
    import pandas as pd
    from torch.utils.data import Subset

    if isinstance(split, Subset):
        parent = split.dataset
        resolved = _resolve_annotations(parent)
        if resolved is None:
            return None
        parent_ann, parent_img, id2label = resolved
        df_images = parent_img.iloc[list(split.indices)].reset_index(drop=True)
        df_annotations = parent_ann[parent_ann["image_id"].isin(df_images["image_id"])]
        return df_annotations, df_images, id2label

    children = getattr(split, "_datasets", None) or getattr(split, "datasets", None)
    if children is not None:
        resolved_children = [_resolve_annotations(c) for c in children]
        resolved_children = [r for r in resolved_children if r is not None]
        if not resolved_children:
            return None
        ann_frames = [r[0] for r in resolved_children]
        img_frames = [r[1] for r in resolved_children]
        id2labels = [r[2] for r in resolved_children]
        if any(m != id2labels[0] for m in id2labels[1:]):
            logger.warning("id2label mismatch across combined children; using first child's mapping")
        return (
            pd.concat(ann_frames, ignore_index=True),
            pd.concat(img_frames, ignore_index=True),
            id2labels[0],
        )

    if hasattr(split, "df_annotations") and hasattr(split, "df_images") and hasattr(split, "id2label"):
        return split.df_annotations, split.df_images, split.id2label

    logger.warning("Unsupported split shape: %s — skipping", type(split).__name__)
    return None
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_logger.py::TestResolveAnnotations -v`
Expected: all 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add mermaidseg/logger.py tests/test_logger.py
git commit -m "feat(logger): _resolve_annotations handles concat wrappers and unknown shapes"
```

---

## Task 5: `_compute_class_counts`

**Files:**
- Modify: `mermaidseg/logger.py`
- Modify: `tests/test_logger.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_logger.py`:

```python
from mermaidseg.logger import _compute_class_counts


class TestComputeClassCounts:
    def _splits(self):
        stub = make_mermaid_stub(
            image_to_classes={
                "img-1": ["Acropora", "Acropora", "Porites"],
                "img-2": ["Acropora"],
                "img-3": ["Other"],  # unclassified row
                "img-4": ["Porites"],
            },
            image_to_region={"img-1": "A", "img-2": "A", "img-3": "B", "img-4": "B"},
            class_subset=["Acropora", "Porites", "Other"],
        )
        from torch.utils.data import Subset
        return {
            "train": Subset(stub, [0, 1]),  # img-1, img-2
            "val": Subset(stub, [2]),       # img-3
            "test": Subset(stub, [3]),      # img-4
        }, stub

    def test_class_counts_schema_and_values(self):
        splits, stub = self._splits()
        resolved = {k: _resolve_annotations(v) for k, v in splits.items()}

        df = _compute_class_counts(resolved, parent_id2label=stub.id2label)

        assert list(df.columns) == [
            "class_id", "class_name", "class_kind",
            "train_annotations", "val_annotations", "test_annotations",
            "train_images", "val_images", "test_images",
            "train_fraction", "val_fraction", "test_fraction",
        ]
        # Background row plus three classes.
        assert df["class_id"].tolist() == [0, 1, 2, 3]
        assert df["class_name"].tolist() == ["background", "Acropora", "Porites", "Other"]
        assert df["class_kind"].tolist() == ["background", "target", "target", "unclassified"]

        acropora = df.set_index("class_name").loc["Acropora"]
        assert acropora["train_annotations"] == 3  # 2 + 1
        assert acropora["val_annotations"] == 0
        assert acropora["test_annotations"] == 0
        assert acropora["train_images"] == 2

        # Fractions sum to 1.0 within each split (zero rows summed to total)
        assert abs(df["train_fraction"].sum() - 1.0) < 1e-9

    def test_class_kind_qualified_other_stays_target(self):
        stub = make_mermaid_stub(
            image_to_classes={"img-1": ["Other Invertebrates"]},
            image_to_region={"img-1": "A"},
            class_subset=["Other Invertebrates"],
        )
        from torch.utils.data import Subset
        resolved = {"train": _resolve_annotations(Subset(stub, [0]))}
        df = _compute_class_counts(resolved, parent_id2label=stub.id2label)
        kind = df.set_index("class_name").loc["Other Invertebrates", "class_kind"]
        assert kind == "target"

    def test_empty_split_yields_zero_row_not_missing(self):
        splits, stub = self._splits()
        # Drop the 'val' split entirely.
        resolved = {"train": _resolve_annotations(splits["train"])}
        df = _compute_class_counts(resolved, parent_id2label=stub.id2label)
        # No val/test columns should appear in the schema when those splits aren't passed.
        for col in df.columns:
            assert "val_" not in col and "test_" not in col
        # Train-only fractions still sum to 1.
        assert abs(df["train_fraction"].sum() - 1.0) < 1e-9
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_logger.py::TestComputeClassCounts -v`
Expected: FAIL with `ImportError: cannot import name '_compute_class_counts'`.

- [ ] **Step 3: Implement `_compute_class_counts`**

Add to `mermaidseg/logger.py` near `_resolve_annotations`:

```python
_UNCLASSIFIED_NAMES = {"other", "unknown", "unclassified"}


def _classify_kind(class_id: int, class_name: str) -> str:
    if class_id == 0:
        return "background"
    if class_name.strip().lower() in _UNCLASSIFIED_NAMES:
        return "unclassified"
    return "target"


def _compute_class_counts(resolved_splits: dict, parent_id2label: dict[int, str]) -> "pd.DataFrame":
    """Per-class × split counts and fractions. Includes implicit background row at id=0.

    ``resolved_splits`` is a mapping from split-name to the tuple returned by
    ``_resolve_annotations`` (callers must drop ``None`` results before calling).
    ``parent_id2label`` is the post-``class_subset`` mapping; id 0 (background)
    is implicit and added to the output.
    """
    import pandas as pd

    rows: list[dict] = []
    # Build the canonical class list: id 0 background, then id2label entries.
    all_classes: list[tuple[int, str]] = [(0, "background")]
    all_classes.extend(sorted(parent_id2label.items()))

    split_names = list(resolved_splits.keys())

    for class_id, class_name in all_classes:
        row: dict = {
            "class_id": class_id,
            "class_name": class_name,
            "class_kind": _classify_kind(class_id, class_name),
        }
        for split_name in split_names:
            df_ann, _df_img, _ = resolved_splits[split_name]
            ann_count = int((df_ann["benthic_attribute_name"] == class_name).sum())
            img_count = int(
                df_ann.loc[df_ann["benthic_attribute_name"] == class_name, "image_id"].nunique()
            )
            row[f"{split_name}_annotations"] = ann_count
            row[f"{split_name}_images"] = img_count
        rows.append(row)

    df = pd.DataFrame(rows)

    # Per-split fraction-of-annotations (relative to all classes within that split).
    for split_name in split_names:
        col = f"{split_name}_annotations"
        total = df[col].sum()
        df[f"{split_name}_fraction"] = df[col] / total if total else 0.0

    # Reorder: identity, then per-split *_annotations / *_images, then *_fraction.
    ordered = ["class_id", "class_name", "class_kind"]
    ordered += [f"{s}_annotations" for s in split_names]
    ordered += [f"{s}_images" for s in split_names]
    ordered += [f"{s}_fraction" for s in split_names]
    return df[ordered]
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_logger.py::TestComputeClassCounts -v`
Expected: all 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add mermaidseg/logger.py tests/test_logger.py
git commit -m "feat(logger): _compute_class_counts emits per-split counts and fractions"
```

---

## Task 6: `_compute_source_stats`

**Files:**
- Modify: `mermaidseg/logger.py`
- Modify: `tests/test_logger.py`

- [ ] **Step 1: Write the failing test**

```python
from mermaidseg.logger import _compute_source_stats


class TestComputeSourceStats:
    def test_mermaid_region_rows(self):
        stub = make_mermaid_stub(
            image_to_classes={
                "img-1": ["Acropora"],
                "img-2": ["Acropora", "Porites"],
            },
            image_to_region={"img-1": "Indonesia", "img-2": "Caribbean"},
            class_subset=["Acropora", "Porites"],
        )
        from torch.utils.data import Subset
        resolved = {
            "train": _resolve_annotations(Subset(stub, [0])),
            "val": _resolve_annotations(Subset(stub, [1])),
        }
        df = _compute_source_stats(resolved)

        assert set(df["source_type"]) == {"region"}
        idx = df.set_index("source_key")
        assert idx.loc["Indonesia", "train_images"] == 1
        assert idx.loc["Indonesia", "val_images"] == 0
        assert idx.loc["Caribbean", "val_annotations"] == 2
        assert "test_images" not in df.columns  # test split not provided

    def test_coralnet_source_rows_cast_to_str(self):
        from tests._dataset_stubs import make_coralnet_stub
        stub = make_coralnet_stub(
            image_to_classes={"img-1": ["Acropora"], "img-2": ["Porites"]},
            image_to_source={"img-1": 42, "img-2": 7},
            class_subset=["Acropora", "Porites"],
        )
        from torch.utils.data import Subset
        resolved = {"train": _resolve_annotations(Subset(stub, [0, 1]))}
        df = _compute_source_stats(resolved)

        assert set(df["source_type"]) == {"source"}
        assert set(df["source_key"]) == {"42", "7"}
        assert df["source_key"].dtype == object  # strings, not ints
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_logger.py::TestComputeSourceStats -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement `_compute_source_stats`**

Add to `mermaidseg/logger.py`:

```python
def _source_columns(df_annotations: "pd.DataFrame") -> tuple[str, str] | None:
    """Return ``(source_type, source_key_column)`` based on which columns are present.

    Mirrors the existing ``BaseCoralDataset.__init__`` branch — ``region_id`` for
    Mermaid, ``source_id`` for CoralNet. Returns ``None`` if neither is present
    (caller will emit a zero-row frame).
    """
    if "region_id" in df_annotations.columns:
        return "region", "region_name"
    if "source_id" in df_annotations.columns:
        return "source", "source_id"
    return None


def _compute_source_stats(resolved_splits: dict) -> "pd.DataFrame":
    """Per-source × split image and annotation counts.

    Mermaid-shaped (``region_*``) and CoralNet-shaped (``source_id``) rows
    coexist in the same frame, distinguished by ``source_type``. ``source_key``
    is always a string (CoralNet ints get cast).
    """
    import pandas as pd

    split_names = list(resolved_splits.keys())
    rows_by_key: dict[tuple[str, str], dict] = {}

    for split_name in split_names:
        df_ann, df_img, _ = resolved_splits[split_name]
        cols = _source_columns(df_ann)
        if cols is None:
            continue
        source_type, key_col = cols

        # Image counts: from df_img (one row per image).
        img_per_source = df_img[key_col].astype(str).value_counts().to_dict()
        # Annotation counts: from df_ann.
        ann_per_source = df_ann[key_col].astype(str).value_counts().to_dict()

        for source_key in set(img_per_source) | set(ann_per_source):
            row = rows_by_key.setdefault(
                (source_type, source_key),
                {"source_key": source_key, "source_type": source_type},
            )
            row[f"{split_name}_images"] = img_per_source.get(source_key, 0)
            row[f"{split_name}_annotations"] = ann_per_source.get(source_key, 0)

    # Zero-fill any missing per-split columns so the schema is uniform.
    for row in rows_by_key.values():
        for split_name in split_names:
            row.setdefault(f"{split_name}_images", 0)
            row.setdefault(f"{split_name}_annotations", 0)

    df = pd.DataFrame(list(rows_by_key.values()))
    if df.empty:
        cols = ["source_key", "source_type"]
        for split_name in split_names:
            cols += [f"{split_name}_images", f"{split_name}_annotations"]
        return pd.DataFrame(columns=cols)

    ordered = ["source_key", "source_type"]
    ordered += [f"{s}_images" for s in split_names]
    ordered += [f"{s}_annotations" for s in split_names]
    return df[ordered].sort_values(["source_type", "source_key"]).reset_index(drop=True)
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_logger.py::TestComputeSourceStats -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add mermaidseg/logger.py tests/test_logger.py
git commit -m "feat(logger): _compute_source_stats emits region/source × split rows"
```

---

## Task 7: `_compute_class_by_source`

**Files:**
- Modify: `mermaidseg/logger.py`
- Modify: `tests/test_logger.py`

- [ ] **Step 1: Write the failing test**

```python
from mermaidseg.logger import _compute_class_by_source


class TestComputeClassBySource:
    def test_long_format_omits_zero_rows(self):
        stub = make_mermaid_stub(
            image_to_classes={
                "img-1": ["Acropora"],
                "img-2": ["Porites"],
            },
            image_to_region={"img-1": "Indonesia", "img-2": "Caribbean"},
            class_subset=["Acropora", "Porites"],
        )
        from torch.utils.data import Subset
        resolved = {
            "train": _resolve_annotations(Subset(stub, [0])),
            "val": _resolve_annotations(Subset(stub, [1])),
        }
        df = _compute_class_by_source(resolved, parent_id2label=stub.id2label)

        assert list(df.columns) == [
            "source_key", "source_type", "class_id", "class_name",
            "split", "annotations", "images",
        ]
        # No zero-count rows: only (Indonesia, Acropora, train) and (Caribbean, Porites, val)
        assert len(df) == 2
        assert df.set_index(["source_key", "class_name", "split"]).loc[
            ("Indonesia", "Acropora", "train"), "annotations"
        ] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_logger.py::TestComputeClassBySource -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement `_compute_class_by_source`**

```python
def _compute_class_by_source(resolved_splits: dict, parent_id2label: dict[int, str]) -> "pd.DataFrame":
    """Long-format source × class × split frame.

    Zero-count rows are omitted to keep file size sane when there are many
    sources × classes × splits. Background (id 0) is excluded — annotation
    rows never carry the background class.
    """
    import pandas as pd

    label2id = {name: cid for cid, name in parent_id2label.items()}

    rows: list[dict] = []
    for split_name, (df_ann, _df_img, _) in resolved_splits.items():
        cols = _source_columns(df_ann)
        if cols is None or df_ann.empty:
            continue
        source_type, key_col = cols
        grouped = (
            df_ann.assign(_source_key=df_ann[key_col].astype(str))
            .groupby(["_source_key", "benthic_attribute_name"], observed=True)
            .agg(annotations=("image_id", "size"), images=("image_id", "nunique"))
            .reset_index()
        )
        for _, r in grouped.iterrows():
            class_name = r["benthic_attribute_name"]
            class_id = label2id.get(class_name)
            if class_id is None:
                # Annotation references a class not in id2label (e.g. filtered post-build).
                continue
            rows.append(
                {
                    "source_key": r["_source_key"],
                    "source_type": source_type,
                    "class_id": class_id,
                    "class_name": class_name,
                    "split": split_name,
                    "annotations": int(r["annotations"]),
                    "images": int(r["images"]),
                }
            )

    cols = ["source_key", "source_type", "class_id", "class_name", "split", "annotations", "images"]
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows)[cols].sort_values(
        ["source_type", "source_key", "class_id", "split"]
    ).reset_index(drop=True)
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_logger.py::TestComputeClassBySource -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add mermaidseg/logger.py tests/test_logger.py
git commit -m "feat(logger): _compute_class_by_source emits long-format drift matrix"
```

---

## Task 8: `_compute_train_summary`

**Files:**
- Modify: `mermaidseg/logger.py`
- Modify: `tests/test_logger.py`

- [ ] **Step 1: Write the failing test**

```python
from mermaidseg.logger import _compute_train_summary


class TestComputeTrainSummary:
    def test_summary_metrics_and_distribution(self):
        stub = make_mermaid_stub(
            image_to_classes={
                "img-1": ["Acropora", "Acropora", "Porites"],
                "img-2": ["Acropora"],
                "img-3": ["Macroalgae"],
                "img-4": [],  # an image with zero annotations (loss-of-points scenario)
            },
            image_to_region={"img-1": "A", "img-2": "A", "img-3": "B", "img-4": "B"},
            class_subset=["Acropora", "Porites", "Macroalgae"],
        )
        # Re-add the zero-annotation image — make_mermaid_stub drops images with no rows
        # because df_images is built from df_annotations. Patch by hand:
        import pandas as pd
        stub.df_images = pd.concat(
            [stub.df_images, pd.DataFrame([{"image_id": "img-4", "region_id": "B", "region_name": "B"}])],
            ignore_index=True,
        )

        from torch.utils.data import Subset
        resolved = {
            "train": _resolve_annotations(Subset(stub, [0, 1])),  # img-1, img-2
            "val": _resolve_annotations(Subset(stub, [2])),       # img-3
            "test": _resolve_annotations(Subset(stub, [3])),      # img-4 (no anns)
        }

        summary = _compute_train_summary(
            resolved, parent_id2label=stub.id2label, class_subset=["Acropora", "Porites", "Macroalgae"]
        )

        assert summary["total_images"] == 4
        assert summary["total_annotations"] == 5
        assert summary["splits"]["train"] == {"images": 2, "annotations": 4}
        assert summary["splits"]["test"] == {"images": 1, "annotations": 0}
        assert summary["class_subset"] == ["Acropora", "Porites", "Macroalgae"]
        assert summary["num_classes"] == 4

        # top-K shares: train has Acropora=3, Porites=1 → top1=0.75
        assert abs(summary["top1_share"] - 0.75) < 1e-9
        # Only 2 eligible classes in train → top3 / top5 clamp to total = 1.0
        assert abs(summary["top3_share"] - 1.0) < 1e-9
        assert abs(summary["top5_share"] - 1.0) < 1e-9

        # effective_num_classes = exp(entropy) ∈ [1, num_eligible]
        assert 1.0 <= summary["effective_num_classes"] <= 3.0

        # Annotations-per-image distribution catches the zero-annotation image
        assert summary["annotations_per_image"]["test"]["min"] == 0
        assert summary["annotations_per_image"]["train"]["max"] == 3
        assert summary["annotations_per_image"]["train"]["mean"] == 2.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_logger.py::TestComputeTrainSummary -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Add `import math` to `mermaidseg/logger.py` if not present**

`numpy` is already imported as `np` at line 24. Add `import math` to the top of the file alongside the other stdlib imports if it's not already there.

- [ ] **Step 4: Implement `_compute_train_summary`**

Add to `mermaidseg/logger.py`:

```python
def _compute_train_summary(
    resolved_splits: dict,
    parent_id2label: dict[int, str],
    class_subset: list[str] | None,
) -> dict:
    """Summary metrics dict serialized to ``train_summary.yaml``.

    Class-balance metrics (``top1/3/5_share``, ``effective_num_classes``) are
    computed over the **training** split only, and exclude classes whose
    ``class_kind`` is ``background`` or ``unclassified``.
    """
    summary: dict = {
        "total_images": 0,
        "total_annotations": 0,
        "splits": {},
        "class_subset": list(class_subset) if class_subset is not None else None,
        "num_classes": len(parent_id2label) + 1,  # +1 for background
    }

    annotations_per_image: dict[str, dict] = {}

    for split_name, (df_ann, df_img, _) in resolved_splits.items():
        n_images = int(len(df_img))
        n_annotations = int(len(df_ann))
        summary["splits"][split_name] = {"images": n_images, "annotations": n_annotations}
        summary["total_images"] += n_images
        summary["total_annotations"] += n_annotations

        # Per-image annotation density distribution. Re-index against df_img so
        # images with zero annotations contribute a 0 (not NaN, not absent).
        if n_images:
            counts = (
                df_ann.groupby("image_id").size()
                .reindex(df_img["image_id"].tolist(), fill_value=0)
                .astype(int)
            )
            annotations_per_image[split_name] = {
                "mean": float(counts.mean()),
                "median": float(counts.median()),
                "p10": float(counts.quantile(0.10)),
                "p90": float(counts.quantile(0.90)),
                "min": int(counts.min()),
                "max": int(counts.max()),
            }
        else:
            annotations_per_image[split_name] = {
                "mean": 0.0, "median": 0.0, "p10": 0.0, "p90": 0.0, "min": 0, "max": 0,
            }

    summary["annotations_per_image"] = annotations_per_image

    # Class-balance metrics over the training split, eligible classes only.
    train = resolved_splits.get("train")
    if train is not None:
        df_ann, _df_img, _ = train
        eligible_names = [
            name for cid, name in parent_id2label.items()
            if _classify_kind(cid, name) == "target"
        ]
        counts = (
            df_ann["benthic_attribute_name"]
            .value_counts()
            .reindex(eligible_names, fill_value=0)
            .sort_values(ascending=False)
        )
        total = int(counts.sum())

        def _topk_share(k: int) -> float:
            if total == 0:
                return 0.0
            return float(counts.head(k).sum() / total)

        summary["top1_share"] = _topk_share(1)
        summary["top3_share"] = _topk_share(3)
        summary["top5_share"] = _topk_share(5)

        if total > 0:
            probs = (counts / total).to_numpy()
            probs = probs[probs > 0]  # Mask zeros — log(0) is -inf.
            entropy = float(-(probs * np.log(probs)).sum())
            summary["effective_num_classes"] = float(math.exp(entropy))
        else:
            summary["effective_num_classes"] = 0.0

    return summary
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/test_logger.py::TestComputeTrainSummary -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add mermaidseg/logger.py tests/test_logger.py
git commit -m "feat(logger): _compute_train_summary emits topK shares and density distribution"
```

---

## Task 9: `Logger.log_dataset_statistics` orchestrator

**Files:**
- Modify: `mermaidseg/logger.py` (add method to `Logger` class)
- Modify: `tests/test_logger.py` (add integration test class)

- [ ] **Step 1: Write the failing test**

```python
from unittest.mock import patch


class TestLogDatasetStatistics:
    def test_happy_path_writes_four_artifacts(self, tmp_mlflow_uri, make_config, fake_meta_model, tmp_path):
        from torch.utils.data import Subset
        stub = make_mermaid_stub(
            image_to_classes={
                "img-1": ["Acropora"],
                "img-2": ["Porites"],
                "img-3": ["Acropora", "Porites"],
            },
            image_to_region={"img-1": "A", "img-2": "B", "img-3": "B"},
            class_subset=["Acropora", "Porites"],
        )
        config = make_config()
        lgr = Logger(config=config, meta_model=fake_meta_model)

        try:
            lgr.log_dataset_statistics(
                {
                    "train": Subset(stub, [0]),
                    "val": Subset(stub, [1]),
                    "test": Subset(stub, [2]),
                }
            )
        finally:
            run_id = lgr.mlflow_run_id
            lgr.end_run()

        client = mlflow.MlflowClient()
        artifacts = {a.path for a in client.list_artifacts(run_id, "dataset_stats")}
        assert artifacts == {
            "dataset_stats/class_counts.csv",
            "dataset_stats/source_stats.csv",
            "dataset_stats/class_by_source.csv",
            "dataset_stats/train_summary.yaml",
        }

    def test_disabled_logger_is_noop(self, tmp_mlflow_uri, make_config, fake_meta_model):
        # No experiment_name → logger disabled.
        config = make_config(logger={"experiment_name": None})
        lgr = Logger(config=config, meta_model=fake_meta_model)
        # Should not raise even with bogus splits.
        lgr.log_dataset_statistics({"train": object()})

    def test_artifact_failure_does_not_drop_others(
        self, tmp_mlflow_uri, make_config, fake_meta_model, caplog
    ):
        from torch.utils.data import Subset
        stub = make_mermaid_stub(
            image_to_classes={"img-1": ["Acropora"]},
            image_to_region={"img-1": "A"},
            class_subset=["Acropora"],
        )
        config = make_config()
        lgr = Logger(config=config, meta_model=fake_meta_model)

        # Make log_text fail only for class_counts.csv; let the others through.
        real_log_text = mlflow.log_text
        def fake_log_text(text, artifact_file):
            if artifact_file.endswith("class_counts.csv"):
                raise RuntimeError("boom")
            return real_log_text(text, artifact_file)

        try:
            with patch("mermaidseg.logger.mlflow.log_text", side_effect=fake_log_text):
                lgr.log_dataset_statistics({"train": Subset(stub, [0])})
            run_id = lgr.mlflow_run_id
        finally:
            lgr.end_run()

        client = mlflow.MlflowClient()
        artifacts = {a.path for a in client.list_artifacts(run_id, "dataset_stats")}
        # class_counts failed; the other three should still be present.
        assert "dataset_stats/class_counts.csv" not in artifacts
        assert "dataset_stats/source_stats.csv" in artifacts
        assert "dataset_stats/class_by_source.csv" in artifacts
        assert "dataset_stats/train_summary.yaml" in artifacts
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_logger.py::TestLogDatasetStatistics -v`
Expected: FAIL with `AttributeError: 'Logger' object has no attribute 'log_dataset_statistics'`.

- [ ] **Step 3: Add `import yaml` to `mermaidseg/logger.py` module-level imports**

Add `import yaml` alongside the other stdlib/third-party imports at the top of the file (don't import inside the method body).

- [ ] **Step 4: Add `log_dataset_statistics` to `Logger`**

Add method to `Logger` class in `mermaidseg/logger.py`, near `log_dataloader_params`:

```python
    def log_dataset_statistics(
        self,
        splits: dict,
        *,
        artifact_dir: str = "dataset_stats",
    ) -> None:
        """Log per-run dataset distribution statistics as MLflow artifacts.

        Emits four artifacts under ``artifact_dir/``:
          * ``class_counts.csv`` — per-class × split counts and fractions
          * ``source_stats.csv`` — per region/source × split image and annotation counts
          * ``class_by_source.csv`` — long-format drift matrix
          * ``train_summary.yaml`` — top-K shares, effective_num_classes, density distribution

        Strictly read-only against datasets. Per-split resolution failures and
        per-artifact write failures are isolated — one bad split or one failed
        artifact does not block the rest.
        """
        if not self._ensure_active_run():
            return
        try:
            resolved: dict = {}
            for split_name, split in splits.items():
                try:
                    r = _resolve_annotations(split)
                except Exception as e:  # noqa: BLE001
                    logger.warning("Failed to resolve split %s: %s", split_name, e)
                    continue
                if r is None:
                    continue
                resolved[split_name] = r

            if not resolved:
                logger.warning("log_dataset_statistics: no splits resolved; nothing to log")
                return

            parent_id2label = next(iter(resolved.values()))[2]
            class_subset = getattr(getattr(self.config, "data", None), "class_subset", None)

            artifacts = {
                f"{artifact_dir}/class_counts.csv": (
                    "csv",
                    lambda: _compute_class_counts(resolved, parent_id2label),
                ),
                f"{artifact_dir}/source_stats.csv": (
                    "csv",
                    lambda: _compute_source_stats(resolved),
                ),
                f"{artifact_dir}/class_by_source.csv": (
                    "csv",
                    lambda: _compute_class_by_source(resolved, parent_id2label),
                ),
                f"{artifact_dir}/train_summary.yaml": (
                    "yaml",
                    lambda: _compute_train_summary(resolved, parent_id2label, class_subset),
                ),
            }

            for path, (kind, builder) in artifacts.items():
                try:
                    payload = builder()
                    if kind == "csv":
                        mlflow.log_text(payload.to_csv(index=False), path)
                    else:
                        mlflow.log_text(yaml.safe_dump(payload, sort_keys=False), path)
                except Exception as e:  # noqa: BLE001
                    logger.warning("Failed to log %s: %s", path, e)

        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to log dataset statistics: %s", e)
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/test_logger.py::TestLogDatasetStatistics -v`
Expected: all 3 tests PASS.

- [ ] **Step 6: Run the full logger test file**

Run: `uv run pytest tests/test_logger.py -v`
Expected: all existing tests still PASS, plus the new ones.

- [ ] **Step 7: Commit**

```bash
git add mermaidseg/logger.py tests/test_logger.py
git commit -m "feat(logger): add Logger.log_dataset_statistics orchestrator"
```

---

## Task 10: Wire call site in `scripts/train.py`

**Files:**
- Modify: `scripts/train.py`

- [ ] **Step 1: Locate the insertion point**

In `scripts/train.py`, find the block (added by PR #88):

```python
logger.log_dataloader_params(train_loader, prefix="train_loader")
logger.log_dataloader_params(val_loader, prefix="val_loader")
logger.log_dataloader_params(test_loader, prefix="test_loader")
```

- [ ] **Step 2: Add the new call below it**

```python
logger.log_dataloader_params(train_loader, prefix="train_loader")
logger.log_dataloader_params(val_loader, prefix="val_loader")
logger.log_dataloader_params(test_loader, prefix="test_loader")
logger.log_dataset_statistics({"train": train_ds, "val": val_ds, "test": test_ds})
```

- [ ] **Step 3: Smoke check**

Run: `uv run python -c "import scripts.train"` — must not raise.
Run: `uv run pytest tests/test_logger.py -v` — must still PASS.

- [ ] **Step 4: Commit**

```bash
git add scripts/train.py
git commit -m "feat(scripts): wire log_dataset_statistics into train.py"
```

---

## Task 11: Wire call site in notebooks

**Files:**
- Modify: `nbs/Base_Pipeline.ipynb`
- Modify: `nbs/Combined_Pipeline.ipynb`
- Modify: `nbs/Concept_Bottleneck_Pipeline.ipynb`

Variable names confirmed by grep:
- `Base_Pipeline.ipynb` → `train_dataset, val_dataset, test_dataset` (line ~212), existing `logger.log_dataset(dataset_dict["train"], context="training")` at line ~359
- `Concept_Bottleneck_Pipeline.ipynb` → same names (line ~194)
- `Combined_Pipeline.ipynb` → `combined_train, combined_val, combined_test` (line ~271), existing `logger.log_datasets(combined_train, context="training")` at line ~433

- [ ] **Step 1: Add `logger.log_dataset_statistics(...)` line after each existing `log_dataset(s)` call**

For `Base_Pipeline.ipynb` and `Concept_Bottleneck_Pipeline.ipynb`:

```python
logger.log_dataset_statistics({"train": train_dataset, "val": val_dataset, "test": test_dataset})
```

For `Combined_Pipeline.ipynb`:

```python
logger.log_dataset_statistics({"train": combined_train, "val": combined_val, "test": combined_test})
```

In each notebook, this line goes in the same cell as the existing `logger.log_dataset(...)` / `logger.log_datasets(...)` call, immediately after it.

- [ ] **Step 2: Verify each notebook still parses as JSON**

Run: `uv run python -c "import json; [json.load(open(p)) for p in ['nbs/Base_Pipeline.ipynb', 'nbs/Combined_Pipeline.ipynb', 'nbs/Concept_Bottleneck_Pipeline.ipynb']]"`
Expected: no error.

- [ ] **Step 3: Commit**

```bash
git add nbs/Base_Pipeline.ipynb nbs/Combined_Pipeline.ipynb nbs/Concept_Bottleneck_Pipeline.ipynb
git commit -m "feat(nbs): wire log_dataset_statistics into all training notebooks"
```

---

## Task 12: Final verification

- [ ] **Step 1: Run the whole test suite**

Run: `uv run pytest -v`
Expected: all tests PASS.

- [ ] **Step 2: Lint**

Run: `uv run ruff check mermaidseg/logger.py tests/test_logger.py tests/_dataset_stubs.py scripts/train.py`
Expected: clean.

Run: `uv run ruff format --check mermaidseg/logger.py tests/test_logger.py tests/_dataset_stubs.py scripts/train.py`
Expected: clean.

- [ ] **Step 3: Push and open PR**

```bash
git push -u origin <branch>
gh pr create --title "feat(logger): add log_dataset_statistics for per-run dataset QA (#72)" --body "$(cat <<'EOF'
## Summary
- Adds `Logger.log_dataset_statistics(splits)` emitting 4 MLflow artifacts: `class_counts.csv`, `source_stats.csv`, `class_by_source.csv`, `train_summary.yaml`
- Mirrors the `mermaid-classifier` repo's per-run statistics pattern, scoped to MermaidSeg shapes (no growth-form, segmentation classes)
- Spec: docs/superpowers/specs/2026-05-01-mlflow-dataset-statistics-design.md
- Closes #72

## Test plan
- [ ] CI passes (unit + integration tests under `TestResolveAnnotations`, `TestComputeClassCounts`, `TestComputeSourceStats`, `TestComputeClassBySource`, `TestComputeTrainSummary`, `TestLogDatasetStatistics`)
- [ ] Manually run `Base_Pipeline.ipynb` against a small dataset; confirm 4 artifacts present in MLflow UI under `dataset_stats/`
- [ ] Manually verify `class_by_source.csv` shows expected geographic split for a multi-region run
EOF
)"
```

- [ ] **Step 4: Open follow-up issues for deferred work**

```bash
gh issue create --title "Add pixel-level mask statistics to MLflow" --body "Follow-up to #72. Capture per-class pixel distribution per split by materializing masks once at training start. Requires benchmarking the cost — likely ~minutes per run for a full mask pass."

gh issue create --title "Add concept_counts.csv for concept-bottleneck training runs" --body "Follow-up to #72. CBM has its own concept hierarchy (from MERMAID API) and pipeline. Mirror the class-level QA artifacts at the concept level so CBM runs have parallel traceability."

gh issue create --title "Add temporal columns (year/season) to dataset stats" --body "Follow-up to #72. Requires expanding the parquet schemas to carry survey date — separate Data ticket. Once available, add temporal axes to source_stats.csv and class_by_source.csv for year-over-year drift QA."
```

---

## Done criteria

- [ ] All 12 tasks complete
- [ ] PR opened and linked to #72
- [ ] Three follow-up issues created for deferred work
- [ ] Spec and plan committed under `docs/superpowers/`
