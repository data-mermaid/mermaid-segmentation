"""Joint source-label registry across multiple source datasets.

The :class:`SourceLabelRegistry` assigns each registered source dataset a
disjoint range of integer IDs in a global source-label space (with ``0``
reserved for background) and pre-computes the GPU lookup tensors used at
training time:

- ``source_to_target``: ``LongTensor`` of shape ``(N+1,)`` mapping global
  source IDs to MERMAID benthic-attribute target IDs.
- ``source_to_concepts``: ``FloatTensor`` of shape ``(N+1, C)`` mapping global
  source IDs to the per-concept multi-hot row (computed against the MERMAID
  benthic-attribute hierarchy).

After construction, each registered dataset has had
:meth:`BaseCoralDataset.set_global_offset` called on it so that its
``__getitem__`` already emits masks in the global source-label space.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from mermaidseg.dataset_reconciliation.concepts import (
    initialize_benthic_concepts,
    initialize_benthic_hierarchy,
)
from mermaidseg.dataset_reconciliation.label_mapping import (
    coralscapes_to_mermaid,
    fetch_coralnet_to_mermaid,
)


def _identity_source_to_target(dataset: Any) -> dict[str, str]:
    """Return an identity name->name mapping for the given dataset's source space."""
    return {name: name for name in dataset.source_name2id}


def _coralscapes_source_to_target(dataset: Any) -> dict[str, str]:
    static = coralscapes_to_mermaid()
    return {name: targets[0] for name, targets in static.items() if name in dataset.source_name2id}


_BUILTIN_DEFAULT_FETCHERS = {
    "mermaid": _identity_source_to_target,
    "coralscapes": _coralscapes_source_to_target,
}


class SourceLabelRegistry:
    """Builds a joint global source-label index across a list of source datasets.

    Args:
        datasets: List of source datasets (each must expose ``SOURCE_NAME``,
            ``source_id2name``, ``source_name2id``, and accept
            :meth:`set_global_offset`).
        target_labels: Optional ordered list of canonical MERMAID
            benthic-attribute target label names (1..M). If ``None``, defaults
            are derived from the union of mapped target names across all
            datasets (sorted alphabetically).
        source_to_target_name_maps: Optional mapping
            ``{SOURCE_NAME: {source_label_name: target_label_name}}``. Defaults
            are looked up per ``SOURCE_NAME``: ``mermaid`` is identity,
            ``coralscapes`` uses the static dict, ``coralnet`` is fetched from
            the MERMAID API.
        concept_hierarchy: Optional benthic-attribute name->parent-name dict.
            If supplied (or if ``compute_concepts=True`` and not supplied), it
            is fetched from the MERMAID API.
        compute_concepts: When ``True``, additionally builds
            ``source_to_concepts``.
        target_label_subset: Optional list/set of MERMAID target label names
            to retain. Source labels mapped to a target outside the subset
            collapse to background (``0``) in ``source_to_target``.
        fetch_remote: When ``True`` (default), missing default mappings are
            fetched from the MERMAID API. Set ``False`` for offline usage —
            in that case ``source_to_target_name_maps`` must be provided for
            every registered ``SOURCE_NAME``.
    """

    datasets: list[Any]
    target_label2id: dict[str, int]
    target_id2label: dict[int, str]
    num_target_classes: int
    source_to_target: torch.LongTensor
    source_to_concepts: torch.Tensor | None
    global_id2source: dict[int, tuple[str, str]]
    dataset_offsets: dict[str, int]

    def __init__(
        self,
        datasets: list[Any],
        target_labels: list[str] | None = None,
        source_to_target_name_maps: dict[str, dict[str, str]] | None = None,
        concept_hierarchy: dict[str, str] | None = None,
        compute_concepts: bool = False,
        target_label_subset: list[str] | set[str] | None = None,
        fetch_remote: bool = True,
    ):
        if not datasets:
            raise ValueError("SourceLabelRegistry requires at least one dataset")

        seen_sources: set[str] = set()
        for ds in datasets:
            name = getattr(ds, "SOURCE_NAME", None)
            if not name:
                raise ValueError(f"Dataset {type(ds).__name__} does not declare SOURCE_NAME")
            if name in seen_sources:
                raise ValueError(f"Duplicate SOURCE_NAME '{name}' across registered datasets")
            seen_sources.add(name)

        self.datasets = list(datasets)
        provided_maps = dict(source_to_target_name_maps or {})
        resolved_maps = self._resolve_source_to_target_maps(
            provided_maps, fetch_remote=fetch_remote
        )

        self.global_id2source = {}
        per_dataset_target_lists: list[list[str | None]] = []
        offsets: dict[str, int] = {}
        running_offset = 0
        for ds in self.datasets:
            offsets[ds.SOURCE_NAME] = running_offset
            ds_target_for_source: list[str | None] = []
            for local_id, source_name in sorted(ds.source_id2name.items()):
                global_id = local_id + running_offset
                self.global_id2source[global_id] = (ds.SOURCE_NAME, source_name)
                ds_target_for_source.append(resolved_maps[ds.SOURCE_NAME].get(source_name))
            per_dataset_target_lists.append(ds_target_for_source)
            running_offset += len(ds.source_id2name)
        self.dataset_offsets = offsets
        num_global_source = running_offset

        if target_labels is None:
            target_label_set: set[str] = set()
            for tgt_list in per_dataset_target_lists:
                for tgt in tgt_list:
                    if tgt:
                        target_label_set.add(tgt)
            target_labels = sorted(target_label_set)

        if target_label_subset is not None:
            subset = set(target_label_subset)
            target_labels = [t for t in target_labels if t in subset]
        else:
            subset = None

        self.target_id2label = dict(enumerate(target_labels, start=1))
        self.target_label2id = {v: k for k, v in self.target_id2label.items()}
        self.num_target_classes = len(self.target_id2label) + 1  # +1 for background

        source_to_target_np = np.zeros(num_global_source + 1, dtype=np.int64)
        for ds, tgt_list in zip(self.datasets, per_dataset_target_lists, strict=False):
            offset = self.dataset_offsets[ds.SOURCE_NAME]
            for local_id, source_name in sorted(ds.source_id2name.items()):
                target_name = resolved_maps[ds.SOURCE_NAME].get(source_name)
                if target_name is None:
                    continue
                if subset is not None and target_name not in subset:
                    continue
                target_id = self.target_label2id.get(target_name, 0)
                source_to_target_np[local_id + offset] = target_id
        self.source_to_target = torch.from_numpy(source_to_target_np).long()

        self.source_to_concepts = None
        self._concept_matrix = None
        self.concept_id2name: dict[int, str] | None = None
        self.concept_name2id: dict[str, int] | None = None
        if compute_concepts:
            if concept_hierarchy is None:
                if not fetch_remote:
                    raise ValueError(
                        "compute_concepts=True requires concept_hierarchy or fetch_remote=True"
                    )
                concept_hierarchy = initialize_benthic_hierarchy()
            self._build_concepts(
                source_to_target_np=source_to_target_np,
                concept_hierarchy=concept_hierarchy,
            )

        for ds in self.datasets:
            ds.set_global_offset(self.dataset_offsets[ds.SOURCE_NAME])

    def _resolve_source_to_target_maps(
        self,
        provided: dict[str, dict[str, str]],
        fetch_remote: bool,
    ) -> dict[str, dict[str, str]]:
        """Resolve a ``SOURCE_NAME -> {source_name: target_name}`` map for every dataset."""
        resolved: dict[str, dict[str, str]] = {}
        for ds in self.datasets:
            name = ds.SOURCE_NAME
            if name in provided:
                resolved[name] = provided[name]
                continue
            if name == "coralnet":
                if not fetch_remote:
                    raise ValueError(
                        "coralnet source-to-target mapping requires fetch_remote=True or "
                        "an explicit entry in source_to_target_name_maps"
                    )
                coralnet_id_to_target = fetch_coralnet_to_mermaid()
                resolved[name] = {
                    src: tgt
                    for src, tgt in coralnet_id_to_target.items()
                    if tgt is not None and src in ds.source_name2id
                }
                continue
            if name in _BUILTIN_DEFAULT_FETCHERS:
                resolved[name] = _BUILTIN_DEFAULT_FETCHERS[name](ds)
                continue
            raise ValueError(
                f"No default source-to-target mapping for SOURCE_NAME='{name}'. "
                "Pass an explicit entry in source_to_target_name_maps."
            )
        return resolved

    def _build_concepts(
        self,
        source_to_target_np: np.ndarray,
        concept_hierarchy: dict[str, str],
    ) -> None:
        """Build the ``source_to_concepts`` lookup tensor and concept metadata."""
        target_label_names = [self.target_id2label[i] for i in sorted(self.target_id2label)]
        concept_set, concept_matrix = initialize_benthic_concepts(
            target_label_names, concept_hierarchy
        )
        self._concept_matrix = concept_matrix
        self.concept_id2name = dict(enumerate(concept_set, start=1))
        self.concept_name2id = {v: k for k, v in self.concept_id2name.items()}

        num_concepts = len(concept_set)
        num_global = source_to_target_np.shape[0]
        source_to_concepts_np = np.zeros((num_global, num_concepts), dtype=np.float32)
        target_to_concepts = concept_matrix.to_numpy().astype(np.float32)
        for global_id in range(1, num_global):
            target_id = int(source_to_target_np[global_id])
            if target_id == 0:
                continue
            source_to_concepts_np[global_id] = target_to_concepts[target_id - 1]
        self.source_to_concepts = torch.from_numpy(source_to_concepts_np).float()

    @property
    def concept_matrix(self):
        """The pandas concept matrix used by ``postprocess_predicted_concepts``."""
        return self._concept_matrix

    @property
    def num_concepts(self) -> int:
        if self.source_to_concepts is None:
            return 0
        return int(self.source_to_concepts.shape[1])

    @property
    def num_global_source_classes(self) -> int:
        """Number of global source classes (excluding background)."""
        return int(self.source_to_target.shape[0]) - 1

    def to(self, device: torch.device | str) -> SourceLabelRegistry:
        """Move the lookup tensors onto ``device`` (in-place) and return self."""
        self.source_to_target = self.source_to_target.to(device)
        if self.source_to_concepts is not None:
            self.source_to_concepts = self.source_to_concepts.to(device)
        return self

    def conceptid2labelid(self) -> dict[int, int] | None:
        """Backward-compat helper used by ``postprocess_predicted_concepts``.

        Returns a dict mapping per-concept-column index (``benthic_concept_matrix``
        column ordering, 0-indexed) to the corresponding target label ID
        (1-indexed, 0 if not a leaf class).
        """
        if self._concept_matrix is None:
            return None
        column_concepts = self._concept_matrix.columns.get_level_values("concept").tolist()
        result: dict[int, int] = {}
        for col_ind, concept_name in enumerate(column_concepts):
            target_id = self.target_label2id.get(concept_name)
            if target_id is not None:
                result[col_ind] = target_id
        return result
