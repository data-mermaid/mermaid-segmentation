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

import logging
from collections.abc import Callable, Mapping
from typing import Any, cast

import numpy as np
import torch

from mermaidseg.dataset_reconciliation.concept_schema import (
    ConceptSchema,
    build_source_to_concepts,
)
from mermaidseg.dataset_reconciliation.concepts import initialize_benthic_hierarchy
from mermaidseg.dataset_reconciliation.label_mapping import (
    coralscapes_v2_to_mermaid,
    fetch_benthos_yuval_to_mermaid,
    fetch_catlin_seaview_to_mermaid,
    fetch_coralnet_to_mermaid,
    fetch_moorea_labeled_corals_to_mermaid,
    fetch_pacific_labeled_corals_to_mermaid,
    fetch_ucsd_mosaics_to_mermaid,
)

logger = logging.getLogger(__name__)


def _identity_source_to_target(dataset: Any) -> dict[str, str]:
    """Return an identity name->name mapping for the given dataset's source space."""
    return {name: name for name in dataset.source_name2id}


def _static_source_to_target(static: dict[str, str], dataset: Any) -> dict[str, str]:
    """Filter a static ``source_name -> target_name`` dict to the dataset's source space."""
    ds_names_lower = {n.lower() for n in dataset.source_name2id}
    return {name: target for name, target in static.items() if name.lower() in ds_names_lower}


_BUILTIN_DEFAULT_FETCHERS = {
    "mermaid": _identity_source_to_target,
    "coralscapes_v2": lambda ds: _static_source_to_target(coralscapes_v2_to_mermaid(), ds),
}


_REMOTE_FETCHERS: dict[str, Callable[..., Mapping[str, str | None]]] = {
    "coralnet": fetch_coralnet_to_mermaid,
    "catlin_seaview": fetch_catlin_seaview_to_mermaid,
    "moorea_labeled_corals": fetch_moorea_labeled_corals_to_mermaid,
    "pacific_labeled_corals": fetch_pacific_labeled_corals_to_mermaid,
    "benthos_yuval": fetch_benthos_yuval_to_mermaid,
    "ucsd_mosaics": fetch_ucsd_mosaics_to_mermaid,
}


def roll_up_label(
    label: str | None, benthic_hierarchy: dict[str, str | None], subset: set[str]
) -> str | None:
    """Return ``label`` if it is in ``subset``, else its nearest ancestor in ``subset``.

    Args:
        label: Target label name, or ``None`` for an unmapped label.
        benthic_hierarchy: Lower-cased name -> parent-name mapping, ``None`` at the roots.
        subset: Names that may be returned.

    Returns:
        The label or ancestor in ``subset``, or ``None`` if there is none.
    """
    if label in subset:
        return label
    if label in benthic_hierarchy:
        parent = benthic_hierarchy[label]
        while parent is not None:
            if parent in subset:
                return parent
            parent = benthic_hierarchy[parent]
    return None


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
        concept_hierarchy: Optional benthic-attribute name->parent-name dict.
            If supplied (or if ``compute_concepts=True`` and not supplied), it
            is fetched from the MERMAID API.
        compute_concepts: When ``True``, additionally builds
            ``source_to_concepts`` from ``concept_schema``.
        target_label_subset: Optional list/set of MERMAID target label names
            to retain. Source labels mapped to a target outside the subset
            collapse to background (``0``) in ``source_to_target``.
    """

    datasets: list[Any]
    target_label2id: dict[str, int]
    target_id2label: dict[int, str]
    num_target_classes: int
    source_to_target: torch.Tensor
    source_to_concepts: torch.Tensor | None
    global_id2source: dict[int, tuple[str, str]]
    dataset_offsets: dict[str, int]
    concept_value2id: dict[str, dict[str, list[int]]] | None

    def __init__(
        self,
        datasets: list[Any],
        target_labels: list[str] | None = None,
        compute_concepts: bool = False,
        label_roll_up: bool = False,
        target_label_subset: list[str] | set[str] | None = None,
        concept_schema: ConceptSchema | None = None,
    ):
        """Assign global offsets and build the source-to-target (and concept) lookups.

        Args:
            datasets: Source datasets to register.
            target_labels: Ordered target label names, derived from the datasets if ``None``.
            compute_concepts: Whether to build ``source_to_concepts``.
            label_roll_up: Whether unmapped targets roll up to the nearest ancestor in
                ``target_label_subset``.
            target_label_subset: Target label names to retain.
            concept_schema: Concept schema, required with ``compute_concepts``.

        Raises:
            ValueError: If there are no datasets, a dataset lacks or repeats ``SOURCE_NAME``,
                ``label_roll_up`` has no ``target_label_subset``, or ``compute_concepts`` has
                no ``concept_schema``.
        """
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
        resolved_maps = self._resolve_source_to_target_maps()

        # Assign each dataset a disjoint offset in the global source-ID space and
        # record the (source, source_name) pair for every global ID. We also
        # accumulate `per_dataset_target_lists` so we can derive a default
        # `target_labels` below when the caller did not supply one.
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
                ds_target_for_source.append(resolved_maps[ds.SOURCE_NAME].get(source_name.lower()))
            per_dataset_target_lists.append(ds_target_for_source)
            running_offset += len(ds.source_id2name)
        self.dataset_offsets = offsets
        num_global_source = running_offset

        # Default target vocabulary: the alphabetical union of every non-null
        # MERMAID target name produced by the resolved per-dataset maps.
        if target_labels is None:
            target_label_set: set[str] = set()
            for tgt_list in per_dataset_target_lists:
                for tgt in tgt_list:
                    if tgt:
                        target_label_set.add(tgt)
            target_labels = sorted(target_label_set)

        target_labels = list(dict.fromkeys(t.lower() for t in target_labels))

        # Optional caller-provided whitelist; targets outside `subset` collapse
        # to background in `source_to_target` below.
        if target_label_subset is not None:
            subset = {t.lower() for t in target_label_subset}
            target_labels = [t for t in target_labels if t in subset]
        else:
            subset = None

        self.target_id2label = dict(enumerate(target_labels, start=1))
        self.target_label2id = {v: k for k, v in self.target_id2label.items()}
        self.num_target_classes = len(self.target_id2label) + 1  # +1 for background

        if label_roll_up:
            benthic_hierarchy = {
                k.lower(): (v.lower() if v is not None else None)
                for k, v in initialize_benthic_hierarchy().items()
            }
            if subset is None:
                raise ValueError("label_roll_up=True requires target_label_subset to be set")

        # Dense lookup `global_source_id -> target_class_id`; index 0 stays
        # background, and any source name without a target (or filtered out by
        # `subset`) either has an attempted roll up to a parent class or
        # is left as 0 so it collapses to background at train time.
        source_to_target_np = np.zeros(num_global_source + 1, dtype=np.int64)
        warned_unmapped: set[tuple[str, str]] = set()
        for ds in self.datasets:
            offset = self.dataset_offsets[ds.SOURCE_NAME]
            for local_id, source_name in sorted(ds.source_id2name.items()):
                target_name = resolved_maps[ds.SOURCE_NAME].get(source_name.lower())
                if label_roll_up:
                    target_name = roll_up_label(
                        target_name, benthic_hierarchy, cast("set[str]", subset)
                    )
                if target_name is None:
                    key = (ds.SOURCE_NAME, source_name.lower())
                    if key not in warned_unmapped:
                        warned_unmapped.add(key)
                        print(
                            f"DEBUG WARNING: unmapped source label (target mapping): "
                            f"source_dataset={ds.SOURCE_NAME!r} label={source_name.lower()!r}",
                            flush=True,
                        )
                    continue
                if subset is not None and target_name not in subset:
                    continue
                target_id = self.target_label2id.get(target_name, 0)
                source_to_target_np[local_id + offset] = target_id
        self._log_target_coverage(source_to_target_np)
        self.source_to_target = torch.from_numpy(source_to_target_np).long()
        self.source_to_concepts = None
        self.concept_id2name: dict[int, str] | None = None
        self.concept_name2id: dict[str, int] | None = None
        self.concept_value2id: dict[str, dict[str, list[int]]] | None = None
        if compute_concepts:
            if concept_schema is None:
                raise ValueError("compute_concepts=True requires concept_schema")
            self._build_concepts(concept_schema)

        for ds in self.datasets:
            ds.set_global_offset(self.dataset_offsets[ds.SOURCE_NAME])

    def _log_target_coverage(self, source_to_target_np: np.ndarray) -> None:
        """Log, per source, how many source labels (and annotations) reach a target class.

        Warns when none do: a label-key mismatch otherwise sends every pixel of that source to
        background with no error.
        """
        for ds in self.datasets:
            offset = self.dataset_offsets[ds.SOURCE_NAME]
            local_ids = sorted(ds.source_id2name)
            if not local_ids:
                continue
            mapped = [source_to_target_np[i + offset] != 0 for i in local_ids]
            n_mapped = sum(mapped)
            msg = f"{ds.SOURCE_NAME}: {n_mapped} of {len(local_ids)} source labels"
            # Dense-mask datasets (Coralscapes, CoralscapesV2, UCSDMosaics) carry no point
            # annotations, so only label counts are logged for them.
            annotations = getattr(ds, "df_annotations", None)
            if annotations is not None and len(annotations):
                counts = annotations["source_label_name"].value_counts()
                mapped_names = {
                    ds.source_id2name[i] for i, ok in zip(local_ids, mapped, strict=True) if ok
                }
                share = counts[counts.index.isin(mapped_names)].sum() / len(annotations)
                msg += f", {share:.1%} of annotations,"
            msg += " map to a target class"
            if n_mapped:
                logger.info(msg)
            else:
                logger.warning(msg + "; check source label keys against the mapping.")

    def _resolve_source_to_target_maps(self) -> dict[str, dict[str, str]]:
        """Resolve a ``SOURCE_NAME -> {source_name: target_name}`` map per dataset."""
        resolved: dict[str, dict[str, str]] = {}
        for ds in self.datasets:
            name = ds.SOURCE_NAME
            ds_names_lower = {n.lower() for n in ds.source_name2id}
            fetcher = _REMOTE_FETCHERS.get(name)
            if fetcher is not None:
                resolved[name] = {
                    src: tgt
                    for src, tgt in fetcher().items()
                    if tgt is not None and src.lower() in ds_names_lower
                }
                continue
            if name in _BUILTIN_DEFAULT_FETCHERS:
                resolved[name] = _BUILTIN_DEFAULT_FETCHERS[name](ds)
                continue
            raise ValueError(f"No default source-to-target mapping for SOURCE_NAME='{name}'.")
        return {
            src_name: {k.lower(): v.lower() for k, v in m.items()}
            for src_name, m in resolved.items()
        }

    def _build_concepts(self, concept_schema: ConceptSchema) -> None:
        self.concept_value2id = concept_schema.concept_value2id
        self.concept_id2name = concept_schema.channel_id2name()
        self.concept_name2id = {v: k for k, v in self.concept_id2name.items()}
        self.source_to_concepts = build_source_to_concepts(self.global_id2source, concept_schema)

    @property
    def num_concepts(self) -> int:
        """Number of concept channels, 0 without concepts."""
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
