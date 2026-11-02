"""Dataset reconciliation: joint source-label registry, label and concept mapping.

Public names resolve lazily (PEP 562) so importing a light submodule such as ``concepts`` does not
import the dataset classes or their heavy dependencies.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mermaidseg.dataset_reconciliation.combined import CombinedCoralDataset
    from mermaidseg.dataset_reconciliation.concept_schema import (
        ConceptSchema,
        build_source_to_concepts,
    )
    from mermaidseg.dataset_reconciliation.concepts import (
        initialize_benthic_concepts,
        initialize_benthic_hierarchy,
        postprocess_predicted_concepts,
        source_labels_to_concepts,
    )
    from mermaidseg.dataset_reconciliation.label_mapping import (
        coralscapes_to_mermaid,
        coralscapes_v2_to_mermaid,
        fetch_coralnet_to_mermaid,
        fetch_mermaid_target_labels,
        source_labels_to_target_labels,
    )
    from mermaidseg.dataset_reconciliation.registry import SourceLabelRegistry
    from mermaidseg.dataset_reconciliation.split_wiring import (
        SourceVocabulary,
        apply_vocabularies,
        attach_registry,
        build_source_vocabularies,
        group_splits,
        prepare_splits_for_registry,
        select_registry_representatives,
    )

_LAZY_ATTRS = {
    "CombinedCoralDataset": "combined",
    "ConceptSchema": "concept_schema",
    "build_source_to_concepts": "concept_schema",
    "initialize_benthic_concepts": "concepts",
    "initialize_benthic_hierarchy": "concepts",
    "postprocess_predicted_concepts": "concepts",
    "source_labels_to_concepts": "concepts",
    "coralscapes_to_mermaid": "label_mapping",
    "coralscapes_v2_to_mermaid": "label_mapping",
    "fetch_coralnet_to_mermaid": "label_mapping",
    "fetch_mermaid_target_labels": "label_mapping",
    "source_labels_to_target_labels": "label_mapping",
    "SourceLabelRegistry": "registry",
    "SourceVocabulary": "split_wiring",
    "apply_vocabularies": "split_wiring",
    "attach_registry": "split_wiring",
    "build_source_vocabularies": "split_wiring",
    "group_splits": "split_wiring",
    "prepare_splits_for_registry": "split_wiring",
    "select_registry_representatives": "split_wiring",
}

__all__ = [
    "CombinedCoralDataset",
    "ConceptSchema",
    "SourceLabelRegistry",
    "SourceVocabulary",
    "apply_vocabularies",
    "attach_registry",
    "build_source_to_concepts",
    "build_source_vocabularies",
    "coralscapes_to_mermaid",
    "coralscapes_v2_to_mermaid",
    "fetch_coralnet_to_mermaid",
    "fetch_mermaid_target_labels",
    "group_splits",
    "initialize_benthic_concepts",
    "initialize_benthic_hierarchy",
    "postprocess_predicted_concepts",
    "prepare_splits_for_registry",
    "select_registry_representatives",
    "source_labels_to_concepts",
    "source_labels_to_target_labels",
]


def __getattr__(name: str) -> Any:
    submodule = _LAZY_ATTRS.get(name)
    if submodule is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(f"{__name__}.{submodule}"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
