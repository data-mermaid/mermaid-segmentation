"""Dataset reconciliation: joint source-label registry, label and concept mapping."""

from mermaidseg.dataset_reconciliation.concept_schema import (
    ConceptSchema,
    build_source_to_concepts,
)
from mermaidseg.dataset_reconciliation.concepts import (
    initialize_benthic_hierarchy,
    source_labels_to_concepts,
)
from mermaidseg.dataset_reconciliation.label_mapping import (
    coralscapes_to_mermaid,
    coralscapes_v2_to_mermaid,
    fetch_coralnet_to_mermaid,
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

__all__ = [
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
    "group_splits",
    "initialize_benthic_hierarchy",
    "prepare_splits_for_registry",
    "select_registry_representatives",
    "source_labels_to_concepts",
    "source_labels_to_target_labels",
]
