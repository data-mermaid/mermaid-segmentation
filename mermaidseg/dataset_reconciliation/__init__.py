"""Dataset reconciliation: joint source-label registry, label and concept mapping."""

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

from mermaidseg.dataset_reconciliation import dataset_stats  # noqa: F401
