"""Dataset reconciliation: joint source-label registry, label and concept mapping."""

from mermaidseg.dataset_reconciliation.combined import CombinedCoralDataset
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

__all__ = [
    "CombinedCoralDataset",
    "SourceLabelRegistry",
    "coralscapes_to_mermaid",
    "fetch_coralnet_to_mermaid",
    "fetch_mermaid_target_labels",
    "initialize_benthic_concepts",
    "initialize_benthic_hierarchy",
    "postprocess_predicted_concepts",
    "source_labels_to_concepts",
    "source_labels_to_target_labels",
]
