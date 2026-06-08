"""Per-source PyTorch datasets.

Each subpackage exposes a dataset that emits labels in its own source label
space. Cross-dataset label/concept mapping lives in
:mod:`mermaidseg.dataset_reconciliation`.
"""

from mermaidseg.datasets.base_dataset import BaseCoralDataset, worker_init_fn
from mermaidseg.datasets.benthos_yuval import BenthosYuvalCoralsDataset
from mermaidseg.datasets.catlin_seaview import CatlinSeaviewDataset
from mermaidseg.datasets.coralnet import CoralNetDataset
from mermaidseg.datasets.coralscapes import CoralscapesDataset
from mermaidseg.datasets.mermaid import MermaidDataset
from mermaidseg.datasets.moorea_labeled_corals import MooreaLabeledCoralsDataset
from mermaidseg.datasets.pacific_labeled_corals import PacificLabeledCoralsDataset

__all__ = [
    "BaseCoralDataset",
    "BenthosYuvalCoralsDataset",
    "CatlinSeaviewDataset",
    "CoralNetDataset",
    "CoralscapesDataset",
    "MermaidDataset",
    "MooreaLabeledCoralsDataset",
    "PacificLabeledCoralsDataset",
    "worker_init_fn",
]
