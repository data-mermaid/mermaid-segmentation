"""Per-source PyTorch datasets.

Each subpackage exposes a dataset that emits labels in its own source label
space. Cross-dataset label/concept mapping lives in
:mod:`mermaidseg.dataset_reconciliation`.
"""

from mermaidseg.datasets.base_dataset import BaseCoralDataset, worker_init_fn
<<<<<<< HEAD
from mermaidseg.datasets.catlin_seaview import CatlinSeaviewDataset
from mermaidseg.datasets.coralnet import CoralNetDataset
from mermaidseg.datasets.coralscapes import CoralscapesDataset
from mermaidseg.datasets.mermaid import MermaidDataset
from mermaidseg.datasets.moorea_labeled_corals import MooreaLabeledCoralsDataset

__all__ = [
    "BaseCoralDataset",
    "CatlinSeaviewDataset",
    "CoralNetDataset",
    "CoralscapesDataset",
    "MermaidDataset",
    "MooreaLabeledCoralsDataset",
=======
from mermaidseg.datasets.coralnet import CoralNetDataset
from mermaidseg.datasets.coralscapes import CoralscapesDataset
from mermaidseg.datasets.mermaid import MermaidDataset

__all__ = [
    "BaseCoralDataset",
    "CoralNetDataset",
    "CoralscapesDataset",
    "MermaidDataset",
>>>>>>> origin/main
    "worker_init_fn",
]
