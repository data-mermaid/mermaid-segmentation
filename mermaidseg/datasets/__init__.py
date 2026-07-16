"""Per-source PyTorch datasets.

Each subpackage exposes a dataset that emits labels in its own source label space.
Cross-dataset label/concept mapping lives in :mod:`mermaidseg.dataset_reconciliation`.
"""

from mermaidseg.datasets.base_dataset import BaseCoralDataset, worker_init_fn
from mermaidseg.datasets.benthos_yuval import BenthosYuvalCoralsDataset
from mermaidseg.datasets.catlin_seaview import CatlinSeaviewDataset
from mermaidseg.datasets.coralnet import CoralNetDataset
from mermaidseg.datasets.coralscapes import CoralscapesDataset
from mermaidseg.datasets.coralscapes_v2 import CoralscapesV2Dataset
from mermaidseg.datasets.mermaid import MermaidDataset
from mermaidseg.datasets.moorea_labeled_corals import MooreaLabeledCoralsDataset
from mermaidseg.datasets.pacific_labeled_corals import PacificLabeledCoralsDataset
from mermaidseg.datasets.ucsd_mosaics import UCSDMosaicsDataset

# Single source of truth mapping a config ``SOURCE_NAME`` to its dataset class, used by the
# training entrypoint to instantiate datasets from ``cfg.data``. Add a training-wired dataset
# here. UCSDMosaicsDataset is exported (above) but intentionally omitted — it is not currently
# wired into training.
# NOTE: not annotated as ``type[BaseCoralDataset]`` because Coralscapes(_v2) currently
# subclasses ``torch.utils.data.Dataset`` directly rather than ``BaseCoralDataset``.
# Order matters: it flows through dataset_dict → ConcatDataset member order → the
# index-to-sample mapping the seeded DataLoader shuffles over, so it must match the prior
# hardcoded DATASET_CLASSES order to keep the same seed reproducing the same run.
DATASET_REGISTRY: dict[str, type] = {
    "pacific_labeled_corals": PacificLabeledCoralsDataset,
    "moorea_labeled_corals": MooreaLabeledCoralsDataset,
    "catlin_seaview": CatlinSeaviewDataset,
    "mermaid": MermaidDataset,
    "coralnet": CoralNetDataset,
    "coralscapes": CoralscapesDataset,
    "coralscapes_v2": CoralscapesV2Dataset,
    "benthos_yuval": BenthosYuvalCoralsDataset,
}

__all__ = [
    "DATASET_REGISTRY",
    "BaseCoralDataset",
    "BenthosYuvalCoralsDataset",
    "CatlinSeaviewDataset",
    "CoralNetDataset",
    "CoralscapesDataset",
    "CoralscapesV2Dataset",
    "MermaidDataset",
    "MooreaLabeledCoralsDataset",
    "PacificLabeledCoralsDataset",
    "UCSDMosaicsDataset",
    "worker_init_fn",
]
