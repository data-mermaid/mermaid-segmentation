"""Deployment-side inference for the concept-bottleneck segmentation model.

Shared by the demos, the evaluation suite and serving lanes. Importing it needs only torch,
transformers, peft, safetensors, OpenCV and NumPy -- never the research stack (albumentations,
pandas, mlflow, wandb, boto3, s3fs, datasets) or the trainer.
"""

from mermaidseg.inference.bundle import (
    ManifestError,
    ModelBundle,
    export_bundle,
    load_bundle,
    write_sidecar,
)
from mermaidseg.inference.checkpoint import (
    SUPPORTED_CBM_MODELS,
    derive_concept_value2id,
    load_checkpoint_model,
)
from mermaidseg.inference.export import ParityError, export_graph
from mermaidseg.inference.predictor import CBMPredictor
from mermaidseg.inference.preprocess import IMAGENET_MEAN, IMAGENET_STD, preprocess

__all__ = [
    "CBMPredictor",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "ManifestError",
    "ModelBundle",
    "ParityError",
    "SUPPORTED_CBM_MODELS",
    "derive_concept_value2id",
    "export_bundle",
    "export_graph",
    "load_bundle",
    "load_checkpoint_model",
    "preprocess",
    "write_sidecar",
]
