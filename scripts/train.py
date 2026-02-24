"""
Training script for coral segmentation models using PyTorch.

This script sets up and trains a semantic segmentation model for coral reef analysis
using datasets from Mermaid and CoralNet. It handles data loading, model initialization,
training configuration, and logging.

The script performs the following key operations:
- Sets up CUDA devices and random seeds for reproducible training
- Loads configuration from YAML files with command-line argument overrides
- Creates data augmentation transforms using Albumentations
- Initializes Mermaid15Dataset and CoralNet15Dataset with specified transforms
- Splits the CoralNet dataset into train/validation/test sets (70%/10%/20%)
- Creates DataLoaders for each dataset split with appropriate batching
- Initializes a MetaModel with segmentation architecture and training parameters
- Sets up evaluation metrics for semantic segmentation
- Configures logging and checkpointing
- Trains the model using the specified configuration

Key components:
- Device detection and GPU utilization
- Configurable data augmentation pipeline
- Train/validation/test split with fixed random seed
- Multi-worker data loading with custom collate functions
- Integrated logging and model checkpointing
- Semantic segmentation evaluation metrics

The training process is fully configurable through YAML configuration files
and command-line arguments, supporting various model architectures and
training hyperparameters.
"""

import copy

import albumentations as A
import torch
from torch.utils.data import DataLoader, random_split

from mermaidseg.datasets.dataset import CoralNet15Dataset, Mermaid15Dataset
from mermaidseg.io import get_parser, setup_config, update_config_with_args
from mermaidseg.logger import Logger
from mermaidseg.model.eval import EvaluatorSemanticSegmentation
from mermaidseg.model.meta import MetaModel
from mermaidseg.model.train import train_model

device_count = torch.cuda.device_count()
for i in range(device_count):
    print(f"CUDA Device {i}: {torch.cuda.get_device_name(i)}")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

cfg = setup_config(
    config_path="../configs/segformer-coralscapes.yaml",
    config_base_path="../configs/base.yaml",
)


parser = get_parser()
args = parser.parse_args()
# args_input = "--run-name=coralscapes_backbone_run --batch-size=8 --epochs=10 --log-epochs=1"
# args_input = args_input.split(" ")
# args = parser.parse_args(args_input)

cfg = update_config_with_args(cfg, args)
cfg_logger = copy.deepcopy(cfg)

transforms = {}
for split in cfg.augmentation:
    transforms[split] = A.Compose(
        [
            getattr(A, transform_name)(**transform_params)
            for transform_name, transform_params in cfg.augmentation[split].items()
        ]
    )

dataset_mermaid = Mermaid15Dataset(transform=transforms["train"], padding=3)
dataset_coralnet = CoralNet15Dataset(transform=transforms["train"], padding=7)

total_size = len(dataset_coralnet)
train_size = int(0.7 * total_size)
val_size = int(0.1 * total_size)
test_size = total_size - train_size - val_size

generator = torch.Generator().manual_seed(42)
train_dataset, val_dataset, test_dataset = random_split(
    dataset_coralnet, [train_size, val_size, test_size], generator=generator
)

train_loader = DataLoader(
    train_dataset,
    batch_size=cfg.data.batch_size,
    shuffle=True,
    num_workers=2,
    drop_last=True,
    collate_fn=dataset_coralnet.collate_fn,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=cfg.data.batch_size,
    shuffle=False,
    num_workers=2,
    drop_last=True,
    collate_fn=dataset_coralnet.collate_fn,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=cfg.data.batch_size,
    shuffle=False,
    num_workers=2,
    drop_last=True,
    collate_fn=dataset_coralnet.collate_fn,
)

print(f"Number of training batches: {len(train_loader)}")
print(f"Number of validation batches: {len(val_loader)}")
print(f"Number of test batches: {len(test_loader)}")

meta_model = MetaModel(
    run_name=cfg.run_name,
    num_classes=dataset_coralnet.num_classes,
    device=device,
    model_kwargs=cfg.model,
    training_kwargs=cfg.training,
)

evaluator = EvaluatorSemanticSegmentation(
    num_classes=dataset_coralnet.num_classes,
    device=device,
)


logger = Logger(
    config=cfg_logger,
    meta_model=meta_model,
    log_epochs=cfg.logger.log_epochs,
    log_checkpoint=cfg.logger.log_checkpoint,
    checkpoint_dir=".",
)

train_model(meta_model, evaluator, train_loader, val_loader, logger=logger)
