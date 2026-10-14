"""Native-resolution evaluation runner for sparse point datasets, with optional --hier.

Run from nbs/:

    uv run --no-sync python eval_sparse.py --dataset pacific_labeled_corals --n 300 --hier \
        --checkpoint model_checkpoints/<checkpoint>

Images are resized to the val ``Resize`` saved in the checkpoint (override with --model-size).
The checkpoint must match the current concept layout (demo/concept_id2name.json), or pass
``--concept-mapping`` for an older one. Needs AWS credentials; set MERMAIDSEG_LOCAL_CACHE_DIR to
also read images through the local cache.
Points are scored at their native location; see ``mermaidseg/model/sparse_eval.py``.
"""

import argparse
import json
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import torch
from torchmetrics.classification import Accuracy, JaccardIndex

from mermaidseg.dataset_reconciliation import (
    ConceptSchema,
    SourceLabelRegistry,
    attach_registry,
    prepare_splits_for_registry,
)
from mermaidseg.datasets import (
    BenthosYuvalCoralsDataset,
    CatlinSeaviewDataset,
    CoralNetDataset,
    CoralscapesDataset,
    CoralscapesV2Dataset,
    MermaidDataset,
    MooreaLabeledCoralsDataset,
    PacificLabeledCoralsDataset,
)
from mermaidseg.io import get_parser, setup_config, update_config_with_args
from mermaidseg.model.class_counts import ClassPixelCounts, format_class_distribution
from mermaidseg.model.eval import Evaluator
from mermaidseg.model.hierarchical_metrics import METRIC_LEGENDS
from mermaidseg.model.meta import MetaModel
from mermaidseg.model.sparse_eval import (
    checkpoint_input_size,
    evaluate_sparse_native,
    strip_transform,
)

DATASET_CLASSES = {
    "pacific_labeled_corals": PacificLabeledCoralsDataset,
    "moorea_labeled_corals": MooreaLabeledCoralsDataset,
    "catlin_seaview": CatlinSeaviewDataset,
    "mermaid": MermaidDataset,
    "coralnet": CoralNetDataset,
    "coralscapes": CoralscapesDataset,
    "coralscapes_v2": CoralscapesV2Dataset,
    "benthos_yuval": BenthosYuvalCoralsDataset,
}
DENSE = ("coralscapes", "coralscapes_v2", "benthos_yuval")

p = argparse.ArgumentParser()
p.add_argument("--dataset", default="pacific_labeled_corals")
p.add_argument("--split", default="val")
p.add_argument("--n", type=int, default=None, help="limit images (default: whole split)")
p.add_argument(
    "--sample",
    default="random",
    choices=["random", "first"],
    help="how --n picks images. 'first' takes a leading slice, which is NOT a random "
    "sample: pacific is stored site-ordered, so its first 300 images are all heron_reef. "
    "Default 'random' samples across the whole split.",
)
p.add_argument("--seed", type=int, default=0, help="seed for --sample random")
p.add_argument("--checkpoint", required=True)
p.add_argument(
    "--model-size", type=int, default=None, help="default: the val Resize saved in the checkpoint"
)
p.add_argument("--max-points", type=int, default=None, help="cap annotated points per image")
p.add_argument("--hier", action="store_true", help="also compute the hierarchical metrics")
p.add_argument(
    "--ambiguity-policy", default="permissive", choices=["permissive", "exclude", "strict"]
)
p.add_argument(
    "--concept-mapping",
    default=None,
    help="override training.concept_mapping_path, e.g. for a checkpoint trained on an older "
    "concept layout",
)
p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
args = p.parse_args()

device = torch.device(args.device)
torch.manual_seed(42)

cfg = setup_config(
    {
        "data": "../configs/data_config.yaml",
        "training": "../configs/training_config_cbm.yaml",
        "model": "../configs/model_config_cbm_dpt_lora_vitl.yaml",
        "logger": "../configs/logger_config.yaml",
    }
)
cfg = update_config_with_args(cfg, get_parser().parse_args(["--run-name=sparse", "--batch-size=1"]))
if args.concept_mapping:
    cfg.training.concept_mapping_path = args.concept_mapping

with open("../demo/id2label.json") as f:
    id2label = json.load(f)
with open("../demo/concept_id2name.json") as f:
    concept_id2name = json.load(f)
pinned_target_labels = [id2label[str(i)] for i in range(1, len(id2label))]

split_cfg = cfg.data[args.dataset].get(args.split)
if split_cfg in (None, "None"):
    raise SystemExit(f"{args.dataset}/{args.split} is not configured in data_config.yaml")

ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
if args.model_size is not None:
    model_size = (args.model_size, args.model_size)
    size_source = "--model-size"
else:
    model_size = checkpoint_input_size(ckpt, args.dataset, args.split)
    size_source = "checkpoint's val Resize"
    if model_size is None:
        raise SystemExit("checkpoint has no saved val Resize; pass --model-size")

cls = DATASET_CLASSES[args.dataset]
extra = {} if args.dataset in DENSE else {"padding": cfg.training.padding}
# No transform: the dataset must yield native-resolution images and masks.
native_dataset = cls(**strip_transform(split_cfg), **extra)

print("=" * 78)
print(f"SPARSE NATIVE EVAL  {args.dataset}/{args.split}")
print("=" * 78)
print(f"  images in split   {len(native_dataset)}")
print(f"  model input size  {model_size}  from {size_source}  (mask NEVER resized)")
print(f"  checkpoint        {args.checkpoint}")

run_sources = set()
for name, split_cfgs in cfg.data.items():
    if name == "default" or name not in DATASET_CLASSES or not hasattr(split_cfgs, "values"):
        continue
    if any(sc not in (None, "None") for sc in split_cfgs.values()):
        run_sources.add(DATASET_CLASSES[name].SOURCE_NAME)
schema = ConceptSchema.from_csv(cfg.training.get("concept_mapping_path"), sources=run_sources)

dataset_dict = {(args.dataset, args.split): native_dataset}
_, registry_datasets = prepare_splits_for_registry(dataset_dict)
registry = SourceLabelRegistry(
    registry_datasets,
    target_labels=pinned_target_labels,
    target_label_subset=cfg.training.class_subset,
    compute_concepts=cfg.training.training_mode != "standard",
    concept_mapping_path=cfg.training.get("concept_mapping_path"),
    concept_schema=schema,
    label_roll_up=cfg.training.get("label_roll_up", False),
).to(device)
attach_registry(registry, dataset_dict.values())

rebuilt = {0: id2label["0"], **registry.target_id2label}
assert rebuilt == {int(k): v for k, v in id2label.items()}, "label space drifted!"
assert registry.num_concepts == schema.num_channels
if not args.concept_mapping:
    assert registry.num_concepts == len(concept_id2name), "concept layout drifted!"
print(
    f"  label space OK    {registry.num_target_classes} classes / {registry.num_concepts} concepts"
)

meta_model = MetaModel(
    run_name=cfg.run_name,
    num_classes=registry.num_target_classes,
    num_concepts=registry.num_concepts or None,
    device=device,
    model_kwargs=cfg.model.copy(),
    training_kwargs=cfg.training.copy(),
    source_to_target_lookup=registry.source_to_target,
    source_to_concepts_lookup=registry.source_to_concepts,
    concept_matrix=registry.concept_matrix,
    conceptid2labelid=registry.conceptid2labelid(),
    concept_value2id=registry.concept_value2id,
)
state_dict = ckpt.get("model_state_dict", ckpt)
model_keys = set(meta_model.model.state_dict())
best = max(
    (
        {
            k.replace("encoder.model.", "encoder.", 1) if k.startswith("encoder.model.") else k: v
            for k, v in state_dict.items()
        },
        state_dict,
    ),
    key=lambda sd: len(model_keys & set(sd)),
)
missing, unexpected = meta_model.model.load_state_dict(best, strict=False)
print(f"  checkpoint loaded missing={len(missing)} unexpected={len(unexpected)}")
meta_model.model.eval()

N = registry.num_target_classes
consumers = None
if args.hier:
    from mermaidseg.dataset_reconciliation.taxonomy_tree import build_taxonomy_tree
    from mermaidseg.model.hierarchical_metrics import (
        HierarchicalMetrics,
        format_calibration_table,
        format_consistency_table,
        format_severity_table,
        format_taxon_summary,
    )

    tree = build_taxonomy_tree(
        cfg.training.get("concept_mapping_path"), ambiguity_policy=args.ambiguity_policy
    )
    consumers = [HierarchicalMetrics(tree, score_transform="none")]

evaluator = Evaluator(
    num_classes=N,
    device=device,
    metric_dict={
        "miou_macro": JaccardIndex(
            task="multiclass", num_classes=N, ignore_index=0, average="macro"
        ).to(device),
        "pixel_accuracy": Accuracy(
            task="multiclass", num_classes=N, ignore_index=0, average="micro"
        ).to(device),
        "class_pixel_counts": ClassPixelCounts(N, ignore_index=0).to(device),
    },
    calculate_concept_metrics=True,
    concept_value2id=registry.concept_value2id,
    rank_output_consumers=consumers,
)

eval_indices = None
if args.n is not None and args.n < len(native_dataset):
    if args.sample == "random":
        import numpy as np

        eval_indices = sorted(
            np.random.default_rng(args.seed)
            .choice(len(native_dataset), args.n, replace=False)
            .tolist()
        )
        print(f"  sampling          {args.n} images at RANDOM (seed={args.seed})")
    else:
        eval_indices = list(range(args.n))
        print(f"  sampling          FIRST {args.n} images -- NOT random if the split is sorted")

results, coverage = evaluate_sparse_native(
    native_dataset,
    meta_model,
    evaluator,
    model_size=model_size,
    indices=eval_indices,
    subset_n=args.n,
    max_points_per_image=args.max_points,
)

print()
print("=" * 78)
print("COVERAGE")
print("=" * 78)
print(f"  images evaluated        {coverage.images}")
print(f"  annotated pixels        {coverage.annotated_pixels:,}")
print(f"  points scored           {coverage.scored_points:,}")
print(f"  images with no labels   {coverage.empty_images}")
print(f"  mean points / image     {results['coverage/mean_points_per_image']:.1f}")

print()
print("=" * 78)
print("METRICS")
print("=" * 78)
for group, title in [
    ("", "SEGMENTATION"),
    ("accuracy/", "PER-RANK ACCURACY"),
    ("consistency/", "HIERARCHICAL CONSISTENCY"),
    ("severity/", "MISTAKE SEVERITY"),
    ("ece/", "CALIBRATION (ECE)"),
    ("rare/", "RARE TAXON (LONG TAIL)"),
]:
    if group == "":
        keys = [k for k in ("pixel_accuracy", "miou_macro") if k in results]
    else:
        keys = sorted(k for k in results if isinstance(k, str) and k.startswith(group))
    if not keys:
        continue
    print(f"\n  --- {title} ---")
    if group in METRIC_LEGENDS:
        print(f"  ({METRIC_LEGENDS[group]})")
    if group == "consistency/":
        print(format_consistency_table(results, consumers[0].ranks))
        continue
    if group == "severity/":
        print(format_severity_table(results, consumers[0].ranks))
        continue
    if group == "ece/":
        print(format_calibration_table(results, consumers[0].ranks))
        continue
    for k in keys:
        v = results[k]
        if isinstance(v, float):
            print(f"  {k:<42} {v:.6f}")
if consumers and consumers[0].last_taxon_summary:
    print("\n  --- PER-TAXON (per rank) ---")
    print(f"  ({METRIC_LEGENDS['taxon/']})")
    print(format_taxon_summary(consumers[0].last_taxon_summary, consumers[0].ranks))

print()
print("=" * 78)
print(
    f"CLASS DISTRIBUTION  (final class head argmax at annotated points, {coverage.images} images)"
)
print("=" * 78)
class_names = {0: id2label["0"], **registry.target_id2label}
print(format_class_distribution(results["class_pixel_counts"], class_names, unit="points"))
print()
