"""Evaluation runner for dense datasets (default coralscapes_v2), with optional --hier.

Run from nbs/:

    uv run --no-sync python diag_eval.py --n 460 --bs 4 --hier \
        --checkpoint model_checkpoints/<checkpoint>

Evaluates the first ``--n`` images of the val split, resized to the val ``Resize`` saved in the
checkpoint (override with --resize). The checkpoint must match the current concept layout
(demo/concept_id2name.json), or pass ``--concept-mapping`` for an older one.
Also prints library versions and checksums of the checkpoint, inputs, loaded weights and
first-batch logits, so runs on different machines can be diffed.
"""

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import albumentations
import cv2
import numpy as np
import PIL
import torch
import transformers
from torch.utils.data import DataLoader
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
from mermaidseg.model.meta import MetaModel
from mermaidseg.model.sparse_eval import IMAGENET_MEAN, IMAGENET_STD, checkpoint_input_size

p = argparse.ArgumentParser()
p.add_argument("--n", type=int, default=32)
p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
p.add_argument("--dataset", default="coralscapes_v2")
p.add_argument(
    "--resize", type=int, default=None, help="override the val Resize saved in the checkpoint"
)
p.add_argument("--bs", type=int, default=2)
p.add_argument(
    "--bn",
    action="store_true",
    help="use BatchNorm2d at dpt_head.head[1], for checkpoints trained before GroupNorm",
)
p.add_argument("--checkpoint", required=True, help="path to the checkpoint to load")
p.add_argument(
    "--hier",
    action="store_true",
    help="also compute the hierarchical metrics "
    "(consistency, mistake severity, ECE per rank, rare taxa)",
)
p.add_argument(
    "--ambiguity-policy",
    default="permissive",
    choices=["permissive", "exclude", "strict"],
    help="how to treat multi-parent taxa (e.g. the turbinaria homonym); --hier only",
)
p.add_argument(
    "--concept-mapping",
    default=None,
    help="override training.concept_mapping_path, e.g. for a checkpoint trained on an older "
    "concept layout",
)
args_cli = p.parse_args()

CHECKPOINT_PATH = args_cli.checkpoint
ID2LABEL_PATH = "../demo/id2label.json"
CONCEPT_ID2NAME_PATH = "../demo/concept_id2name.json"

device = torch.device(args_cli.device)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


def sha(x: bytes) -> str:
    return hashlib.sha256(x).hexdigest()[:16]


print("=" * 78)
print("ENVIRONMENT")
print("=" * 78)
try:
    import huggingface_hub

    hub_v = huggingface_hub.__version__
except Exception:
    hub_v = "?"

print(
    f"  platform      {platform.system()} {platform.machine()}  python {platform.python_version()}"
)
print(f"  torch         {torch.__version__}   device={device}")
print(f"  transformers  {transformers.__version__}")
print(f"  hf-hub        {hub_v}")
print(f"  albumentations{albumentations.__version__}")
print(f"  opencv        {cv2.__version__}")
print(f"  pillow        {PIL.__version__}")
print(f"  numpy         {np.__version__}")
ck = Path(CHECKPOINT_PATH)
print(f"  checkpoint    {ck.stat().st_size} bytes  md5={hashlib.md5(ck.read_bytes()).hexdigest()}")

# ---------------------------------------------------------------- config
config_path_dict = {
    "data": "../configs/data_config.yaml",
    "training": "../configs/training_config_cbm.yaml",
    "model": "../configs/model_config_cbm_dpt_lora_vitl.yaml",
    "logger": "../configs/logger_config.yaml",
}
cfg = setup_config(config_path_dict)
parser = get_parser()
cfg = update_config_with_args(cfg, parser.parse_args(["--run-name=diag", "--batch-size=2"]))
if args_cli.concept_mapping:
    cfg.training.concept_mapping_path = args_cli.concept_mapping

with open(ID2LABEL_PATH) as f:
    id2label = json.load(f)
with open(CONCEPT_ID2NAME_PATH) as f:
    concept_id2name = json.load(f)
pinned_target_labels = [id2label[str(i)] for i in range(1, len(id2label))]

run_sources = set()
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
for name, split_cfgs in cfg.data.items():
    # data_config.yaml also holds scalar keys (e.g. local_cache_dir) beside the datasets.
    if name == "default" or name not in DATASET_CLASSES or not hasattr(split_cfgs, "values"):
        continue
    if any(sc not in (None, "None") for sc in split_cfgs.values()):
        run_sources.add(DATASET_CLASSES[name].SOURCE_NAME)
schema = ConceptSchema.from_csv(cfg.training.get("concept_mapping_path"), sources=run_sources)

# ---------------------------------------------------------------- data
print()
print("=" * 78)
print("DATA FINGERPRINT")
print("=" * 78)
ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
if args_cli.resize is not None:
    size, size_source = (args_cli.resize, args_cli.resize), "--resize"
else:
    size, size_source = checkpoint_input_size(ckpt, args_cli.dataset), "checkpoint's val Resize"
    if size is None:
        raise SystemExit("checkpoint has no saved val Resize; pass --resize")
split_cfg = dict(cfg.data[args_cli.dataset].get("val"))
split_cfg["transform"] = albumentations.Compose(
    [
        albumentations.Resize(height=size[0], width=size[1]),
        albumentations.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)
print(f"  input size    {size} from {size_source}")
print(
    f"  {args_cli.dataset}/val split_cfg = { {k: (str(v)[:80] if k == 'transform' else v) for k, v in dict(split_cfg).items()} }"
)
# Point datasets take a `padding` argument; dense ones do not.
_DENSE = ("coralscapes", "coralscapes_v2", "benthos_yuval")
if args_cli.dataset in _DENSE:
    eval_dataset = DATASET_CLASSES[args_cli.dataset](**split_cfg)
else:
    eval_dataset = DATASET_CLASSES[args_cli.dataset](**split_cfg, padding=cfg.training.padding)
print(f"  len(dataset)  {len(eval_dataset)}")

n = min(args_cli.n, len(eval_dataset))
sub = torch.utils.data.Subset(eval_dataset, range(n))

dataset_dict = {(args_cli.dataset, "val"): eval_dataset}
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
expected = {int(k): v for k, v in id2label.items()}
print(f"  label space matches snapshot: {rebuilt == expected}")
print(f"  num_target_classes={registry.num_target_classes}  num_concepts={registry.num_concepts}")

loader = DataLoader(sub, batch_size=args_cli.bs, shuffle=False, num_workers=0)

# hash the exact input tensors + GT the model will see
h_in = hashlib.sha256()
h_gt = hashlib.sha256()
first_stats = None
for inputs, src in loader:
    x = inputs.float()
    h_in.update(np.ascontiguousarray(x.numpy()).tobytes())
    h_gt.update(np.ascontiguousarray(src.numpy().astype(np.int64)).tobytes())
    if first_stats is None:
        first_stats = (
            tuple(x.shape),
            x.dtype,
            float(x.min()),
            float(x.max()),
            float(x.mean()),
            float(x.std()),
        )
print(f"  first batch   shape={first_stats[0]} dtype={first_stats[1]}")
print(
    f"                min={first_stats[2]:.6f} max={first_stats[3]:.6f} mean={first_stats[4]:.6f} std={first_stats[5]:.6f}"
)
print(f"  INPUT  sha256[:16] over {n} imgs = {h_in.hexdigest()[:16]}")
print(f"  GT src sha256[:16] over {n} imgs = {h_gt.hexdigest()[:16]}")

# ---------------------------------------------------------------- model
print()
print("=" * 78)
print("CHECKPOINT LOAD")
print("=" * 78)
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

if args_cli.bn:
    # BatchNorm and GroupNorm share weight/bias shapes, so without this strict=False would load
    # an old checkpoint's BN affine params into GroupNorm and silently drop running_mean/var.
    old = meta_model.model.dpt_head.head[1]
    print(f"  *** dpt_head.head[1] was {type(old).__name__}; restoring BatchNorm2d ***")
    meta_model.model.dpt_head.head[1] = torch.nn.BatchNorm2d(old.num_channels).to(device)

model_keys = set(meta_model.model.state_dict().keys())
state_dict = (
    ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
)


def _rewrite(sd, strip_encoder_model, collapse_double_model):
    out = {}
    for key, value in sd.items():
        if strip_encoder_model and key.startswith("encoder.model."):
            key = key.replace("encoder.model.", "encoder.", 1)
        if collapse_double_model and ".base_model.model.model." in key:
            key = key.replace(".base_model.model.model.", ".base_model.model.", 1)
        out[key] = value
    return out


# Checkpoint key nesting depends on the transformers version; use the best-matching rewrite.
variants = {
    "notebook (strip encoder.model. + collapse .model.model.)": (True, True),
    "strip encoder.model. only": (True, False),
    "collapse .model.model. only": (False, True),
    "no rewrite": (False, False),
}
print("  key-rewrite variants (overlap with the model's own keys):")
best, best_name, best_hit = None, None, -1
for vname, (a, b) in variants.items():
    cand = _rewrite(state_dict, a, b)
    hit = len(model_keys & set(cand))
    print(f"      {hit:>4}/{len(model_keys)} matched   {vname}")
    if hit > best_hit:
        best, best_name, best_hit = cand, vname, hit
print(f"  -> using: {best_name}")
state_dict = best
missing, unexpected = meta_model.model.load_state_dict(state_dict, strict=False)
meta_model.model.eval()
print(f"  model params in state_dict: {len(state_dict)}   model expects: {len(model_keys)}")
print(f"  MISSING keys   : {len(missing)}")
for k in missing[:20]:
    print(f"      - {k}")
print(f"  UNEXPECTED keys: {len(unexpected)}")
for k in unexpected[:20]:
    print(f"      + {k}")

# weight fingerprint after loading
sd_after = meta_model.model.state_dict()
h_w = hashlib.sha256()
for k in sorted(sd_after):
    v = sd_after[k]
    if v.is_floating_point():
        h_w.update(k.encode())
        h_w.update(np.ascontiguousarray(v.detach().float().cpu().numpy()).tobytes())
print(f"  LOADED-WEIGHTS sha256[:16] = {h_w.hexdigest()[:16]}")
tot = sum(v.numel() for v in sd_after.values())
print(f"  total params in model: {tot:,}")

# ---------------------------------------------------------------- forward
print()
print("=" * 78)
print(f"FORWARD + METRICS  (n={n} images, device={device})")
print("=" * 78)
N = registry.num_target_classes


metric_dict = {
    "miou_macro": JaccardIndex(
        task="multiclass", num_classes=N, ignore_index=0, average="macro"
    ).to(device),
    "pixel_accuracy": Accuracy(
        task="multiclass", num_classes=N, ignore_index=0, average="micro"
    ).to(device),
    "class_pixel_counts": ClassPixelCounts(N, ignore_index=0).to(device),
}
hier_metrics = None
if args_cli.hier:
    from mermaidseg.dataset_reconciliation.taxonomy_tree import build_taxonomy_tree, summarise
    from mermaidseg.model.hierarchical_metrics import (
        METRIC_LEGENDS,
        HierarchicalMetrics,
        format_calibration_table,
        format_consistency_table,
        format_severity_table,
        format_taxon_summary,
    )

    tree = build_taxonomy_tree(
        cfg.training.get("concept_mapping_path"),
        ambiguity_policy=args_cli.ambiguity_policy,
    )
    print()
    print(summarise(tree))
    # The CBM already applies a per-rank softmax, so the scores are probabilities.
    hier_metrics = HierarchicalMetrics(tree, score_transform="none")

evaluator = Evaluator(
    num_classes=N,
    device=device,
    metric_dict=metric_dict,
    calculate_concept_metrics=True,
    concept_value2id=registry.concept_value2id,
    rank_output_consumers=[hier_metrics] if hier_metrics else None,
)
loader2 = DataLoader(sub, batch_size=args_cli.bs, shuffle=False, num_workers=0)
results = evaluator.evaluate_model(dataloader=loader2, meta_model=meta_model)

print(f"  pixel_accuracy {float(results['pixel_accuracy']):.6f}")
print(f"  miou_macro     {float(results['miou_macro']):.6f}")
for k in sorted(results):
    if isinstance(k, str) and k.startswith("accuracy/"):
        print(f"  {k:<30} {float(results[k]):.6f}")

if hier_metrics is not None:
    for prefix, title in [
        ("consistency/", "HIERARCHICAL CONSISTENCY"),
        ("severity/", "MISTAKE SEVERITY"),
        ("ece/", "CALIBRATION (ECE)"),
        ("rare/", "RARE TAXON (LONG TAIL)"),
    ]:
        keys = sorted(k for k in results if isinstance(k, str) and k.startswith(prefix))
        if not keys:
            continue
        print()
        print(f"  --- {title} ---")
        if prefix in METRIC_LEGENDS:
            print(f"  ({METRIC_LEGENDS[prefix]})")
        if prefix == "consistency/":
            print(format_consistency_table(results, hier_metrics.ranks))
            continue
        if prefix == "severity/":
            print(format_severity_table(results, hier_metrics.ranks))
            continue
        if prefix == "ece/":
            print(format_calibration_table(results, hier_metrics.ranks))
            continue
        for k in keys:
            print(f"  {k:<40} {float(results[k]):.6f}")

    if hier_metrics.last_taxon_summary:
        print()
        print("  --- PER-TAXON (per rank) ---")
        print(f"  ({METRIC_LEGENDS['taxon/']})")
        print(format_taxon_summary(hier_metrics.last_taxon_summary, hier_metrics.ranks))

# logits fingerprint on the first batch
loader3 = DataLoader(sub, batch_size=args_cli.bs, shuffle=False, num_workers=0)
inputs, src = next(iter(loader3))
with torch.no_grad():
    logits, _ = meta_model.batch_predict(inputs.float().to(device))
lg = logits.detach().float().cpu()
print()
print(f"  logits shape   {tuple(lg.shape)}")
print(f"  logits  min/max/mean/std  {lg.min():.5f} {lg.max():.5f} {lg.mean():.5f} {lg.std():.5f}")
print(f"  LOGITS sha256[:16] (batch0, rounded 3dp) = {sha(np.round(lg.numpy(), 3).tobytes())}")

print()
print("=" * 78)
print(f"CLASS DISTRIBUTION  (final class head argmax, all n={n} images)")
print("=" * 78)
class_names = {0: id2label["0"], **registry.target_id2label}
print(format_class_distribution(results["class_pixel_counts"], class_names, unit="px"))
print()
print("=" * 78)
