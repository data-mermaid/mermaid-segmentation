"""Train the concept-bottleneck segmentation pipeline (ViT-L, LoRA, DPT head)."""

import argparse
import contextlib
import copy
import json
import os
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.utils.data import ConcatDataset, DataLoader

from mermaidseg.dataset_reconciliation.concept_source_mask import attach_source_rows
from mermaidseg.datasets import build_datasets, make_worker_init_fn, setup_local_cache
from mermaidseg.io import setup_config, update_config_with_args
from mermaidseg.logger import Logger, prune_rolling_checkpoints
from mermaidseg.model.distributed import (
    DistContext,
    ShardedBatchSampler,
    broadcast_from_rank0,
    init_distributed_from_env,
    make_rank_worker_init_fn,
    on_main_process,
)
from mermaidseg.model.eval import Evaluator
from mermaidseg.model.meta import MetaModel, next_epoch_from_checkpoint, restore_warmup_progress
from mermaidseg.model.models import align_optimizer_state_dict
from mermaidseg.model.train import train_model
from mermaidseg.run_setup import (
    build_meta_model,
    build_registry,
    check_env_wandb,
    concept_mapping_path,
    load_model_weights,
    load_run_concept_source_mask,
    resolve_repo_path,
)

# LoRA + DPT concept-bottleneck entrypoint. The encoder comes from the model config.


def print0(dist_ctx: DistContext, *args: Any, **kwargs: Any) -> None:
    """Print only on the main rank.

    Args:
        dist_ctx: Rank layout of this process.
        *args: Positional arguments forwarded to ``print``.
        **kwargs: Keyword arguments forwarded to ``print``.
    """
    if dist_ctx.is_main:
        print(*args, **kwargs)


def load_training_checkpoint(
    meta_model: MetaModel,
    checkpoint_path: str,
    device: torch.device | str,
    dist_ctx: DistContext,
    *,
    load_optimizer: bool = True,
) -> int:
    """Load model weights and, when resuming, optimizer and scheduler state.

    ``load_optimizer=False`` keeps the freshly constructed optimizer and schedule
    (hi-res fine-tune from cooled weights).
    """
    checkpoint = load_model_weights(meta_model, checkpoint_path, device)
    raw_state_dict = checkpoint.get("model_state_dict", checkpoint)

    if not load_optimizer:
        return 0

    # Restore optimizer state. torch.load(map_location=device) already placed the
    # state tensors on `device`, and the model was loaded in place (same Parameter
    # objects the optimizer references), so the param<->state mapping stays valid.
    # A checkpoint saved before the unused fusion residual was removed has four
    # extra Adam slots in the middle of the non-LoRA group; drop those only.
    if "optimizer_state_dict" in checkpoint:
        optimizer_state, n_dropped = align_optimizer_state_dict(
            checkpoint["optimizer_state_dict"],
            meta_model.optimizer,
            meta_model.model,
            raw_state_dict,
        )
        if n_dropped:
            print0(
                dist_ctx, f"Dropped {n_dropped} optimizer states for the removed fusion residual"
            )
        meta_model.optimizer.load_state_dict(optimizer_state)

    # Restore the LR scheduler so an epoch schedule continues from the correct step
    # instead of restarting its decay. `scheduler` may be None when no scheduler was
    # configured.
    scheduler = getattr(meta_model, "scheduler", None)
    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    # Step checkpoints store how far warmup got. Older epoch checkpoints omit it;
    # those are past warmup (it finishes inside epoch 0), so the helper marks it done.
    restore_warmup_progress(meta_model, checkpoint)

    if "global_step" in checkpoint:
        meta_model.global_step = int(checkpoint["global_step"])

    # Missing epoch_complete means an older epoch-boundary checkpoint: start the next epoch.
    # A mid-epoch model_step* file sets epoch_complete false and continues this epoch.
    meta_model.resuming_within_epoch = not bool(checkpoint.get("epoch_complete", True))  # type: ignore[attr-defined]  # set dynamically, read via getattr
    return next_epoch_from_checkpoint(checkpoint)


def make_shuffled_loader(
    dist_ctx: DistContext, dataset: ConcatDataset, **kwargs: Any
) -> DataLoader:
    """Shuffled minibatches of ``batch_size``; under torchrun, this rank's shard.

    Every rank uses rank 0's shuffle seed, so the global minibatches are the
    ones a single-process ``shuffle=True`` loader draws.

    Args:
        dist_ctx: Rank layout of this process.
        dataset: Dataset to draw minibatches from.
        **kwargs: Keyword arguments forwarded to ``DataLoader``.

    Returns:
        The shuffled loader, sharded across ranks when distributed.
    """
    if not dist_ctx.distributed:
        return DataLoader(dataset, shuffle=True, **kwargs)
    kwargs = dict(kwargs)
    batch_size = kwargs.pop("batch_size")
    kwargs.pop("drop_last")
    sampler = ShardedBatchSampler(
        dataset,
        batch_size,
        dist_ctx.rank,
        dist_ctx.world_size,
        broadcast_seed=broadcast_from_rank0,
    )
    return DataLoader(dataset, batch_sampler=sampler, **kwargs)


def run(args: argparse.Namespace) -> None:
    """Run one training job from parsed command-line arguments.

    Args:
        args: Parsed arguments from ``scripts/train.py``.
    """
    # -- 0. Environment --------------------------------------------------------
    # wandb-backed, AWS-free run: data is served from the pre-warmed local cache
    # (scripts/prefetch_cache.py) and metrics go to wandb. No MLflow / SageMaker
    # / AWS SSO is required. Set MERMAIDSEG_S3_OFFLINE=1 to fail loudly on any cache
    # miss instead of silently falling back to S3.
    check_env_wandb()

    # Under torchrun (WORLD_SIZE > 1) this process trains one GPU's shard of each
    # minibatch. Only rank 0 logs to wandb, writes checkpoints, and prints.
    dist_ctx = init_distributed_from_env()

    if dist_ctx.distributed:
        device = torch.device("cuda", dist_ctx.local_rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for i in range(torch.cuda.device_count()):
        print0(dist_ctx, f"CUDA Device {i}: {torch.cuda.get_device_name(i)}")

    # Match the SLURM CPU allocation when running under Slurm (falls back to 50 for
    # local runs). MERMAIDSEG_NUM_WORKERS / MERMAIDSEG_PREFETCH_FACTOR override both
    # (slurm/train.sbatch sets MERMAIDSEG_NUM_WORKERS). Under torchrun these are per rank.
    NUM_WORKERS = int(
        os.environ.get("MERMAIDSEG_NUM_WORKERS", os.environ.get("SLURM_CPUS_PER_TASK", "50"))
    )
    PREFETCH_FACTOR = int(os.environ.get("MERMAIDSEG_PREFETCH_FACTOR", "3"))
    PERSISTENT_WORKERS = NUM_WORKERS > 0

    # -- 1. Config -------------------------------------------------------------
    # CLI overrides used by the slurm/launch_*.sh experiment launchers.

    SEED = 4
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # slurm/train.sbatch resumes the latest checkpoint through --checkpoint.
    CHECKPOINT = args.checkpoint

    cfg = setup_config(
        {
            "data": str(resolve_repo_path(args.data_config)),
            "training": str(resolve_repo_path(args.training_config)),
            "model": str(resolve_repo_path(args.model_config)),
            "logger": str(resolve_repo_path(args.logger_config)),
        },
        resolution=args.resolution,
    )
    cfg = update_config_with_args(cfg, args)
    # torchrun decides: one process per GPU trains its shard of each minibatch.
    cfg.training.distributed = dist_ctx.distributed
    # MetaModel opens the taxonomy CSV from this value, so it must not depend on the cwd.
    cfg.training.concept_mapping_path = concept_mapping_path(cfg.training)

    # Leave the config value untouched when a flag is not given.
    if args.per_pixel_loss_weight is not None:
        cfg.training.loss.per_pixel_loss_weight = args.per_pixel_loss_weight
    if args.per_image_loss_weight is not None:
        cfg.training.loss.per_image_loss_weight = args.per_image_loss_weight
    if args.grad_accum_steps is not None:
        cfg.training.grad_accum_steps = args.grad_accum_steps
    taxonomic_cfg = cfg.training.get("taxonomic_loss")
    if taxonomic_cfg is not None and not taxonomic_cfg.get("kl_weight", 0.0) > 0:
        cfg.training.pop("taxonomic_loss")

    # Encoder, iterations, batch size, and epochs come from the model and training configs.

    # Set experiment on the config the Logger actually reads.
    cfg_logger = copy.deepcopy(cfg)
    cfg_logger.logger.experiment_name = "mermaid"

    # -- 2. Datasets -----------------------------------------------------------
    cache_stats = setup_local_cache(cfg.data)

    # Shared build loop (also used by scripts/prefetch_cache.py) so the training job
    # and the cache-prefetch job always agree on which source objects are needed.
    dataset_dict = build_datasets(cfg.data, padding=cfg.training.padding)

    loader_kwargs = {
        "batch_size": cfg.training.batch_size,
        "num_workers": NUM_WORKERS,
        "pin_memory": True,
        "persistent_workers": PERSISTENT_WORKERS,
        "drop_last": True,
        "prefetch_factor": PREFETCH_FACTOR,
    }
    if NUM_WORKERS > 0:
        loader_kwargs["worker_init_fn"] = make_worker_init_fn(cache_stats)
        if dist_ctx.distributed:
            loader_kwargs["worker_init_fn"] = make_rank_worker_init_fn(
                dist_ctx.rank, loader_kwargs["worker_init_fn"]
            )

    # -- 3. Registry / model / evaluator --------------------------------------
    registry, schema = build_registry(
        dataset_dict,
        cfg.training,
        schema_config=resolve_repo_path(args.concept_schema_config),
        device=device,
    )
    print0(dist_ctx, f"Froze concept channels to the datasets of {args.concept_schema_config}")

    # repeat coralscapes_v2 train dataset 5 times
    train_datasets = [ds for (_, split), ds in dataset_dict.items() if split == "train"] + [
        ds
        for (name, split), ds in dataset_dict.items()
        if name == "coralscapes_v2" and split == "train"
    ] * 5
    val_datasets = [ds for (_, split), ds in dataset_dict.items() if split == "val"]

    # training.concept_source_mask: zero the targets of source x binary concept pairs listed in the
    # mask CSV. Train batches then carry each sample's keep-table row; validation is not masked.
    concept_source_mask = load_run_concept_source_mask(cfg.training, schema)
    if concept_source_mask is not None:
        train_datasets = attach_source_rows(train_datasets, concept_source_mask)
        print0(
            dist_ctx,
            f"Concept source mask {concept_source_mask.path}: {concept_source_mask.num_masked} "
            f"{concept_source_mask.dataset} source x concept pairs over "
            f"{len(concept_source_mask.source_ids)} sources",
        )

    train_loader = make_shuffled_loader(dist_ctx, ConcatDataset(train_datasets), **loader_kwargs)
    loader_kwargs["prefetch_factor"] = 2
    val_loader = make_shuffled_loader(dist_ctx, ConcatDataset(val_datasets), **loader_kwargs)

    print0(dist_ctx, f"train batches: {len(train_loader)}   val batches: {len(val_loader)}")
    concept_id2name = schema.channel_id2name()
    if dist_ctx.is_main:
        # scripts/evaluate.py reads the class and concept layouts next to the checkpoint.
        run_dir = Path("model_checkpoints") / cfg.run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "concept_id2name.json").write_text(
            json.dumps({str(k): v for k, v in concept_id2name.items()})
        )
        (run_dir / "id2label.json").write_text(
            json.dumps({"0": "ignore", **{str(k): v for k, v in registry.target_id2label.items()}})
        )
        print(f"Wrote class and concept layouts ({len(concept_id2name)} channels) to {run_dir}")

    meta_model = build_meta_model(cfg, registry, device)
    if concept_source_mask is not None:
        meta_model.set_concept_source_mask(concept_source_mask.keep)

    start_epoch = 0
    init_weights = args.init_weights
    if init_weights:
        load_training_checkpoint(meta_model, init_weights, device, dist_ctx, load_optimizer=False)
        # Optimizer and warmup were built with the training config before the load.
        # Leave them alone: a cooled source has LR 0 and Adam moments we do not want.
        fresh_lrs = [group["lr"] for group in meta_model.optimizer.param_groups]
        print0(
            dist_ctx,
            f"Loaded weights from {init_weights!r}; fresh optimizer lrs={fresh_lrs} "
            "(epoch 0, global_step 0)",
        )
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)
        print0(dist_ctx, f"Re-seeded RNG to {SEED} for the new run's data order")
    elif CHECKPOINT:
        start_epoch = load_training_checkpoint(meta_model, CHECKPOINT, device, dist_ctx)
        print0(dist_ctx, f"Loaded checkpoint {CHECKPOINT!r}; resuming at epoch {start_epoch}")
        if start_epoch >= int(cfg.training.epochs):
            print0(
                dist_ctx,
                f"Checkpoint already covers all {int(cfg.training.epochs)} epoch(s). "
                "Exiting without further training.",
            )
            raise SystemExit(0)
        # Offset the RNG by the number of already-trained epochs so the resumed run
        # sees a different data-shuffling stream instead of replaying the epoch
        # 0..N-1 orderings. The DataLoader's RandomSampler draws from the global
        # torch generator (re-seeded to SEED at startup), and the training loop is
        # iterated below, so re-seeding here changes the permutations for every
        # resumed epoch. (Augmentations already vary per run via Albumentations'
        # own entropy-seeded RNG.)
        if getattr(meta_model, "resuming_within_epoch", False):
            resume_seed = SEED + int(meta_model.global_step)
            seed_note = f"{meta_model.global_step} completed updates"
        else:
            resume_seed = SEED + start_epoch
            seed_note = f"{start_epoch} trained epochs"
        torch.manual_seed(resume_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(resume_seed)
        print0(dist_ctx, f"Re-seeded RNG to {resume_seed} (SEED={SEED} + {seed_note})")

    evaluator = Evaluator(
        num_classes=registry.num_target_classes,
        device=device,
        calculate_concept_metrics=cfg.training.training_mode != "standard",
        concept_value2id=registry.concept_value2id,
    )
    # -- 5. Train (run lifecycle managed by `with`) ---------------------------
    # Backend selection is config-driven (configs/logger_config.yaml sets
    # enable_wandb: true). Under torchrun only rank 0 has a
    # Logger; the other ranks train with logger=None and write nothing.
    logger_context: contextlib.AbstractContextManager[Logger | None]
    if dist_ctx.is_main:
        logger_context = Logger(
            config=cfg_logger,
            meta_model=meta_model,
            log_epochs=cfg_logger.logger.get("log_epochs", 1),
            log_checkpoint=1,
            checkpoint_dir=".",
            id2label={0: "ignore", **registry.target_id2label},
            save_local_checkpoints=True,
        )
    else:
        logger_context = contextlib.nullcontext(None)
    with logger_context as run_logger:
        if run_logger is not None:
            assert run_logger.enable_wandb, (
                "wandb logging was not started — check WANDB_API_KEY and "
                "config.logger.enable_wandb (see warnings above)"
            )
            run_logger.log_dict(
                {str(k): v for k, v in concept_id2name.items()},
                "metadata/concept_id2name.json",
            )
            run_logger.log_params(
                {
                    "model/encoder_name": cfg.model.encoder_name,
                    "model/head": "dpt",
                    "model/adapter": "lora",
                    "model/lora_r": cfg.model.lora_r,
                    "model/lora_alpha": cfg.model.lora_alpha,
                    "loss/per_pixel_loss_weight": cfg.training.loss.per_pixel_loss_weight,
                    "loss/per_image_loss_weight": cfg.training.loss.per_image_loss_weight,
                    "loss/kl_weight": (cfg.training.get("taxonomic_loss") or {}).get(
                        "kl_weight", 0.0
                    ),
                    "loss/partial_weight": (cfg.training.get("taxonomic_loss") or {}).get(
                        "partial_weight", 0.0
                    ),
                    "loss/distance_weight": (cfg.training.get("distance_loss") or {}).get(
                        "weight", 0.0
                    ),
                    "data/resolution": args.resolution,
                    "train/batch_size": cfg.training.batch_size,
                    "train/grad_accum_steps": int(cfg.training.get("grad_accum_steps", 1)),
                    "train/world_size": dist_ctx.world_size,
                }
            )
            if init_weights:
                run_logger.log_params(
                    {
                        "init/weights": init_weights,
                        "init/optimizer": "fresh",
                        "init/start_epoch": 0,
                    }
                )
            elif CHECKPOINT:
                run_logger.log_params(
                    {"init/checkpoint": CHECKPOINT, "init/start_epoch": start_epoch}
                )
            print(f"\nwandb run_id : {run_logger.wandb_run_id}")
            print(f"wandb run URL: {run_logger.run_url}\n")

            # Reuse the logger helpers instead of hand-rolling concept extraction.
            run_logger.log_dataloader_params(train_loader, prefix="train_loader")
            run_logger.log_dataloader_params(val_loader, prefix="val_loader")
            run_logger.log_reconciliation(registry)  # writes metadata/concept_id2name.json

            train_size = sum(len(d) for d in train_datasets)
            val_size = sum(len(d) for d in val_datasets)
            run_logger.log_params(
                {
                    "data/train_size": train_size,
                    "data/val_size": val_size,
                    "data/total_size": sum(len(d) for d in dataset_dict.values()),
                    "data/seed": SEED,
                    "data/dataset_name": "ALL",
                }
            )

        metrics_all: dict[int, dict] = {}
        checkpoint_every = int(cfg.training.get("checkpoint_every_steps", 0) or 0)
        preempt_every = int(args.preempt_checkpoint_every_steps or 0)
        total_updates = int(cfg.training.epochs) * int(cfg.training.iterations_per_train_epoch)

        for epoch in range(start_epoch, cfg.training.epochs):

            def _save_step_checkpoint(meta: MetaModel, epoch: int = epoch) -> None:
                step = int(meta.global_step)
                # Same predicate on every rank: global_step is updated together.
                # Steps that do not write return immediately, with no barrier.
                write_interval = checkpoint_every > 0 and step % checkpoint_every == 0
                write_preempt = preempt_every > 0 and step > 0 and step % preempt_every == 0
                if not write_interval and not write_preempt:
                    return

                def _write() -> None:
                    assert (
                        run_logger is not None
                    )  # on_main_process runs only where the Logger exists
                    # Interval checkpoints (eval cadence). Unchanged when preempt_every is 0,
                    # which is the default.
                    if write_interval:
                        run_logger.save_model_checkpoint(
                            meta,
                            epoch,
                            {},
                            checkpoint_name=f"model_step{step}",
                            epoch_complete=step >= total_updates,
                        )
                    # One rolling file for preemption. epoch_complete stays false until the
                    # run is finished, so a resume reseeds with SEED + global_step and
                    # continues this epoch. Older model_preempt* files are deleted after
                    # the rename; model_step* and model_epoch* are not.
                    if write_preempt:
                        name = f"model_preempt{step}"
                        run_logger.save_model_checkpoint(
                            meta,
                            epoch,
                            {},
                            checkpoint_name=name,
                            epoch_complete=step >= total_updates,
                            atomic=True,
                        )
                        prune_rolling_checkpoints(
                            os.path.join(
                                run_logger.checkpoint_dir, "model_checkpoints", meta.run_name
                            ),
                            keep_name=name,
                        )

                on_main_process(_write)

            metrics = train_model(
                meta_model=meta_model,
                evaluator=evaluator,
                train_loader=train_loader,
                val_loader=val_loader,
                logger=run_logger,
                start_epoch=epoch,
                end_epoch=epoch + 1,
                metric_of_interest="accuracy",
                train_log_interval=cfg.logger.get("train_log_interval", 1),
                on_optimizer_step=(
                    _save_step_checkpoint
                    if (dist_ctx.distributed or run_logger is not None)
                    else None
                ),
                save_best_checkpoint=checkpoint_every <= 0,
            )
            metrics_all.update(metrics)
            # Step-checkpointed runs already saved model_step*.
            # Other runs still write model_epoch{N} at the end of each epoch.
            if (
                run_logger is not None
                and checkpoint_every <= 0
                and cfg.training.iterations_per_train_epoch > 1000
            ):
                run_logger.save_model_checkpoint(
                    meta_model,
                    epoch,
                    metrics[epoch].get("validation_metrics", {}),
                )
        if not metrics_all:
            print0(dist_ctx, "No training epochs ran.")
        else:
            final_epoch = max(metrics_all)
            print0(
                dist_ctx, "Final train metrics     :", metrics_all[final_epoch].get("train_metrics")
            )
            print0(
                dist_ctx,
                "Final validation metrics:",
                metrics_all[final_epoch].get("validation_metrics"),
            )
        if run_logger is not None:
            print(f"wandb run URL: {run_logger.run_url}")

    if dist_ctx.distributed:
        dist.destroy_process_group()
