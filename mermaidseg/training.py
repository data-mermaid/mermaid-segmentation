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
    broadcast_generator_from_rank0,
    init_distributed_from_env,
    on_main_process,
)
from mermaidseg.model.eval import Evaluator
from mermaidseg.model.meta import MetaModel, next_epoch_from_checkpoint, restore_warmup_progress
from mermaidseg.model.train import train_model
from mermaidseg.run_setup import (
    MODEL_CHECKPOINT_KEYS,
    RESUME_CHECKPOINT_KEYS,
    build_meta_model,
    build_registry,
    check_env_wandb,
    concept_mapping_path,
    load_model_weights,
    load_run_concept_source_mask,
    read_checkpoint,
    resolve_repo_path,
    write_resolved_config,
)
from mermaidseg.seeding import make_worker_seeder, seed_everything

# LoRA + DPT concept-bottleneck trainer that scripts/train.py calls. The encoder comes from the model config.


# The val generator seed is the run seed plus this, so val and train shuffle independently.
VAL_GENERATOR_SEED_OFFSET = 1_000_003


def seed_run(seed: int, train_generator: torch.Generator, val_generator: torch.Generator) -> None:
    """Seed the global RNGs and both loaders' generators for one run (or resume).

    The loaders draw their shuffle order and worker seeds from these generators on
    their first iteration, so re-seeding a built loader that has not started
    iterating changes its whole data order.

    Args:
        seed: Run seed; the train generator gets it and the val generator gets
            ``seed + VAL_GENERATOR_SEED_OFFSET``.
        train_generator: Generator held by the train loader.
        val_generator: Generator held by the val loader.
    """
    seed_everything(seed)
    train_generator.manual_seed(seed)
    val_generator.manual_seed(seed + VAL_GENERATOR_SEED_OFFSET)


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
    checkpoint: dict,
    dist_ctx: DistContext,
    *,
    load_optimizer: bool = True,
) -> int:
    """Load model weights and, when resuming, optimizer and scheduler state from ``checkpoint``.

    ``load_optimizer=False`` keeps the freshly constructed optimizer and schedule
    (hi-res fine-tune from cooled weights).
    """
    load_model_weights(meta_model, checkpoint)

    if not load_optimizer:
        return 0

    # The state tensors are on the CPU; `optimizer.load_state_dict` casts them to each param's
    # device. The model was loaded in place (same Parameter objects the optimizer references).
    meta_model.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    meta_model.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    restore_warmup_progress(meta_model, checkpoint)
    meta_model.global_step = int(checkpoint["global_step"])
    # A mid-epoch model_step* file sets epoch_complete false and continues this epoch.
    meta_model.resuming_within_epoch = not bool(checkpoint["epoch_complete"])
    return next_epoch_from_checkpoint(checkpoint)


def _read_run_checkpoint(args: argparse.Namespace) -> dict | None:
    """The ``--init-weights`` or ``--checkpoint`` file, read and key-checked; ``None`` if neither."""
    if args.init_weights:
        return read_checkpoint(args.init_weights, MODEL_CHECKPOINT_KEYS)
    if args.checkpoint:
        return read_checkpoint(args.checkpoint, RESUME_CHECKPOINT_KEYS)
    return None


def make_shuffled_loader(
    dist_ctx: DistContext, dataset: ConcatDataset, generator: torch.Generator, **kwargs: Any
) -> DataLoader:
    """Shuffled minibatches of ``batch_size``; under torchrun, this rank's shard.

    Every rank adopts rank 0's generator state before each pass, so the global
    minibatches are those of a single-process ``DataLoader(shuffle=True,
    batch_size=batch_size, drop_last=True, generator=<generator in the same state>)``.

    Args:
        dist_ctx: Rank layout of this process.
        dataset: Dataset to draw minibatches from.
        generator: Generator for the shuffle order and the worker base seed.
        **kwargs: Keyword arguments forwarded to ``DataLoader``.

    Returns:
        The shuffled loader, sharded across ranks when distributed.
    """
    if not dist_ctx.distributed:
        return DataLoader(dataset, shuffle=True, generator=generator, **kwargs)
    kwargs = dict(kwargs)
    batch_size = kwargs.pop("batch_size")
    kwargs.pop("drop_last")
    sampler = ShardedBatchSampler(
        dataset,
        batch_size,
        dist_ctx.rank,
        dist_ctx.world_size,
        generator=generator,
        broadcast_state=broadcast_generator_from_rank0,
    )
    return DataLoader(dataset, batch_sampler=sampler, generator=generator, **kwargs)


def _assemble_train_datasets(dataset_dict: dict[tuple[str, str], Any], data_cfg: Any) -> list[Any]:
    """Every train dataset, then each extra copy that ``repeat`` asks for.

    ``repeat: N`` on a train split is the total number of copies of that dataset in
    the result: the first sits among the train datasets and ``N - 1`` more are
    appended after all of them. A missing ``repeat`` means 1.

    Args:
        dataset_dict: Built datasets keyed by ``(name, split)``.
        data_cfg: The ``data`` section of the run config.

    Returns:
        The train datasets in concatenation order.

    Raises:
        ValueError: A train split's ``repeat`` is not an int (bools and floats
            included) or is below 1; the message names the dataset.
    """
    train_items = [(name, ds) for (name, split), ds in dataset_dict.items() if split == "train"]
    datasets = [ds for _, ds in train_items]
    for name, ds in train_items:
        repeat = data_cfg[name]["train"].get("repeat", 1)
        if isinstance(repeat, bool) or not isinstance(repeat, int) or repeat < 1:
            raise ValueError(f"dataset {name!r}: train repeat must be an int >= 1, got {repeat!r}")
        datasets.extend([ds] * (repeat - 1))
    return datasets


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

    # -- 1. Config -------------------------------------------------------------
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

    # Under torchrun (WORLD_SIZE > 1) this process trains one GPU's shard of each
    # minibatch. Only rank 0 logs to wandb, writes checkpoints, and prints.
    dist_ctx = init_distributed_from_env(cfg.training.world_size, args.nccl_timeout_min)

    if dist_ctx.distributed:
        device = torch.device("cuda", dist_ctx.local_rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for i in range(torch.cuda.device_count()):
        print0(dist_ctx, f"CUDA Device {i}: {torch.cuda.get_device_name(i)}")

    seed = int(cfg.training.seed)
    train_generator, val_generator = torch.Generator(), torch.Generator()
    seed_run(seed, train_generator, val_generator)

    # slurm/train.sbatch resumes the latest checkpoint through --checkpoint.
    checkpoint = args.checkpoint
    # Read before any dataset work so an unloadable file fails in seconds.
    checkpoint_data = _read_run_checkpoint(args)

    # torchrun decides: one process per GPU trains its shard of each minibatch.
    cfg.training.distributed = dist_ctx.distributed
    # MetaModel opens the taxonomy CSV from this value, so it must not depend on the cwd.
    cfg.training.concept_mapping_path = concept_mapping_path(cfg.training)

    # Leave the config value untouched when a flag is not given.
    if args.grad_accum_steps is not None:
        cfg.training.grad_accum_steps = args.grad_accum_steps
    if not cfg.training.taxonomic_loss.kl_weight > 0:
        cfg.training.pop("taxonomic_loss")

    # Encoder, iterations, batch size, and epochs come from the model and training configs.

    # Snapshot before MetaModel construction so the Logger records the config as loaded.
    cfg_logger = copy.deepcopy(cfg)

    # -- 2. Datasets -----------------------------------------------------------
    if args.cache_dir is not None:
        cfg.data.local_cache_dir = args.cache_dir
    cache_stats = setup_local_cache(cfg.data)

    # Shared build loop (also used by scripts/prefetch_cache.py) so the training job
    # and the cache-prefetch job always agree on which source objects are needed.
    dataset_dict = build_datasets(cfg.data, padding=cfg.training.padding)

    loader_kwargs = {
        "batch_size": cfg.training.batch_size,
        # Under torchrun the worker count is per rank.
        "num_workers": args.num_workers,
        "pin_memory": True,
        "persistent_workers": True,
        "drop_last": True,
        "prefetch_factor": cfg.training.loader.prefetch_factor,
    }
    loader_kwargs["worker_init_fn"] = make_worker_seeder(
        dist_ctx.rank, make_worker_init_fn(cache_stats)
    )

    # -- 3. Registry / model / evaluator --------------------------------------
    registry, schema = build_registry(
        dataset_dict,
        cfg.training,
        schema_config=resolve_repo_path(args.concept_schema_config),
        device=device,
    )
    print0(dist_ctx, f"Froze concept channels to the datasets of {args.concept_schema_config}")

    train_datasets = _assemble_train_datasets(dataset_dict, cfg.data)
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

    train_loader = make_shuffled_loader(
        dist_ctx, ConcatDataset(train_datasets), generator=train_generator, **loader_kwargs
    )
    loader_kwargs["prefetch_factor"] = cfg.training.loader.val_prefetch_factor
    val_loader = make_shuffled_loader(
        dist_ctx, ConcatDataset(val_datasets), generator=val_generator, **loader_kwargs
    )

    print0(dist_ctx, f"train batches: {len(train_loader)}   val batches: {len(val_loader)}")
    concept_id2name = schema.channel_id2name()
    if dist_ctx.is_main:
        # scripts/evaluate.py reads the class and concept layouts next to the checkpoint.
        run_dir = Path(args.output_dir) / "model_checkpoints" / cfg.run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "concept_id2name.json").write_text(
            json.dumps({str(k): v for k, v in concept_id2name.items()})
        )
        (run_dir / "id2label.json").write_text(
            json.dumps({"0": "ignore", **{str(k): v for k, v in registry.target_id2label.items()}})
        )
        write_resolved_config(cfg_logger, run_dir)
        print(f"Wrote class and concept layouts ({len(concept_id2name)} channels) to {run_dir}")

    meta_model = build_meta_model(cfg, registry, device)
    if concept_source_mask is not None:
        meta_model.set_concept_source_mask(concept_source_mask.keep)

    start_epoch = 0
    init_weights = args.init_weights
    if init_weights:
        assert checkpoint_data is not None
        load_training_checkpoint(meta_model, checkpoint_data, dist_ctx, load_optimizer=False)
        # Optimizer and warmup were built with the training config before the load.
        # Leave them alone: a cooled source has LR 0 and Adam moments we do not want.
        fresh_lrs = [group["lr"] for group in meta_model.optimizer.param_groups]
        print0(
            dist_ctx,
            f"Loaded weights from {init_weights!r}; fresh optimizer lrs={fresh_lrs} "
            "(epoch 0, global_step 0)",
        )
        seed_run(seed, train_generator, val_generator)
        print0(dist_ctx, f"Re-seeded RNG to {seed} for the new run's data order")
    elif checkpoint:
        assert checkpoint_data is not None
        start_epoch = load_training_checkpoint(meta_model, checkpoint_data, dist_ctx)
        print0(dist_ctx, f"Loaded checkpoint {checkpoint!r}; resuming at epoch {start_epoch}")
        if start_epoch >= int(cfg.training.epochs):
            print0(
                dist_ctx,
                f"Checkpoint already covers all {int(cfg.training.epochs)} epoch(s). "
                "Exiting without further training.",
            )
            raise SystemExit(0)
        # Offset the seed by the progress so far, so the resumed run does not replay
        # the orderings already trained on. The loaders have not iterated yet, so
        # re-seeding their generators here sets every resumed epoch's permutation.
        # (Augmentations vary per run via Albumentations' own entropy-seeded RNG.)
        if meta_model.resuming_within_epoch:
            resume_seed = seed + int(meta_model.global_step)
            seed_note = f"{meta_model.global_step} completed updates"
        else:
            resume_seed = seed + start_epoch
            seed_note = f"{start_epoch} trained epochs"
        seed_run(resume_seed, train_generator, val_generator)
        print0(dist_ctx, f"Re-seeded RNG to {resume_seed} (seed={seed} + {seed_note})")

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
            log_checkpoint=1,
            checkpoint_dir=args.output_dir,
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
                    "loss/kl_weight": (
                        cfg.training.taxonomic_loss.kl_weight
                        if "taxonomic_loss" in cfg.training
                        else 0.0
                    ),
                    "data/resolution": args.resolution,
                    "train/batch_size": cfg.training.batch_size,
                    "train/grad_accum_steps": int(cfg.training.grad_accum_steps),
                    "train/world_size": dist_ctx.world_size,
                }
            )
            run_logger.log_params({f"cli/{k}": v for k, v in vars(args).items()})
            if init_weights:
                run_logger.log_params(
                    {
                        "init/weights": init_weights,
                        "init/optimizer": "fresh",
                        "init/start_epoch": 0,
                    }
                )
            elif checkpoint:
                run_logger.log_params(
                    {"init/checkpoint": checkpoint, "init/start_epoch": start_epoch}
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
                    "data/seed": seed,
                    "data/dataset_name": "ALL",
                }
            )

        metrics_all: dict[int, dict] = {}
        checkpoint_every = int(cfg.training.checkpoint_every_steps)
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
                    # run is finished, so a resume reseeds with seed + global_step and
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
                metric_of_interest=cfg.training.metric_of_interest,
                max_load_failure_rate=cfg.training.max_load_failure_rate,
                train_log_interval=cfg.logger.train_log_interval,
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
