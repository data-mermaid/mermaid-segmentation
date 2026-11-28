"""Train the concept-bottleneck segmentation pipeline (ViT-L, LoRA, DPT head)."""

import argparse
import contextlib
import copy
import functools
import json
import os
import shutil
import signal
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.utils.data import ConcatDataset, DataLoader

from mermaidseg.checkpoint import checkpoint_dir, prune_rolling_checkpoints, write_checkpoint
from mermaidseg.config import to_plain_dict
from mermaidseg.data.config import DataSection
from mermaidseg.data.factory import build_datasets
from mermaidseg.dataset_reconciliation.artifact_meta import meta_path
from mermaidseg.dataset_reconciliation.concept_source_mask import attach_source_rows
from mermaidseg.dataset_reconciliation.label_space import (
    CLASSES_CSV,
    CONCEPTS_CSV,
    META_YAML,
    SOURCE_LABELS_CSV,
)
from mermaidseg.export import write_sidecar
from mermaidseg.inference.bundle import sha256_file
from mermaidseg.inference.preprocess import IMAGENET_MEAN, IMAGENET_STD
from mermaidseg.io import REPO_ROOT, resolve_repo_path, setup_config, update_config_with_args
from mermaidseg.lineage import check_same_data, collect_lineage, data_version, inherit_lineage
from mermaidseg.logger import Logger
from mermaidseg.model.distributed import (
    DistContext,
    ShardedBatchSampler,
    broadcast_generator_from_rank0,
    init_distributed_from_env,
    on_main_process,
)
from mermaidseg.model.eval import Evaluator
from mermaidseg.model.meta import MetaModel, next_epoch_from_checkpoint, restore_warmup_progress
from mermaidseg.model.train import run_epoch
from mermaidseg.run_setup import (
    MODEL_CHECKPOINT_KEYS,
    RESUME_CHECKPOINT_KEYS,
    build_meta_model,
    build_registry,
    check_env,
    load_model_weights,
    load_run_concept_source_mask,
    read_checkpoint,
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


# The run's copy of its label space, inside the run directory.
_RUN_LABEL_SPACE = "label_space"
_LABEL_SPACE_FILES = (CLASSES_CSV, CONCEPTS_CSV, SOURCE_LABELS_CSV, META_YAML)
_ACKNOWLEDGED_CSV = "acknowledged.csv"


def _label_space_files(source: Path) -> list[Path]:
    """The generated files of the label space ``source``, then ``acknowledged.csv`` if present."""
    files = [source / name for name in _LABEL_SPACE_FILES]
    acknowledged = source / _ACKNOWLEDGED_CSV
    return files + [acknowledged] if acknowledged.is_file() else files


def _lineage_inputs(
    args: argparse.Namespace, training_cfg: Any, section: DataSection
) -> list[Path]:
    """The files a run reads from the repo whose content lineage does not record.

    These are its training config, the files the training config names, the
    sidecars of its removal list and mask, and each dataset's split file. The label space
    contributes its listed files, so one untracked file fails the check. Empty config entries are
    skipped.
    """
    named = [
        training_cfg.concept_mapping_path,
        training_cfg.concept_removal,
        training_cfg.concept_source_mask,
    ]
    sidecars = [training_cfg.concept_removal, training_cfg.concept_source_mask]
    return [
        resolve_repo_path(args.training_config),
        *_label_space_files(resolve_repo_path(training_cfg.label_space)),
        *(resolve_repo_path(path) for path in named if path),
        *(meta_path(resolve_repo_path(path)) for path in sidecars if path),
        *(REPO_ROOT / e.split_file for e in section.datasets.values() if e.split_file is not None),
    ]


def _check_label_space_copy(files: list[Path], target: Path) -> None:
    """Raise ValueError if the existing directory ``target`` differs from the label space ``files``.

    An absent ``target`` passes. Reads only.
    """
    if not target.exists():
        return
    differing = [
        p.name
        for p in files
        if not (target / p.name).is_file() or sha256_file(target / p.name) != sha256_file(p)
    ]
    if differing:
        raise ValueError(
            f"the run's label space copy {target} differs from the configured label space "
            f"in {differing}; remove the copy or restore the label space"
        )


def _copy_label_space(files: list[Path], target: Path) -> dict[str, str]:
    """Copy the label space ``files`` into the directory ``target``; the sha256 of each copy.

    The copy is staged in ``.{target name}.partial`` beside ``target`` and renamed onto it, so a
    crash leaves no partial ``target``. An existing ``target`` (a resumed or relaunched run) is
    kept; the caller checks it against ``files`` first (:func:`_check_label_space_copy`).
    """
    if target.exists():
        return {path.name: sha256_file(target / path.name) for path in files}
    staging = target.with_name(f".{target.name}.partial")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir()
    for path in files:
        shutil.copy2(path, staging / path.name)
    os.replace(staging, target)
    return {path.name: sha256_file(target / path.name) for path in files}


def _assemble_train_datasets(
    dataset_dict: dict[tuple[str, str], Any], section: DataSection
) -> list[Any]:
    """Every train dataset, then each extra copy that ``repeat`` asks for.

    ``repeat: N`` on a train block is the total number of copies of that dataset in
    the result: the first sits among the train datasets and ``N - 1`` more are
    appended after all of them. A missing ``repeat`` means 1.

    Args:
        dataset_dict: Built datasets keyed by ``(name, split)``.
        section: The validated ``data`` section of the run.

    Returns:
        The train datasets in concatenation order.
    """
    train_items = [(name, ds) for (name, split), ds in dataset_dict.items() if split == "train"]
    datasets = [ds for _, ds in train_items]
    for name, ds in train_items:
        block = section.datasets[name].train
        repeat = block.repeat if block is not None and block.repeat is not None else 1
        datasets.extend([ds] * (repeat - 1))
    return datasets


# Environment-only config keys; lineage.versions records the transformers version.
_ENV_ENCODER_CONFIG_KEYS = ("transformers_version", "_name_or_path")


def _encoder_config(model: torch.nn.Module) -> dict[str, Any]:
    """Architecture config of the pretrained encoder under the PEFT wrapper of ``model.encoder``."""
    encoder: Any = model.encoder  # type: ignore[attr-defined]
    config = encoder.get_base_model().config.to_dict()
    return {k: v for k, v in config.items() if k not in _ENV_ENCODER_CONFIG_KEYS}


def _lineage_params(lineage: dict[str, Any]) -> dict[str, Any]:
    """Flat ``lineage/...`` tracker params from a lineage dict."""
    version = lineage["data_version"]
    params = {
        "lineage/git_sha": lineage["git_sha"],
        "lineage/git_dirty": lineage["git_dirty"],
        "lineage/uv_lock_sha256": lineage["uv_lock_sha256"],
    }
    for name, pin in version["datasets"].items():
        params[f"lineage/datasets/{name}/version"] = pin["version"]
        params[f"lineage/datasets/{name}/manifest_sha256"] = pin["manifest_sha256"]
    params["lineage/taxonomy/version"] = version["taxonomy"]["version"]
    params["lineage/taxonomy/manifest_sha256"] = version["taxonomy"]["manifest_sha256"]
    return params


# Exit status of a process ended by SIGTERM (128 + signal number).
SIGTERM_EXIT_CODE = 128 + signal.SIGTERM


@contextlib.contextmanager
def _sigterm_raises_system_exit(enabled: bool) -> Iterator[None]:
    """Turn SIGTERM into ``SystemExit(143)`` inside the block when ``enabled``.

    Python's default SIGTERM action ends the process without unwinding, so an enclosing
    ``with`` block never exits. Raising lets it unwind. The previous handler is restored
    on leaving.
    """
    if not enabled:
        yield
        return

    def _raise_exit(signum: int, frame: Any) -> None:
        # A group-wide SIGTERM also kills the DataLoader workers; torch's SIGCHLD handler would
        # then raise inside the unwinding tracker flush. A repeated SIGTERM must not cut it short.
        signal.signal(signal.SIGCHLD, signal.SIG_DFL)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        raise SystemExit(SIGTERM_EXIT_CODE)

    previous = signal.signal(signal.SIGTERM, _raise_exit)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


def run(args: argparse.Namespace) -> None:
    """Run one training job from parsed command-line arguments.

    Args:
        args: Parsed arguments from ``scripts/train.py``.
    """
    # -- 0. Environment --------------------------------------------------------
    # Data is read from --cache-dir, which scripts/sync_data.py fills from the pinned dataset
    # versions; metrics go to the tracker that the logger config names.
    check_env()

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
    # minibatch. Only rank 0 logs to the tracker, writes checkpoints, and prints.
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

    # Leave the config value untouched when a flag is not given.
    if args.grad_accum_steps is not None:
        cfg.training.grad_accum_steps = args.grad_accum_steps

    # Encoder, iterations, batch size, and epochs come from the model and training configs.

    # Snapshot before MetaModel construction so the Logger records the config as loaded.
    cfg_logger = copy.deepcopy(cfg)

    # -- 2. Datasets -----------------------------------------------------------
    cache_dir = Path(args.cache_dir)
    section = DataSection.model_validate(cfg.data)
    if args.checkpoint:
        assert checkpoint_data is not None
        # A resume continues one run on one dataset; --init-weights may change data.
        check_same_data(data_version(section), checkpoint_data["lineage"])
    dataset_dict = build_datasets(
        section,
        cache_dir,
        repo_root=REPO_ROOT,
        padding=cfg.training.padding,
        input_size=cfg.model.input_size,
    )

    loader_kwargs = {
        "batch_size": cfg.training.batch_size,
        # Under torchrun the worker count is per rank.
        "num_workers": args.num_workers,
        "pin_memory": True,
        "persistent_workers": True,
        "drop_last": True,
        "prefetch_factor": cfg.training.loader.prefetch_factor,
    }
    loader_kwargs["worker_init_fn"] = make_worker_seeder(dist_ctx.rank)

    # -- 3. Registry / model / evaluator --------------------------------------
    registry, schema = build_registry(
        dataset_dict,
        cfg.training,
        section=section,
        device=device,
    )
    print0(
        dist_ctx,
        f"Classes and concept channels from {cfg.training.label_space} "
        f"({registry.num_target_classes - 1} classes, {schema.num_channels} channels)",
    )

    train_datasets = _assemble_train_datasets(dataset_dict, section)
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
    run_dir = checkpoint_dir(args.output_dir, cfg.run_name)
    label_space_sha256: dict[str, str] = {}
    label_space_files = _label_space_files(resolve_repo_path(cfg.training.label_space))
    _check_label_space_copy(label_space_files, run_dir / _RUN_LABEL_SPACE)
    if dist_ctx.is_main:
        run_dir.mkdir(parents=True, exist_ok=True)
        write_resolved_config(cfg_logger, run_dir)
        label_space_sha256 = _copy_label_space(label_space_files, run_dir / _RUN_LABEL_SPACE)

    meta_model = build_meta_model(cfg, registry, device)
    if concept_source_mask is not None:
        meta_model.set_concept_source_mask(concept_source_mask.keep)

    sidecar: dict[str, Any] = {}
    lineage: dict[str, Any] | None = None
    if dist_ctx.is_main:
        lineage = collect_lineage(
            resolved_config=to_plain_dict(cfg_logger),
            seed=seed,
            data_version=data_version(section),
            inputs=_lineage_inputs(args, cfg.training, section),
        )
        if checkpoint_data is not None:
            lineage = inherit_lineage(lineage, checkpoint_data["lineage"])
        sidecar_path = write_sidecar(
            run_dir,
            run_name=cfg.run_name,
            model_class=meta_model.model_name,
            model_kwargs=meta_model.model_kwargs,
            encoder_config=_encoder_config(meta_model.model),
            id2label={0: "ignore", **registry.target_id2label},
            concept_id2name=concept_id2name,
            input_size=cfg.model.input_size,
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
            label_space=label_space_sha256,
            lineage=lineage,
        )
        sidecar = json.loads(sidecar_path.read_text())
        print(f"Wrote {sidecar_path} ({len(concept_id2name)} concept channels)")

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
    # Release the host copy of the checkpoint before the loaders fork their workers.
    del checkpoint_data

    evaluator = Evaluator(
        num_classes=registry.num_target_classes,
        device=device,
        concept_value2id=registry.concept_value2id,
    )
    # -- 5. Train (run lifecycle managed by `with`) ---------------------------
    # The logger config's `tracker` key selects the backend. Under torchrun only rank 0 has a
    # Logger; the other ranks train with logger=None and write nothing.
    logger_context: contextlib.AbstractContextManager[Logger | None]
    if dist_ctx.is_main:
        logger_context = Logger(
            config=cfg_logger,
            meta_model=meta_model,
            id2label={0: "ignore", **registry.target_id2label},
        )
    else:
        logger_context = contextlib.nullcontext(None)
    # SIGTERM (SageMaker stop, Slurm) must unwind `logger_context` so the tracker run is ended
    # as failed. Only rank 0 owns a tracker run. This holds for single-process runs; under
    # torchrun (multi-GPU) rank 0 is typically blocked in a collective when its peers die, so
    # its handler never runs and torchrun SIGKILLs it after 30 s, leaving the run RUNNING.
    with _sigterm_raises_system_exit(dist_ctx.is_main), logger_context as run_logger:
        if run_logger is not None:
            run_logger.log_dict(sidecar, "metadata/model.json")
            run_logger.log_params(_lineage_params(sidecar["lineage"]))
            run_logger.log_params(
                {
                    "model/encoder_name": cfg.model.encoder_name,
                    "model/head": "dpt",
                    "model/adapter": "lora",
                    "model/lora_r": cfg.model.lora_r,
                    "model/lora_alpha": cfg.model.lora_alpha,
                    "loss/per_pixel_loss_weight": cfg.training.loss.per_pixel_loss_weight,
                    "loss/per_image_loss_weight": cfg.training.loss.per_image_loss_weight,
                    "loss/kl_weight": cfg.training.taxonomic_loss.kl_weight,
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
            print(f"\ntracker run id : {run_logger.run_id}")
            print(f"tracker run URL: {run_logger.run_url}\n")

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
                write_interval = step % checkpoint_every == 0
                write_preempt = preempt_every > 0 and step % preempt_every == 0
                if not write_interval and not write_preempt:
                    return

                def _write() -> None:
                    assert lineage is not None  # on_main_process runs only on rank 0
                    save = functools.partial(
                        write_checkpoint,
                        meta_model=meta,
                        config=cfg_logger,
                        lineage=lineage,
                        epoch=epoch,
                        epoch_complete=step >= total_updates,
                        metrics={},
                    )
                    if write_interval:
                        save(run_dir / f"model_step{step}")
                    # One rolling file for preemption. epoch_complete stays false until the
                    # run is finished, so a resume reseeds with seed + global_step and
                    # continues this epoch. Older model_preempt* files are deleted after
                    # the rename; model_step* files are not.
                    if write_preempt:
                        name = f"model_preempt{step}"
                        save(run_dir / name, atomic=True)
                        prune_rolling_checkpoints(run_dir, keep_name=name)

                on_main_process(_write)

            metrics_all[epoch] = run_epoch(
                meta_model,
                evaluator,
                train_loader,
                val_loader,
                run_logger,
                epoch=epoch,
                train_log_interval=cfg.logger.train_log_interval,
                on_optimizer_step=_save_step_checkpoint,
            )
        if not metrics_all:
            print0(dist_ctx, "No training epochs ran.")
        else:
            final_epoch = max(metrics_all)

            # Skipped windows leave global_step short of the schedule, so no interval step
            # checkpoint is guaranteed to cover the last updates; this one always does.
            def _write_final() -> None:
                assert lineage is not None  # on_main_process runs only on rank 0
                write_checkpoint(
                    run_dir / "model_final",
                    meta_model,
                    config=cfg_logger,
                    lineage=lineage,
                    epoch=final_epoch,
                    epoch_complete=True,
                    metrics={},
                    atomic=True,
                )

            on_main_process(_write_final)
            print0(
                dist_ctx, "Final train metrics     :", metrics_all[final_epoch].get("train_metrics")
            )
            print0(
                dist_ctx,
                "Final validation metrics:",
                metrics_all[final_epoch].get("validation_metrics"),
            )
        if run_logger is not None:
            print(f"tracker run URL: {run_logger.run_url}")

    if dist_ctx.distributed:
        dist.destroy_process_group()
