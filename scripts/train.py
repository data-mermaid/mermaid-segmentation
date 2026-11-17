"""Command-line entrypoint for the concept-bottleneck training pipeline."""

import argparse

from mermaidseg.io import get_parser
from mermaidseg.training import run


def build_parser() -> argparse.ArgumentParser:
    """Build the training argument parser.

    Returns:
        The shared ``mermaidseg.io`` parser extended with the trainer's options.
    """
    # CLI overrides used by the slurm/launch_*.sh experiment launchers.
    parser = get_parser()
    parser.add_argument(
        "--training-config",
        default="configs/training_config_cbm_final.yaml",
        help="path to the training config YAML",
    )
    parser.add_argument(
        "--model-config",
        default="configs/model_config_cbm_dpt_lora_vitl.yaml",
        help="path to the model config YAML (default: rank-16 ViT-L LoRA)",
    )
    parser.add_argument(
        "--data-config",
        default="configs/data_config.yaml",
        help="path to the data config YAML (default: train split, deduplicated CoralNet splits)",
    )
    parser.add_argument(
        "--concept-schema-config",
        default="configs/data_config.yaml",
        help=(
            "Data config whose configured datasets freeze the concept channels, so every run "
            "shares one layout whatever subset of the data it trains on."
        ),
    )
    parser.add_argument(
        "--logger-config",
        default="configs/logger_config.yaml",
        help="path to the logger config YAML",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=256,
        help="square train crop and val resize in pixels; also sets model.input_size",
    )
    parser.add_argument("--per-pixel-loss-weight", type=float, default=None)
    parser.add_argument("--per-image-loss-weight", type=float, default=None)
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=None,
        help="microbatches per optimizer step (default: training config, usually 1)",
    )
    parser.add_argument(
        "--preempt-checkpoint-every-steps",
        type=int,
        default=0,
        help=(
            "Also write one rolling model_preempt{step} checkpoint every N optimizer "
            "updates (0 disables). model_step / model_epoch interval checkpoints are unchanged."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="path to a checkpoint to resume from",
    )
    parser.add_argument(
        "--init-weights",
        default=None,
        help=(
            "path to a checkpoint whose model weights initialize a new run. The "
            "optimizer and warmup schedule stay as just constructed. Mutually "
            "exclusive with --checkpoint"
        ),
    )
    parser.set_defaults(run_name="mermaid_base_run_dinov3_lora_dpt_all")
    return parser


def main() -> None:
    """Parse arguments, validate them, and run training."""
    parser = build_parser()
    args = parser.parse_args()
    if args.preempt_checkpoint_every_steps < 0:
        parser.error("--preempt-checkpoint-every-steps must be >= 0")
    if args.checkpoint is not None and args.init_weights is not None:
        parser.error("pass only one of --checkpoint and --init-weights")
    run(args)


if __name__ == "__main__":
    main()
