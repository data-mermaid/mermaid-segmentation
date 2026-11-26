"""Command-line entrypoint for the concept-bottleneck training pipeline."""

from mermaidseg.io import build_train_parser, check_train_args
from mermaidseg.training import run


def main() -> None:
    """Parse arguments, validate them, and run training."""
    parser = build_train_parser()
    args = parser.parse_args()
    check_train_args(parser, args)
    run(args)


if __name__ == "__main__":
    main()
