"""In-container entrypoint for a mermaid-segmentation ProcessingJob.

ProcessingJobs are diverse (eval, inference, batch transform, mask generation, etc.).
This entrypoint reads the run YAML, dispatches on the `processing:` block's
`--task=<name>` container_arg, and routes to the matching subroutine in mermaidseg/. Add
new tasks as the team needs them.
"""

from __future__ import annotations

import argparse
import logging
import sys

log = logging.getLogger("seg_processing_entrypoint")


def _configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )


def main():
    _configure_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        required=True,
        choices=["eval", "inference", "coralnet-etl"],
        help="Which processing routine to run.",
    )
    # Pass-through args specific to each task.
    args, extra = parser.parse_known_args()
    log.info("Running processing task=%s extra=%s", args.task, extra)
    if args.task == "eval":
        from mermaidseg.eval import run_eval  # implemented in mermaidseg

        run_eval(extra)
    elif args.task == "inference":
        from mermaidseg.inference import run_inference

        run_inference(extra)
    elif args.task == "coralnet-etl":
        from mermaidseg.datasets.coralnet.etl.__main__ import main as etl_main

        etl_main(extra)
    else:
        raise SystemExit(f"Unknown task: {args.task}")


if __name__ == "__main__":
    main()
