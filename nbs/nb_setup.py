"""Environment checks for the wandb-backed training pipeline.

Import at the top of the training script so a missing wandb login surfaces
before training starts::

    from nb_setup import check_env_wandb
    check_env_wandb()
"""

import os


def check_wandb_auth() -> None:
    """Confirm wandb is authenticated in this environment.

    Checks for a resolved API key (via ``WANDB_API_KEY``, ``wandb login``, or
    ``~/.netrc``) without starting a run. Call before
    ``Logger(..., enable_wandb=True)`` so a missing login surfaces immediately
    instead of mid-training.

    Raises:
        RuntimeError: If no wandb API key can be resolved.
    """
    import wandb

    api_key = wandb.api.api_key
    if not api_key:
        raise RuntimeError(
            "wandb is not authenticated. Run `wandb login` in a terminal, "
            "set WANDB_API_KEY, or call wandb.login(key=...) in a notebook cell."
        )
    print(f"wandb authenticated (API key ...{api_key[-4:]})")


def check_env_wandb() -> None:
    """Validate the environment for wandb-backed, AWS-free training runs.

    Prints the resolved wandb / HF / cache-offline settings and confirms wandb
    authentication (via ``check_wandb_auth``).

    Raises:
        RuntimeError: If wandb cannot resolve an API key.
    """
    print(f"WANDB_PROJECT       : {os.getenv('WANDB_PROJECT', '(not set — using config default)')}")
    print(f"WANDB_ENTITY        : {os.getenv('WANDB_ENTITY', '(not set — using default entity)')}")
    print(f"WANDB_DIR           : {os.getenv('WANDB_DIR', '(not set — using ./wandb)')}")
    print(f"HF_TOKEN            : {'set' if os.getenv('HF_TOKEN') else '(not set)'}")
    offline = os.getenv("MERMAIDSEG_S3_OFFLINE")
    print(
        "MERMAIDSEG_S3_OFFLINE: "
        + (f"{offline} (cache misses will raise)" if offline else "(not set — S3 fallback enabled)")
    )
    check_wandb_auth()
