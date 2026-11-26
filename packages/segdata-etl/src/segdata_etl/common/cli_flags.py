"""Argparse flag groups shared by the ``segdata`` commands."""

from __future__ import annotations

import argparse


def pin_flags(
    parser: argparse.ArgumentParser, prefix: str, what: str, *, required: bool = True
) -> None:
    """Add the required ``--<prefix>version`` and ``--<prefix>manifest-sha256`` pin flags."""
    parser.add_argument(f"--{prefix}version", required=required, help=f"Version of {what}")
    parser.add_argument(
        f"--{prefix}manifest-sha256", required=required, help=f"Manifest sha256 of {what}"
    )


def pin_values(args: argparse.Namespace, prefix: str) -> tuple[str | None, str | None]:
    """The ``(version, manifest sha256)`` parsed for the ``--<prefix>version`` pin flags."""
    attr = prefix.replace("-", "_")
    return getattr(args, f"{attr}version"), getattr(args, f"{attr}manifest_sha256")


def pin_flag_names(prefix: str) -> str:
    """``--<prefix>version and --<prefix>manifest-sha256``, for usage errors."""
    return f"--{prefix}version and --{prefix}manifest-sha256"


def half_pin_error(args: argparse.Namespace, prefix: str) -> str | None:
    """A usage error when only one of the pin's two flags is given, else None."""
    version, sha = pin_values(args, prefix)
    if (version is None) != (sha is None):
        return f"{pin_flag_names(prefix)} go together"
    return None


def pin_command_parts(args: argparse.Namespace, prefix: str) -> list[str]:
    """``--<prefix>version <v> --<prefix>manifest-sha256 <sha>`` for an ``etl_command``."""
    version, sha = pin_values(args, prefix)
    return [f"--{prefix}version", str(version), f"--{prefix}manifest-sha256", str(sha)]
