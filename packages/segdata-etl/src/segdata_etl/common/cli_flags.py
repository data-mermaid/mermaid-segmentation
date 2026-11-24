"""Argparse flag groups shared by the ``segdata`` commands."""

from __future__ import annotations

import argparse


def _pin_flags(parser: argparse.ArgumentParser, prefix: str, what: str) -> None:
    parser.add_argument(f"--{prefix}version", required=True, help=f"Version of {what}")
    parser.add_argument(
        f"--{prefix}manifest-sha256", required=True, help=f"Manifest sha256 of {what}"
    )
