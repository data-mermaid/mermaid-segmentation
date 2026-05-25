"""CoralNet image resizing preprocessing pipeline.

Resize images exceeding 2048px longest edge, store on S3, and maintain checkpoint/manifest.
"""

try:
    from .manifest import build_manifest
except ImportError:
    build_manifest = None

from .resize import run_phase_1_scan, run_phase_2_resize

__all__ = ["run_phase_1_scan", "run_phase_2_resize", "build_manifest"]
