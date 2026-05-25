"""Phase 1 (scan) and Phase 2 (resize) image preprocessing pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ResizeConfig:
    """Configuration for image resizing."""

    bucket: str
    output_prefix: str
    threshold: int = 2048
    workers: int = 16
    checkpoint_every: int = 500
    temp_dir: str = "/tmp/coralnet-resize-checkpoint"
