"""Phase 1 (scan) and Phase 2 (resize) image preprocessing pipeline."""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass

from PIL import Image

logger = logging.getLogger(__name__)


def resize_image_to_threshold(
    image_bytes: io.BytesIO,
    threshold: int = 2048,
) -> io.BytesIO:
    """Resize image so longest edge = threshold, maintaining aspect ratio.

    If longest edge <= threshold, returns original bytes unchanged.

    Args:
        image_bytes: Seeked BytesIO with JPEG image data
        threshold: Target longest edge (pixels)

    Returns:
        BytesIO with resized JPEG (or original if no resize needed)
    """
    img = Image.open(image_bytes)
    img.load()

    width, height = img.size
    longest_edge = max(width, height)

    # No resize needed
    if longest_edge <= threshold:
        image_bytes.seek(0)
        return image_bytes

    # Calculate new dimensions
    scale = threshold / longest_edge
    new_width = int(width * scale)
    new_height = int(height * scale)

    # Resize
    img_resized = img.resize((new_width, new_height), Image.LANCZOS)

    # Save to BytesIO
    output = io.BytesIO()
    img_resized.save(output, format="JPEG", quality=95)
    output.seek(0)
    return output


@dataclass
class ResizeConfig:
    """Configuration for image resizing."""

    bucket: str
    output_prefix: str
    threshold: int = 2048
    workers: int = 16
    checkpoint_every: int = 500
    temp_dir: str = "/tmp/coralnet-resize-checkpoint"
