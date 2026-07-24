from __future__ import annotations

import io

import numpy as np
import pytest
from botocore.exceptions import ClientError
from PIL import Image

from mermaidseg.datasets.mermaid.mermaid_dataset import MermaidDataset


class _S3:
    def __init__(self, jpeg: bytes, available_extension: str):
        self.jpeg = jpeg
        self.available_extension = available_extension
        self.requested_keys: list[str] = []

    def get_object(self, **kwargs):
        key = kwargs["Key"]
        self.requested_keys.append(key)
        if key.endswith(self.available_extension):
            return {"Body": io.BytesIO(self.jpeg)}
        raise ClientError(
            {"Error": {"Code": "NoSuchKey", "Message": "missing"}},
            "GetObject",
        )


def _jpeg_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (16, 12), color=(20, 40, 60)).save(buffer, format="JPEG")
    return buffer.getvalue()


@pytest.mark.parametrize("available_extension", [".jpg", ".jpeg"])
def test_mermaid_read_image_resolves_alternate_object_extension(available_extension):
    dataset = object.__new__(MermaidDataset)
    dataset.source_bucket = "bucket"
    dataset.s3 = _S3(_jpeg_bytes(), available_extension)

    image = dataset.read_image("image-id")

    assert isinstance(image, np.ndarray)
    assert image.shape == (12, 16, 3)
    attempted_extensions = [".png", ".jpg"]
    if available_extension == ".jpeg":
        attempted_extensions.append(".jpeg")
    assert dataset.s3.requested_keys == [
        f"mermaid/image-id{extension}" for extension in attempted_extensions
    ]
