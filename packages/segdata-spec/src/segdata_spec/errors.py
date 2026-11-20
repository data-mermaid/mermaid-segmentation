"""Exceptions raised by segdata-spec. Callers catch :class:`SegdataError`."""

from __future__ import annotations


class SegdataError(Exception):
    """Base class for every segdata-spec failure."""


class PublishRefused(SegdataError):
    """Publishing is refused: dirty or unpushed tree, or the version already exists."""


class ObjectExists(SegdataError):
    """A write-once S3 key already exists with different bytes."""


class PinMismatch(SegdataError):
    """A manifest's sha256 differs from the pin that names it."""


class ChecksumMismatch(SegdataError):
    """A file's content differs from the checksum or size recorded for it."""


class ImageChanged(SegdataError):
    """An image object no longer matches the ETag recorded in ``images.parquet``."""


class ImageMissing(SegdataError):
    """An image object listed in ``images.parquet`` is absent from its store."""


class SchemaValidationError(SegdataError, ValueError):
    """A table does not satisfy its declared schema."""
