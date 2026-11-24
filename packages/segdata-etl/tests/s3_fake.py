"""In-memory S3 client for the segdata-etl tests, and the ETag and checksum values S3 reports.

Imported as ``s3_fake``: pytest's default import mode puts ``tests/`` on ``sys.path``.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from botocore.exceptions import ClientError

from segdata_spec.sync import _crc64nvme_b64


def client_error(code: str, status: int, op: str) -> ClientError:
    return ClientError(
        {"Error": {"Code": code, "Message": code}, "ResponseMetadata": {"HTTPStatusCode": status}},
        op,
    )


def md5_hex(data: bytes) -> str:
    """The unquoted MD5 hex digest of ``data``, the ETag of a single-part upload."""
    return hashlib.md5(data, usedforsecurity=False).hexdigest()


def quoted_etag(data: bytes, *, multipart: bool = False) -> str:
    """The quoted ETag S3 gives ``data``: its MD5, suffixed ``-2`` for a two-part upload."""
    suffix = "-2" if multipart else ""
    return f'"{md5_hex(data)}{suffix}"'


def crc64nvme(data: bytes) -> str:
    """The base64 full-object CRC64NVME S3 reports for ``data``."""
    return _crc64nvme_b64(data)


class _Body:
    def __init__(self, data: bytes):
        self._data = data

    def read(self) -> bytes:
        return self._data


def _as_bytes(body: Any) -> bytes:
    if isinstance(body, str):
        return body.encode("utf-8")
    if isinstance(body, bytes | bytearray | memoryview):
        return bytes(body)
    raw = body.read()
    return raw.encode("utf-8") if isinstance(raw, str) else bytes(raw)


@dataclass
class FakeS3:
    """In-memory S3 client; ``objects`` maps ``(bucket, key)`` to bytes.

    ``calls`` logs every operation as ``(op, bucket, key_or_prefix, kwargs)``; ``ops(op)`` filters
    it. ``etags`` and ``crcs`` override the ETag and CRC64NVME computed from an object's bytes;
    ``multipart`` holds the objects whose ETag is a two-part upload's. An object has a CRC64NVME
    unless it is in ``no_crc``: a write through ``put_object`` or ``copy_object`` stores one only
    when it passes ``ChecksumAlgorithm="CRC64NVME"``, as S3 does, and ``put`` seeds one unless
    told ``crc=False``; objects placed straight into ``objects`` have one. ``put_object`` and
    ``copy_object`` responses carry ``ChecksumCRC64NVME`` when the object has one.

    ``Range`` on ``get_object`` follows ``bytes=start-end`` (inclusive, end optional) and
    ``IfMatch`` raises 412 on a different ETag; ``full_reads`` records the key of every
    ``get_object`` without a ``Range``. ``copy_calls`` holds the keyword arguments of every
    ``copy_object``; ``copy_error`` makes every ``copy_object`` raise that error code with status
    403. ``source_changes_on_copy`` maps a source ``(bucket, key)`` to the bytes it holds by the
    time ``copy_object`` reads it, as a source overwritten between the HEAD and the copy does.
    ``page_size`` caps the entries of a ``list_objects_v2`` page below ``MaxKeys`` and issues
    ``NextContinuationToken``; ``None`` leaves ``MaxKeys`` as the cap.
    """

    objects: dict[tuple[str, str], bytes] = field(default_factory=dict)
    crcs: dict[tuple[str, str], str] = field(default_factory=dict)
    etags: dict[tuple[str, str], str] = field(default_factory=dict)
    calls: list[tuple[str, str, str, dict[str, Any]]] = field(default_factory=list)
    full_reads: list[str] = field(default_factory=list)
    multipart: set[tuple[str, str]] = field(default_factory=set)
    no_crc: set[tuple[str, str]] = field(default_factory=set)
    copy_calls: list[dict[str, Any]] = field(default_factory=list)
    copy_error: str | None = None
    source_changes_on_copy: dict[tuple[str, str], bytes] = field(default_factory=dict)
    page_size: int | None = None

    @property
    def puts(self) -> list[str]:
        """The key of every ``put_object`` and ``upload_file``, in call order."""
        return [key for op, _, key, _ in self.calls if op == "put"]

    def ops(self, op: str) -> list[tuple[str, str, str, dict[str, Any]]]:
        return [call for call in self.calls if call[0] == op]

    def etag(self, bucket: str, key: str) -> str:
        """The quoted ETag S3 gives the object."""
        if (bucket, key) in self.etags:
            return self.etags[(bucket, key)]
        return quoted_etag(self.objects[(bucket, key)], multipart=(bucket, key) in self.multipart)

    def crc(self, bucket: str, key: str) -> str | None:
        """The object's CRC64NVME, or ``None`` when it was written without one."""
        if (bucket, key) in self.no_crc:
            return None
        return self.crcs.get((bucket, key)) or crc64nvme(self.objects[(bucket, key)])

    def put(
        self, bucket: str, key: str, data: bytes, *, etag: str | None = None, crc: bool = True
    ) -> None:
        """Seed an object without logging a call."""
        location = (bucket, key)
        self.objects[location] = data
        self.multipart.discard(location)
        self.crcs.pop(location, None)
        if etag is None:
            self.etags.pop(location, None)
        else:
            self.etags[location] = etag
        if crc:
            self.no_crc.discard(location)
            self.crcs[location] = crc64nvme(data)
        else:
            self.no_crc.add(location)

    def put_object(self, *, Bucket: str, Key: str, Body: Any, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("put", Bucket, Key, dict(kwargs)))
        if kwargs.get("IfNoneMatch") == "*" and (Bucket, Key) in self.objects:
            raise client_error("PreconditionFailed", 412, "PutObject")
        self.put(Bucket, Key, _as_bytes(Body), crc=kwargs.get("ChecksumAlgorithm") == "CRC64NVME")
        response: dict[str, Any] = {"ETag": self.etag(Bucket, Key)}
        if (crc := self.crc(Bucket, Key)) is not None:
            response["ChecksumCRC64NVME"] = crc
        return response

    def upload_file(self, Filename: str, Bucket: str, Key: str) -> None:
        self.calls.append(("put", Bucket, Key, {}))
        with open(Filename, "rb") as f:
            self.put(Bucket, Key, f.read(), crc=False)

    def copy_object(
        self, *, CopySource: dict[str, str], Bucket: str, Key: str, **kwargs: Any
    ) -> dict[str, Any]:
        self.copy_calls.append({"CopySource": CopySource, "Bucket": Bucket, "Key": Key, **kwargs})
        self.calls.append(("copy", Bucket, Key, {"CopySource": CopySource, **kwargs}))
        if self.copy_error is not None:
            raise client_error(self.copy_error, 403, "CopyObject")
        source = (CopySource["Bucket"], CopySource["Key"])
        if source in self.source_changes_on_copy:
            self.objects[source] = self.source_changes_on_copy[source]
        if source not in self.objects:
            raise client_error("NoSuchKey", 404, "CopyObject")
        wanted = kwargs.get("CopySourceIfMatch")
        # S3 accepts the ETag with or without its quotes.
        if wanted is not None and wanted.strip('"') != self.etag(*source).strip('"'):
            raise client_error("PreconditionFailed", 412, "CopyObject")
        if kwargs.get("IfNoneMatch") == "*" and (Bucket, Key) in self.objects:
            raise client_error("PreconditionFailed", 412, "CopyObject")
        # A single CopyObject writes a single-part object, so a multipart source ETag is not
        # carried over.
        self.put(
            Bucket, Key, self.objects[source], crc=kwargs.get("ChecksumAlgorithm") == "CRC64NVME"
        )
        result: dict[str, Any] = {"ETag": self.etag(Bucket, Key)}
        if (crc := self.crc(Bucket, Key)) is not None:
            result["ChecksumCRC64NVME"] = crc
        return {"CopyObjectResult": result}

    def head_object(self, *, Bucket: str, Key: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("head", Bucket, Key, dict(kwargs)))
        if (Bucket, Key) not in self.objects:
            raise client_error("404", 404, "HeadObject")
        response: dict[str, Any] = {
            "ContentLength": len(self.objects[(Bucket, Key)]),
            "ETag": self.etag(Bucket, Key),
            "LastModified": datetime(2026, 8, 12, 6, 30, tzinfo=UTC),
        }
        crc = self.crc(Bucket, Key)
        if kwargs.get("ChecksumMode") == "ENABLED" and crc is not None:
            response["ChecksumCRC64NVME"] = crc
        return response

    def get_object(self, *, Bucket: str, Key: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("get", Bucket, Key, dict(kwargs)))
        if (Bucket, Key) not in self.objects:
            raise client_error("NoSuchKey", 404, "GetObject")
        full = self.objects[(Bucket, Key)]
        etag = self.etag(Bucket, Key)
        if kwargs.get("IfMatch") not in (None, etag):
            raise client_error("PreconditionFailed", 412, "GetObject")
        range_header = kwargs.get("Range")
        if not range_header:
            self.full_reads.append(Key)
            return {"ETag": etag, "Body": _Body(full), "ContentLength": len(full)}
        assert range_header.startswith("bytes=")
        start_str, end_str = range_header[len("bytes=") :].split("-")
        start = int(start_str)
        end = min(int(end_str), len(full) - 1) if end_str else len(full) - 1
        data = full[start : end + 1]
        return {
            "ETag": etag,
            "Body": _Body(data),
            "ContentLength": len(data),
            "ContentRange": f"bytes {start}-{end}/{len(full)}",
        }

    def list_objects_v2(
        self,
        *,
        Bucket: str,
        Prefix: str = "",
        Delimiter: str | None = None,
        MaxKeys: int = 1000,
        ContinuationToken: str | None = None,
    ) -> dict[str, Any]:
        """One page; ``Contents`` and ``CommonPrefixes`` are absent when empty, as in S3.

        The token is the index of the page's first entry among all entries, in key order.
        """
        extras = {"Delimiter": Delimiter, "ContinuationToken": ContinuationToken}
        logged = {name: value for name, value in extras.items() if value is not None}
        self.calls.append(("list", Bucket, Prefix, {"MaxKeys": MaxKeys, **logged}))
        entries: list[tuple[str, str]] = []
        folders: set[str] = set()
        for bucket, key in sorted(self.objects):
            if bucket != Bucket or not key.startswith(Prefix):
                continue
            rest = key[len(Prefix) :]
            if Delimiter and Delimiter in rest:
                folder = Prefix + rest.split(Delimiter, 1)[0] + Delimiter
                if folder not in folders:
                    folders.add(folder)
                    entries.append(("folder", folder))
            else:
                entries.append(("object", key))
        start = int(ContinuationToken) if ContinuationToken else 0
        limit = MaxKeys if self.page_size is None else min(self.page_size, MaxKeys)
        end = start + limit
        contents = [
            {
                "Key": key,
                "ETag": self.etag(Bucket, key),
                "Size": len(self.objects[(Bucket, key)]),
            }
            for kind, key in entries[start:end]
            if kind == "object"
        ]
        prefixes = [{"Prefix": name} for kind, name in entries[start:end] if kind == "folder"]
        response: dict[str, Any] = {
            "KeyCount": len(contents) + len(prefixes),
            "IsTruncated": end < len(entries),
        }
        if end < len(entries):
            response["NextContinuationToken"] = str(end)
        if contents:
            response["Contents"] = contents
        if prefixes:
            response["CommonPrefixes"] = prefixes
        return response

    def get_paginator(self, operation: str) -> _Paginator:
        if operation != "list_objects_v2":
            raise NotImplementedError(operation)
        return _Paginator(self)


class _Paginator:
    def __init__(self, client: FakeS3):
        self._client = client

    def paginate(self, **kwargs: Any) -> Iterator[dict[str, Any]]:
        token: str | None = None
        while True:
            page = self._client.list_objects_v2(**kwargs, ContinuationToken=token)
            yield page
            if not page["IsTruncated"]:
                return
            token = page["NextContinuationToken"]
