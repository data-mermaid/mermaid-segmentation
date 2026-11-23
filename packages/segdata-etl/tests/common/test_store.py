"""``open_store`` and ``add_store_argument``: the shared ``--store`` option."""

from __future__ import annotations

import argparse
from pathlib import Path

from segdata_etl.common.local_store import LocalStore
from segdata_etl.common.store import DEFAULT_STORE, add_store_argument, open_store
from segdata_spec import DATASET_BUCKET


def test_open_store_directory_gives_a_local_store_on_the_dataset_bucket(tmp_path: Path) -> None:
    client, bucket = open_store(tmp_path, source_client=object())
    assert isinstance(client, LocalStore)
    assert bucket == "mermaid-ic-datasets"
    client.put_object(Bucket=bucket, Key="a/b.txt", Body=b"x")
    assert (tmp_path / "mermaid-ic-datasets" / "a" / "b.txt").read_bytes() == b"x"


def test_open_store_s3_returns_the_source_client_itself() -> None:
    source = object()
    client, bucket = open_store("s3", source_client=source)
    assert client is source
    assert bucket == DATASET_BUCKET


def test_store_argument_defaults_to_the_local_directory() -> None:
    parser = argparse.ArgumentParser()
    add_store_argument(parser)
    assert parser.parse_args([]).store == DEFAULT_STORE
