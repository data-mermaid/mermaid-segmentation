"""Tests for the S3 helpers."""

from __future__ import annotations

import io
from unittest.mock import MagicMock

import pandas as pd
import pytest
from botocore.exceptions import ClientError
from pandas.testing import assert_frame_equal

from coralnet_scraper import s3_io


def _client_error(code: str) -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": "m"}}, "GetObject")


def test_read_csv_s3_returns_dataframe():
    client = MagicMock()
    client.get_object.return_value = {"Body": io.BytesIO(b"a,b\n1,x\n2,y\n")}
    df = s3_io.read_csv_s3(client, "bkt", "k.csv")
    assert_frame_equal(df, pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}))
    client.get_object.assert_called_once_with(Bucket="bkt", Key="k.csv")


@pytest.mark.parametrize("code", ["404", "NoSuchKey"])
def test_read_csv_s3_returns_none_when_key_missing(code):
    client = MagicMock()
    client.get_object.side_effect = _client_error(code)
    assert s3_io.read_csv_s3(client, "bkt", "k.csv") is None


def test_read_csv_s3_propagates_other_client_errors():
    client = MagicMock()
    client.get_object.side_effect = _client_error("AccessDenied")
    with pytest.raises(ClientError):
        s3_io.read_csv_s3(client, "bkt", "k.csv")


@pytest.mark.parametrize(
    ("func", "args", "expected"),
    [
        (
            s3_io.object_key_image_list_csv,
            ("coralnet-public-images", 7),
            "coralnet-public-images/s7/image_list.csv",
        ),
        (
            s3_io.object_key_labelset_csv,
            ("coralnet-public-images", 7),
            "coralnet-public-images/s7/labelset.csv",
        ),
        (
            s3_io.object_key_metadata_csv,
            ("coralnet-public-images", 7),
            "coralnet-public-images/s7/metadata.csv",
        ),
        (s3_io.object_key_image, ("p", 7, "123"), "p/s7/images/123.jpg"),
    ],
)
def test_object_key_helpers(func, args, expected):
    assert func(*args) == expected


@pytest.mark.parametrize(
    ("base_prefix", "expected"),
    [
        ("/coralnet-public-images/", "coralnet-public-images/s7/"),
        ("a/b//", "a/b/s7/"),
        ("plain", "plain/s7/"),
    ],
)
def test_source_prefix_strips_surrounding_slashes_from_base_prefix(base_prefix, expected):
    assert s3_io.source_prefix(base_prefix, 7) == expected
