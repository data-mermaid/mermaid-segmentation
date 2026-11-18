"""Tests for the S3 helpers."""

from __future__ import annotations

import io
from unittest.mock import MagicMock

import pandas as pd
import pytest
from botocore.exceptions import ClientError
from coralnet_scraper import s3_io
from pandas.testing import assert_frame_equal


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
