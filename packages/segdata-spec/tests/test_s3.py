import pytest
from botocore.exceptions import ClientError
from spec_helpers import FakeS3, _client_error

from segdata_spec.s3 import exists, is_not_found


def _error(code: str, status: int) -> ClientError:
    return _client_error(code, status, "HeadObject")


@pytest.mark.parametrize(
    ("code", "status", "expected"),
    [
        ("404", 404, True),
        ("NoSuchKey", 404, True),
        ("NotFound", 404, True),
        ("", 404, True),
        ("NoSuchBucket", 404, False),
        ("AccessDenied", 403, False),
        ("InternalError", 500, False),
    ],
)
def test_is_not_found_classifies_the_error_code_and_status(code, status, expected):
    assert is_not_found(_error(code, status)) is expected


def test_exists_is_true_for_a_present_key():
    client = FakeS3()
    client.put("b", "k", b"x")
    assert exists(client, "b", "k") is True


def test_exists_is_false_for_a_missing_key():
    assert exists(FakeS3(), "b", "k") is False


def test_exists_propagates_other_client_errors():
    class Denied:
        def head_object(self, **_):
            raise _error("AccessDenied", 403)

    with pytest.raises(ClientError):
        exists(Denied(), "b", "k")
