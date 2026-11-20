import pytest

from segdata_spec import errors


@pytest.mark.parametrize(
    "name",
    [
        "PublishRefused",
        "ObjectExists",
        "PinMismatch",
        "ChecksumMismatch",
        "ImageChanged",
        "ImageMissing",
        "SchemaValidationError",
    ],
)
def test_specific_errors_are_segdata_errors(name):
    assert issubclass(getattr(errors, name), errors.SegdataError)
