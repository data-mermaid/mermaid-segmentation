import hashlib
import json
import re
from pathlib import Path

import pyarrow as pa
import pytest
from spec_helpers import (
    BUCKET,
    EXTRACTOR,
    FEATURE_FAMILY,
    FEATURE_STORE,
    IMAGES,
    POINTS,
    READY_ETL,
    VERSION,
    FakeS3,
    publish_version,
)

from segdata_spec.errors import (
    ExtractorMismatch,
    FeatureIndexInvalid,
    PinMismatch,
    SchemaValidationError,
)
from segdata_spec.features import (
    claim_extractor,
    feature_file_key,
    feature_store_name,
    publish_feature_set,
)
from segdata_spec.manifest import FeatureSetPin, ImageStore, Pin

SIDECAR = "features/ds/effnet/_extractor_spec.json"
SET_PREFIX = f"datasets/ds/{VERSION}/features/effnet/"
TWO_IMAGES = {"a": (b"aaaa", "g1"), "b": (b"bbbbbbbb", "g1")}


def feature_index(ids: list[str], keys: list[str] | None = None) -> pa.Table:
    return pa.table(
        {
            "image_id": ids,
            "key": keys or [f"features/ds/effnet/{i}.featurevector" for i in ids],
            "etag": [f"etag-{i}" for i in ids],
            "size": [10] * len(ids),
            "crc64nvme": pa.nulls(len(ids), pa.string()),
            "n_points": [3] * len(ids),
        }
    )


@pytest.fixture
def pin(fake_s3: FakeS3, tmp_path: Path) -> Pin:
    _, pin = publish_version(
        fake_s3, tmp_path, images=TWO_IMAGES, points=POINTS, image_store=FEATURE_STORE
    )
    return pin


def publish(client: FakeS3, pin: Pin, index: pa.Table, **kwargs) -> FeatureSetPin:
    return publish_feature_set(
        client,
        pin,
        FEATURE_FAMILY,
        index,
        kwargs.pop("extractor", EXTRACTOR),
        bucket=BUCKET,
        etl=READY_ETL,
        **kwargs,
    )


def _put_keys(client: FakeS3) -> list[str]:
    return [key for op, _, key, _ in client.calls if op == "put"]


def test_publish_writes_sidecar_index_then_manifest_and_returns_its_pin(fake_s3, tmp_path, pin):
    feature_pin = publish(fake_s3, pin, feature_index(["a", "b"]))

    assert _put_keys(fake_s3) == [
        SIDECAR,
        SET_PREFIX + "index.parquet",
        SET_PREFIX + "manifest.json",
    ]
    manifest_bytes = fake_s3.objects[(BUCKET, SET_PREFIX + "manifest.json")]
    index_bytes = fake_s3.objects[(BUCKET, SET_PREFIX + "index.parquet")]
    assert feature_pin == FeatureSetPin(
        dataset=pin,
        family="effnet",
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
    )
    manifest = json.loads(manifest_bytes)
    assert manifest["stats"] == {"n_images": 2, "n_missing": 0}
    assert manifest["files"]["index.parquet"]["sha256"] == hashlib.sha256(index_bytes).hexdigest()
    assert manifest["files"]["index.parquet"]["rows"] == 2
    assert manifest["dataset"] == pin.model_dump()
    assert manifest["family"] == "effnet"
    assert manifest["extractor"] == EXTRACTOR
    assert json.loads(fake_s3.objects[(BUCKET, SIDECAR)]) == EXTRACTOR


def test_claim_extractor_stores_the_spec_in_a_fresh_family(fake_s3):
    claim_extractor(fake_s3, BUCKET, FEATURE_STORE, FEATURE_FAMILY, EXTRACTOR)

    assert json.loads(fake_s3.objects[(BUCKET, SIDECAR)]) == EXTRACTOR


def test_claim_extractor_keeps_a_sidecar_whose_identity_matches(fake_s3):
    stored = json.dumps({**EXTRACTOR, "weights_uri": "s3://elsewhere/b.pt"}).encode()
    fake_s3.put(BUCKET, SIDECAR, stored)

    claim_extractor(fake_s3, BUCKET, FEATURE_STORE, FEATURE_FAMILY, EXTRACTOR)

    assert fake_s3.objects[(BUCKET, SIDECAR)] == stored


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("class_path", "spacer.extract_features.VGG16CaffeExtractor"),
        ("crop_size", 168),
        ("feature_dim", 4096),
        ("weights_sha256", "x" * 64),
    ],
)
def test_claim_extractor_refuses_an_identity_that_differs(fake_s3, field, value):
    fake_s3.put(BUCKET, SIDECAR, json.dumps({**EXTRACTOR, field: value}).encode())

    with pytest.raises(ExtractorMismatch, match=field):
        claim_extractor(fake_s3, BUCKET, FEATURE_STORE, FEATURE_FAMILY, EXTRACTOR)


def test_claim_extractor_refuses_a_spec_without_an_identity_key_before_any_s3_call(fake_s3):
    extractor = {k: v for k, v in EXTRACTOR.items() if k != "feature_dim"}

    with pytest.raises(ValueError, match="feature_dim"):
        claim_extractor(fake_s3, BUCKET, FEATURE_STORE, FEATURE_FAMILY, extractor)

    assert fake_s3.calls == []


def test_publish_refuses_a_spec_without_an_identity_key_before_any_s3_call(fake_s3, tmp_path, pin):
    extractor = {k: v for k, v in EXTRACTOR.items() if k != "feature_dim"}
    fake_s3.calls.clear()

    with pytest.raises(ValueError, match="feature_dim"):
        publish(fake_s3, pin, feature_index(["a", "b"]), extractor=extractor)

    assert fake_s3.calls == []


def _assert_no_set_files(client: FakeS3) -> None:
    assert not any(key.startswith(SET_PREFIX) for _, key in client.objects)


def test_publish_with_a_stored_extractor_of_another_identity_leaves_no_index_or_manifest(
    fake_s3, tmp_path, pin
):
    fake_s3.put(BUCKET, SIDECAR, json.dumps({**EXTRACTOR, "crop_size": 168}).encode())

    with pytest.raises(ExtractorMismatch):
        publish(fake_s3, pin, feature_index(["a", "b"]))

    _assert_no_set_files(fake_s3)


def test_publish_loses_a_sidecar_race_to_another_identity_and_leaves_no_index_or_manifest(
    fake_s3, tmp_path, pin, monkeypatch
):
    real_put_object = fake_s3.put_object
    rival = json.dumps({**EXTRACTOR, "weights_sha256": "x" * 64}).encode()

    def put_after_rival_claims(*, Bucket, Key, **kwargs):
        if Key == SIDECAR:
            fake_s3.put(Bucket, Key, rival)
        return real_put_object(Bucket=Bucket, Key=Key, **kwargs)

    monkeypatch.setattr(fake_s3, "put_object", put_after_rival_claims)

    with pytest.raises(ExtractorMismatch, match="weights_sha256"):
        publish(fake_s3, pin, feature_index(["a", "b"]))

    assert fake_s3.objects[(BUCKET, SIDECAR)] == rival
    _assert_no_set_files(fake_s3)


@pytest.mark.parametrize(
    ("ids", "a_key", "message"),
    [
        (["a", "b", "z"], None, "not in"),
        (["a", "a", "b"], None, "duplicate"),
        (["a", "b"], "features/ds/effnet/b.featurevector", "features/ds/effnet/b.featurevector"),
        (["a", "b"], "features/ds/other/a.featurevector", "features/ds/other/a.featurevector"),
        (["a", "b"], "features/xx/effnet/a.featurevector", "features/xx/effnet/a.featurevector"),
        (["a", "b"], "features/ds/effnet/a~../x.featurevector", "a~../x"),
    ],
    ids=[
        "unknown-id",
        "duplicate-id",
        "key-of-another-image",
        "key-in-another-family",
        "key-in-another-store",
        "malformed-suffix",
    ],
)
def test_publish_refuses_an_index_with_bad_ids(fake_s3, tmp_path, pin, ids, a_key, message):
    index = feature_index(ids)
    if a_key is not None:
        index = feature_index(ids, [a_key, *index.column("key").to_pylist()[1:]])

    with pytest.raises(FeatureIndexInvalid, match=re.escape(message)):
        publish(fake_s3, pin, index)

    assert _put_keys(fake_s3) == []


def test_publish_refuses_a_partial_index_unless_allowed(fake_s3, tmp_path, pin):
    with pytest.raises(FeatureIndexInvalid, match="1 image"):
        publish(fake_s3, pin, feature_index(["a"]))
    assert _put_keys(fake_s3) == []

    publish(fake_s3, pin, feature_index(["a"]), allow_missing=True)

    manifest = json.loads(fake_s3.objects[(BUCKET, SET_PREFIX + "manifest.json")])
    assert manifest["stats"] == {"n_images": 1, "n_missing": 1}


def test_publish_refuses_a_dataset_pin_with_the_wrong_sha(fake_s3, tmp_path, pin):
    wrong = pin.model_copy(update={"manifest_sha256": "0" * 64})

    with pytest.raises(PinMismatch):
        publish(fake_s3, wrong, feature_index(["a", "b"]))

    assert _put_keys(fake_s3) == []


def test_publish_refuses_an_index_missing_a_column(fake_s3, tmp_path, pin):
    index = feature_index(["a", "b"]).drop_columns(["n_points"])

    with pytest.raises(SchemaValidationError, match="n_points"):
        publish(fake_s3, pin, index)

    assert _put_keys(fake_s3) == []


def test_publish_with_a_ready_etl_records_it_verbatim(fake_s3, pin):
    publish_feature_set(
        fake_s3,
        pin,
        FEATURE_FAMILY,
        feature_index(["a", "b"]),
        EXTRACTOR,
        bucket=BUCKET,
        etl=READY_ETL,
    )

    manifest = json.loads(fake_s3.objects[(BUCKET, SET_PREFIX + "manifest.json")])
    assert manifest["etl"] == READY_ETL.model_dump()


def test_publish_accepts_a_key_with_a_suffix(fake_s3, tmp_path, pin):
    keys = ["features/ds/effnet/a~p2.featurevector", "features/ds/effnet/b.featurevector"]

    publish(fake_s3, pin, feature_index(["a", "b"], keys))

    assert (BUCKET, SET_PREFIX + "manifest.json") in fake_s3.objects


def test_publish_refuses_keys_that_collide_across_images_and_writes_nothing(fake_s3, tmp_path):
    # Image a with suffix x and image a~x without one name the same file.
    _, pin = publish_version(
        fake_s3,
        tmp_path,
        images={"a": IMAGES["a"], "a~x": IMAGES["b"]},
        points=POINTS,
        image_store=FEATURE_STORE,
    )
    shared = "features/ds/effnet/a~x.featurevector"

    with pytest.raises(FeatureIndexInvalid, match="duplicate key"):
        publish(fake_s3, pin, feature_index(["a", "a~x"], [shared, shared]))

    assert _put_keys(fake_s3) == []


def test_publish_refuses_a_dataset_whose_images_are_outside_the_bucket_and_writes_nothing(
    fake_s3, tmp_path
):
    _, external = publish_version(fake_s3, tmp_path, images=TWO_IMAGES, points=POINTS)

    with pytest.raises(ValueError, match="bucket"):
        publish(fake_s3, external, feature_index(["a", "b"]))

    assert _put_keys(fake_s3) == []


def test_feature_store_name_is_the_segment_of_an_image_store_in_the_bucket():
    store = ImageStore(bucket=BUCKET, prefix="images/coralnet/", owner="segdata")

    assert feature_store_name(store, BUCKET) == "coralnet"


@pytest.mark.parametrize(
    ("bucket", "prefix"),
    [
        ("other-bucket", "images/coralnet/"),
        (BUCKET, "x/images/coralnet/"),
        (BUCKET, "images/"),
        (BUCKET, "images/a/b/"),
        (BUCKET, "images/Coral/"),
    ],
    ids=["other-bucket", "nested-under-a-prefix", "no-segment", "two-segments", "invalid-segment"],
)
def test_feature_store_name_refuses_a_store_outside_images_segment_in_the_bucket(bucket, prefix):
    store = ImageStore(bucket=bucket, prefix=prefix, owner="segdata")

    with pytest.raises(ValueError, match="image store"):
        feature_store_name(store, BUCKET)


@pytest.mark.parametrize(
    ("suffix", "expected"),
    [
        ("c1", "features/coralnet/effnet/img1~c1.featurevector"),
        (None, "features/coralnet/effnet/img1.featurevector"),
    ],
)
def test_feature_file_key_names_the_image_file_under_its_store_and_family(suffix, expected):
    store = ImageStore(bucket=BUCKET, prefix="images/coralnet/", owner="segdata")

    assert feature_file_key(store, BUCKET, "effnet", "img1", suffix) == expected
