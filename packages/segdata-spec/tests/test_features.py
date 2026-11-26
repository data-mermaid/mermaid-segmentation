import hashlib
import json
import re
from pathlib import Path

import pytest
from spec_helpers import (
    BUCKET,
    EXTRACTOR,
    FEATURE_FAMILY,
    FEATURE_STORE,
    IMAGES,
    POINTS,
    VERSION,
    FakeS3,
    feature_index,
    publish_version,
    stage_features,
)

from segdata_spec.errors import (
    ExtractorMismatch,
    FeatureIndexInvalid,
    ManifestNotFound,
    PinMismatch,
    PublishRefused,
    SchemaValidationError,
)
from segdata_spec.features import (
    check_claimed_extractor,
    claim_extractor,
    feature_file_key,
    feature_store_name,
)
from segdata_spec.manifest import FeatureSetManifest, FeatureSetPin, ImageStore, Pin

SIDECAR = "features/ds/effnet/_extractor_spec.json"
SET_PREFIX = f"datasets/ds/{VERSION}/features/effnet/"
TWO_IMAGES = {"a": IMAGES["a"], "b": IMAGES["b"]}


@pytest.fixture
def pin(fake_s3: FakeS3, tmp_path: Path) -> Pin:
    _, pin = publish_version(
        fake_s3, tmp_path, images=TWO_IMAGES, points=POINTS, image_store=FEATURE_STORE
    )
    return pin


@pytest.fixture
def stage(tmp_path: Path) -> Path:
    return tmp_path / "stage"


def set_dir(stage: Path) -> Path:
    return stage / BUCKET / SET_PREFIX


def test_stage_writes_the_index_then_a_manifest_that_pins_the_dataset(tmp_path, stage, pin):
    feature_pin = stage_features(stage, tmp_path, pin, feature_index(["a", "b"]))

    manifest_bytes = (set_dir(stage) / "manifest.json").read_bytes()
    index_bytes = (set_dir(stage) / "index.parquet").read_bytes()
    assert sorted(p.name for p in set_dir(stage).iterdir()) == ["index.parquet", "manifest.json"]
    assert feature_pin == FeatureSetPin(
        dataset=pin,
        family="effnet",
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
    )
    manifest = FeatureSetManifest.model_validate_json(manifest_bytes)
    assert manifest.stats.model_dump() == {"n_images": 2, "n_missing": 0}
    assert manifest.files["index.parquet"].sha256 == hashlib.sha256(index_bytes).hexdigest()
    assert manifest.files["index.parquet"].rows == 2
    assert (manifest.dataset, manifest.family, manifest.extractor) == (pin, "effnet", EXTRACTOR)
    assert (manifest.etl.package, manifest.etl.git_sha) == ("segdata-etl", "abcdef1234567")


def test_stage_reads_the_dataset_from_the_first_dir_that_holds_it(tmp_path, stage, pin):
    empty = tmp_path / "empty"

    stage_features(
        stage,
        tmp_path,
        pin,
        feature_index(["a", "b"]),
        dataset_dirs=(empty, tmp_path / "fixture-stage"),
    )

    assert (set_dir(stage) / "manifest.json").is_file()


def test_stage_refuses_a_dataset_in_none_of_the_dirs(tmp_path, stage, pin):
    with pytest.raises(ManifestNotFound):
        stage_features(
            stage, tmp_path, pin, feature_index(["a", "b"]), dataset_dirs=(tmp_path / "x",)
        )

    assert not set_dir(stage).exists()


def test_stage_refuses_a_set_already_staged(tmp_path, stage, pin):
    stage_features(stage, tmp_path, pin, feature_index(["a", "b"]))

    with pytest.raises(PublishRefused, match="already staged"):
        stage_features(stage, tmp_path, pin, feature_index(["a", "b"]))


def test_stage_refuses_a_spec_without_an_identity_key_and_writes_nothing(tmp_path, stage, pin):
    extractor = {k: v for k, v in EXTRACTOR.items() if k != "feature_dim"}

    with pytest.raises(ValueError, match="feature_dim"):
        stage_features(stage, tmp_path, pin, feature_index(["a", "b"]), extractor=extractor)

    assert not set_dir(stage).exists()


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
def test_stage_refuses_an_index_with_bad_ids(tmp_path, stage, pin, ids, a_key, message):
    index = feature_index(ids)
    if a_key is not None:
        index = feature_index(ids, [a_key, *index.column("key").to_pylist()[1:]])

    with pytest.raises(FeatureIndexInvalid, match=re.escape(message)):
        stage_features(stage, tmp_path, pin, index)

    assert not set_dir(stage).exists()


def test_stage_refuses_a_partial_index_unless_allowed(tmp_path, stage, pin):
    with pytest.raises(FeatureIndexInvalid, match="1 image"):
        stage_features(stage, tmp_path, pin, feature_index(["a"]))
    assert not set_dir(stage).exists()

    stage_features(stage, tmp_path, pin, feature_index(["a"]), allow_missing=True)

    manifest = json.loads((set_dir(stage) / "manifest.json").read_bytes())
    assert manifest["stats"] == {"n_images": 1, "n_missing": 1}


def test_stage_refuses_a_dataset_pin_with_the_wrong_sha(tmp_path, stage, pin):
    wrong = pin.model_copy(update={"manifest_sha256": "0" * 64})

    with pytest.raises(PinMismatch):
        stage_features(stage, tmp_path, wrong, feature_index(["a", "b"]))

    assert not (stage / BUCKET).exists()


def test_stage_refuses_an_index_missing_a_column(tmp_path, stage, pin):
    index = feature_index(["a", "b"]).drop_columns(["n_points"])

    with pytest.raises(SchemaValidationError, match="n_points"):
        stage_features(stage, tmp_path, pin, index)

    assert not set_dir(stage).exists()


def test_stage_accepts_a_key_with_a_suffix(tmp_path, stage, pin):
    keys = ["features/ds/effnet/a~p2.featurevector", "features/ds/effnet/b.featurevector"]

    stage_features(stage, tmp_path, pin, feature_index(["a", "b"], keys))

    assert (set_dir(stage) / "manifest.json").is_file()


def test_stage_refuses_keys_that_collide_across_images(fake_s3, tmp_path, stage):
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
        stage_features(stage, tmp_path, pin, feature_index(["a", "a~x"], [shared, shared]))

    assert not set_dir(stage).exists()


def test_stage_refuses_a_dataset_whose_images_are_outside_the_bucket(fake_s3, tmp_path, stage):
    _, external = publish_version(fake_s3, tmp_path, images=TWO_IMAGES, points=POINTS)

    with pytest.raises(ValueError, match="bucket"):
        stage_features(stage, tmp_path, external, feature_index(["a", "b"]))

    assert not (stage / BUCKET).exists()


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


def test_check_claimed_extractor_passes_an_unclaimed_family_and_writes_nothing(fake_s3):
    check_claimed_extractor(fake_s3, BUCKET, FEATURE_STORE, FEATURE_FAMILY, EXTRACTOR)

    assert [op for op, *_ in fake_s3.calls if op == "put"] == []


def test_check_claimed_extractor_passes_a_stored_spec_with_the_same_identity(fake_s3):
    fake_s3.put(BUCKET, SIDECAR, json.dumps({**EXTRACTOR, "weights_uri": "s3://b/c.pt"}).encode())

    check_claimed_extractor(fake_s3, BUCKET, FEATURE_STORE, FEATURE_FAMILY, EXTRACTOR)


def test_check_claimed_extractor_refuses_a_stored_spec_of_another_identity(fake_s3):
    fake_s3.put(BUCKET, SIDECAR, json.dumps({**EXTRACTOR, "crop_size": 168}).encode())

    with pytest.raises(ExtractorMismatch, match="crop_size"):
        check_claimed_extractor(fake_s3, BUCKET, FEATURE_STORE, FEATURE_FAMILY, EXTRACTOR)


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
