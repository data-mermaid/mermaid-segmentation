"""The ``architecture`` section of ``model.json`` is a wire contract with delivered bundles.

``load_bundle`` rebuilds a model as ``MODEL_CLASSES[class](base_encoder=..., **kwargs)``, so the
class name and every kwarg key must stay stable for a given ``SCHEMA_VERSION``. The fixtures in
``fixtures/model_json_architecture/`` record that shape, one file per schema version.

Never edit an existing ``vN.json``: bump ``SCHEMA_VERSION`` in ``mermaidseg/inference/bundle.py``
and add ``v{N+1}.json``.
"""

from __future__ import annotations

import inspect
import json
import re
from pathlib import Path

import torch

from mermaidseg.export import write_sidecar
from mermaidseg.inference.bundle import MODEL_JSON, SCHEMA_VERSION
from mermaidseg.io import REPO_ROOT, setup_config
from mermaidseg.model.meta import MetaModel
from mermaidseg.model.models import MODEL_CLASSES
from tests.conftest import (
    CBM_CONCEPT_VALUE2ID,
    CBM_NUM_CONCEPTS,
    NUM_CLASSES,
    cbm_concept_lookup,
    cbm_test_config,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "model_json_architecture"
MODEL_CONFIG = REPO_ROOT / "configs" / "model_config_cbm_dpt_lora_vitl.yaml"


def _load_fixture(version: int) -> dict:
    return json.loads((FIXTURE_DIR / f"v{version}.json").read_text())


def _sidecar_architecture(tmp_path: Path) -> dict:
    """The ``architecture`` section written for the committed model config, via the real path."""
    cfg = setup_config({"model": str(MODEL_CONFIG)})
    meta = MetaModel(
        run_name="contract",
        num_classes=NUM_CLASSES,
        num_concepts=CBM_NUM_CONCEPTS,
        model_kwargs=cfg.model.copy(),
        training_kwargs=cbm_test_config().training,
        device="cpu",
        concept_value2id=CBM_CONCEPT_VALUE2ID,  # type: ignore[arg-type]
        source_to_target_lookup=torch.arange(NUM_CLASSES),
        source_to_concepts_lookup=cbm_concept_lookup(),
        model=torch.nn.Linear(1, 1),
    )
    write_sidecar(
        tmp_path,
        run_name="contract",
        model_class=meta.model_name,
        model_kwargs=meta.model_kwargs,
        encoder_config={},
        id2label={0: "ignore"},
        concept_id2name={0: "c"},
        input_size=cfg.model.input_size,
        mean=(0.5,) * 3,
        std=(0.25,) * 3,
        lineage={},
    )
    return json.loads((tmp_path / MODEL_JSON).read_text())["architecture"]


def test_sidecar_architecture_matches_the_fixture_for_the_current_schema_version(tmp_path: Path):
    fixture = _load_fixture(SCHEMA_VERSION)
    architecture = _sidecar_architecture(tmp_path)

    actual = {
        "class": architecture["class"],
        "architecture_keys": sorted(architecture),
        "kwargs_keys": sorted(architecture["kwargs"]),
    }
    expected = {key: fixture[key] for key in actual}
    assert actual == expected, (
        "architecture changed: bump SCHEMA_VERSION in mermaidseg/inference/bundle.py and add "
        f"fixtures/model_json_architecture/v{SCHEMA_VERSION + 1}.json"
    )


def test_fixture_schema_version_matches_its_filename():
    paths = sorted(FIXTURE_DIR.glob("v*.json"))
    assert paths
    for path in paths:
        match = re.fullmatch(r"v(\d+)\.json", path.name)
        assert match, path.name
        assert json.loads(path.read_text())["schema_version"] == int(match.group(1)), path.name


def test_model_constructor_takes_exactly_the_fixture_kwargs():
    fixture = _load_fixture(SCHEMA_VERSION)
    parameters = inspect.signature(MODEL_CLASSES[fixture["class"]].__init__).parameters

    assert set(parameters) - {"self"} == set(fixture["kwargs_keys"]) | {"base_encoder"}
