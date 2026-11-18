"""SageMaker run-YAML schema."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from scripts.sagemaker.run_config import load_run_config

_RUNS = Path(__file__).resolve().parents[3] / "sagemaker" / "runs"

_VALID: dict[str, dict[str, object]] = {
    "job": {
        "name_prefix": "p",
        "image": "img:tag",
        "instance_type": "ml.g6e.8xlarge",
        "volume_gb": 100,
        "max_runtime_hours": 1,
    },
    "train": {"run_name": "r", "init_weights": None, "args": []},
}


def _variant(section: str | None, key: str, value: object = None, drop: bool = False) -> dict:
    data: dict[str, object] = {name: dict(block) for name, block in _VALID.items()}
    target: dict[str, object] = data if section is None else dict(_VALID[section])
    if section is not None:
        data[section] = target
    if drop:
        del target[key]
    else:
        target[key] = value
    return data


@pytest.mark.parametrize("name", ["cbm_final", "cbm_final_res768"])
def test_committed_run_yamls_load(name):
    config = load_run_config(_RUNS / f"{name}.yaml")
    assert config.train.run_name == name


@pytest.mark.parametrize(
    "data",
    [
        _variant("job", "spot", True),
        _variant("train", "hyperparameters", {}),
        _variant(None, "env", {}),
        _variant("job", "volume_gb", drop=True),
        _variant("train", "args", drop=True),
        _variant("train", "init_weights", drop=True),
    ],
    ids=[
        "unknown-job-key",
        "unknown-train-key",
        "unknown-top-key",
        "missing-job-field",
        "missing-train-field",
        "init-weights-absent",
    ],
)
def test_invalid_run_yaml_raises_naming_the_file(tmp_path, data):
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match=str(path)):
        load_run_config(path)
