"""SageMaker run-YAML schema."""

from __future__ import annotations

import pytest
import yaml

from scripts.sagemaker.run_config import load_run_config

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


@pytest.mark.parametrize(
    "data",
    [
        _variant("job", "spot", True),
        _variant("train", "init_weights", drop=True),
    ],
    ids=["unknown-job-key", "init-weights-absent"],
)
def test_invalid_run_yaml_raises_naming_the_file(tmp_path, data):
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match=str(path)):
        load_run_config(path)
