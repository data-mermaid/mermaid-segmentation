"""Run settings of the trainer: dataset repetition."""

from __future__ import annotations

from mermaidseg.datasets.factory import DATASET_CLASSES, build_datasets
from mermaidseg.io import ConfigDict, load_config
from mermaidseg.run_setup import resolve_repo_path
from mermaidseg.training import _assemble_train_datasets


def _datasets() -> dict[tuple[str, str], str]:
    return {
        ("a", "train"): "a",
        ("coralscapes_v2", "train"): "cs",
        ("b", "train"): "b",
        ("b", "val"): "bv",
    }


class _Stub:
    def __init__(self, **kwargs):
        pass

    def __len__(self) -> int:
        return 1


def test_train_datasets_repeat_the_configured_dataset_after_all_trains():
    data_cfg = ConfigDict(
        {"a": {"train": {}}, "coralscapes_v2": {"train": {"repeat": 6}}, "b": {"train": {}}}
    )
    assert _assemble_train_datasets(_datasets(), data_cfg) == [
        "a",
        "cs",
        "b",
        "cs",
        "cs",
        "cs",
        "cs",
        "cs",
    ]


def test_train_datasets_without_repeat_are_not_repeated():
    data_cfg = ConfigDict(
        {"a": {"train": {}}, "coralscapes_v2": {"train": {"split": "train"}}, "b": {"train": {}}}
    )
    assert _assemble_train_datasets(_datasets(), data_cfg) == ["a", "cs", "b"]


def test_trainval_config_trains_on_six_coralscapes_v2_copies():
    data_cfg = load_config(resolve_repo_path("configs/data_config_trainval.yaml"))["data"]
    classes = dict.fromkeys(DATASET_CLASSES, _Stub)
    built = build_datasets(data_cfg, dataset_classes=classes, splits=["train"], verbose=False)
    train = _assemble_train_datasets(built, data_cfg)
    assert sum(ds is built[("coralscapes_v2", "train")] for ds in train) == 6
