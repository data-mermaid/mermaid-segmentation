"""Tests for the helpers behind nbs/Concept_Bottleneck_Pipeline_vitL_lora_dpt_all.py."""

from __future__ import annotations

import sys
from pathlib import Path

import albumentations as A
import pytest

_NBS = Path(__file__).resolve().parent.parent / "nbs"
if str(_NBS) not in sys.path:
    sys.path.insert(0, str(_NBS))

import cbm_run_setup as setup  # noqa: E402


def _val(width=256, height=256, mean=(0.1, 0.2, 0.3), std=(0.4, 0.5, 0.6)):
    return A.Compose(
        [A.Resize(height=height, width=width), A.Normalize(mean=mean, std=std)]
    )


class TestId2Label:
    def test_class_zero_is_ignore(self):
        assert setup.logger_id2label({1: "coral", 2: "sand"}) == {
            0: "ignore",
            1: "coral",
            2: "sand",
        }

    def test_matches_logger_inputs(self):
        # The script passes this same function's result to Logger(id2label=...).
        target = {1: "coral"}
        kwargs = setup.sidecar_kwargs(
            run_name="r",
            model_cfg={"name": "X"},
            target_id2label=target,
            concept_id2name={0: "a"},
            concept_value2id={},
            val_transform=_val(),
        )
        assert kwargs["id2label"] == setup.logger_id2label(target)
        assert kwargs["id2label"][0] == "ignore"


class TestSidecarSizes:
    def _kwargs(self, model_cfg, **val):
        return setup.sidecar_kwargs(
            run_name="r",
            model_cfg=model_cfg,
            target_id2label={1: "coral"},
            concept_id2name={1: "a"},
            concept_value2id={},
            val_transform=_val(**val),
        )

    def test_records_validated_and_model_config_sizes(self):
        kwargs = self._kwargs({"name": "X", "input_size": [512, 512]}, width=256, height=256)
        assert kwargs["input_size"] == (256, 256)
        assert kwargs["model_config_input_size"] == (512, 512)

    def test_int_model_config_size_is_square(self):
        assert self._kwargs({"name": "X", "input_size": 384})["model_config_input_size"] == (
            384,
            384,
        )

    def test_absent_model_config_size_is_none(self):
        assert self._kwargs({"name": "X"})["model_config_input_size"] is None


class TestValInputSpec:
    def test_taken_from_val_transform(self):
        size, mean, std = setup.val_input_spec(
            _val(width=128, height=192, mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2)),
            fallback_size=[512, 512],
        )
        assert size == (192, 128)
        assert mean == (0.5, 0.5, 0.5)
        assert std == (0.2, 0.2, 0.2)

    def test_fallback_without_resize(self):
        t = A.Compose([A.Normalize(mean=(0, 0, 0), std=(1, 1, 1))])
        assert setup.val_input_spec(t, [512, 384])[0] == (512, 384)
        assert setup.val_input_spec(t, 224)[0] == (224, 224)

    def test_no_resize_no_fallback_raises(self):
        t = A.Compose([A.Normalize(mean=(0, 0, 0), std=(1, 1, 1))])
        with pytest.raises(ValueError):
            setup.val_input_spec(t, None)

    def test_missing_normalize_raises(self):
        with pytest.raises(ValueError):
            setup.val_input_spec(A.Compose([A.Resize(height=8, width=8)]), None)


class TestOutputDir:
    def test_flag_beats_env(self):
        assert setup.resolve_output_dir("flag", {setup.OUTPUT_DIR_ENV: "env"}) == "flag"

    def test_env_beats_default(self):
        assert setup.resolve_output_dir(None, {setup.OUTPUT_DIR_ENV: "env"}) == "env"

    def test_default_is_cwd(self):
        assert setup.resolve_output_dir(None, {}) == "."

    def test_empty_env_is_unset(self):
        assert setup.resolve_output_dir(None, {setup.OUTPUT_DIR_ENV: ""}) == "."

    def test_run_checkpoint_dir_matches_logger_layout(self):
        assert setup.run_checkpoint_dir("out", "r1") == "out/model_checkpoints/r1"


class TestConceptMappingDefault:
    def test_env_wins(self):
        assert setup.resolve_concept_mapping_default({setup.CONCEPT_MAPPING_ENV: "/x.csv"}) == (
            "/x.csv"
        )

    def test_repo_default_exists(self):
        assert Path(setup.resolve_concept_mapping_default({})).is_file()
