"""Tests for mermaidseg.run_setup, the helpers behind scripts/train.py."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import albumentations as A
import pytest

import mermaidseg.run_setup as setup

REPO_ROOT = Path(__file__).resolve().parent.parent


def _val(width=256, height=256, mean=(0.1, 0.2, 0.3), std=(0.4, 0.5, 0.6)):
    return A.Compose([A.Resize(height=height, width=width), A.Normalize(mean=mean, std=std)])


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
            val_transforms=[_val()],
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
            val_transforms=[_val(**val)],
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

    def test_default_is_repo_outputs_independent_of_cwd(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        resolved = setup.resolve_output_dir(None, {})
        assert resolved == str(REPO_ROOT / "outputs")
        assert Path(resolved).is_absolute()

    def test_empty_env_is_unset(self):
        assert (
            setup.resolve_output_dir(None, {setup.OUTPUT_DIR_ENV: ""}) == setup.DEFAULT_OUTPUT_DIR
        )

    def test_run_checkpoint_dir_matches_logger_layout(self):
        assert setup.run_checkpoint_dir("out", "r1") == "out/model_checkpoints/r1"


class TestConceptMappingDefault:
    def test_env_wins(self):
        assert setup.resolve_concept_mapping_default({setup.CONCEPT_MAPPING_ENV: "/x.csv"}) == (
            "/x.csv"
        )

    def test_repo_default_exists(self):
        resolved = Path(setup.resolve_concept_mapping_default({}))
        assert resolved == REPO_ROOT / "configs" / "class_to_concepts.csv"
        assert resolved.is_file()


class TestValSpecStrictness:
    def _kwargs(self, transforms):
        return setup.sidecar_kwargs(
            run_name="r",
            model_cfg={"name": "X"},
            target_id2label={1: "coral"},
            concept_id2name={1: "a"},
            concept_value2id={},
            val_transforms=transforms,
        )

    def test_agreeing_datasets_pass(self):
        assert self._kwargs([_val(), _val()])["input_size"] == (256, 256)

    def test_differing_sizes_raise(self):
        with pytest.raises(ValueError, match="differ"):
            self._kwargs([_val(width=256), _val(width=128)])

    def test_differing_normalisation_raises(self):
        with pytest.raises(ValueError, match="differ"):
            self._kwargs([_val(), _val(mean=(0.9, 0.9, 0.9))])

    @pytest.mark.parametrize(
        "spatial",
        [A.SmallestMaxSize(max_size=64), A.LongestMaxSize(max_size=64), A.CenterCrop(8, 8)],
    )
    def test_unrecognised_spatial_transform_raises(self, spatial):
        t = A.Compose([spatial, A.Normalize(mean=(0, 0, 0), std=(1, 1, 1))])
        with pytest.raises(ValueError, match=type(spatial).__name__):
            setup.val_input_spec(t, [512, 512])


class TestWriteRunSidecar:
    def test_non_exportable_model_class_is_skipped_with_a_warning(self, tmp_path, caplog):
        result = setup.write_run_sidecar(
            tmp_path, model_cfg={"name": "ConceptBottleneckDINOv3"}, sidecar_kwargs={}
        )
        assert result is None
        assert "ConceptBottleneckDINOv3" in caplog.text
        assert not list(tmp_path.iterdir())


class TestTrainerDefaults:
    def test_default_config_paths_exist_and_are_repo_anchored(self):
        source = (REPO_ROOT / "scripts" / "train.py").read_text()
        for name in (
            "training_config_cbm.yaml",
            "model_config_cbm_dpt_lora_vitl.yaml",
            "data_config_dedup.yaml",
            "logger_config.yaml",
        ):
            assert (REPO_ROOT / "configs" / name).is_file(), name
            assert f'str(REPO_ROOT / "configs" / "{name}")' in source, name

    def test_trainer_help_runs_from_another_cwd(self, tmp_path):
        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts" / "train.py"), "--help"],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert result.returncode == 0, result.stderr
        assert "--training-config" in result.stdout
