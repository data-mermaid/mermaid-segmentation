"""Tests for scripts.launch_training (mermaid-segmentation)."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path
from unittest.mock import MagicMock, patch

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import launch_training as lt  # type: ignore  # noqa: E402


def _minimal_yaml(*, use_spot: bool = False) -> str:
    spot_line = "  use_spot: true\n" if use_spot else ""
    return f"""
job:
  name_prefix: mermaidseg-test
  image: mermaid-segmentation-jobs:training-smoke
  entrypoint: scripts/sagemaker_train_entrypoint.py
  instance_type: ml.g5.2xlarge
  volume_gb: 200
  max_runtime_hours: 24
{spot_line}"""


def _build_kwargs(yaml_text: str, run_id: str = "mermaidseg-test-20260525T120000Z") -> dict:
    from mermaidseg.sagemaker.launcher_config import parse_run_config

    cfg = parse_run_config(yaml_text, kind="training", strict=False)
    return lt.build_training_job_kwargs(
        cfg=cfg,
        run_id=run_id,
        staging_bucket="dev-datamermaid-sm-data",
        mlflow_uri="arn:aws:sagemaker:us-east-1:554812291621:mlflow-app/app-EJVJ6AVFDWW2",
        role=lt.EXEC_ROLE,
    )


# The builder imports sagemaker.core lazily, so the extra is only needed by tests that call it.
def _has_sagemaker_core() -> bool:
    # find_spec raises ModuleNotFoundError when the parent `sagemaker` package is absent.
    try:
        return find_spec("sagemaker.core.resources") is not None
    except ModuleNotFoundError:
        return False


requires_sagemaker_core = unittest.skipUnless(
    _has_sagemaker_core(), "sagemaker-core (the `sagemaker` extra) not installed"
)


class ExpandImageTest(unittest.TestCase):
    def test_short_form_expands(self):
        result = lt.expand_image_uri("mermaid-segmentation-jobs:training-latest")
        self.assertEqual(
            result,
            "554812291621.dkr.ecr.us-east-1.amazonaws.com"
            "/mermaid-segmentation-jobs:training-latest",
        )

    def test_full_uri_passes_through(self):
        full = "111111111111.dkr.ecr.us-east-1.amazonaws.com/other:tag"
        self.assertEqual(lt.expand_image_uri(full), full)

    def test_short_form_rejects_unknown_repo(self):
        with self.assertRaises(ValueError):
            lt.expand_image_uri("mermaid-classifier-jobs:training-latest")


@requires_sagemaker_core
class BuildTrainingJobKwargsTest(unittest.TestCase):
    def test_kwargs_match_expectation(self):
        kwargs = _build_kwargs(_minimal_yaml())
        self.assertEqual(kwargs["training_job_name"], "mermaidseg-test-20260525T120000Z")
        self.assertEqual(kwargs["role_arn"], "arn:aws:iam::554812291621:role/dev-sm-execution-role")
        self.assertEqual(
            kwargs["algorithm_specification"].training_image,
            "554812291621.dkr.ecr.us-east-1.amazonaws.com/mermaid-segmentation-jobs:training-smoke",
        )
        self.assertEqual(kwargs["algorithm_specification"].training_input_mode, "File")
        rc = kwargs["resource_config"]
        self.assertEqual(
            (rc.instance_type, rc.instance_count, rc.volume_size_in_gb),
            ("ml.g5.2xlarge", 1, 200),
        )
        self.assertEqual(kwargs["stopping_condition"].max_runtime_in_seconds, 24 * 3600)
        self.assertEqual(
            kwargs["output_data_config"].s3_output_path,
            "s3://dev-datamermaid-sm-data/runs/mermaidseg-test-20260525T120000Z/output/",
        )
        env = kwargs["environment"]
        self.assertEqual(
            env["MLFLOW_TRACKING_URI"],
            "arn:aws:sagemaker:us-east-1:554812291621:mlflow-app/app-EJVJ6AVFDWW2",
        )
        self.assertEqual(
            env["CONTAINER_ENTRYPOINT_SCRIPT"], "scripts/sagemaker_train_entrypoint.py"
        )

    def test_hf_secret_id_only_set_when_given(self):
        env = _build_kwargs(_minimal_yaml())["environment"]
        self.assertNotIn("HF_TOKEN_SECRET_ID", env)
        from mermaidseg.sagemaker.launcher_config import parse_run_config

        cfg = parse_run_config(_minimal_yaml(), kind="training", strict=False)
        kwargs = lt.build_training_job_kwargs(
            cfg=cfg,
            run_id="r",
            staging_bucket="b",
            mlflow_uri="m",
            role="role",
            hf_token_secret_id="hf-secret",
        )
        self.assertEqual(kwargs["environment"]["HF_TOKEN_SECRET_ID"], "hf-secret")
        self.assertNotIn("HF_TOKEN", kwargs["environment"])

    def test_tags_are_resources_tag_objects(self):
        from sagemaker.core.resources import Tag

        kwargs = _build_kwargs(_minimal_yaml() + "  tags:\n    team: seg\n")
        self.assertEqual(kwargs["tags"], [Tag(key="team", value="seg")])


@requires_sagemaker_core
class ChannelsTest(unittest.TestCase):
    def test_config_channel_points_at_run_config_prefix(self):
        channels = {c.channel_name: c for c in _build_kwargs(_minimal_yaml())["input_data_config"]}
        self.assertEqual(list(channels), ["config"])
        cfg_ch = channels["config"]
        self.assertEqual(cfg_ch.input_mode, "File")
        src = cfg_ch.data_source.s3_data_source
        self.assertEqual(
            src.s3_uri,
            "s3://dev-datamermaid-sm-data/runs/mermaidseg-test-20260525T120000Z/config/",
        )
        self.assertEqual(src.s3_data_type, "S3Prefix")
        self.assertEqual(src.s3_data_distribution_type, "FullyReplicated")

    def test_training_channels_keep_their_input_mode(self):
        yaml_text = (
            _minimal_yaml()
            + """training:
  channels:
    data:
      s3_uri: s3://bucket/data/
      input_mode: FastFile
"""
        )
        channels = {c.channel_name: c for c in _build_kwargs(yaml_text)["input_data_config"]}
        self.assertEqual(set(channels), {"config", "data"})
        self.assertEqual(channels["data"].input_mode, "FastFile")
        self.assertEqual(channels["data"].data_source.s3_data_source.s3_uri, "s3://bucket/data/")


@requires_sagemaker_core
class SpotAndRunIdTest(unittest.TestCase):
    def test_run_id_always_injected_for_resume(self):
        """MERMAIDSEG_RUN_ID is set on every job so a spot restart can resume its wandb
        run."""
        kwargs = _build_kwargs(_minimal_yaml(), run_id="mermaidseg-test-abc")
        self.assertEqual(kwargs["environment"]["MERMAIDSEG_RUN_ID"], "mermaidseg-test-abc")

    def test_on_demand_sets_no_spot_fields(self):
        kwargs = _build_kwargs(_minimal_yaml(use_spot=False))
        self.assertNotIn("enable_managed_spot_training", kwargs)
        # Unset optional shape fields hold an Unassigned sentinel, not None.
        self.assertNotIsInstance(kwargs["stopping_condition"].max_wait_time_in_seconds, int)

    def test_spot_enables_managed_spot_and_max_wait(self):
        kwargs = _build_kwargs(_minimal_yaml(use_spot=True))
        self.assertTrue(kwargs["enable_managed_spot_training"])
        # max_wait = max_run (+1h buffer) so a job can wait for spot capacity; billed for compute.
        stop = kwargs["stopping_condition"]
        self.assertEqual(stop.max_runtime_in_seconds, 24 * 3600)
        self.assertEqual(stop.max_wait_time_in_seconds, stop.max_runtime_in_seconds + 3600)


class UploadRunConfigTest(unittest.TestCase):
    def test_uploads_run_yaml_under_config_prefix(self):
        s3 = MagicMock()
        with tempfile.TemporaryDirectory() as d:
            yaml_path = Path(d) / "run.yaml"
            yaml_path.write_text(_minimal_yaml())
            uri = lt._upload_run_config(yaml_path, "rid-1", s3)
        self.assertEqual(uri, f"s3://{lt.STAGING_BUCKET}/runs/rid-1/config/")
        s3.upload_file.assert_called_once_with(
            str(yaml_path), lt.STAGING_BUCKET, "runs/rid-1/config/run.yaml"
        )


@requires_sagemaker_core
class MainSubmitTest(unittest.TestCase):
    def _run_main(self, extra_args: list[str]):
        region_at_wait = []
        with tempfile.TemporaryDirectory() as d:
            yaml_path = Path(d) / "run.yaml"
            yaml_path.write_text(_minimal_yaml() + "config: {}\n")
            with (
                patch.dict(os.environ),
                patch("launch_training.make_run_id", return_value="mermaidseg-test-rid"),
                patch("launch_training.boto3") as mock_boto3,
                patch("sagemaker.core.resources.TrainingJob.create") as mock_create,
            ):
                mock_create.return_value.wait.side_effect = lambda **_: region_at_wait.append(
                    os.environ.get("AWS_DEFAULT_REGION")
                )
                lt.main(
                    [
                        "--run-config",
                        str(yaml_path),
                        "--mlflow-tracking-uri",
                        "arn:mlflow",
                        *extra_args,
                    ]
                )
        return mock_create, mock_boto3, region_at_wait

    def test_submits_with_job_name_equal_to_run_id_and_waits(self):
        mock_create, mock_boto3, region_at_wait = self._run_main([])
        kwargs = mock_create.call_args.kwargs
        self.assertEqual(kwargs["training_job_name"], "mermaidseg-test-rid")
        self.assertEqual(kwargs["region"], lt.REGION)
        self.assertIs(kwargs["session"], mock_boto3.Session.return_value)
        mock_create.return_value.wait.assert_called_once_with(logs=True)
        self.assertEqual(region_at_wait, [lt.REGION])

    def test_uploads_run_config_under_the_run_id_prefix(self):
        _, mock_boto3, _ = self._run_main([])
        mock_boto3.client.return_value.upload_file.assert_called_once()
        _, bucket, key = mock_boto3.client.return_value.upload_file.call_args.args
        self.assertEqual(bucket, lt.STAGING_BUCKET)
        self.assertEqual(key, "runs/mermaidseg-test-rid/config/run.yaml")

    def test_cli_values_are_forwarded_into_the_job_request(self):
        mock_create, _, _ = self._run_main(
            [
                "--hf-token-secret-id",
                "hf-secret",
                "--wandb-secret-id",
                "wb-secret",
                "--role-arn",
                "arn:aws:iam::1:role/custom",
            ]
        )
        kwargs = mock_create.call_args.kwargs
        env = kwargs["environment"]
        self.assertEqual(env["MLFLOW_TRACKING_URI"], "arn:mlflow")
        self.assertEqual(env["HF_TOKEN_SECRET_ID"], "hf-secret")
        self.assertEqual(env["WANDB_API_KEY_SECRET_ID"], "wb-secret")
        self.assertEqual(kwargs["role_arn"], "arn:aws:iam::1:role/custom")

    def test_no_wait_submits_without_waiting(self):
        mock_create, _, _ = self._run_main(["--no-wait"])
        mock_create.assert_called_once()
        mock_create.return_value.wait.assert_not_called()


EXAMPLE_RUN = Path(__file__).resolve().parents[2] / "sagemaker" / "runs" / "cbm_vitl_example.yaml"


class ValidateTrainerConfigTest(unittest.TestCase):
    def test_example_run_config_is_valid(self):
        import yaml

        seg = yaml.safe_load(EXAMPLE_RUN.read_text())["config"]
        self.assertEqual(lt.validate_trainer_config(seg), [])

    def test_missing_block_and_missing_file_are_errors(self):
        self.assertTrue(lt.validate_trainer_config(None))
        errors = lt.validate_trainer_config({"config_model": "configs/nope.yaml"})
        self.assertEqual(errors, ["Config file not found: configs/nope.yaml"])

    def test_dry_run_on_example_makes_no_aws_call(self):
        with patch("launch_training.boto3") as mock_boto3:
            lt.main(["--run-config", str(EXAMPLE_RUN), "--dry-run"])
        mock_boto3.client.assert_not_called()
        mock_boto3.Session.assert_not_called()


@requires_sagemaker_core
class WandbEnvTest(unittest.TestCase):
    def test_wandb_secret_id_is_passed_never_the_key(self):
        from mermaidseg.sagemaker.launcher_config import parse_run_config

        cfg = parse_run_config(_minimal_yaml(), kind="training", strict=False)
        kwargs = lt.build_training_job_kwargs(
            cfg=cfg,
            run_id="r",
            staging_bucket="b",
            mlflow_uri=None,
            role="role",
            wandb_secret_id="wb-secret",
        )
        self.assertEqual(kwargs["environment"]["WANDB_API_KEY_SECRET_ID"], "wb-secret")
        self.assertNotIn("WANDB_API_KEY", kwargs["environment"])
        self.assertNotIn("MLFLOW_TRACKING_URI", kwargs["environment"])


class NoRawSecretFlagsTest(unittest.TestCase):
    def test_raw_value_flags_are_gone(self):
        for flag in ("--hf-token", "--wandb-api-key"):
            with self.assertRaises(SystemExit):
                lt.main(["--run-config", str(EXAMPLE_RUN), "--dry-run", flag, "v"])


@requires_sagemaker_core
class CheckpointConfigTest(unittest.TestCase):
    def test_spot_adds_checkpoint_config_under_the_run_prefix(self):
        kwargs = _build_kwargs(_minimal_yaml(use_spot=True), run_id="mermaidseg-test-abc")
        ckpt = kwargs["checkpoint_config"]
        self.assertEqual(
            ckpt.s3_uri, "s3://dev-datamermaid-sm-data/runs/mermaidseg-test-abc/checkpoints/"
        )
        self.assertEqual(ckpt.local_path, "/opt/ml/checkpoints")

    def test_on_demand_has_no_checkpoint_config(self):
        self.assertNotIn("checkpoint_config", _build_kwargs(_minimal_yaml(use_spot=False)))


class RunIdTest(unittest.TestCase):
    def test_run_ids_from_the_same_second_differ(self):
        frozen = MagicMock()
        frozen.now.return_value.strftime.return_value = "20261102T120000Z"
        with patch("launch_training.datetime", frozen):
            first, second = lt.make_run_id("p"), lt.make_run_id("p")
        self.assertNotEqual(first, second)
        self.assertRegex(first, r"^p-20261102T120000Z-[0-9a-f]{6}$")


class ConfigVariableTest(unittest.TestCase):
    def test_a_variable_in_the_config_block_is_an_error(self):
        errors = lt.validate_trainer_config({"overrides": {"run-name": "${USER}-run"}})
        self.assertTrue(any("$" in e for e in errors), errors)

    def test_dry_run_rejects_a_variable_the_trainer_would_receive_literally(self):
        with tempfile.TemporaryDirectory() as d:
            yaml_path = Path(d) / "run.yaml"
            yaml_path.write_text(
                _minimal_yaml() + "config:\n  overrides:\n    run-name: ${HOME}-run\n"
            )
            with patch("launch_training.boto3"), self.assertRaises(SystemExit) as exit_info:
                lt.main(["--run-config", str(yaml_path), "--dry-run"])
        self.assertEqual(exit_info.exception.code, 1)


class RunIdValidationTest(unittest.TestCase):
    def test_a_prefix_that_overflows_the_job_name_limit_is_rejected(self):
        with self.assertRaises(ValueError):
            lt.make_run_id("p" * 40)
        self.assertTrue(lt.make_run_id("p" * 39))

    def test_a_prefix_with_an_underscore_is_rejected(self):
        with self.assertRaises(ValueError):
            lt.make_run_id("bad_prefix")

    def test_dry_run_reports_a_bad_prefix(self):
        with tempfile.TemporaryDirectory() as d:
            yaml_path = Path(d) / "run.yaml"
            yaml_path.write_text(_minimal_yaml().replace("mermaidseg-test", "bad_prefix"))
            with patch("launch_training.boto3"), self.assertRaises(SystemExit) as exit_info:
                lt.main(["--run-config", str(yaml_path), "--dry-run"])
        self.assertNotEqual(exit_info.exception.code, 0)


class RealLaunchValidationTest(unittest.TestCase):
    def _launch(self, yaml_text):
        with tempfile.TemporaryDirectory() as d:
            yaml_path = Path(d) / "run.yaml"
            yaml_path.write_text(yaml_text)
            with (
                patch("launch_training.boto3") as mock_boto3,
                self.assertRaises(SystemExit) as exit_info,
            ):
                lt.main(["--run-config", str(yaml_path)])
        return exit_info.exception.code, mock_boto3

    def test_a_variable_in_the_config_block_stops_the_launch_before_any_upload(self):
        code, mock_boto3 = self._launch(
            _minimal_yaml() + "config:\n  overrides:\n    run-name: ${HOME}\n"
        )
        self.assertEqual(code, 1)
        mock_boto3.client.assert_not_called()

    def test_a_spot_run_that_moves_the_output_dir_stops_the_launch(self):
        code, mock_boto3 = self._launch(
            _minimal_yaml(use_spot=True) + "config:\n  overrides:\n    output-dir: /x\n"
        )
        self.assertNotEqual(code, 0)
        mock_boto3.client.assert_not_called()


RESERVED_ENV_KEYS = ["MERMAIDSEG_RUN_ID", "CONTAINER_ENTRYPOINT_SCRIPT", "AWS_DEFAULT_REGION"]


@requires_sagemaker_core
class ReservedJobEnvTest(unittest.TestCase):
    def test_a_reserved_key_in_job_env_is_rejected_naming_the_key(self):
        for key in RESERVED_ENV_KEYS:
            with self.subTest(key=key):
                with self.assertRaises(ValueError) as ctx:
                    _build_kwargs(_minimal_yaml() + f"  env:\n    {key}: x\n")
                self.assertIn(key, str(ctx.exception))

    def test_an_allowed_job_env_key_reaches_the_environment_beside_the_run_id(self):
        env = _build_kwargs(_minimal_yaml() + "  env:\n    FOO: bar\n", run_id="rid-9")[
            "environment"
        ]
        self.assertEqual(env["FOO"], "bar")
        self.assertEqual(env["MERMAIDSEG_RUN_ID"], "rid-9")

    def test_main_rejects_a_reserved_key_before_any_upload(self):
        for key in RESERVED_ENV_KEYS:
            with self.subTest(key=key), tempfile.TemporaryDirectory() as d:
                yaml_path = Path(d) / "run.yaml"
                yaml_path.write_text(_minimal_yaml() + f"  env:\n    {key}: x\nconfig: {{}}\n")
                with (
                    patch("launch_training.boto3") as mock_boto3,
                    self.assertRaises(SystemExit) as ctx,
                ):
                    lt.main(["--run-config", str(yaml_path)])
                self.assertIn(key, str(ctx.exception.code))
                mock_boto3.client.assert_not_called()


class ConfigFlagOverrideTest(unittest.TestCase):
    def test_validate_reports_an_override_that_shadows_a_config_flag(self):
        errors = lt.validate_trainer_config({"overrides": {"training-config": "x.yaml"}})
        self.assertTrue(any("training-config" in e for e in errors), errors)
