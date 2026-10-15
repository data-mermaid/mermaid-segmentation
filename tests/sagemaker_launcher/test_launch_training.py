"""Tests for scripts.launch_training (mermaid-segmentation)."""

from __future__ import annotations

import sys
import unittest
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
    return lt.build_estimator_kwargs(
        cfg=cfg,
        run_id=run_id,
        staging_bucket="dev-datamermaid-sm-data",
        mlflow_uri="arn:aws:sagemaker:us-east-1:554812291621:mlflow-app/app-EJVJ6AVFDWW2",
        sm_session=MagicMock(),
        role=lt.EXEC_ROLE,
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


class BuildEstimatorKwargsTest(unittest.TestCase):
    @patch("launch_training.datetime")
    def test_kwargs_match_expectation(self, mock_dt):
        mock_dt.now.return_value.strftime.return_value = "20260525T120000Z"
        from mermaidseg.sagemaker.launcher_config import parse_run_config

        cfg = parse_run_config(_minimal_yaml(), kind="training", strict=False)
        kwargs = lt.build_estimator_kwargs(
            cfg=cfg,
            run_id="mermaidseg-test-20260525T120000Z",
            staging_bucket="dev-datamermaid-sm-data",
            mlflow_uri="arn:aws:sagemaker:us-east-1:554812291621:mlflow-app/app-EJVJ6AVFDWW2",
            sm_session=MagicMock(),
            role=lt.EXEC_ROLE,
        )
        self.assertEqual(kwargs["instance_type"], "ml.g5.2xlarge")
        self.assertEqual(kwargs["volume_size"], 200)
        self.assertEqual(
            kwargs["role"],
            "arn:aws:iam::554812291621:role/dev-sm-execution-role",
        )
        self.assertEqual(
            kwargs["image_uri"],
            "554812291621.dkr.ecr.us-east-1.amazonaws.com/mermaid-segmentation-jobs:training-smoke",
        )
        self.assertEqual(
            kwargs["environment"]["MLFLOW_TRACKING_URI"],
            "arn:aws:sagemaker:us-east-1:554812291621:mlflow-app/app-EJVJ6AVFDWW2",
        )


class SpotAndRunIdTest(unittest.TestCase):
    def test_run_id_always_injected_for_resume(self):
        """MERMAIDSEG_RUN_ID is set on every job so a spot restart can resume its MLflow
        run."""
        kwargs = _build_kwargs(_minimal_yaml(), run_id="mermaidseg-test-abc")
        self.assertEqual(kwargs["environment"]["MERMAIDSEG_RUN_ID"], "mermaidseg-test-abc")

    def test_on_demand_sets_no_spot_kwargs(self):
        kwargs = _build_kwargs(_minimal_yaml(use_spot=False))
        self.assertNotIn("use_spot_instances", kwargs)
        self.assertNotIn("max_wait", kwargs)

    def test_spot_sets_use_spot_and_max_wait(self):
        kwargs = _build_kwargs(_minimal_yaml(use_spot=True))
        self.assertTrue(kwargs["use_spot_instances"])
        # max_wait = max_run (+1h buffer) so a job can wait for spot capacity; billed for compute.
        self.assertEqual(kwargs["max_wait"], kwargs["max_run"] + 3600)
        self.assertEqual(kwargs["max_run"], 24 * 3600)


class CliArgsTest(unittest.TestCase):
    def test_no_wait_flag_recognized(self):
        """--no-wait must parse without error alongside required args."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--run-config", required=True, type=Path)
        parser.add_argument("--mlflow-tracking-uri", required=True)
        parser.add_argument("--dry-run", action="store_true")
        parser.add_argument("--no-wait", action="store_true")
        args = parser.parse_args(
            [
                "--run-config",
                "sagemaker/runs/example-training.yaml",
                "--mlflow-tracking-uri",
                "arn:aws:sagemaker:us-east-1:554812291621:mlflow-app/app-X",
                "--no-wait",
            ]
        )
        self.assertTrue(args.no_wait)
        self.assertFalse(args.dry_run)
