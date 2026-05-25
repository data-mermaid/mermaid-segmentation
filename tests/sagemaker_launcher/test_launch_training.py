"""Tests for scripts.launch_training (mermaid-segmentation)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import launch_training as lt  # type: ignore


def _minimal_yaml() -> str:
    return """
job:
  name_prefix: mermaidseg-test
  image: mermaid-segmentation-jobs:training-smoke
  entrypoint: scripts/sagemaker_train_entrypoint.py
  instance_type: ml.g5.2xlarge
  volume_gb: 200
  max_runtime_hours: 24
"""


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
        )
        self.assertEqual(kwargs["instance_type"], "ml.g5.2xlarge")
        self.assertEqual(kwargs["volume_size"], 200)
        self.assertEqual(
            kwargs["role"],
            "arn:aws:iam::554812291621:role/dev-sm-execution-role",
        )
        self.assertEqual(
            kwargs["image_uri"],
            "554812291621.dkr.ecr.us-east-1.amazonaws.com"
            "/mermaid-segmentation-jobs:training-smoke",
        )
        self.assertEqual(kwargs["environment"]["MLFLOW_TRACKING_SERVER"],
            "arn:aws:sagemaker:us-east-1:554812291621:mlflow-app/app-EJVJ6AVFDWW2")


class CliArgsTest(unittest.TestCase):
    def test_no_wait_flag_recognized(self):
        """--no-wait must parse without error alongside required args."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--run-config", required=True, type=Path)
        parser.add_argument("--config-dir", required=True, type=Path)
        parser.add_argument("--mlflow-tracking-uri", required=True)
        parser.add_argument("--dry-run", action="store_true")
        parser.add_argument("--no-wait", action="store_true")
        args = parser.parse_args([
            "--run-config", "sagemaker/runs/example-training.yaml",
            "--config-dir", "sagemaker/configs/example/",
            "--mlflow-tracking-uri", "arn:aws:sagemaker:us-east-1:554812291621:mlflow-app/app-X",
            "--no-wait",
        ])
        self.assertTrue(args.no_wait)
        self.assertFalse(args.dry_run)
