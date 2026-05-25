"""Tests for mermaidseg.sagemaker.launcher_config."""
from __future__ import annotations

import unittest

import yaml
from pydantic import ValidationError

from mermaidseg.sagemaker.launcher_config import (
    JobConfig, ProcessingConfig, ShardConfig, TrainingConfig,
    parse_run_config,
)


class JobConfigTest(unittest.TestCase):
    def test_required_fields_minimal(self):
        cfg = JobConfig(
            name_prefix="my-run",
            image="mermaid-segmentation-jobs:training-latest",
            entrypoint="scripts/sagemaker_train_entrypoint.py",
            instance_type="ml.g5.2xlarge",
            volume_gb=200,
            max_runtime_hours=24,
        )
        self.assertEqual(cfg.instance_count, 1)
        self.assertFalse(cfg.use_spot)
        self.assertEqual(cfg.env, {})
        self.assertEqual(cfg.tags, {})

    def test_missing_required_field_raises(self):
        with self.assertRaises(ValidationError) as ctx:
            JobConfig(
                name_prefix="x", entrypoint="x", instance_type="x",
                volume_gb=1, max_runtime_hours=1)
        self.assertIn("image", str(ctx.exception))


class ShardConfigTest(unittest.TestCase):
    def test_workers_must_be_positive(self):
        with self.assertRaises(ValidationError):
            ShardConfig(items_from="x.csv", workers=0, per_worker_arg="--ids")


class ParseRunConfigTest(unittest.TestCase):
    def test_training_only(self):
        y = yaml.safe_dump({
            "job": {
                "name_prefix": "x", "image": "y", "entrypoint": "z",
                "instance_type": "ml.g5.2xlarge",
                "volume_gb": 200, "max_runtime_hours": 24,
            },
            "training": {"hyperparameters": {"k": "v"}},
        })
        cfg = parse_run_config(y, kind="training")
        self.assertIsNotNone(cfg.training)
        self.assertIsNone(cfg.processing)

    def test_unknown_top_level_key_raises_in_strict(self):
        y = yaml.safe_dump({
            "job": {
                "name_prefix": "x", "image": "y", "entrypoint": "z",
                "instance_type": "a", "volume_gb": 1, "max_runtime_hours": 1,
            },
            "garbage": {"foo": "bar"},
        })
        with self.assertRaises(ValidationError):
            parse_run_config(y, kind="training", strict=True)

    def test_unknown_top_level_key_ignored_in_loose(self):
        y = yaml.safe_dump({
            "job": {
                "name_prefix": "x", "image": "y", "entrypoint": "z",
                "instance_type": "a", "volume_gb": 1, "max_runtime_hours": 1,
            },
            "config": {"model_name": "LinearDINOv3"},   # seg-specific block,
                                                         # consumed by the entrypoint
                                                         # not the launcher.
        })
        cfg = parse_run_config(y, kind="training", strict=False)
        self.assertIsNotNone(cfg.job)
