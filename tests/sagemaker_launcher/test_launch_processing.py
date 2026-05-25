"""Tests for scripts.launch_processing (mermaid-segmentation)."""
from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import launch_processing as lp  # type: ignore


class ChunkItemsTest(unittest.TestCase):
    def test_round_robin(self):
        self.assertEqual(
            len(lp.chunk_items(["a", "b", "c", "d", "e"], n_workers=2)), 2)

    def test_drop_empty_when_workers_exceed_items(self):
        self.assertEqual(len(lp.chunk_items(["a", "b"], n_workers=5)), 2)

    def test_zero_workers_raises(self):
        with self.assertRaises(ValueError):
            lp.chunk_items(["a"], n_workers=0)


class LoadItemsTest(unittest.TestCase):
    def test_auto_detect_first_column(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "ids.csv"
            with open(p, "w") as f:
                w = csv.writer(f)
                w.writerow(["site_id", "extra"])
                w.writerow(["1", "x"])
                w.writerow(["2", "y"])
            self.assertEqual(lp.load_items(p, column=None), ["1", "2"])


class BuildProcessingRequestTest(unittest.TestCase):
    @patch("launch_processing.datetime")
    def test_request_shape(self, mock_dt):
        mock_dt.now.return_value.strftime.return_value = "20260525T120000Z"
        from mermaidseg.sagemaker.launcher_config import parse_run_config
        cfg = parse_run_config(
            """
job:
  name_prefix: mermaidseg-eval
  image: mermaid-segmentation-jobs:processing-latest
  entrypoint: scripts/sagemaker_processing_entrypoint.py
  instance_type: ml.g5.xlarge
  volume_gb: 100
  max_runtime_hours: 6
processing:
  container_args:
    - --task=eval
""",
            kind="processing", strict=True)
        req = lp.build_processing_request(
            cfg=cfg, run_id="mermaidseg-eval-20260525T120000Z",
            worker_idx=0, worker_items=None)
        self.assertEqual(req["ProcessingJobName"], "mermaidseg-eval-20260525T120000Z-0")
        self.assertEqual(req["RoleArn"], "arn:aws:iam::554812291621:role/dev-sm-execution-role")
        self.assertEqual(
            req["AppSpecification"]["ImageUri"],
            "554812291621.dkr.ecr.us-east-1.amazonaws.com/mermaid-segmentation-jobs:processing-latest")
        self.assertIn("--task=eval", req["AppSpecification"]["ContainerArguments"])
