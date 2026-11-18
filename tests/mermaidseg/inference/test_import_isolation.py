"""Loading a bundle and predicting imports no training, logging, data or cloud code."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tests.mermaidseg.inference import _tiny

REPO_ROOT = Path(__file__).resolve().parents[3]
BLOCKED = ("wandb", "albumentations", "datasets", "boto3", "s3fs")
FORBIDDEN_PREFIXES = (
    "mermaidseg.training",
    "mermaidseg.logger",
    "mermaidseg.io",
    "mermaidseg.model.meta",
    "mermaidseg.datasets",
    "mermaidseg.evaluation",
    "mermaidseg.dataset_reconciliation.concepts",
    "mermaidseg.dataset_reconciliation.registry",
    "mermaidseg.dataset_reconciliation.concept_schema",
    "mermaidseg.lineage",
    "mermaidseg.run_setup",
)

SCRIPT = """
import json, sys
for name in {blocked!r}:
    sys.modules[name] = None
import numpy as np
from mermaidseg.inference.bundle import load_bundle
from mermaidseg.inference.predictor import CBMPredictor
bundle = load_bundle(sys.argv[1], "cpu")
predictor = CBMPredictor.from_bundle(bundle, amp_dtype=None)
image = np.random.default_rng(0).integers(0, 255, (40, 50, 3), dtype=np.uint8)
class_probs, concept_probs = predictor.predict_points(image, [[5, 5], [30, 40], [10, 20]])
print(json.dumps({{
    "class_shape": list(class_probs.shape),
    "concept_shape": list(concept_probs.shape),
    "modules": sorted(m for m in sys.modules if m.startswith("mermaidseg")),
}}))
"""


def _is_forbidden(module: str) -> bool:
    return any(module == p or module.startswith(p + ".") for p in FORBIDDEN_PREFIXES)


def test_predicting_from_a_bundle_imports_no_training_or_cloud_code(bundle_dir):
    result = subprocess.run(
        [sys.executable, "-c", SCRIPT.format(blocked=BLOCKED), str(bundle_dir)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert report["class_shape"] == [3, len(_tiny.CLASS_NAMES)]
    assert report["concept_shape"] == [3, len(_tiny.CONCEPT_NAMES)]
    leaked = [m for m in report["modules"] if _is_forbidden(m)]
    assert leaked == []
