"""mermaidseg.inference loads a bundle and predicts with the research stack blocked."""

from __future__ import annotations

import subprocess
import sys

BLOCKED = ["albumentations", "pandas", "mlflow", "wandb", "boto3", "s3fs", "datasets"]
NOT_IMPORTED = [
    "mermaidseg.datasets",
    "mermaidseg.logger",
    "mermaidseg.model.meta",
    "mermaidseg.io",
]

# A None entry in sys.modules makes `import x` raise ImportError while
# importlib.util.find_spec(x) reports it missing, as on a slim install.
SCRIPT = f"""
import sys
for name in {BLOCKED!r}:
    sys.modules[name] = None

import numpy as np
from mermaidseg.inference import CBMPredictor, load_bundle

bundle = load_bundle(sys.argv[1], format=sys.argv[2])
predictor = CBMPredictor.from_bundle(bundle)
image = np.zeros((40, 50, 3), dtype=np.uint8)
class_probs, concept_probs = predictor.predict_points(image, np.array([[1, 2], [30, 45]]))
print(tuple(class_probs.shape), tuple(concept_probs.shape))
print(sorted(m for m in {BLOCKED + NOT_IMPORTED!r} if sys.modules.get(m) is not None))
"""


def test_bundle_loads_and_predicts_without_the_research_stack(graph_bundle):
    for fmt in ("checkpoint", "graph"):
        result = subprocess.run(
            [sys.executable, "-c", SCRIPT, str(graph_bundle), fmt],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        shapes, leaked = result.stdout.strip().splitlines()[-2:]
        assert shapes == "(2, 5) (2, 14)"
        assert leaked == "[]"
