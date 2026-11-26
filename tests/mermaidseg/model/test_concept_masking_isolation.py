"""The model package reaches no dataset-reconciliation concept-source-mask code."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = """
import sys
import mermaidseg.model.models, mermaidseg.model.meta, mermaidseg.model.data_parallel
print("mermaidseg.dataset_reconciliation.concept_source_mask" in sys.modules)
"""


def test_importing_the_model_does_not_import_the_concept_source_mask_module():
    result = subprocess.run(
        [sys.executable, "-c", SCRIPT], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    )

    assert result.stdout.strip() == "False"
