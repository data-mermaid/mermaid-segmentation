"""Import isolation of the model package, so a slim inference install can load it.

``mermaidseg.model.models`` needs only the concept-rank constants. Importing it must not pull in
the dataset classes (``mermaidseg.datasets``) or the augmentation stack (``albumentations``,
``cv2``). ``pandas`` and ``requests`` are asserted absent only for the ``concepts`` module alone:
transformers and peft import them transitively (sklearn pulls in pandas, huggingface-hub pulls in
requests), so asserting on them for ``models`` would test third-party code, not ours.
"""

import subprocess
import sys

import pytest

FORBIDDEN = [
    "mermaidseg.datasets",
    "mermaidseg.dataset_reconciliation.combined",
    "albumentations",
    "cv2",
]


def _run(code: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)


def test_model_import_excludes_research_modules():
    code = (
        "import sys, json\n"
        "import mermaidseg.model.models\n"
        f"print(json.dumps([m for m in {FORBIDDEN!r} if m in sys.modules]))\n"
    )
    result = _run(code)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "[]"


def test_concepts_constants_import_excludes_pandas_and_requests():
    code = (
        "import sys\n"
        "from mermaidseg.dataset_reconciliation.concepts import TAXONOMIC_CONCEPTS, parse_concept_rank\n"
        "assert TAXONOMIC_CONCEPTS\n"
        "print([m for m in ('pandas', 'requests') if m in sys.modules])\n"
    )
    result = _run(code)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "[]"


@pytest.mark.parametrize(
    "name",
    ["SourceLabelRegistry", "ConceptSchema", "CombinedCoralDataset", "source_labels_to_concepts"],
)
def test_lazy_package_resolves_public_names(name):
    code = (
        "import mermaidseg.dataset_reconciliation as dr\n"
        f"assert dr.{name} is not None\n"
        f"assert '{name}' in dir(dr)\n"
    )
    result = _run(code)
    assert result.returncode == 0, result.stderr
