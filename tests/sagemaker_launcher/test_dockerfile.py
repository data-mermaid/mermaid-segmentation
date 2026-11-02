"""Static checks on docker/jobs/Dockerfile (Docker cannot build in CI)."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = (ROOT / "docker" / "jobs" / "Dockerfile").read_text()
INSTRUCTIONS = "\n".join(
    line for line in DOCKERFILE.splitlines() if not line.lstrip().startswith("#")
)


def test_dependencies_install_from_the_lock():
    assert re.search(r"COPY .*uv\.lock", INSTRUCTIONS)
    assert "uv sync --locked" in INSTRUCTIONS


def test_image_does_not_resolve_dependencies_with_pip():
    assert "pip install" not in INSTRUCTIONS


def test_uv_and_python_versions_are_pinned_to_the_project():
    assert re.search(r"astral-sh/uv:\d+\.\d+\.\d+", INSTRUCTIONS)
    python_version = (ROOT / ".python-version").read_text().strip()
    assert f"python {python_version}" in INSTRUCTIONS.lower() or (
        f"UV_PYTHON={python_version}" in INSTRUCTIONS
    )


def test_git_sha_build_arg_reaches_the_environment():
    assert "ARG GIT_SHA=unknown" in INSTRUCTIONS
    assert "MERMAIDSEG_GIT_SHA=$GIT_SHA" in INSTRUCTIONS


def test_build_context_excludes_local_artefacts_and_secrets():
    dockerignore = ROOT / ".dockerignore"
    assert dockerignore.is_file()
    patterns = {line.strip().rstrip("/") for line in dockerignore.read_text().splitlines()}
    for required in (".venv", ".env", ".git", "nbs/model_checkpoints", "logs", "**/wandb"):
        assert required in patterns, required


def test_builds_record_a_dirty_tree_in_the_git_sha():
    # `git diff` misses untracked files, which COPY still bakes into the image.
    status = "git status --porcelain -- mermaidseg scripts configs nbs pyproject.toml uv.lock"
    for path in ("Makefile", "docker/jobs/local_smoke.sh"):
        text = (ROOT / path).read_text()
        assert status in text, path
        assert "echo -dirty" in text, path
        assert "git diff --quiet" not in text, path
