# Contributing to MermaidSeg

Thank you for your interest in contributing. This document covers how to set up a development environment, run tests, and submit changes.

## Development setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
# Clone and install in editable mode with dev dependencies
git clone https://github.com/data-mermaid/mermaid-segmentation
cd mermaid-segmentation
uv sync --group tests --group lint --group notebooks
```

## Running tests

```bash
uv run pytest
```

To run only fast tests (excluding slow/integration):

```bash
uv run pytest -m "not slow and not integration"
```

## Code style

Linting and formatting are handled by [Ruff](https://docs.astral.sh/ruff/). The pre-commit hooks run these automatically on commit.

```bash
# Install hooks
uv run pre-commit install

# Run manually
uv run ruff check .
uv run ruff format .
```

## AWS credentials

Some dataset classes and tests require access to S3. Before running integration tests or notebooks that read from S3, verify your AWS session:

```bash
aws sts get-caller-identity
```

## Submitting changes

1. Fork the repository and create a branch from `main`.
2. Make your changes and add tests where appropriate.
3. Ensure `uv run pytest` and `uv run ruff check .` pass.
4. Open a pull request with a clear description of what changed and why.

## Reporting issues

Please open a GitHub issue with a minimal reproducer, the Python version, and any relevant error output.
