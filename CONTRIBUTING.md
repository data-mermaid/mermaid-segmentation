# Contributing to MermaidSeg

Thank you for contributing. This document covers the technical checklist for getting set up and submitting changes. For onboarding, team process, and workflow guides, see the [project Wiki](https://github.com/data-mermaid/mermaid-segmentation/wiki).

## Ways to contribute

All contributions are valued, not just code. You can help by:

- **Reporting bugs** — open a Bug issue with a minimal reproducer
- **Improving documentation** — fix a confusing explanation, add an example, correct a broken step
- **Writing or improving tests** — test coverage helps everyone work with more confidence
- **Reviewing pull requests** — a second pair of eyes catches things the author misses
- **Opening issues** — a well-written issue describing a problem or idea is a real contribution

If you're not sure where to start, look for open issues on the [project board](https://github.com/data-mermaid/mermaid-segmentation/projects) or ask in Slack.

## Development setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
git clone https://github.com/data-mermaid/mermaid-segmentation
cd mermaid-segmentation
uv sync --group tests --group lint --group notebooks
uv run pre-commit install
```

## Running tests

```bash
# All tests
uv run pytest

# Fast tests only (skip slow and integration)
uv run pytest -m "not slow and not integration"
```

## Code style

Linting and formatting are handled by [Ruff](https://docs.astral.sh/ruff/) and run automatically via pre-commit on each commit.

```bash
# Run manually
uv run ruff check .
uv run ruff format .
```

## AWS credentials

Verify your AWS session before running integration tests or notebooks that read from S3:

```bash
aws sso login --profile mermaid-core
aws sts get-caller-identity
```

## Submitting changes

1. Create a branch from `main` (`feature/<description>` or `fix/<description>`).
2. Make your changes and add tests where appropriate.
3. Ensure `uv run pytest` and `uv run ruff check .` pass.
4. Add tests for any new behaviour — tests serve as living documentation of how code is expected to work.
5. Open a pull request with a clear title and description of what changed and why, linked to the relevant issue.
6. Assign a reviewer — all PRs require review. Expect feedback within 1–2 business days.

## Reporting issues

Open a GitHub issue using the appropriate [issue template](https://github.com/data-mermaid/mermaid-segmentation/issues/new/choose).
