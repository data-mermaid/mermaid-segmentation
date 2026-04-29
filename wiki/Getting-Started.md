# Getting Started

This page walks you from a fresh machine to a working development environment. Follow each step in order.

**Prerequisites:** Python 3.11+, git, and an AWS account that has been granted access to the project's S3 bucket by the Sparkgeo and Mermaid.


---

## 1. Clone the repository

```bash
git clone https://github.com/data-mermaid/mermaid-segmentation.git
cd mermaid-segmentation
```

## 2. Install uv

[uv](https://docs.astral.sh/uv/) is the package manager for this project.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## 3. Install dependencies

```bash
uv sync --group tests --group lint --group notebooks
```

## 4. Install pre-commit hooks

Pre-commit runs Ruff (linting and formatting) automatically on each commit. Install it once:

```bash
uv run pre-commit install
```

## 5. Set up your environment file

Copy the example environment file and fill in your values:

```bash
cp .env.example .env
```

Open `.env` and set:

```
AWS_PROFILE=mermaid-core
HF_TOKEN=hf_xxxxxxxxxxxx        # see step 8
MLFLOW_TRACKING_URI=...         # get this value from the team lead
```

## 6. Authenticate with AWS

This project uses AWS SSO. Run:

```bash
aws sso login --profile mermaid-core
```

Follow the browser prompt to complete authentication.

## 7. Verify your AWS session

```bash
aws sts get-caller-identity
```

You should see a JSON response with your account ID and user ARN. If this fails, see [Common issues](#common-issues) below.

## 8. Set up your HuggingFace token

DINOv3 models are gated on HuggingFace and require authentication.

1. Visit the [model page](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m), sign in, and request access.
2. Once approved, authenticate using one of:
   - **CLI** (recommended): `hf auth login` and paste your token when prompted
   - **Environment variable**: add `HF_TOKEN=hf_xxxx` to your `.env` file (create a token at [hf.co/settings/tokens](https://hf.co/settings/tokens))

## 9. Get your MLflow tracking URI

Ask the team lead for the `MLFLOW_TRACKING_URI` value (it points to the SageMaker MLflow app) and add it to your `.env`.

## 10. Confirm everything works

```bash
uv run pytest -m "not slow and not integration"
```

All tests should pass. If they do, you're set up correctly.

---

## Common issues

**`.env` changes not picked up by Jupyter**
Run `direnv allow` in the repo root, then restart the Jupyter kernel.

**HuggingFace "403 Forbidden"**
Your access request hasn't been approved yet, or you haven't added the token. Visit the model page on HuggingFace to check your request status.

**AWS "ExpiredTokenException"**
Your session has expired. Re-run `aws sso login --profile mermaid-core`.

**Still stuck on S3 or AWS access?**
Post in the project Slack channel with the exact error message. Don't spend more than 30 minutes troubleshooting credentials alone — bucket access requires permissions that only the team lead can grant.
