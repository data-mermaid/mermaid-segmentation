# Git Workflow

This page covers how we use git in this project. It assumes you can clone a repo and make commits — if git is new to you, start with [GitHub's beginner guide](https://docs.github.com/en/get-started/quickstart/hello-world) and come back here for the project-specific conventions.

---

## The one rule: never push directly to `main`

`main` is the shared branch that everyone's work is based on. Pushing directly to it bypasses review and can introduce bugs or conflicts that affect the whole team. Branch protection is enforced — direct pushes will be rejected.

Always work on a branch.

---

## Branch naming

Create branches from `main` using this naming convention:

```
feature/<short-description>    # new functionality
fix/<short-description>        # bug fix or correction
docs/<short-description>       # documentation only
```

Examples:
- `feature/add-coralnet-augmentation`
- `fix/mask-padding-off-by-one`
- `docs/update-reproducibility-guide`

```bash
git checkout main
git pull origin main
git checkout -b feature/your-description
```

## Commit messages

Write commit messages in the imperative mood, present tense. Keep them brief and specific.

| Good | Avoid |
|------|-------|
| `Add DINOv2 loss function` | `added some stuff` |
| `Fix mask padding for edge annotations` | `bug fix` |
| `Update SegFormer config for run3` | `changes` |

---

## Opening a pull request

When your work is ready for review:

1. Push your branch: `git push origin feature/your-description`
2. Open a PR on GitHub against `main`
3. Write a clear **title** (what changed) and **description** (what changed and why)
4. Link the related issue (GitHub auto-links `Closes #42` or `Related to #42`)
5. Assign a reviewer

A good PR description answers: *What problem does this solve? How did you solve it? Is there anything the reviewer should pay special attention to?*

---

## Code review

**All PRs require review** — no exceptions, including documentation fixes. This keeps quality consistent and helps spread knowledge across the team.

- Expect feedback within **1–2 business days**
- Respond to each comment: either address it or explain why not, then mark it as resolved
- If your PR is blocking someone else's work, say so in the description or ping in Slack

When you receive a review, treat it as a conversation, not a verdict. Reviewers are looking out for the whole team.

---

## Merging

- Squash-merge is preferred (combines all your commits into one clean commit on `main`)
- Delete your branch after merging — it keeps the repo tidy

---

## Keeping your branch up to date

If `main` has moved while you've been working:

```bash
git fetch origin
git rebase origin/main
```

Resolve any conflicts, then continue. Ask in Slack if you're unsure.
