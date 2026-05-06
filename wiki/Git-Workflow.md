# Git Workflow

This page covers how we use git in this project. It assumes you can clone a repo and make commits — if git is new to you, start with [GitHub's beginner guide](https://docs.github.com/en/get-started/quickstart/hello-world) and come back here for the project-specific conventions.

---

## The one rule: never push directly to `main`

`main` is the shared branch that everyone's work is based on. Pushing directly to it bypasses review and can introduce bugs or conflicts that affect the whole team. Branch protection is enforced — direct pushes will be rejected.

Always work on a branch.

---

## Branch naming

```
<issue-number>-<short-description>
```

- `123-add-coralnet-augmentation`
- `456-mask-padding-off-by-one`
- `789-update-reproducibility-guide`

```bash
git checkout main && git pull origin main
git checkout -b 123-your-description
```

## Commit messages

Generally, add a commit message that describes the main change.

[Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) has great suggestions for how to write commits.


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

For PR descriptions, you may find AI-assistance worth exploring as a short-cut, as long as you review and agree with
the output.

---

## Code review

**All PRs require review** — no exceptions, including documentation fixes. This keeps quality consistent and helps spread knowledge across the team.

For clarity on PR comments, we use [Conventional Comments](https://conventionalcomments.org)

- Assume good intent, be a collaborator
- Expect feedback within **1–3 business days**
- Respond to each comment: either address it or explain why not, then mark it as resolved
- If your PR is blocking someone else's work, say so in the description or ping in Slack

When you receive a review, treat it as a conversation, not a verdict. Reviewers are looking out for the whole team.

Additional context for PR reviews and how to handle them :
- [](https://hackernoon.com/pull-request-etiquette-20-core-principles-for-handling-prs-as-a-software-developer-a76l3yek)

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
