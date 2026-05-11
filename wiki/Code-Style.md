# Code Style

Good code is easy to read and hard to misunderstand. This page covers the tools and habits that keep the codebase consistent and maintainable.

---

## Automated formatting and linting

Pre-commit runs [Ruff](https://docs.astral.sh/ruff/) on every commit, handling formatting and linting automatically.

If a commit fails because of a Ruff error, fix the flagged issue and commit again. You can also run Ruff manually:

```bash
uv run ruff check .       # show linting errors
uv run ruff format .      # auto-format code
```

Ruff handles: import order, unused imports, unused variables, formatting consistency, and a broad set of common Python mistakes. You don't need to think about most of this — just let the hook run.

**Why this matters beyond just style:** Automated formatting eliminates style debates in code review entirely. When a reviewer reads your PR, their attention goes to logic, correctness, and design — not spacing or import order. Consistent style across the codebase also means you can read any file without mentally adjusting to a different author's habits. As [pyOpenSci](https://www.pyopensci.org/python-package-guide/package-structure-code/code-style-linting-format.html) puts it, automated tools "reduce manual format review work" and "minimize purely visual edits during review cycles."

---

## On comments

Write comments for *why*, not *what*. If your code is named clearly, it already says what it does — a comment repeating that is noise.

**Write a comment for:**
- A non-obvious constraint (`# DINOv3 patch tokens start at index 5, not 1`)
- A workaround for a specific bug or upstream limitation
- A decision that looks wrong but is intentional

**Skip the comment for:**
- Anything the function name already communicates
- Step-by-step narration of what the code does
- References to the current task or PR ("added for the CoralNet refactor")

---

## Keep scope tight

Don't add error handling for scenarios that can't happen. Don't add features nobody asked for. Don't refactor code that isn't related to what you're changing.

A focused change is easier to review, easier to revert if something goes wrong, and less likely to introduce unexpected side effects.

---

## Before opening a PR

- Run `uv run pytest` — all tests should pass
- Run `uv run ruff check .` — no errors
- Run `uv run pre-commit .` - to ensure local hooks are run before pushing (expensive in Github Actions)
- Read through your own diff before assigning a reviewer — catch the obvious things yourself first
