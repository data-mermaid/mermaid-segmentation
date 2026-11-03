# Code Style

Good code is easy to read and hard to misunderstand. This page covers the tools and habits that keep the codebase consistent and maintainable.

---

## Automated formatting and linting

[Ruff](https://docs.astral.sh/ruff/) handles linting and formatting. Run it by hand before you commit. CI runs `ruff check` and `ruff format --check` on the Python and notebook files changed in a pull request.

If CI fails on a Ruff error, fix the flagged issue and push again:

```bash
uv run ruff check .       # show linting errors
uv run ruff format .      # auto-format code
```

Ruff handles: import order, unused imports, unused variables, formatting consistency, and a broad set of common Python mistakes. Run the commands above before pushing so CI does not reject the change.

A separate CI hygiene check runs on the files changed in a pull request. It flags files over 1 MB, file-name case conflicts, invalid YAML, trailing whitespace, and missing final newlines. Docstrings are wrapped at 100 columns by `make docformat`, which edits files in place. Run it by hand before you commit.

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
- Run `uv run ruff format --check .` — no formatting changes needed
- Run `make docformat` — wraps docstrings
- Read through your own diff before assigning a reviewer — catch the obvious things yourself first
