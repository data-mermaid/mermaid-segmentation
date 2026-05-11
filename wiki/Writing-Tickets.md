# Writing Tickets

Good GitHub issues make collaboration possible. They capture why work is being done, what done looks like, and enough context for someone else to pick it up. Poor issues lead to duplicated effort, misaligned expectations, and work that doesn't connect to the team's goals.

All contributions are valued — not just code. A well-written bug report, a documentation fix, or a question that surfaces a gap in our onboarding is a genuine contribution. If you're not sure whether something is worth an issue, open one anyway. The worst outcome is a short conversation.

Ticket writing for data science projects is still a work in progress, if you have ideas for making a smoother process please surface these concerns to Lauren directly! The DSLP approach described below, orients the higher level goals and deliverables for a data science project with multiple collaborators.

---

## The DSLP workflow

This project follows the [Data Science Lifecycle Process (DSLP)](https://github.com/dslp/dslp). Every piece of work flows through a sequence of issue types:

```
Ask (problem statement)
  ↓
Data Acquisition  ←→  Data Create
  ↓
Explore
  ↓
Experiment  →  (iterate)
  ↓
Model (productionalize)
```

See the [README](https://github.com/data-mermaid/mermaid-segmentation#dslp-issue-workflow) for the full flowchart.

---

## Which template to use

When creating a new issue, select the template that matches where you are in the workflow:

| Template | Use when... |
|----------|-------------|
| **Ask** | You have a question or problem to investigate. Start here if unsure. |
| **Data Acquisition** | You need access to an existing dataset. |
| **Data Create** | You're creating a new derived dataset. |
| **Explore** | You're doing exploratory analysis on data. |
| **Experiment** | You're testing a modeling approach. |
| **Model** | You're preparing a successful experiment for deployment. |
| **Bug** | Something is broken and shouldn't be. |

**When in doubt, open an Ask ticket first.** It's always better to start a conversation than to build in the wrong direction.

---

## Anatomy of a good issue

Every issue should answer four questions:

**Title**: What specifically are you doing?
Make the title specific and actionable. Someone should understand the work from the title alone.

**Context**: Why does this matter? What do we already know?
Link to prior experiments, relevant papers, or related issues. Don't assume the reader has your full mental model.

**Goal**: What does success look like?
Describe the outcome, not the process.

**Definition of done**: How will you know when it's finished?
Make this testable and specific. Avoid vague endings like "investigate further."

---

## Bad vs. good: an example

**Experiment template — Bad:**

> **Title:** Train a model on new data
>
> **Description:** I want to try training SegFormer on the CoralNet data we got.

Problems: the title is vague, there's no context about which data or split, no goal, no definition of done.

---

**Experiment template — Good:**

> **Title:** Experiment: Evaluate SegFormer on CoralNet run1 val split — baseline before augmentation tuning
>
> **Context:** We have run1 train/val/test splits in `configs/`. Before tuning augmentations, we need a baseline SegFormer performance on the val split to compare against.
>
> **Goal:** Establish a baseline F1 score for SegFormer (mit-b2) on the CoralNet run1 val split with `class_subset=[Acropora, Porites, Rubble]`.
>
> **Definition of done:**
> - [ ] Run logged to MLflow with config, metrics, and model checkpoint
> - [ ] Val F1 reported per class in the issue comments
> - [ ] Config committed to `configs/run1_segformer_baseline.yaml`

---

## Tips

- **Link issues to PRs.** When you open a PR that addresses an issue, write `Closes #42` in the PR description. GitHub will close the issue automatically when the PR merges.
- **Update the issue as work progresses.** If your approach changes, note it. The issue is a record of the work, not just the initial plan.
- **Keep one issue per distinct piece of work.** Don't bundle unrelated changes into one issue.
