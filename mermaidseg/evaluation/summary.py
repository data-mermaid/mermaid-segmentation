"""The evaluation summary: ``summary.json`` and a Markdown block per evaluated section."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

from mermaidseg.evaluation.metrics import REPORTED_BINARY_CONCEPTS
from mermaidseg.evaluation.reporting import fmt, write_json

if TYPE_CHECKING:
    from mermaidseg.evaluation.run import Evaluation

#: Summary key of the loss on ``--split train``; the ``val_loss`` entry renders it.
TRAIN_LOSS_KEY = "train_loss"
VAL_LOSS_KEY = "val_loss"
TEST_LOSS_KEY = "test_loss"


def _concept_lines(section: dict) -> list[str]:
    """Explicit binary concept and hierarchical lines for one eval section."""
    lines = []
    for concept in REPORTED_BINARY_CONCEPTS:
        b = section.get(concept)
        if not isinstance(b, dict):
            continue
        lines.append(
            f"- {concept} acc/precision/recall/F1: "
            f"{fmt(b.get('accuracy'))}/{fmt(b.get('precision'))}/{fmt(b.get('recall'))}/{fmt(b.get('f1'))} "
            f"(n_true={b.get('n_true')} n_false={b.get('n_false')} "
            f"n_not_given={b.get('n_not_given')} n_unannotated={b.get('n_unannotated')})"
        )
    h = section.get("hierarchical")
    if isinstance(h, dict):
        lines.append(
            f"- hierarchical: consistency {fmt(h.get('consistency/overall'))}, "
            f"severity (mean normalized) {fmt(h.get('severity/mean_normalized'))}, "
            f"ECE (mean) {fmt(h.get('ece/mean'))}"
        )
    return lines


def _point_lines(title: str, pooled: dict) -> list[str]:
    b = pooled.get("binary") or {}
    return [
        f"## {title}",
        f"- class accuracy: {fmt(pooled['class_accuracy'])}",
        "- taxonomic acc_all/acc_living: "
        + ", ".join(
            f"{r}={fmt(v['acc_all'])}/{fmt(v['acc_living'])}"
            for r, v in pooled["taxonomic"].items()
        ),
        "- taxonomic macro F1 all/living, none-class F1: "
        + ", ".join(
            f"{r}={fmt(v.get('macro_f1_all'))}/{fmt(v.get('macro_f1_living'))}, "
            f"{fmt(v.get('f1_none'))}"
            for r, v in pooled["taxonomic"].items()
        ),
        "- live coral taxonomic acc/macro F1: "
        + ", ".join(
            f"{r}={fmt(v.get('acc_coral'))}/{fmt(v.get('macro_f1_coral'))} (n={v.get('n_coral')})"
            for r, v in pooled["taxonomic"].items()
        ),
        f"- binary macro acc/F1: {fmt(b.get('macro_accuracy'))}/{fmt(b.get('macro_f1'))}",
        *_concept_lines(pooled),
        "",
    ]


def _val_loss_lines(section: dict, title: str = "Validation loss") -> list[str]:
    overall = section.get("overall") or {}
    focus = [f"multi_hot/{concept}" for concept in REPORTED_BINARY_CONCEPTS]
    headline = ["loss", "classification", "concepts", "multi_hot", *focus]
    ranks = [k for k in overall if "/" not in k and k not in headline]
    lines = [f"## {title} (overall)"]
    lines += [f"- {key}: {fmt(overall.get(key))}" for key in headline if key in overall]
    if ranks:
        lines.append("- ranks: " + ", ".join(f"{k}={fmt(overall[k])}" for k in ranks))
    for name, block in (section.get("per_dataset") or {}).items():
        parts = [f"{key}={fmt(block.get(key))}" for key in ("loss", *focus) if key in block]
        lines.append(f"- {name}: " + ", ".join(parts))
    lines.append("")
    return lines


def _probe_lines(title: str, section: dict, split: str) -> list[str]:
    lines = [f"## {title}"]
    for k, block in section.get("by_k", {}).items():
        scores = block.get(split) or {}
        probe = block.get("probe") or {}
        lines.append(
            f"- k={k}: {split} accuracy {fmt(scores.get('accuracy'))}, "
            f"mIoU {fmt(scores.get('miou'))} "
            f"(probe train acc {fmt(probe.get('train_accuracy'))}, "
            f"{probe.get('num_points')} pixels)"
        )
    return lines


def summarize_coralnet(key: str, section: dict) -> list[str]:
    """Markdown lines of a ``coralnet_<split>`` section."""
    split = key.removeprefix("coralnet_")
    return _point_lines(f"CoralNet (pooled over {split} sources)", section["pooled"])


def summarize_pacific(key: str, section: dict) -> list[str]:
    """Markdown lines of the ``pacific`` section."""
    return _point_lines("Pacific Labeled Corals (pooled over regions)", section["pooled"])


def summarize_coralscapes(key: str, section: dict) -> list[str]:
    """Markdown lines of a ``coralscapes_<split>`` section: probe scores, then concepts."""
    split = key.removeprefix("coralscapes_")
    title = f"CoralscapesV2 95-class linear probe ({split})"
    return [*_probe_lines(title, section, split), *_concept_lines(section), ""]


def summarize_coralscapes39(key: str, section: dict) -> list[str]:
    """Markdown lines of a ``coralscapes39_<split>`` section."""
    split = key.removeprefix("coralscapes39_")
    return [*_probe_lines(f"CoralscapesV2 39-class linear probe ({split})", section, split), ""]


def summarize_loss(key: str, section: dict) -> list[str]:
    """Markdown lines of the ``val_loss``, ``train_loss`` or ``test_loss`` section."""
    if key == TRAIN_LOSS_KEY:
        return _val_loss_lines(section, "CoralNet train-source loss")
    if key == TEST_LOSS_KEY:
        return _val_loss_lines(section, "Test loss")
    return _val_loss_lines(section)


def _entry_of(key: str, registry: Mapping[str, Evaluation]) -> Evaluation | None:
    """The registry entry whose name is ``key`` or prefixes it as ``<name>_``, if any."""
    name = VAL_LOSS_KEY if key in (TRAIN_LOSS_KEY, TEST_LOSS_KEY) else key
    for entry_name, entry in registry.items():
        if name == entry_name or name.startswith(f"{entry_name}_"):
            return entry
    return None


def write_summary(summary: dict, output_dir: Path, registry: Mapping[str, Evaluation]) -> None:
    """Write ``summary.json`` and ``summary.md`` into ``output_dir``.

    Each dict section is rendered by the ``summarize`` of its ``registry`` entry, or as an
    error line when it records ``{"error": ...}``; other entries stay out of the Markdown.
    """
    write_json(output_dir / "summary.json", summary)
    lines = ["# CBM evaluation summary", ""]
    for key, section in summary.items():
        if not isinstance(section, dict):
            continue
        if "error" in section:
            lines += [f"## {key}: ERROR - {section['error']}", ""]
            continue
        entry = _entry_of(key, registry)
        if entry is not None:
            lines += entry.summarize(key, section)
    (output_dir / "summary.md").write_text("\n".join(lines))
