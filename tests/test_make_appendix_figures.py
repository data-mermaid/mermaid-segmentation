"""Appendix figure rendering on a synthetic bundle. No cache or cluster access."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from scripts import make_appendix_figures as figs
from tests._appendix_stubs import build_bundle


@pytest.fixture(autouse=True)
def _style():
    figs.setup_style()


def _lab_distance(a, b) -> float:
    lab = figs.srgb_to_lab(np.array([a, b]) / 255.0)
    return float(np.linalg.norm(lab[0] - lab[1]))


def test_assign_colors_distinct_within_group_and_away_from_black():
    labels = [f"l{i}" for i in range(30)]
    colors = figs.assign_colors(labels)
    assert len(set(colors.values())) == 30
    assert all(_lab_distance(c, (0, 0, 0)) > 25 for c in colors.values())
    assert colors == figs.assign_colors(labels)


def test_assign_colors_shared_label_keeps_one_color_and_groups_stay_distinct():
    groups = [{"cca", "turf", "a1", "a2"}, {"cca", "turf", "b1", "b2"}]
    order = ["cca", "turf", "a1", "b1", "a2", "b2"]
    colors = figs.assign_colors(order, groups=groups)
    for group in groups:
        group_colors = [colors[label] for label in group]
        assert len(set(group_colors)) == len(group)
    # Labels never shown together may reuse a color.
    assert colors["a1"] == colors["b1"]


def test_fmt_text_replaces_glyphs_cmr10_cannot_render():
    assert figs.fmt_text("turf_algae <5cm> | {x}") == r"turf algae $<$5cm$>$ $|$ $\{$x$\}$"
    assert figs.fmt_text("cost $5") == r"cost \$5"


def test_mapped_label_totals_sum_splits_and_drop_unmapped(tmp_path):
    build_bundle(tmp_path)
    counts = pd.read_csv(tmp_path / "label_counts.csv", keep_default_na=False)
    totals = figs.mapped_label_totals(counts, "annotations")
    assert totals["source_label"].tolist() == ["cca", "Sand", "coral"]
    assert totals["annotations"].tolist() == [54, 30, 7]


def test_load_source_table_orders_datasets_and_adds_totals(tmp_path):
    build_bundle(tmp_path)
    table = figs.load_source_table(tmp_path / "dataset_table.csv")
    assert list(table.index)[0] == "moorea_labeled_corals"
    assert "" not in table.index
    coralnet = table.loc["coralnet"]
    assert coralnet["images_total"] == (
        coralnet["images_train"] + coralnet["images_val"] + coralnet["images_test"]
    )


def test_colorize_mask_uses_palette_and_white_for_ignore():
    label = np.array([[0, 1], [2, 3]], dtype=np.uint8)
    names = {1: "coral", 2: "sand", 3: "unlabeled / unidentified"}
    out = figs.colorize_mask(label, names, {"coral": (10, 20, 30), "sand": (40, 50, 60)})
    assert out[0, 1].tolist() == [10, 20, 30]
    assert out[1, 0].tolist() == [40, 50, 60]
    assert out[0, 0].tolist() == [255, 255, 255] and out[1, 1].tolist() == [255, 255, 255]


def test_main_renders_every_figure(tmp_path):
    bundle = build_bundle(tmp_path / "bundle")
    out = tmp_path / "figs"
    figs.main(["--data-dir", str(bundle), "--output-dir", str(out), "--max-edge", "48"])

    for metric in figs.METRICS:
        assert (out / f"labels_{metric}.pdf").stat().st_size > 0
        for split in figs.PIE_SPLITS:
            assert (out / f"pie_{metric}_{split}.pdf").stat().st_size > 0
    assert (out / "legend_datasets.pdf").is_file()

    moorea = out / "moorea_labeled_corals"
    assert sorted(p.name for p in moorea.glob("*_overlay.jpg")) == [
        "00_overlay.jpg",
        "01_overlay.jpg",
    ]
    assert (moorea / "legend_moorea_labeled_corals.pdf").is_file()

    ucsd = out / "ucsd_mosaics"
    image = Image.open(ucsd / "00_image.jpg")
    assert Image.open(ucsd / "00_label.png").size == image.size
    assert (ucsd / "legend_ucsd_mosaics.pdf").is_file()

    coralnet = out / "coralnet"
    lines = (coralnet / "sources.txt").read_text().strip().splitlines()
    assert len(lines) == 3
    for line in lines[1:]:
        sid = line.split("\t")[0]
        assert (coralnet / f"legend_s{sid}.pdf").is_file()
        assert len(list((coralnet / f"s{sid}").glob("*_overlay.jpg"))) == 2


def test_concept_state_weights_follow_dataset_counts_and_section_order():
    counts = pd.DataFrame(
        {
            "dataset_key": [
                "CoralNet",
                "coralnet",
                "mermaid",
                "ucsd_mosaics",
                "mermaid",
                "coralnet",
            ],
            "split": ["train", "val", "train", "train", "train", "train"],
            "source_label": ["CCA", "cca", "Sand", "coral", "Rock", "unmapped thing"],
            "mapped": ["True", True, "1", "true", True, False],
            "annotations": [50, 4, 30, 7, 10, 90],
            "pixels": [50, 4, 30, 700, 10, 90],
        }
    )
    # Second coralnet/cca row is ignored (first row wins). Rock has no concept row.
    concepts = pd.DataFrame(
        {
            "source_dataset_source": ["coralnet", "coralnet", "mermaid", "ucsd_mosaics"],
            "source_label_class_name": ["cca", "cca", "Sand", "Coral"],
            "phylum": ["Cnidaria", "porifera", "porifera", "cnidaria"],
            "live": ["TRUE", "FALSE", False, True],
            "bleached": ["false", "TRUE", "none", "FALSE"],
            "branching": ["true", "false", "not_given", ""],
            "sand": ["not_given", "FALSE", "TRUE", np.nan],
        }
    )
    assert figs.concept_display_name("hard_substrate") == "Hard Substrate"
    weights = figs.concept_state_weights(counts, concepts).set_index("concept")

    expected = [(group, name) for group, names in figs.CONCEPT_SECTIONS for name in names]
    assert list(zip(weights["group"], weights.index, strict=True)) == expected

    def states(concept: str, metric: str) -> list[int]:
        columns = [f"{metric}_{state}" for state in ("true", "false", "not_given")]
        return [int(weights.loc[concept, column]) for column in columns]

    # cca 54/54, Sand 30/30, coral 7/700, Rock 10/10; unmapped 90 is dropped.
    assert all(sum(states(concept, "annotations")) == 101 for concept in weights.index)
    assert all(sum(states(concept, "pixels")) == 794 for concept in weights.index)
    assert states("live", "annotations") == [61, 30, 10]
    assert states("live", "pixels") == [754, 30, 10]
    assert states("bleached", "annotations") == [0, 61, 40]
    assert states("branching", "annotations") == [54, 0, 47]
    assert states("sand", "pixels") == [30, 0, 764]
    assert states("algae", "annotations") == [0, 0, 101]

    # cca 54/54 and coral 7/700; Sand is porifera and Rock has no phylum.
    cnidarian = figs.concept_state_weights(counts, concepts, phylum="cnidaria").set_index("concept")

    def cnidarian_states(concept: str, metric: str) -> list[int]:
        columns = [f"{metric}_{state}" for state in ("true", "false", "not_given")]
        return [int(cnidarian.loc[concept, column]) for column in columns]

    assert all(sum(cnidarian_states(concept, "annotations")) == 61 for concept in cnidarian.index)
    assert all(sum(cnidarian_states(concept, "pixels")) == 754 for concept in cnidarian.index)
    assert cnidarian_states("live", "annotations") == [61, 0, 0]
    assert cnidarian_states("live", "pixels") == [754, 0, 0]
    assert cnidarian_states("bleached", "annotations") == [0, 61, 0]
    assert cnidarian_states("branching", "annotations") == [54, 0, 7]
    assert cnidarian_states("sand", "pixels") == [0, 0, 754]


def test_plot_concept_balance_writes_pdf(tmp_path):
    counts = pd.DataFrame(
        {
            "dataset_key": ["coralnet"],
            "source_label": ["cca"],
            "mapped": [True],
            "annotations": [4],
            "pixels": [4],
        }
    )
    concepts = pd.DataFrame(
        {
            "source_dataset_source": ["coralnet"],
            "source_label_class_name": ["cca"],
            "live": ["true"],
            "bleached": ["false"],
        }
    )
    path = tmp_path / "concepts_balance.pdf"
    weights = figs.concept_state_weights(counts, concepts)
    cnidarian = figs.concept_state_weights(counts, concepts, phylum="cnidaria")
    figs.plot_concept_balance(weights, cnidarian, path)
    assert path.stat().st_size > 0


def test_save_legend_fits_requested_width(tmp_path):
    handles = [figs.patch_handle(f"label number {i}", (0.2, 0.4, 0.6)) for i in range(40)]
    ncol = figs.save_legend(handles, tmp_path / "legend.pdf", max_width_in=3.0)
    assert 1 <= ncol < 40
    wide = figs.save_legend(handles, tmp_path / "wide.pdf", max_width_in=20.0)
    assert wide > ncol
