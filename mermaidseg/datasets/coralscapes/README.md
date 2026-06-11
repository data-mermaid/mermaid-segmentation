# Coralscapes dataset

PyTorch dataset and supporting code for the
[Coralscapes](https://huggingface.co/datasets/EPFL-ECEO/coralscapes) source
dataset.

The dataset class ([`CoralscapesDataset`](coralscapes_dataset.py)) emits
annotation masks in the **native Coralscapes 1..39 label space**. The static
mapping from Coralscapes labels to the MERMAID benthic attribute space lives
in
[`mermaidseg.dataset_reconciliation.label_mapping.coralscapes_to_mermaid`](../../dataset_reconciliation/label_mapping.py).

## Layout

- `coralscapes_dataset.py` — `CoralscapesDataset` class.
- `nbs/` — exploration notebooks for Coralscapes-specific data analysis.

## Description

The Coralscapes dataset (https://huggingface.co/datasets/EPFL-ECEO/coralscapes) is a general-purpose dense semantic segmentation dataset for coral reefs. Similar in scope and with the same structure as the widely used Cityscapes dataset for urban scene understanding, Coralscapes allows for the benchmarking of semantic segmentation models in a new challenging domain.

The Coralscapes dataset spans 2075 images at 1024×2048px resolution gathered from 35 dive sites in 5 countries in the Red Sea, labeled in a consistent and speculation-free manner containing 174k polygons over 39 benthic classes.
