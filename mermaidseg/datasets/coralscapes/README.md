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
