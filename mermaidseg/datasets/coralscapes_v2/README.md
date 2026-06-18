# Coralscapes V2 dataset

PyTorch dataset and supporting code for the
[Coralscapes V2](https://huggingface.co/datasets/josauder/314d3951853dad8855bd06248987f626)
source dataset.

The dataset class ([`CoralscapesV2Dataset`](coralscapes_v2_dataset.py)) emits
annotation masks in the **native Coralscapes V2 1..95 label space**. The static
mapping from Coralscapes V2 labels to the MERMAID benthic attribute space lives
in
[`mermaidseg.dataset_reconciliation.label_mapping.coralscapes_v2_to_mermaid`](../../dataset_reconciliation/label_mapping.py).

## Layout

- `coralscapes_v2_dataset.py` — `CoralscapesV2Dataset` class.

## Description

Coralscapes V2 extends the original Coralscapes dataset with a finer-grained
95-class taxonomy. It has the same HuggingFace structure as V1 (`train` /
`validation` / `test` splits, `image` + `label` columns) and is hosted at
https://huggingface.co/datasets/josauder/314d3951853dad8855bd06248987f626.
