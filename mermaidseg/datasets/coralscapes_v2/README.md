# Coralscapes V2 dataset

PyTorch dataset and supporting code for the
[Coralscapes V2](https://huggingface.co/datasets/josauder/coralscapesV2)
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
95-class taxonomy. It spans 2433 images at 1024x2048px resolution gathered from
41 dive sites, split into `train` / `validation` / `test`. Each split exposes
`image` + `label` columns (the `default` HuggingFace config); the dataset also
publishes optional `instances` and `sequences` configs that we do not use. It is
hosted at https://huggingface.co/datasets/josauder/coralscapesV2.

The native `id -> name` label space (1..95) is defined by the dataset's
[`id2label.json`](https://huggingface.co/datasets/josauder/coralscapesV2/raw/main/id2label.json)
and mirrored in `CORALSCAPES_V2_ID2NAME`.

## References

- Project page: https://josauder.github.io/coralscapesv2/
- Paper: https://arxiv.org/pdf/2609.12826
