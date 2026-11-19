# Coralscapes V2

Dense label masks over 95 classes for video frames of Red Sea reefs, read from Hugging Face by `CoralscapesV2Dataset` in `coralscapes_v2_dataset.py`.

## Source and citation

CoralscapesV2: Panoptic and Fine-Grained Visual Scene Understanding in Coral Reefs (ECCV 2026 Workshop on Marine Vision), [arXiv:2609.12826](https://arxiv.org/abs/2609.12826). It extends [Coralscapes](https://huggingface.co/datasets/EPFL-ECEO/coralscapes) from 39 to 95 classes. Each of the 95 classes maps onto one of the 39 Coralscapes classes; `CORALSCAPES_V2_TO_LEGACY_NAME` in `coralscapes_v2_dataset.py` holds that mapping.

The Hugging Face dataset card gives 2433 frames at 1024x2048 px from 41 dive sites:

| Split | Images | Dive sites |
|---|---|---|
| `train` | 1790 | 34 |
| `validation` | 170 | 3 |
| `test` | 473 | 5 |

## Licence

Apache 2.0, as stated on the Hugging Face dataset card.

## Where the class reads from

The default configuration of the Hugging Face dataset [`josauder/coralscapesV2`](https://huggingface.co/datasets/josauder/coralscapesV2). The class reads the `image` and `label` columns. `split` selects `train`, `validation`, `test` or a list of them; the default is all three. To read a local copy, pass its directory as `hf_repo_id`.

## How to rebuild

There is no ingestion step. The class downloads it into `HF_HOME` the first time it is built. See [Get the Hugging Face datasets](../../../docs/runbooks/training-datasets.md#get-the-hugging-face-datasets).
