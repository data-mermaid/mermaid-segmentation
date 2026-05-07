# CoralNet dataset

PyTorch dataset and supporting code for the CoralNet source dataset.

The dataset class ([`CoralNetDataset`](coralnet_dataset.py)) emits annotation
masks in the **CoralNet provider label space** (i.e. CoralNet `Label ID` /
`provider_id` values). Mapping into the unified target MERMAID benthic
attribute space is done at training time by the
[`SourceLabelRegistry`](../../dataset_reconciliation/registry.py) and
[`fetch_coralnet_to_mermaid`](../../dataset_reconciliation/label_mapping.py).

## Layout

- `coralnet_dataset.py` — `CoralNetDataset` class.
- `scraper/` — scraping utilities and CSV artifacts originally located under
  the repo-root `scraping/` directory.
- `nbs/` — exploration and annotation-aggregation notebooks for the CoralNet
  sources.
