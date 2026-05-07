# MERMAID dataset

PyTorch dataset and supporting code for the MERMAID source dataset.

The dataset class ([`MermaidDataset`](mermaid_dataset.py)) emits annotation
masks in the **MERMAID benthic attribute label space**. The MERMAID label
space is the canonical target space used by
[`SourceLabelRegistry`](../../dataset_reconciliation/registry.py); the
`source -> target` mapping for MERMAID is therefore the identity (modulo the
global offset assigned by the registry).

## Layout

- `mermaid_dataset.py` — `MermaidDataset` class.
- `nbs/` — exploration notebooks for MERMAID-specific data analysis.
