# `dataset_reconciliation`

Disentangles the per-source dataset code (in
[`mermaidseg/datasets/`](../datasets/)) from cross-dataset label and concept
mapping.

Each source dataset emits `source_labels` in its own per-source label space
(e.g. CoralNet provider IDs, MERMAID benthic attribute IDs, Coralscapes
1..39). At training time, a [`SourceLabelRegistry`](registry.py) is built from
a list of source datasets and produces:

- A global, jointly indexed source label space (0 reserved for background;
  positive ints unique across all datasets).
- A `source_to_target` GPU `LongTensor` of shape `(N+1,)` mapping global
  source IDs to target MERMAID benthic attribute label IDs.
- A `source_to_concepts` GPU `FloatTensor` of shape `(N+1, num_concepts)`
  mapping global source IDs to concept rows.

The dataset's `__getitem__` emits masks already offset into the global source
space, so the model trainer only has to:

```python
source_labels = batch_labels.long().to(device)
target_labels = source_labels_to_target_labels(source_labels, source_to_target_lookup)
concept_labels = source_labels_to_concepts(source_labels, source_to_concepts_lookup)
```

## Modules

- `registry.py` — `SourceLabelRegistry`: builds global IDs and lookup tensors.
- `label_mapping.py` — fetches source→target dictionaries (CoralNet provider
  API, Coralscapes static dict, MERMAID benthic-attribute target labels) and
  exposes `source_labels_to_target_labels`.
- `concepts.py` — benthic-attribute concept hierarchy + the GPU helper
  `source_labels_to_concepts` (renamed from the legacy `labels_to_concepts`).
- `combined.py` — `CombinedCoralDataset` wrapping multiple registered
  datasets into one.
