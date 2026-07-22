# Class-weight artifacts

Build from annotation parquet/manifests (no image decode):

```bash
uv run python scripts/build_class_weight_artifact.py \
  --class-subset-from configs/training_config_dinov3_base.yaml \
  --mapping configs/coralnet_to_mermaid_mapping_temporary.json \
  --coralnet-parquet /path/to/coralnet_training_manifest.parquet \
  --coralnet-label-column source_label_name \
  --mermaid-parquet /path/to/mermaid_confirmed_annotations.parquet \
  --mermaid-label-column benthic_attribute_name \
  --output artifacts/class_weights_dinov3_base.json
```

Point `training.loss.weight_path` at the resulting JSON. Counts are point-label frequencies, not padded pixel masks.
