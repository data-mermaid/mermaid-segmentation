# Unit test audit: mermaid-segmentation-rebuild (branch `main-rebuild`, f5a254e)

> Note: this report records the suite as it stood at f5a254e, before the fixes. `hierarchical_eval.py` and the dead WandbLogger methods it lists have since been deleted.

Date: 2026-11-04. The audit only read code; nothing in the repo was changed. It followed the spk-overture `unit-test-auditor` procedure, run as six parallel auditors, one per area. Line numbers are relative to the repo root.

## Totals

| Area | Tests | Valuable | Weak | Bloat |
|---|---|---|---|---|
| CoralNet scraper / preprocessing / live | 86 | 63 | 19 | 4 |
| CoralNet dedup / ETL / CLI / dataset | 48 | 30 | 10 | 8 |
| Model (losses, LR, grad accum, train loop) | 70 | 58 | 6 | 6 |
| Inference + evaluation | 109 | 94 | 5 | 10 |
| Launcher / entrypoint / run_setup / integration | 153 | 129 | 14 | 10 |
| Logger / datasets / cache / stats / concept_expr | 165 | 133 | 18 | 14 |
| **Total** | **631** | **507** | **72** | **52** |

Counting units differ slightly: some areas counted test functions, others counted parametrized items.

## Broken or never run (fix first)

1. **7 integration tests fail today.** `tests/conftest.py:101-118`: `minimal_config` has no `training.training_mode`, and `mermaidseg/model/meta.py:172-176` asserts that value. Verified: `pytest tests/test_integration.py -m integration` gives 7 failed, 2 passed, 1 skipped. CI never runs them because `.github/workflows/tests.yml:23` uses `-m "not slow and not integration"`.
2. **CI skips 13 launcher tests.** The `requires_sagemaker_core` classes in `tests/sagemaker_launcher/test_launch_training.py` need the `sagemaker` extra, and CI does not sync it.
3. **Live preprocessing tests are broken.** `tests/datasets/coralnet/test_preprocessing_live.py:123,249` call `build_manifest(df_images=..., df_checkpoint=...)`, but the signature is `(images: ibis.Table, checkpoint: ibis.Table, ...)` (verified). Also:
   - `test_checkpoint_recovery` (:325) expects skipped=2 on the second pass, but `_summarise_checkpoint` now reports resized=2.
   - The tests use fixed S3 prefixes with no cleanup, so reruns fail.
4. **`--refresh-fixtures` is unusable.** `conftest_live.py:12` registers it through `pytest_plugins` in a test module, which loads too late; pytest rejects the flag. As a result, `save_fixture` never writes.
5. **Dead code with tests.** Only `tests/evaluation/test_hierarchical_eval.py` imports `mermaidseg/evaluation/hierarchical_eval.py`. Its docstring cites an `evaluate.py --hierarchical` flag that does not exist. Either delete the module and its 3 tests, or wire it in.

## Weak: recurring patterns

- **The fixture never reaches the case the test names** (the most common pattern):
  - dedup `test_config_helpers`: no (split, unused) pair.
  - `test_anneal_fixed_groups`: the pinned group is not eligible.
  - `test_build_annotations_skips_incomplete_sources`: the source is skipped for a different reason.
  - CLI `--source-ids` tests: S3 only contains the requested ids.
  - `test_next_part_index_*`: no gap between part numbers.
  - `test_early_stopping_triggers_with_patience_and_min_delta`: min_delta is 0.
  - `test_bf16_does_not_enable_grad_scaler`: on CPU, AMP is always off.
  - `test_flags_read_from_config`: the config value equals the default.
  - `test_load_stage_stats_flush_and_snapshot`: the snapshot flushes anyway.
- **Identical inputs hide selection bugs:**
  - Taxonomy KL tests give every pixel the same logits with B=1, so label-dependent or per-pair masks are not caught (`test_taxonomy_loss.py:295,365`).
  - `test_microbatches_match_a_single_batch` uses a zeros encoder, so feeding the same microbatch twice is not caught.
- **Mirror assertions:**
  - `test_pick_points_quota_one_is_the_distance_transform_maximum` compares against `interior_point()`, the code under test.
  - `test_benthos_scoring_and_nodata` rebuilds the scoring inline and never calls `evaluate_benthos`.
  - `test_run_checkpoint_dir_matches_logger_layout` compares against a literal instead of the Logger.
- **Assertions too loose to fail:**
  - Four logger "noop" tests assert `mlflow_run_id is None`, a field those methods never write.
  - `test_atomic_checkpoint_leaves_no_partial` passes with `atomic=False` (the mutation was run).
  - `test_concept_matrix_logging_does_not_disable_logger` passes with a no-op (the mutation was run).
  - `test_audit_sources_three_sources` asserts `isinstance(bool(x), bool)`.
  - Integration tests check only `isinstance`.
  - `test_remediate_probe_writes_report` checks that columns exist, not their values.
- **Clone/merge aliasing.** `test_class_confusion_pooling` and `test_bundle_merge_pooling` do not check that the source accumulator is unchanged after a merge (`metrics.py:73,495-505`).
- **Dockerfile and Makefile text tests** (`test_dockerfile.py`, `test_makefile_does_not_pass_a_token`): each misses a concrete regression:
  - `--locked` dropped from one of the two `uv sync` lines.
  - `COPY . .` (copies `nbs/`).
  - `|| echo -dirty` changed to `&&`.
  - ENV placed before ARG.

## Weak: full list

### CoralNet scraper / preprocessing
- `test_scraper.py:169` `test_classify_manual_when_website_fewer_than_s3_images`: deleting `classify.py:54-55` still falls through to MANUAL_REVIEW.
- `test_scraper.py:186`: `n_annotations=0` masks the `annotations_csv_read_failed` clause (`classify.py:42`).
- `test_scraper.py:107` `test_parse_image_status_zero_confirmed_like_public_source`: colliding fixture values, so swapping uq/cf or uc/total (`parsers.py:65-71`) passes.
- `test_scraper.py:719` `test_paginated_export_checkpoint_uploads_partial`: the checkpoint body is never inspected (`downloader.py:817-822`).
- `test_scraper.py:521` `test_paginated_export_three_pages_concatenated`: compares sets, so page order is never checked (`downloader.py:589`).
- `test_preprocessing.py:314`: no image has two non-null timestamps, so the timestamp ordering (`manifest.py:29`) and the `_run_order` tiebreak (:30) go unprotected.
- `test_preprocessing.py:426`: 1365.33 rounds the same either way, so a floor→round change at `manifest.py:73/78` passes; landscape-only data also leaves `greatest` at :69 unprotected.
- `test_preprocessing.py:52`: deleting the no-resize fast path (`resize.py:93-95`) passes.
- `test_preprocessing.py:514` `test_inspect_image_truncated`: accepts DECODE_FAILURE, so the TRUNCATED branch (`inspect.py:113-114`) is unprotected.
- `test_preprocessing.py:524` `test_inspect_image_corrupted_header`: the fixture is already unreadable, so the CORRUPTED_HEADER branch is never reached.
- `test_jpeg_header.py:48`: accepts either `corrupt` or `pil_fallback` (`jpeg_header.py:165`).
- `test_images_pipeline.py:66`: writes the same frame twice, so removing the sort at `io.py:144` passes.
- `test_images_pipeline.py:124`: with no gap in part numbers, `len(parts)` passes.
- Live tests:
  - `test_live_smoke.py:84`: the pagination assertion is unreachable.
  - `test_live_smoke.py:194` and `:319`: use a hand-copied export payload, so drift in the production form goes undetected.
  - `test_preprocessing_live.py` :40, :159 and :325: broken (see above).

### CoralNet dedup / ETL / CLI
- `test_dedup.py:101` `test_config_helpers`: dropping `used &` (`dedup/common.py:89-90`) passes.
- `test_dedup.py:291` `test_anneal_fixed_groups`: dropping `& (fixed < 0)` (`splits.py:142`) passes.
- `test_dedup.py:353` `test_leakage`: removing `("test","val")` (`leakage.py:47`) passes.
- `test_annotations.py:53`: deleting `sort_values` (`etl/annotations.py:267`) passes, because the input is pre-sorted.
- `test_annotations.py:65`: the status filter masks the regex (`annotations.py:101`).
- `test_annotations.py:107`: removing the `is_complete` filter (`annotations.py:218`) passes.
- `test_audit.py:18`: the always-true `isinstance` assertion; read-failure flags (`audit.py:184-187`) and the `is_complete` clauses (:208-213) are unprotected.
- `test_cli.py:84` and `:123`: ignoring `--source-ids` / `--source-ids-file` (`etl/__main__.py:114`) passes.
- `test_remediate_cli.py:57`: a sign flip in `images_gap` or a constant `action` (`remediate_incomplete.py:104-120`) passes.

### Model
- `test_taxonomy_loss.py:295` and `:365`: identical-logit pixels, so label-based masks and `labelled[-1]` (`taxonomy_loss.py:368-379`) pass.
- `test_train_model_integration.py:231`: min_delta=0, so dropping or sign-flipping it (`train.py:208/210`) passes.
- `test_grad_accum.py:56`: deleting the `amp_dtype == float16` gate (`meta.py:198`) passes on CPU.
- `test_grad_accum.py:66`: deleting the fp32 cast (`meta.py:396-397`) passes.
- `test_grad_accum.py:97`: hoisting `next(loader)` out of the micro loop (`meta.py:648-656`) passes.

### Inference / evaluation
- `test_metrics.py:41` and `:191`: clone aliasing and a deleted binary merge (`metrics.py:73,495-505`) pass.
- `test_metrics.py:126`: neither `none_local_index` detection (`metrics.py:440`) nor `binary.update` (:487) is checked.
- `test_eval_helpers.py:151`: dropping `& ~nodata` (`benthos_eval.py:178`) passes.
- `test_probe_points.py:99`: mirror assertion, so `interior_point` returning the first True pixel passes.

### Launcher / entrypoint / run_setup / integration
- `test_dockerfile.py`:
  - :15: `--locked` can be dropped from one of the two lines.
  - :20: `pip3 install` or `uv add` passes.
  - :32: ENV placed before ARG passes.
  - :53: `COPY . .` passes.
  - :57: `&&` in place of `||` passes.
- `test_launch_training.py:226`: `upload_file` is asserted with no args, so a wrong run id or dropped kwargs (`launch_training.py:316-326`) pass.
- `test_launch_training.py:312`: Makefile text test; `HF_TOKEN=$(HF_TOKEN)` passes.
- `test_sagemaker_train_entrypoint.py:271`: deleting the `global_step >= total_steps` clause (`entrypoint:202`) passes.
- `test_run_setup.py:113`: compares a literal against itself, so Logger layout drift (`logger.py:907`) passes.
- `test_run_setup.py:172`: greps the source; swapping the config defaults (`train.py:123-136`) passes.
- `test_integration.py` :27, :114, :163, :183: `isinstance`-only checks (and currently failing).

### Logger / datasets / cache / stats / concept_expr
- `test_concept_expr.py:54`: operator tokens are not checked.
- `test_concept_expr.py:208`: blend formula changes and the 0.4 boundary (`video_demo.py:114`) pass.
- `test_concept_expr.py:220`: a constant `banner_height = 120` passes.
- `test_dataset_stats.py:167`: `img_per_source` taken from annotations (`dataset_stats.py:305`) passes (mutation was run).
- `test_dataset_stats.py:218`: `effective_num_classes` is only bounds-checked; top3/top5 shares are indistinguishable with 2 classes.
- `test_local_cache.py:150`: deleting the threshold flush (`local_cache.py:246-247`) passes.
- `test_logger.py`:
  - :58: deleting `set_tracking_uri` (`logger.py:113`) passes.
  - :184: a no-op `_log_concept_matrix_artifact` passes (mutation was run).
  - :275: raw `log_dict` forwarded to wandb passes.
  - :288: the config value equals the default.
  - :333: 3 of 5 reconciliation artifacts are unchecked.
  - :359: the dataset tags (`logger.py:521-531`) are unchecked.
  - :510: `atomic=False` passes (mutation was run).
  - :572: the `"best-model"` filter is satisfied by `metadata.json`.
  - Noop tests at :706, :853, :900, :955: the `_ensure_active_run()` guards are unprotected.

## Bloat

- **CoralNet scraper/preprocessing:**
  - `test_preprocessing.py:69` (square; covered by the aspect-ratio test).
  - `:807` (round-trips its own df).
  - `test_jpeg_header.py:13` (covered by the parametrized test at :56).
  - `test_live_smoke.py:163` (exercises only stdlib `urljoin`).
- **CoralNet dedup/CLI:**
  - Schema change detectors: `test_schemas.py` :39, :45, :51, :58.
  - Argparse `_actions` tests: `test_cli.py:20`, `test_remediate_cli.py:21`.
  - `test_remediate_cli.py:36` (re-implements `main()`'s defaulting).
  - `test_cli.py:159` (a schema mirror).
- **Model:**
  - `test_taxonomy_loss.py:136, :177, :336`.
  - `test_distance_loss.py:116`.
  - `test_train_model_integration.py:110` (4 cases; epoch 0 always "improves").
  - `:215` (covered by `test_metric_policy`).
- **Inference/evaluation:**
  - `test_predictor_sampling.py:27`.
  - `test_bundle.py:60, :298, :305`.
  - `test_checkpoint_loader.py:60, :109`.
  - `test_predictor.py:45` (medium confidence).
  - The 3 `test_hierarchical_eval.py` tests (dead code).
- **Launcher/run_setup:**
  - `test_launch_training.py:243` (builds its own parser).
  - `test_launch_training.py:395`.
  - `test_sagemaker_train_entrypoint.py` :217, :227, :232, :460, :468, :554 (fold them into the :87/:98 table).
  - `test_integration.py:41, :52`.
- **Logger/datasets:**
  - `test_metric_policy.py:31` (constant tuple).
  - `test_concept_expr.py:69`.
  - `test_datasets.py` :186, :203 (2 cases), :381.
  - `test_logger.py` :295, :475, :586, :619, :644, :810, :737, :754.

## Gaps (no test covers these)

- **MetaModel loss wiring** (`meta.py:257-294`, `:529-544`): the CBM-only ValueError, `setdefault` weight inheritance, the distance `weight>0` gate, and adding the taxonomic/distance values to the loss. `TaxonomicLoss` also forwards its constructor kwargs untested (`taxonomy_loss.py:453-468,499`).
- **Loss reduction:** only tested at the defaults with B=1. Untested: `per_pixel_loss_weight>0`, multi-image batches, images with no valid pixels, the uniform-anneal midpoint (`:206`), the `boundary_px` masking inside the loss, and the underflow clamp (`:135`).
- **Training loop:**
  - `train_model`: the incomplete-warmup scheduler skip (`train.py:230-237`), the load-failure guard boundary (`:52-56`) and the test-eval cadence.
  - `train_epoch`: resume-offset reset, empty-batch skip, non-finite grad-norm skip, and the returned average loss (`meta.py:631-797`).
- **HierarchicalMetrics** (`model/hierarchical_metrics.py`): no direct tests. Untested: non-zero ECE (`:453-488`), rare-taxon strata, consistency variants, severity normalization, and the `max_distance` off-by-one (`:421`).
- **metrics.py:** out-of-range ground truth / predictions (`:52-58`), mIoU NaN and ignore handling (`:95-103`), the 0.5 threshold edge (`:240`), and the `to_dict` macros.
- **Inference:**
  - `CBMPredictor` eager `forward_features` / `forward_concept_logits` (`predictor.py:181-232`).
  - The `predict_tiled` `processing_resolution` branch.
  - Most `bundle.py` / `checkpoint.py` validation raises.
  - The forced `format="checkpoint"` in `scripts/evaluate.py:133-140`.
- **CoralNet dedup:**
  - Keep-copy tie-breaks (`dedup.py:227-231`).
  - The entire near-dup branch of `main()` (`:406-503`).
  - `near_dup_merge` failure paths and `load_labels`.
  - The `splits` anneal balancing.
  - The ETL `build-annotations` / `build-images` subcommands, which never run.
  - Remediation in non-dry-run mode.
- **CoralNet scraper:**
  - Most of `download_source` (`downloader.py:1159-1214`).
  - `_with_retry` give-up.
  - The bulk annotations CSV upload (`:515-521`).
- **Resize / inspect:**
  - `resize._validate_and_download` records transient S3 errors as `corrupted_unknown` skips (`resize.py:337-350,413-419`). This is likely unintended behavior, not only a gap.
  - `build_todo_from_checkpoint`.
  - Portrait images.
  - The INVALID_CHANNELS and CORRUPTED_HEADER branches in `inspect.py`.
- **Launcher:**
  - `job.env` merge (`launch_training.py:129`). It silently overrides `MERMAIDSEG_RUN_ID` and `CONTAINER_ENTRYPOINT_SCRIPT`, which breaks spot wandb resume. Worth a look as a bug.
  - `launcher_config` `kind` is ignored (`:86`).
  - `expandvars` behaves differently in the `job:` and `config:` blocks (`:89`).
  - Most of the entrypoint `main()` (`:512-570`).
- **concept_expr:**
  - Operator precedence without parentheses: swapping `+` and `*` passes everything.
  - Clipping (`:222-227`).
  - Most parser error branches.
- **local_cache / base_dataset / dataset_stats:**
  - Error and fallback branches (see the per-file lines in the logger/datasets section).
  - The reconciliation modules (`registry`, `concepts`, `label_mapping`, `taxonomy_tree`) are not reached by those tests.

## Unprotected functions (the body can be deleted with no test failing)

- **Scraper:**
  - Most of `downloader.py` (metadata, labelset, image URL resolution and downloads, login, cleanup).
  - All of `batch_download.py`.
  - `s3_io.delete_source_prefix`.
  - `parsers.extract_csrf_token`.
  - `resize.build_todo_from_checkpoint`.
- **Dedup:**
  - All of `hashing.py`.
  - `conflicts.main` (the source-ranking `recommend` logic).
  - `review.point_status`.
  - `splits.dedup_annotations` and `splits.main`.
  - `near_dup_merge.load_labels`.
  - `near_duplicates.candidate_pairs` and `phash_bits`.
  - `near_dup_review.label_progress`.
- **ETL:** `_cmd_build_annotations`, `_cmd_build_images` and `_maybe_upload`.
- **Remediate:** `cmd_all`, `_action_download_kwargs` and `main`.
- **Model:**
  - `MetaModel._abort_accumulation`.
  - The `_step_iteration_scheduler` warmup guard (dead).
  - `train.evaluate_and_log` (always monkeypatched).
- **Inference/eval:**
  - `CBMPredictor.activate_concept_logits` (used in `coralscapes_probe.py:355`).
  - `evaluation/predictor.make_eval_transform` and `preprocess_image`. Nothing checks that they match `inference.preprocess`.
  - `ClassConfusion.to_dict`.
  - The `HierarchicalMetrics` rare-taxon and table formatters.
- **Launcher:**
  - `run_setup.check_wandb_auth` and `check_env_wandb`.
  - `_cloudwatch_url`.
  - `_torch_load` (always monkeypatched).
- **Logger:**
  - `run_url`, `resume_run`, `resume_wandb_run`, `_safe_wandb_artifact_name` and `_log_concept_matrix_artifact`.
  - Every `WandbLogger` method except `__init__`. `save_checkpoint` and `log_datasets` are dead code.
- **local_cache:** `read_pil_image`, `read_json`, `setup_local_cache` and `configure_from_env`.
- **base_dataset:** `make_worker_init_fn` and `worker_init_fn`.

## Optional suggestions

- Add `"training_mode": "standard"` to `minimal_config`. Run `-m integration` in CI, and sync the `sagemaker` extra in `tests.yml`.
- Replace mirror expectations with literals:
  - `test_run_setup.py:29`.
  - The split-wiring source ids.
  - `test_datasets.py:174, :232`.
  - `test_checkpoint_loader.py:41`.
- Add `match=` to bare `pytest.raises` calls, for example `test_datasets.py:307` and `TestValInputSpec`. `test_required_args_enforced` should check that the exit code is 2.
- Give the loss and grad-accum tests distinct per-pixel and per-image inputs.
- Move the tile, banner and inferno tests out of `test_concept_expr.py`.
- Collapse the three `_reject_batchnorm_head` tests into one direct test of the helper.
- Remove the unnecessary `fresh_module` reload fixture in `test_coralnet_dataset_defaults.py:10`.
- `test_lr_schedule.py:120` is missing its `@pytest.mark.integration` marker.

## Follow-ups (out of scope for the 2026-11-04 fix pass)

These gaps were deliberately left for later:
- Scraper and downloader functions, and `batch_download.py`.
- Dedup `hashing.py`, `conflicts`, the near-dup branch of `dedup.main()`, and the `splits` anneal.
- The ETL `build-annotations` / `build-images` subcommands, and remediate in non-dry-run mode.
- `WandbLogger` methods, the `local_cache` readers, and `base_dataset` worker init.
- The `dataset_reconciliation` modules (`registry`, `concepts`, `label_mapping`, `taxonomy_tree`).
- `launcher_config.parse_run_config` accepts but ignores `kind`. It is kept as is to match mermaid-classifier's schema.
- The `MetaModel._step_iteration_scheduler` warmup guard can never fire from its only caller. It is kept as a defensive check, so deleting it is an equivalent mutation, not a gap.
- The preprocessing live tests (`test_preprocessing_live.py`) were removed on 2026-11-04. The offline resize tests in `test_preprocessing.py` cover the same logic, and the live copies had gone stale unnoticed.
