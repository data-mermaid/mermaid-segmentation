# DataLoader host-RAM leak — diagnosis + fix validation

**Question:** did `persistent_workers=True` + copy-on-write (CoW) churn on `BaseCoralDataset`'s
Python-object structures cause the `dinov3-lora-qv-r8` OOM (host RAM 8→25 GB over ~45h, GPU flat)?
And does the numpy-native hot path fix it while keeping `persistent_workers=True`?

## Method

`scripts/diagnostics/dataloader_rss_repro.py` — a synthetic `BaseCoralDataset` (large
`df_annotations`, `read_image` stubbed to zeros) so we exercise ONLY the structure-access hot path,
fast, with no S3. DataLoaders forced to `multiprocessing_context="fork"` (the Linux/SageMaker
mechanism). Matrix: `num_workers=0`, `=N persistent=False`, `=N persistent=True`.

**Metric = USS (unique/private set size), not summed RSS.** Fork shares pages; summed worker RSS
double-counts inherited shared pages (the frame, the numpy arrays), which inflates the figure with
memory that is *shared, not leaked*. USS counts only pages a process actually privatized — i.e. the
real CoW cost. (An earlier RSS-summed pass over-stated everything; USS is authoritative.)

## Result 1 — persistent workers retain what non-persistent workers reclaim

At 3M rows / 6 workers / 6 epochs, worker USS at each epoch END:

- `num_workers=0`: flat (~no workers) → growth is worker/fork-specific.
- `persistent=False`: workers reach ~500 MB mid-epoch, **reset to ~0 every epoch boundary** → fully reclaimed.
- `persistent=True`: **monotonic, never resets** → the accumulation lives in the persistent workers.

## Result 2 — the leak scales with `df_annotations` size, and the fix removes most of it

Worker USS (epoch 5, `num_workers=6, persistent=True`), baseline vs the numpy-native hot path:

| `df_annotations` rows | baseline | fixed | Δ |
|---|---|---|---|
| 3,000,000  | 2288 MB | 2131 MB | −157 MB |
| 12,000,000 | **4238 MB** | **2619 MB** | **−1619 MB** |

- **Baseline scales ~217 MB per 1M rows** — per-sample access to the object columns of
  `df_annotations` (`image_id`, `source_label_name`) and the str-keyed
  `_annotation_positions_by_image` dict increfs scattered Python objects, privatizing their pages in
  every worker. The cost is paid **per worker** (6×) and **grows** as more of the frame is touched.
- **Fixed scales ~54 MB per 1M rows** (~75% less). The hot path now slices contiguous numpy arrays
  (`_ann_offsets/_ann_row/_ann_col/_ann_label_id`) — no Python-object refcounts, so nothing
  privatizes. The benefit **grows with frame size**: at 12M rows the fix already saves 1.6 GB, so at
  production scale (tens of millions of rows) it saves many GB — the margin that prevents the OOM.

## Result 3 — the residual is shared, not a leak (and it isn't `df_images`)

Ablation at 12M rows, `--skip-dfimages` (bypass the per-sample `df_images.iloc[idx].to_dict()`):
worker USS = 2612 MB vs 2619 MB with it — **identical**. So `df_images` is not the residual.

What remains in "fixed" is (a) workers reading the large **shared, read-only** numpy annotation
arrays (288 MB at 12M) and (b) fixed per-worker working set (torch runtime, prefetch batch buffers,
the per-sample mask allocation). Critically, the numpy arrays are **shared across all workers via
fork** — on Linux they cost ~288 MB *once*, never privatized. The harness *sums* per-worker USS, so
macOS over-counts these shared reads; on the real Linux box the fixed footprint is lower than the
table shows.

## Result 4 — authoritative Linux run (in the training image, real fork + /proc USS)

macOS `psutil.uss` over-counts fork-shared pages; the true test is real Linux. Run inside the
training image (`make rss-repro-docker` — Docker Desktop is a real Linux kernel, so fork CoW and
`/proc` USS are genuine; only CPU is amd64-emulated, irrelevant to a memory measurement),
6M rows / 100k images / 4 workers / 5 epochs, worker USS at each epoch END:

| config | epoch 0→4 | net drift |
|---|---|---|
| baseline `persist1` | 983 → 998 → 1009 → 1018 → 1025 MB | **+42 MB** (Δ 15,11,9,7) |
| fixed `persist1`    | 935 → 942 → 947 → 951 → 954 MB | **+19 MB** (Δ 7,5,4,3) |

`num_workers=0` and `persist=False` are flat / fully reclaimed in both. On real Linux the fix
**more than halves** the per-epoch worker drift, and both curves decelerate (CoW saturates as the
frame's pages get touched).

## Conclusion

- **Cause confirmed on real Linux:** persistent forked workers accumulate CoW pages from
  per-sample Python-object access; `num_workers=0` / `persistent=False` don't. The fix converts a
  per-worker-private, string-refcount cost into a shared, read-only numpy cost — Linux worker drift
  halves and per-sample hot-path CPU drops **8.7×** (0.231 → 0.027 ms/sample; frees the DataLoader).
- **Magnitude caveat / definitive next test.** This harness stubs `read_image` to zeros, so it
  isolates the *structure-access* driver but omits real JPEG-decode allocator churn and uses short
  synthetic strings. At synthetic scale the absolute drift is modest; the production 17 GB climb is
  reproduced only by *extrapolating the row-scaling* (baseline ~217 MB/1M rows) to the real frame's
  tens of millions of rows. So the fix is a **validated, large improvement**, but the harness has
  **not** reproduced the full 17 GB magnitude. The definitive confirmation is the next real training
  run instrumented with `MERMAIDSEG_LOG_WORKER_RSS=1` — watch the MLflow
  `system/system_memory_usage_megabytes` curve stay flat — or the harness `--from-run` mode on real
  data/hardware (real image decode).
- **Backstops kept:** `persistent_workers=False` (B2 knob, full reclaim each epoch) and larger-RAM
  instances (B4 advisory) remain available until the real-run canary confirms the fix alone holds.
- **`_load_failures`** (a genuinely unbounded list, distinct from CoW which saturates) is bounded in B3.

Repro: `make rss-repro-docker` (authoritative Linux), or locally on macOS with
`OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES python scripts/diagnostics/dataloader_rss_repro.py ...`
(fork semantics are what matter). Add `--skip-dfimages` for the df_images ablation.
