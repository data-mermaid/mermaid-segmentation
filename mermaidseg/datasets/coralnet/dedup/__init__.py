"""CoralNet deduplication: find images uploaded to several sources, keep one copy, merge their
annotations and split the result into leak-free train/val/test sets.

Run in this order (each step is ``python -m mermaidseg.datasets.coralnet.dedup.<step>``):

1. ``hashing``: content-hash every image and list the source pairs sharing images (~35 min).
2. ``conflicts``: compare the labels on shared points and rank the sources of each cluster.
3. ``dedup``: image blacklist, merged annotations, point provenance and manual-review files.
4. ``near_duplicates``: perceptual near duplicates (re-encoded, resized, cropped copies) among the
   images exact deduplication kept (~1 h). Label the pairs in ``nbs/Near_Duplicate_Review.ipynb``
   (``configs/coralnet/near_dup_labels.csv``), then rerun ``dedup``: it also removes the pairs
   labelled ``dup`` and maps their points onto the kept copy (``near_dup_merge``).
5. ``mermaidseg.datasets.coralnet.scraper.source_metadata``: CoralNet source names, owners and
   locations (once; resumable).
6. ``splits``: grouped, class-balanced train/val/test split; writes the data config. The groups of
   ``configs/coralnet/test_sources.txt`` go to test.
7. ``leakage``: check that no image is shared between the splits of a data config.

``review`` / ``near_dup_review`` hold the helpers of ``nbs/Dedup_Review.ipynb`` /
``nbs/Near_Duplicate_Review.ipynb``. Intermediate outputs go to ``data/coralnet_dedup/``; the files
training reads (``image_blacklist.parquet``, ``annotation_overrides.parquet``), the curated inputs
and ``source_splits.csv`` go to ``configs/coralnet/``.
"""
