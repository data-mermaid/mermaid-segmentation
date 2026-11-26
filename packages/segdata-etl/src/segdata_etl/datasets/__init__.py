"""Dataset builds: one module per source, each staging a version for ``segdata promote``.

``mermaid``, ``coralnet`` (a package), ``coralscapes_v2`` and ``ucsd_mosaics`` build from their own
sources. ``catlin_seaview``, ``moorea_labeled_corals``, ``pacific_labeled_corals`` and
``benthos_yuval`` build from the prefix ``segdata ingest`` published; ``external.py`` is their
shared helper.
"""
