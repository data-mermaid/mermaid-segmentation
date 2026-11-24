"""Load a dataset's curated label map from the packaged CSV."""

from __future__ import annotations

from importlib import resources
from pathlib import Path

import pyarrow as pa
import pyarrow.csv as pacsv

CURATED_SCHEMA = pa.schema(
    [
        ("label_id", pa.int32()),
        ("name", pa.string()),
        ("mermaid_benthic_attribute", pa.string()),
    ]
)


def load_curated(dataset: str, *, root: Path | None = None) -> pa.Table:
    """Read ``<dataset>.csv`` from the packaged ``curated`` directory, or from ``root`` when given.

    An empty cell is null.
    """
    directory = root if root is not None else resources.files(__package__) / "curated"
    with resources.as_file(directory / f"{dataset}.csv") as path:
        return pacsv.read_csv(
            path,
            convert_options=pacsv.ConvertOptions(
                column_types=CURATED_SCHEMA, strings_can_be_null=True
            ),
        )
