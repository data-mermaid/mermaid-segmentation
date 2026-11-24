from __future__ import annotations

import pyarrow as pa

from segdata_etl.label_maps.curated import load_curated


def test_load_curated_types_columns_and_nulls_empty_cells(tmp_path):
    (tmp_path / "demo.csv").write_text(
        "label_id,name,mermaid_benthic_attribute\n1,Hard coral,Hard coral\n2,Sand,\n"
    )
    table = load_curated("demo", root=tmp_path)
    assert table.schema == pa.schema(
        [
            ("label_id", pa.int32()),
            ("name", pa.string()),
            ("mermaid_benthic_attribute", pa.string()),
        ]
    )
    assert table.to_pydict() == {
        "label_id": [1, 2],
        "name": ["Hard coral", "Sand"],
        "mermaid_benthic_attribute": ["Hard coral", None],
    }
