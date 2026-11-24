"""CoralNet labels mapped to MERMAID benthic attributes by the MERMAID LabelMapping API.

Fetching, parsing and digesting match ``mermaidseg.dataset_reconciliation.label_mapping``; the
code is duplicated because ``segdata_etl`` never imports ``mermaidseg``. Rows join on the integer
CoralNet label id, so a label renamed in CoralNet still maps.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Final, Literal, TypedDict

import pyarrow as pa
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

from segdata_spec.manifest import LabelMapping
from segdata_spec.schema import LABELS_SCHEMA, conform

logger = logging.getLogger(__name__)

CORALNET_LABELMAPPINGS_URL = (
    "https://api.datamermaid.org/v1/classification/labelmappings/?provider=CoralNet&limit=1000"
)
MAPPING_SOURCE: Final[Literal["mermaid_api"]] = "mermaid_api"
_REQUEST_TIMEOUT_SEC = 30
_RETRY_BACKOFF_SEC = 1


class CoralNetLabelMapping(TypedDict):
    """One CoralNet row from the MERMAID LabelMapping API."""

    id: str | None
    benthic_attribute_id: str | None
    benthic_attribute_name: str | None
    growth_form_id: str | None
    growth_form_name: str | None
    provider: str | None
    provider_id: int
    provider_label: str | None


@dataclass(frozen=True)
class CoralNetLabelMappings:
    """The CoralNet label mappings a run used, keyed by CoralNet label id.

    Attributes:
        records: API rows keyed by integer CoralNet label id (``provider_id``).
        source: ``"api"`` when fetched for this run, ``"snapshot"`` when read offline.
        fetched_at: UTC ISO time the rows were fetched from the API.
    """

    records: dict[int, CoralNetLabelMapping]
    source: Literal["api", "snapshot"]
    fetched_at: str

    @property
    def sha256(self) -> str:
        """Digest of the records, independent of their order."""
        return coralnet_mapping_digest(self.records)

    def to_json(self) -> dict[str, Any]:
        """The snapshot JSON shape: ``fetched_at``, ``sha256`` and the records by id."""
        return {
            "fetched_at": self.fetched_at,
            "sha256": self.sha256,
            "records": [self.records[k] for k in sorted(self.records)],
        }


def snapshot_json_text(mappings: CoralNetLabelMappings) -> str:
    """Serialise ``mappings`` as the snapshot file text, for saving beside a run's outputs."""
    return json.dumps(mappings.to_json(), indent=1, sort_keys=True)


def _retrying_session() -> requests.Session:
    """A session that retries a GET up to 3 times on connection errors and 502/503/504."""
    retry = Retry(
        total=3,
        backoff_factor=_RETRY_BACKOFF_SEC,
        status_forcelist=(502, 503, 504),
        allowed_methods=frozenset({"GET"}),
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.mount("http://", HTTPAdapter(max_retries=retry))
    return session


def _fetch_paginated_results(url: str) -> list[dict]:
    """Page through a MERMAID list endpoint until ``next`` is exhausted.

    Raises:
        requests.RequestException: A request still fails after its retries.
        ValueError: A ``next`` URL repeats a page already fetched.
    """
    records: list[dict] = []
    seen: set[str] = set()
    next_url: str | None = url
    with _retrying_session() as session:
        while next_url:
            if next_url in seen:
                raise ValueError(f"Pagination loop: {next_url} was already fetched")
            seen.add(next_url)
            response = session.get(next_url, timeout=_REQUEST_TIMEOUT_SEC)
            response.raise_for_status()
            data = response.json()
            records.extend(data["results"])
            next_url = data.get("next")
    return records


def _normalize_provider_id(provider_id: object) -> int | None:
    """Return ``provider_id`` as an int (``82``, ``82.0`` and ``"82.0"`` give 82), else None."""
    try:
        value = float(provider_id)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not value.is_integer():
        return None
    return int(value)


def parse_coralnet_label_mappings(rows: list[dict]) -> dict[int, CoralNetLabelMapping]:
    """Key raw LabelMapping rows by integer CoralNet label id.

    Raises:
        ValueError: If a row's ``provider_id`` is not an integer, or two rows share a CoralNet
            label id.
    """
    records: dict[int, CoralNetLabelMapping] = {}
    for row in rows:
        provider_id = _normalize_provider_id(row.get("provider_id"))
        if provider_id is None:
            raise ValueError(
                f"CoralNet label mapping has non-integer provider_id {row.get('provider_id')!r}"
            )
        if provider_id in records:
            raise ValueError(f"Duplicate CoralNet label mapping for provider_id {provider_id}")
        records[provider_id] = CoralNetLabelMapping(
            id=row.get("id"),
            benthic_attribute_id=row.get("benthic_attribute_id"),
            benthic_attribute_name=row.get("benthic_attribute_name"),
            growth_form_id=row.get("growth_form_id"),
            growth_form_name=row.get("growth_form_name"),
            provider=row.get("provider"),
            provider_id=provider_id,
            provider_label=row.get("provider_label"),
        )
    return records


def coralnet_mapping_digest(records: Mapping[int, CoralNetLabelMapping]) -> str:
    """sha256 of the records as canonical JSON (sorted ids and keys)."""
    canonical = json.dumps([records[k] for k in sorted(records)], sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


def fetch_coralnet_label_mappings(url: str = CORALNET_LABELMAPPINGS_URL) -> CoralNetLabelMappings:
    """Fetch every CoralNet row from the MERMAID LabelMapping API.

    Network and HTTP errors propagate: a run never falls back to a stale mapping silently.
    """
    records = parse_coralnet_label_mappings(_fetch_paginated_results(url))
    logger.info("Fetched %d CoralNet label mappings from the MERMAID API", len(records))
    return CoralNetLabelMappings(
        records=records, source="api", fetched_at=datetime.now(UTC).isoformat()
    )


def api_labels(
    id2name: Mapping[int, str | None], mappings: CoralNetLabelMappings
) -> tuple[pa.Table, LabelMapping]:
    """Build ``labels.parquet`` rows and the mapping provenance for the labels in ``id2name``.

    A label is unmapped when its name is not a string, it has no API record, or the record has no
    benthic attribute. A None name is stored as an empty string because ``name`` is not nullable.

    Args:
        id2name: CoralNet label id -> name, for the ids the dataset uses.
        mappings: The API rows, keyed by CoralNet label id.
    """
    label_ids = list(id2name)
    mapped: list[CoralNetLabelMapping | None] = []
    for label_id in label_ids:
        record = mappings.records.get(label_id) if isinstance(id2name[label_id], str) else None
        has_target = record is not None and record["benthic_attribute_name"] is not None
        mapped.append(record if has_target else None)
    table = conform(
        pa.table(
            {
                "label_id": label_ids,
                "name": [id2name[i] or "" for i in label_ids],
                "mermaid_benthic_attribute_id": [r and r["benthic_attribute_id"] for r in mapped],
                "mermaid_benthic_attribute": [r and r["benthic_attribute_name"] for r in mapped],
                "mermaid_growth_form": [r and r["growth_form_name"] for r in mapped],
                "mapping_source": [MAPPING_SOURCE] * len(label_ids),
            }
        ),
        LABELS_SCHEMA,
    )
    mapping = LabelMapping(
        source=MAPPING_SOURCE,
        sha256=mappings.sha256,
        fetched_at=datetime.fromisoformat(mappings.fetched_at),
        n_labels=len(label_ids),
        n_mapped=sum(r is not None for r in mapped),
    )
    return table, mapping
