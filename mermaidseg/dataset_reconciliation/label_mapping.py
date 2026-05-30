"""Source-dataset to MERMAID benthic-attribute target-label mappings.

Provides the HTTP fetchers + static dicts that translate a source-space label
(CoralNet provider IDs, Coralscapes 1..39 names, MERMAID benthic-attribute
names) into MERMAID benthic-attribute target names, plus the GPU helper
``source_labels_to_target_labels`` used at training time.
"""

from __future__ import annotations

import requests
import torch


def fetch_mermaid_target_labels(
    benthicattributes_url: str = "https://api.datamermaid.org/v1/benthicattributes/",
) -> list[str]:
    """Fetch the canonical MERMAID benthic-attribute label names.

    Returns:
        Alphabetically sorted list of unique benthic-attribute names.
    """
    response = requests.get(benthicattributes_url, timeout=30)
    response.raise_for_status()
    data = response.json()
    records = list(data["results"])
    while data.get("next"):
        response = requests.get(data["next"], timeout=30)
        response.raise_for_status()
        data = response.json()
        records.extend(data["results"])
    return sorted({rec["name"] for rec in records if rec.get("name") is not None})


def fetch_coralnet_to_mermaid(
    mapping_endpoint: str = "https://api.datamermaid.org/v1/classification/labelmappings/?provider=CoralNet",
) -> dict[str, str]:
    """Fetch the CoralNet provider label -> MERMAID benthic-attribute name mapping.

    Returns a dict keyed by stringified CoralNet provider label with values equal
    to the MERMAID benthic-attribute name (or ``None`` if the CoralNet label
    is not yet mapped).
    """
    response = requests.get(mapping_endpoint, timeout=30)
    response.raise_for_status()
    data = response.json()
    labelset = list(data["results"])
    while data.get("next"):
        response = requests.get(data["next"], timeout=30)
        response.raise_for_status()
        data = response.json()
        labelset.extend(data["results"])
    return {str(label["provider_label"]): label["benthic_attribute_name"] for label in labelset}


def fetch_catlin_seaview_to_mermaid(
    mapping_endpoint: str = "https://api.datamermaid.org/v1/classification/labelmappings/?provider=Catlin%20Seaview",
) -> dict[str, str]:
    """Fetch the Catlin Seaview label-name -> MERMAID benthic-attribute name mapping.

    Mirrors :func:`fetch_coralnet_to_mermaid`: pages through the MERMAID API
    label-mappings endpoint filtered to ``provider=Catlin Seaview`` and
    returns a dict keyed by the Catlin Seaview ``provider_id`` (which holds
    the original Catlin label name) with values equal to the MERMAID
    benthic-attribute name (or ``None`` if the Catlin label is not yet
    mapped, in which case it collapses to background at training time).
    """
    response = requests.get(mapping_endpoint, timeout=30)
    response.raise_for_status()
    data = response.json()
    labelset = list(data["results"])
    while data.get("next"):
        response = requests.get(data["next"], timeout=30)
        response.raise_for_status()
        data = response.json()
        labelset.extend(data["results"])
    return {str(label["provider_id"]): label["benthic_attribute_name"] for label in labelset}


def fetch_moorea_labeled_corals_to_mermaid(
    mapping_endpoint: str = "https://api.datamermaid.org/v1/classification/labelmappings/?provider=Moorea%20Labeled%20Corals",
) -> dict[str, str]:
    """Fetch the Moorea Labeled Corals label-name -> MERMAID benthic-attribute name mapping.

    Mirrors :func:`fetch_catlin_seaview_to_mermaid`: pages through the MERMAID
    API label-mappings endpoint filtered to ``provider=Moorea Labeled
    Corals`` and returns a dict keyed by the Moorea ``provider_id`` (which
    holds the original Moorea label name, e.g. ``"Acropora"`` or ``"Turf"``)
    with values equal to the MERMAID benthic-attribute name (or ``None`` if
    the Moorea label is not yet mapped, in which case it collapses to
    background at training time).

    Note: the Moorea Labeled Corals provider mapping has not yet been
    populated on the MERMAID side at the time of writing -- this fetcher will
    return an empty dict in that case, which causes every Moorea source label
    to fall back to background through :class:`SourceLabelRegistry`.
    """
    response = requests.get(mapping_endpoint, timeout=30)
    response.raise_for_status()
    data = response.json()
    labelset = list(data["results"])
    while data.get("next"):
        response = requests.get(data["next"], timeout=30)
        response.raise_for_status()
        data = response.json()
        labelset.extend(data["results"])
    return {str(label["provider_id"]): label["benthic_attribute_name"] for label in labelset}


def fetch_pacific_labeled_corals_to_mermaid(
    mapping_endpoint: str = "https://api.datamermaid.org/v1/classification/labelmappings/?provider=Pacific%20Labeled%20Corals",
) -> dict[str, str]:
    """Fetch the Pacific Labeled Corals label-name -> MERMAID benthic-attribute name mapping.

    Mirrors :func:`fetch_moorea_labeled_corals_to_mermaid`: pages through the
    MERMAID API label-mappings endpoint filtered to ``provider=Pacific
    Labeled Corals`` and returns a dict keyed by the Pacific Labeled Corals
    ``provider_id`` (which holds the original Pacific label name from the
    per-site ``labelmap.txt``, e.g. ``"Acropora"`` or ``"CCA"``) with values
    equal to the MERMAID benthic-attribute name (or ``None`` if the Pacific
    label is not yet mapped, in which case it collapses to background at
    training time).

    Note: the Pacific Labeled Corals provider mapping has not yet been
    populated on the MERMAID side at the time of writing -- this fetcher
    will return an empty dict in that case, which causes every Pacific
    Labeled Corals source label to fall back to background through
    :class:`SourceLabelRegistry`.
    """
    response = requests.get(mapping_endpoint, timeout=30)
    response.raise_for_status()
    data = response.json()
    labelset = list(data["results"])
    while data.get("next"):
        response = requests.get(data["next"], timeout=30)
        response.raise_for_status()
        data = response.json()
        labelset.extend(data["results"])
    return {str(label["provider_id"]): label["benthic_attribute_name"] for label in labelset}


def fetch_benthos_yuval_to_mermaid(
    mapping_endpoint: str = "https://api.datamermaid.org/v1/classification/labelmappings/?provider=Benthos%20Yuval",
) -> dict[str, str]:
    """Fetch the Benthos Yuval label-name -> MERMAID benthic-attribute name mapping.

    Mirrors :func:`fetch_moorea_labeled_corals_to_mermaid`: pages through the
    MERMAID API label-mappings endpoint filtered to ``provider=Benthos
    Yuval`` and returns a dict keyed by the Benthos Yuval ``provider_id``
    (which holds the original Benthos label name from
    ``dictionary_labels.txt`` / ``dictionary_labelsDW.txt``, e.g.
    ``"Acropora"`` or ``"Sand"``) with values equal to the MERMAID
    benthic-attribute name (or ``None`` if the Benthos label is not yet
    mapped, in which case it collapses to background at training time).

    Note: the Benthos Yuval provider mapping has not yet been populated on
    the MERMAID side at the time of writing -- this fetcher will return an
    empty dict in that case, which causes every Benthos Yuval source label
    to fall back to background through :class:`SourceLabelRegistry`.
    """
    response = requests.get(mapping_endpoint, timeout=30)
    response.raise_for_status()
    data = response.json()
    labelset = list(data["results"])
    while data.get("next"):
        response = requests.get(data["next"], timeout=30)
        response.raise_for_status()
        data = response.json()
        labelset.extend(data["results"])
    return {str(label["provider_id"]): label["benthic_attribute_name"] for label in labelset}


def coralscapes_to_mermaid() -> dict[str, list[str]]:
    """Static Coralscapes 39-class -> MERMAID benthic-attribute mapping.

    Mapping was previously embedded inside ``CoralscapesDataset``. The first
    element of each value list is treated as the canonical MERMAID label;
    subsequent elements are alternative interpretations not currently used.
    """
    return {
        "human": ["Unknown"],
        "background": ["Unknown", "Obscured"],
        "fish": ["Unknown"],
        "sand": ["Sand"],
        "rubble": ["Rubble"],
        "unknown hard substrate": ["Bare substrate"],
        "algae covered substrate": ["Turf algae"],
        "dark": ["Unknown"],
        "branching bleached": ["Bleached coral"],
        "branching dead": ["Dead coral"],
        "branching alive": ["Hard coral"],
        "stylophora alive": ["Stylophora"],
        "pocillopora alive": ["Pocillopora"],
        "acropora alive": ["Acropora"],
        "table acropora alive": ["Acropora"],
        "table acropora dead": ["Dead coral"],
        "millepora": ["Milleporidae"],
        "turbinaria": ["Turbinaria reniformis"],
        "other coral": ["Bleached coral"],
        "other coral dead": ["Dead coral"],
        "other coral alive": ["Hard coral"],
        "other coral bleached": ["Bleached coral"],
        "massive/meandering alive": ["Hard coral"],
        "massive/meandering dead": ["Dead coral"],
        "massive/meandering bleached": ["Bleached coral"],
        "meandering alive": ["Hard coral"],
        "meandering dead": ["Dead coral"],
        "meandering bleached": ["Bleached coral"],
        "transect line": ["Tape"],
        "transect tools": ["Unknown"],
        "sea urchin": ["Sea urchin"],
        "sea cucumber": ["Sea cucumber"],
        "anemone": ["Anemone"],
        "sponge": ["Sponge"],
        "clam": ["Tridacna giant clam"],
        "other animal": ["Other invertebrates"],
        "trash": ["Trash"],
        "seagrass": ["Seagrass"],
        "crown of thorn": ["Acanthaster planci"],
        "dead clam": ["Unknown"],
    }


def source_labels_to_target_labels(
    source_labels: torch.Tensor,
    lookup: torch.Tensor,
) -> torch.Tensor:
    """Map a global source-label segmentation map to target MERMAID label IDs.

    Pure GPU index op; ``lookup`` must live on the same device as
    ``source_labels``.

    Args:
        source_labels: Integer source-label map of shape ``(B, H, W)`` with
            values in ``[0, N]``.
        lookup: 1-D long tensor of shape ``(N+1,)`` mapping each global source
            label (0 = background) to its MERMAID target label ID
            (0 = background / unmapped).

    Returns:
        Long tensor of shape ``(B, H, W)`` containing target MERMAID label IDs.
    """
    if source_labels.dtype != torch.long:
        source_labels = source_labels.long()
    return lookup[source_labels]
