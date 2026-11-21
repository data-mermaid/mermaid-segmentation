"""Source links that keep sources on one side of the split, and the groups they form.

Sources are linked when they share a byte-identical image (same ``(crc64nvme, size)``), when a
pair of their images is labelled ``dup`` or ``unsure`` in the near-duplicate review, or when they
belong to one project (a shared admin, and close in ID rank or in location).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

Edge = tuple[int, int]
IMAGE_KEY = ["crc64nvme", "size"]
LINKING_LABELS = ("dup", "unsure")


def haversine_km(lat1: Any, lon1: Any, lat2: Any, lon2: Any) -> Any:
    """Great-circle distance in km; NaN when any coordinate is NaN."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    )
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def project_edges(
    meta: pd.DataFrame, sources: list[int], max_id_gap: int, max_km: float
) -> list[tuple[int, int, str]]:
    """Pairs of sources sharing an admin that are close in ID or in location."""
    rank = {s: i for i, s in enumerate(sources)}
    meta = meta[meta["source_id"].isin(rank) & meta["admins"].notna()].set_index("source_id")
    by_admin = meta["admins"].str.split(";").explode()
    edges: dict[Edge, str] = {}
    for _, ids in by_admin.groupby(by_admin):
        ids = sorted(set(ids.index))
        for i, a in enumerate(ids):
            for b in ids[i + 1 :]:
                if (a, b) in edges:
                    continue
                near_id = abs(rank[a] - rank[b]) <= max_id_gap
                near_loc = (
                    haversine_km(
                        meta.loc[a, "latitude"],
                        meta.loc[a, "longitude"],
                        meta.loc[b, "latitude"],
                        meta.loc[b, "longitude"],
                    )
                    <= max_km
                )  # NaN -> False
                if near_id or near_loc:
                    edges[a, b] = "id" if near_id else "location"
    return [(a, b, why) for (a, b), why in edges.items()]


def shared_image_edges(keys: pd.DataFrame, sources: Iterable[int]) -> tuple[set[Edge], set[int]]:
    """Link sources that hold an image with the same ``(crc64nvme, size)``.

    Args:
        keys: One row per image with ``source_id``, ``crc64nvme`` and ``size``.
        sources: The universe; images of other sources are ignored.

    Returns:
        The edges, and the sources that share at least one image with another source. Images
        without a CRC64NVME are not compared.
    """
    keys = keys[keys["source_id"].isin(set(sources)) & keys["crc64nvme"].notna()]
    keys = keys[["source_id", *IMAGE_KEY]].drop_duplicates()
    shared = keys[keys.duplicated(IMAGE_KEY, keep=False)]
    by_key = shared.groupby(IMAGE_KEY)["source_id"]
    cross = set(shared.loc[by_key.transform("nunique") > 1, "source_id"])
    edges = {
        (int(min(a, b)), int(max(a, b)))
        for _, ids in by_key
        for a, b in zip(ids.iloc[:-1], ids.iloc[1:], strict=True)
    }
    return edges, {int(s) for s in cross}


def load_labels(path: Path | str) -> pd.DataFrame:
    """Latest label per pair (rows of the append-only labels CSV), keyed by ``(image_a, image_b)``."""
    d = pd.read_csv(path, dtype={"image_a": str, "image_b": str})
    return d.drop_duplicates(["image_a", "image_b"], keep="last").set_index(["image_a", "image_b"])


def near_dup_edges(labels: pd.DataFrame, sources: Iterable[int]) -> set[Edge]:
    """Link the two sources of every pair labelled ``dup`` or ``unsure``.

    Same-source pairs and pairs with a source outside ``sources`` are ignored.
    """
    universe = set(sources)
    near = labels[
        labels["label"].isin(LINKING_LABELS)
        & (labels["source_a"] != labels["source_b"])
        & labels["source_a"].isin(universe)
        & labels["source_b"].isin(universe)
    ]
    return {(min(a, b), max(a, b)) for a, b in near[["source_a", "source_b"]].to_numpy().tolist()}


def components(sources: Sequence[int], edges: Iterable[Edge]) -> list[set[int]]:
    """Connected components of ``sources`` under ``edges``, largest first, then by lowest id."""
    graph = nx.Graph()
    graph.add_nodes_from(sources)
    graph.add_edges_from((a, b) for a, b in edges if a != b)
    return sorted(nx.connected_components(graph), key=lambda c: (-len(c), min(c)))
