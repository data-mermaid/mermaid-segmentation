"""Concept helpers: label map to per-pixel concept vectors, and the benthic-attribute hierarchy fetch."""

from __future__ import annotations

import torch

from mermaidseg.dataset_reconciliation import concepts as concepts_module
from mermaidseg.dataset_reconciliation.concepts import source_labels_to_concepts

# Row i is the concept vector of label i: label 0 (background) is all zeros.
LOOKUP = torch.tensor([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])


def test_non_square_label_map_becomes_channels_first_concepts():
    labels = torch.tensor([[[1, 0, 2], [2, 2, 1]]])  # (B=1, H=2, W=3)

    out = source_labels_to_concepts(labels, LOOKUP)

    assert out.shape == (1, 3, 2, 3)
    assert out[0, :, 0, 0].tolist() == [1.0, 2.0, 3.0]  # label 1 at (0, 0)
    assert out[0, :, 0, 1].tolist() == [0.0, 0.0, 0.0]  # background at (0, 1)
    assert out[0, :, 1, 0].tolist() == [4.0, 5.0, 6.0]  # label 2 at (1, 0)
    assert out[0, :, 1, 2].tolist() == [1.0, 2.0, 3.0]  # label 1 at (1, 2)


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload
        self.raise_called = False

    def raise_for_status(self) -> None:
        self.raise_called = True

    def json(self) -> dict:
        return self._payload


def test_benthic_hierarchy_follows_pagination_and_maps_names_to_parent_names(monkeypatch):
    pages = {
        "https://api.test/ba/": {
            "next": "https://api.test/ba/?page=2",
            "results": [
                {"id": 1, "name": "Hard coral", "parent": None},
                {"id": 2, "name": "Acropora", "parent": 1},
            ],
        },
        "https://api.test/ba/?page=2": {
            "next": None,
            "results": [{"id": 3, "name": "Acropora palmata", "parent": 2}],
        },
    }
    calls: list[tuple[str, dict]] = []
    responses: list[_FakeResponse] = []

    def fake_get(url: str, **kwargs):
        calls.append((url, kwargs))
        responses.append(_FakeResponse(pages[url]))
        return responses[-1]

    monkeypatch.setattr(concepts_module.requests, "get", fake_get)

    hierarchy = concepts_module.initialize_benthic_hierarchy("https://api.test/ba/")

    assert hierarchy == {
        "Hard coral": None,
        "Acropora": "Hard coral",
        "Acropora palmata": "Acropora",
    }
    assert [url for url, _ in calls] == list(pages)
    assert all(kwargs.get("timeout") for _, kwargs in calls)  # a request never waits forever
    assert all(r.raise_called for r in responses)
