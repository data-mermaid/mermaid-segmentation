"""torch.export graph format: dynamic-size parity with the eager bundle, and the parity gate."""

from __future__ import annotations

import pytest
import torch

from mermaidseg.inference import ParityError, export_graph, load_bundle


@pytest.mark.parametrize("size_hw", [(64, 64), (96, 96)])
def test_graph_matches_eager_at_several_input_sizes(graph_bundle, size_hw):
    eager = load_bundle(graph_bundle, format="checkpoint")
    graph = load_bundle(graph_bundle, format="graph")
    assert graph.format == "graph"

    x = torch.randn(1, 3, *size_hw, generator=torch.Generator().manual_seed(2))
    with torch.no_grad():
        expected = eager.model(x)
        class_logits, concept_probs = graph.model.module()(x)
    assert torch.allclose(class_logits, expected.logits, atol=1e-5, rtol=0)
    assert torch.allclose(concept_probs, expected.hidden_states, atol=1e-5, rtol=0)


def test_auto_format_prefers_the_graph_when_present(graph_bundle):
    assert load_bundle(graph_bundle).format == "graph"


def test_export_graph_refuses_a_graph_outside_tolerance(graph_bundle, tmp_path):
    eager = load_bundle(graph_bundle, format="checkpoint")
    with pytest.raises(ParityError):
        export_graph(eager.model, (64, 64), tmp_path / "model.pt2", atol=-1.0)
    assert not (tmp_path / "model.pt2").exists()
