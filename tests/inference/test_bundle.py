"""Sidecar → export → load round trip, and the load-time guards on a bundle."""

from __future__ import annotations

import json
import logging

import pytest
import torch
from safetensors.torch import load_file, save_file

import mermaidseg.inference.bundle as bundle_module
import mermaidseg.model.models as mm_models
from mermaidseg.inference import ManifestError, ParityError, export_bundle, load_bundle
from mermaidseg.io import ConfigDict
from mermaidseg.model.models import ConceptBottleneckDPTDINOv3
from tests.inference._tiny import (
    CLASS_NAMES,
    INPUT_SIZE,
    RUN_NAME,
    NestedLayersCBM,
    build_tiny_cbm,
    write_tiny_sidecar,
)


@pytest.fixture(scope="module")
def bundle_dir(trained_run, checkpoint_path, sidecar_path):
    return export_bundle(checkpoint_path, sidecar_path, trained_run["root"] / "bundle")


def _rewrite_manifest(src, dst, edit):
    import shutil

    shutil.copytree(src, dst)
    manifest = json.loads((dst / "model.json").read_text())
    edit(manifest)
    (dst / "model.json").write_text(json.dumps(manifest))
    return dst


@pytest.mark.parametrize("wrapped", [True, False], ids=["checkpoint_dict", "bare_state_dict"])
def test_bundle_reproduces_the_lora_model_outputs(trained_run, sidecar_path, tmp_path, wrapped):
    model = trained_run["model"]
    state_dict = model.state_dict()
    ckpt = tmp_path / "ckpt"
    torch.save({"model_state_dict": state_dict} if wrapped else state_dict, ckpt)

    bundle = load_bundle(export_bundle(ckpt, sidecar_path, tmp_path / "bundle"))

    assert type(bundle.model) is ConceptBottleneckDPTDINOv3
    x = torch.randn(1, 3, *INPUT_SIZE, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        expected = model(x)
        got = bundle.model(x)
    assert torch.allclose(got.logits, expected.logits, atol=1e-5, rtol=0)
    assert torch.allclose(got.hidden_states, expected.hidden_states, atol=1e-5, rtol=0)


def test_bundle_weights_hold_no_lora_adapters(bundle_dir):
    keys = load_file(bundle_dir / "weights.safetensors").keys()
    assert not [k for k in keys if "lora_" in k]


def test_bundle_metadata_describes_the_model(bundle_dir):
    bundle = load_bundle(bundle_dir)
    assert bundle.classes == CLASS_NAMES
    assert bundle.concept_names[:2] == ["kingdom__v0", "kingdom__v1"]
    assert bundle.concept_names[-2:] == ["live", "sand"]
    assert bundle.input_size == INPUT_SIZE
    assert bundle.metadata["architecture"]["trained_class"] == "ConceptBottleneckDPTLoRADINOv3"
    environment = {"mermaidseg", "mermaidseg_dirty", "torch", "transformers", "peft"}
    assert set(bundle.metadata["trained_with"]) == environment
    assert set(bundle.metadata["exported_with"]) == environment


def _move_one_concept_to_another_rank(manifest):
    names = manifest["concepts"]["names"]
    names[names.index("kingdom__v1")] = "phylum__v2"


def _drop_one_concept(manifest):
    manifest["concepts"]["names"].pop()


def _drop_one_class(manifest):
    manifest["classes"].pop()


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (_move_one_concept_to_another_rank, "rank_widths"),
        (_drop_one_concept, "concept"),
        (_drop_one_class, "classes"),
        (lambda m: m.update(schema_version=2), "schema_version"),
        (lambda m: m["architecture"].update({"class": "ConceptBottleneckDINOv3"}), "architecture"),
    ],
    ids=["stale_rank_widths", "concept_count", "class_count", "schema_version", "architecture"],
)
def test_load_rejects_a_manifest_that_does_not_fit_the_weights(bundle_dir, tmp_path, edit, message):
    broken = _rewrite_manifest(bundle_dir, tmp_path / "broken", edit)
    with pytest.raises(ManifestError, match=message):
        load_bundle(broken)


def _batchnorm_buffers(state_dict):
    channels = state_dict["dpt_head.head.1.weight"].shape[0]
    return {
        "dpt_head.head.1.running_mean": torch.zeros(channels),
        "dpt_head.head.1.running_var": torch.ones(channels),
        "dpt_head.head.1.num_batches_tracked": torch.tensor(0),
    }


def test_export_rejects_a_batchnorm_head_checkpoint(trained_run, sidecar_path, tmp_path):
    state_dict = dict(trained_run["model"].state_dict())
    state_dict.update(_batchnorm_buffers(state_dict))
    ckpt = tmp_path / "bn_ckpt"
    torch.save({"model_state_dict": state_dict}, ckpt)
    with pytest.raises(ManifestError, match="BatchNorm"):
        export_bundle(ckpt, sidecar_path, tmp_path / "bundle")


def test_load_rejects_batchnorm_head_weights(bundle_dir, tmp_path):
    broken = _rewrite_manifest(bundle_dir, tmp_path / "broken", lambda m: None)
    weights = load_file(broken / "weights.safetensors")
    weights.update(_batchnorm_buffers(weights))
    save_file(weights, broken / "weights.safetensors")
    with pytest.raises(ManifestError, match="BatchNorm"):
        load_bundle(broken)


def _bfloat16_encoder_config(manifest):
    manifest["encoder_config"].pop("torch_dtype", None)
    manifest["encoder_config"]["dtype"] = "bfloat16"


def test_load_builds_float32_from_a_bfloat16_encoder_config(bundle_dir, trained_run, tmp_path):
    bf16 = _rewrite_manifest(bundle_dir, tmp_path / "bf16", _bfloat16_encoder_config)
    bundle = load_bundle(bf16, format="checkpoint")

    assert {p.dtype for p in bundle.model.parameters()} == {torch.float32}
    x = torch.randn(1, 3, *INPUT_SIZE, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        expected = trained_run["model"](x).logits
        got = bundle.model(x).logits
    assert torch.allclose(got, expected, atol=1e-5, rtol=0)


def test_failed_re_export_leaves_the_previous_bundle_intact(
    sidecar_path, checkpoint_path, tmp_path, monkeypatch
):
    out = export_bundle(checkpoint_path, sidecar_path, tmp_path / "bundle")
    before = {p.name: p.read_bytes() for p in out.iterdir()}
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    checkpoint["model_state_dict"]["concept_classifier.weight"] += 1.0
    retrained = tmp_path / "retrained"
    torch.save(checkpoint, retrained)

    def refuse_graph(*args, **kwargs):
        raise ParityError("graph refused")

    monkeypatch.setattr(bundle_module, "export_graph", refuse_graph)
    with pytest.raises(ParityError):
        export_bundle(retrained, sidecar_path, out, graph=True)

    assert {p.name: p.read_bytes() for p in out.iterdir()} == before
    assert not (tmp_path / "bundle.partial").exists()


def test_bundle_records_training_and_export_environments_separately(
    trained_run, checkpoint_path, tmp_path, monkeypatch
):
    monkeypatch.setattr(bundle_module, "_installed_version", lambda name: f"train-{name}")
    sidecar = write_tiny_sidecar(tmp_path / "run", trained_run["model_cfg"])
    monkeypatch.setattr(bundle_module, "_installed_version", lambda name: f"export-{name}")
    metadata = load_bundle(export_bundle(checkpoint_path, sidecar, tmp_path / "bundle")).metadata

    assert metadata["trained_with"]["transformers"] == "train-transformers"
    assert metadata["exported_with"]["transformers"] == "export-transformers"


def test_export_rejects_a_sidecar_from_another_run(trained_run, checkpoint_path, tmp_path):
    sidecar = write_tiny_sidecar(tmp_path / "other", trained_run["model_cfg"], run_name="other")
    with pytest.raises(ManifestError, match="run_name"):
        export_bundle(checkpoint_path, sidecar, tmp_path / "bundle")


def test_export_rejects_a_sidecar_whose_architecture_differs_from_the_checkpoint(
    trained_run, sidecar_path, tmp_path
):
    other_cfg = {**trained_run["model_cfg"], "encoder_name": "facebook/other-encoder"}
    ckpt = tmp_path / "ckpt"
    torch.save(
        {
            "model_state_dict": trained_run["model"].state_dict(),
            "config": ConfigDict({"run_name": RUN_NAME, "model": other_cfg}),
        },
        ckpt,
    )
    with pytest.raises(ManifestError, match="encoder_name"):
        export_bundle(ckpt, sidecar_path, tmp_path / "bundle")


def test_export_warns_that_a_bare_state_dict_is_not_checked_against_the_run(
    trained_run, sidecar_path, tmp_path, caplog
):
    ckpt = tmp_path / "bare"
    torch.save(trained_run["model"].state_dict(), ckpt)
    with caplog.at_level(logging.WARNING, logger=bundle_module.__name__):
        export_bundle(ckpt, sidecar_path, tmp_path / "bundle")
    assert "not checked" in caplog.text


def _assert_same_logits(model, reference):
    x = torch.randn(1, 3, *INPUT_SIZE, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        assert torch.allclose(model(x).logits, reference(x).logits, atol=1e-5, rtol=0)


def test_bundle_loads_where_transformers_nests_the_encoder_layers(
    bundle_dir, trained_run, monkeypatch
):
    monkeypatch.setattr(bundle_module, "ConceptBottleneckDPTDINOv3", NestedLayersCBM)
    bundle = load_bundle(bundle_dir, format="checkpoint")

    assert any(k.startswith("encoder.model.layer.") for k in bundle.model.state_dict())
    _assert_same_logits(bundle.model, trained_run["model"])


def test_bundle_exported_where_transformers_nests_the_encoder_layers_loads_here(
    trained_run, tmp_path, monkeypatch
):
    cfg = {k: v for k, v in trained_run["model_cfg"].items() if k not in bundle_module.LORA_KEYS}
    cfg["name"] = "ConceptBottleneckDPTDINOv3"
    source = build_tiny_cbm(NestedLayersCBM, cfg).eval()
    ckpt = tmp_path / "ckpt"
    torch.save(
        {
            "model_state_dict": source.state_dict(),
            "config": ConfigDict({"run_name": RUN_NAME, "model": cfg}),
        },
        ckpt,
    )
    sidecar = write_tiny_sidecar(tmp_path / "run", cfg)
    with monkeypatch.context() as patch:
        patch.setattr(bundle_module, "ConceptBottleneckDPTDINOv3", NestedLayersCBM)
        patch.setattr(mm_models, "ConceptBottleneckDPTDINOv3", NestedLayersCBM)
        out = export_bundle(ckpt, sidecar, tmp_path / "bundle")

    keys = load_file(out / "weights.safetensors").keys()
    assert not [k for k in keys if k.startswith("encoder.model.")]
    _assert_same_logits(load_bundle(out, format="checkpoint").model, source)


def test_load_names_the_transformers_versions_when_weights_do_not_fit(bundle_dir, tmp_path):
    broken = _rewrite_manifest(bundle_dir, tmp_path / "broken", lambda m: None)
    weights = load_file(broken / "weights.safetensors")
    weights["encoder.renamed.weight"] = weights.pop("encoder.norm.weight")
    save_file(weights, broken / "weights.safetensors")
    with pytest.raises(ManifestError, match="running transformers"):
        load_bundle(broken, format="checkpoint")
