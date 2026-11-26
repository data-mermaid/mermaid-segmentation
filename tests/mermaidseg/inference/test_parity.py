"""The inference path reproduces the evaluation path on the same image."""

from __future__ import annotations

import numpy as np
import torch

from mermaidseg.inference.bundle import load_bundle
from mermaidseg.inference.predictor import CBMPredictor, class_predictions, class_probabilities
from mermaidseg.io import build_eval_transform
from tests.mermaidseg.inference import _tiny


def test_inference_path_matches_the_eval_path(run, bundle_dir):
    image = np.random.default_rng(0).integers(0, 256, (70, 90, 3), dtype=np.uint8)

    eval_input = build_eval_transform(_tiny.INPUT_SIZE)(image=image)["image"]
    eval_batch = torch.from_numpy(eval_input).permute(2, 0, 1).float()[None]
    with torch.no_grad():
        outputs = run[1](eval_batch)
    eval_class = class_probabilities(outputs.logits)
    eval_concepts = outputs.concept_probs.float()

    predictor = CBMPredictor.from_bundle(load_bundle(bundle_dir, "cpu"), amp_dtype=None)
    inf_class, inf_concepts = predictor.forward_probs(predictor.preprocess(image)[None])

    torch.testing.assert_close(inf_class, eval_class, atol=1e-5, rtol=0)
    torch.testing.assert_close(inf_concepts, eval_concepts, atol=1e-5, rtol=0)
    assert torch.equal(class_predictions(inf_class), class_predictions(eval_class))
