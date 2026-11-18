"""SourceLabelRegistry vocabulary normalisation."""

from __future__ import annotations

from mermaidseg.dataset_reconciliation.registry import SourceLabelRegistry


def test_source_label_registry_normalizes_target_names_to_lowercase() -> None:
    """Lowercase static map values must match a capitalized class_subset."""

    class CoralscapesV2Stub:
        SOURCE_NAME = "coralscapes_v2"
        source_id2name = {1: "pocillopora alive", 2: "sand", 3: "background"}
        source_name2id = {"pocillopora alive": 1, "sand": 2, "background": 3}

        def set_global_offset(self, offset: int) -> None:
            self._global_offset = offset

    registry = SourceLabelRegistry(
        [CoralscapesV2Stub()],
        target_label_subset=["Pocillopora", "Sand", "Background"],
    )

    assert registry.target_label2id == {"background": 1, "pocillopora": 2, "sand": 3}
    assert registry.target_id2label == {1: "background", 2: "pocillopora", 3: "sand"}

    offset = registry.dataset_offsets["coralscapes_v2"]
    pocillopora_gid = 1 + offset
    sand_gid = 2 + offset
    background_gid = 3 + offset
    assert int(registry.source_to_target[pocillopora_gid]) == 2
    assert int(registry.source_to_target[sand_gid]) == 3
    assert int(registry.source_to_target[background_gid]) == 1
