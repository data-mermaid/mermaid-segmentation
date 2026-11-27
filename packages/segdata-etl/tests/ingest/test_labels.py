from segdata_etl.common.labels import build_classes_and_colors


def test_classes_are_deduplicated_alphabetical_and_colors_follow_seed_order():
    classes, colors = build_classes_and_colors(["b", "a", "a"])
    assert classes == {"unlabeled": 0, "a": 1, "b": 2}
    assert colors["unlabeled"] == [0, 0, 0]
    assert colors["a"] != colors["b"]
    assert colors == build_classes_and_colors(["a", "b"])[1]
