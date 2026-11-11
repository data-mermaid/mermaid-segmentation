"""Derive configs/data_config_trainval.yaml (the final-run mixture) from configs/data_config.yaml.

CoralNet's validation sources move onto the train whitelist, and the Coralscapes
v2 validation split joins its train split. Test splits stay held out. Benthos
and Pacific labeled corals stay validation monitors. The source configs are not
modified.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS = REPO_ROOT / "configs"

HEADER = """\
# Final-run mixture, derived from data_config.yaml.
# CoralNet validation sources are on the train whitelist, and the Coralscapes v2
# validation split is part of train. Test splits stay held out.
"""

# Character classes, not '.', so a wrapped flow list can span lines without the
# match running into the next block.
_WHITELIST = re.compile(r"whitelist_sources: \[[^\]]*\]")
_CORALNET_VAL = re.compile(
    r"    val:\n"
    r"      whitelist_sources: \[[^\]]*\]\n"
    r"      image_blacklist: [^\n]*\n"
    r"      annotation_overrides: [^\n]*\n"
)
_CORALSCAPES_VAL = re.compile(
    r"    val:\n      split: validation\n      transform:\n(?:        [^\n]*\n)+"
)
_LIST_INDENT = " " * 26


def _flow_list(values: list[int], width: int = 96) -> str:
    """Format an id list in the same wrapped flow style as the source configs."""
    lines: list[str] = []
    current = ""
    for index, value in enumerate(values):
        piece = str(value) + ("," if index < len(values) - 1 else "")
        candidate = piece if not current else f"{current} {piece}"
        if current and len(_LIST_INDENT + candidate) > width:
            lines.append(current)
            current = piece
        else:
            current = candidate
    lines.append(current)
    if len(lines) == 1:
        return f"[{lines[0]}]"
    continuation = ("\n" + _LIST_INDENT).join(lines[1:])
    return f"[{lines[0]}\n{_LIST_INDENT}{continuation}]"


def _section(text: str, name: str) -> tuple[int, int, str]:
    pattern = re.compile(rf"(  {re.escape(name)}:\n)(.*?)(?=\n  [a-z_]+:|\Z)", re.S)
    found = pattern.search(text)
    if found is None:
        raise ValueError(f"section {name!r} not found")
    return found.start(2), found.end(2), found.group(2)


def _replace_section(text: str, name: str, body: str) -> str:
    start, end, _ = _section(text, name)
    return text[:start] + body + text[end:]


def _whitelist_ids(body: str) -> list[int]:
    match = _WHITELIST.search(body)
    if match is None:
        raise ValueError("whitelist_sources not found")
    inner = match.group(0).split("[", 1)[1].rsplit("]", 1)[0]
    return [int(piece) for piece in inner.replace("\n", " ").split(",") if piece.strip()]


def build_final_data_config(text: str, *, header: str) -> str:
    """Return ``text`` with the final-run train/validation merge applied."""
    _, _, coralnet = _section(text, "coralnet")
    if _CORALNET_VAL.search(coralnet) is None:
        raise ValueError("coralnet val block not found")
    train_ids = _whitelist_ids(coralnet)
    val_body = _CORALNET_VAL.search(coralnet)
    assert val_body is not None
    val_ids = _whitelist_ids(val_body.group(0))
    seen = set(train_ids)
    merged = train_ids + [source_id for source_id in val_ids if source_id not in seen]
    if len(merged) != len(set(merged)):
        raise ValueError("merged CoralNet train whitelist has duplicate sources")
    coralnet = _WHITELIST.sub(f"whitelist_sources: {_flow_list(merged)}", coralnet, count=1)
    coralnet = _CORALNET_VAL.sub("    val: None\n", coralnet, count=1)
    text = _replace_section(text, "coralnet", coralnet)

    _, _, coralscapes = _section(text, "coralscapes_v2")
    if "      split: train\n" not in coralscapes:
        raise ValueError("coralscapes_v2 train split not found")
    if _CORALSCAPES_VAL.search(coralscapes) is None:
        raise ValueError("coralscapes_v2 val block not found")
    coralscapes = coralscapes.replace(
        "      split: train\n", "      split: [train, validation]\n", 1
    )
    coralscapes = _CORALSCAPES_VAL.sub("    val: None\n", coralscapes, count=1)
    text = _replace_section(text, "coralscapes_v2", coralscapes)

    if text.startswith("#"):
        # Rebuilding an already-derived file is not supported; sources have no banner.
        raise ValueError("source data config unexpectedly starts with a comment")
    return header + text


def _check(path: Path) -> None:
    loaded = yaml.safe_load(path.read_text())
    coralnet = loaded["data"]["coralnet"]
    if coralnet["val"] not in (None, "None"):
        raise ValueError(f"{path.name}: coralnet.val was not cleared")
    coralscapes = loaded["data"]["coralscapes_v2"]
    if coralscapes["train"]["split"] != ["train", "validation"]:
        raise ValueError(f"{path.name}: coralscapes_v2 train split was not merged")
    if coralscapes["val"] not in (None, "None"):
        raise ValueError(f"{path.name}: coralscapes_v2.val was not cleared")
    if coralscapes["test"]["split"] != "test":
        raise ValueError(f"{path.name}: coralscapes_v2 test split changed")


def write_trainval_data_config() -> None:
    """Write configs/data_config_trainval.yaml from configs/data_config.yaml."""
    source = CONFIGS / "data_config.yaml"
    destination = CONFIGS / "data_config_trainval.yaml"
    destination.write_text(build_final_data_config(source.read_text(), header=HEADER))
    _check(destination)
    print(f"wrote {destination}")


def main() -> None:
    write_trainval_data_config()


if __name__ == "__main__":
    main()
