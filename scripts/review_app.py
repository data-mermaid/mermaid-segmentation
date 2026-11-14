#!/usr/bin/env python3
"""Browser app to review model-vs-annotation disagreements at the annotated points.

Loads what ``scripts/review_export.py`` saved (no model, no GPU) and serves a single page
(``scripts/review_app/index.html``). Each point is classed from the annotation and the model's call for the target:
TP / TN (agree), FP (model positive, annotation negative), FN (annotation positive, model negative), NA (annotation
not given). The reviewer gives the points a verdict, by default only the FP / FN ones:

- ``model_wrong``: the annotation is right, the model's error is real;
- ``annotation_wrong``: the model is right, the annotation is wrong;
- ``ambiguous``: cannot tell from the image.

Points without a verdict keep their automatic outcome (so TP / TN stay as they are). Every verdict is appended to
``<data>/reviews.csv`` at once (one row per change, with a time stamp); the latest row per point wins, and an empty
verdict clears it. ``load_reviews(data_dir)`` returns that latest state as a DataFrame.

Usage:
    python scripts/review_app.py --data eval_out/coral_cover/review/base_taxv2_live [--port 8765] [--reviewer NAME]
then open http://127.0.0.1:8765 (VS Code forwards the port when it sees it; otherwise use the Ports panel or
``ssh -L 8765:127.0.0.1:8765 <host>``).
"""

from __future__ import annotations

import argparse
import csv
import json
import threading
import time
from pathlib import Path

import pandas as pd
from flask import Flask, abort, jsonify, request, send_from_directory

APP_DIR = Path(__file__).resolve().parent / "review_app"
VERDICTS = {"model_wrong", "annotation_wrong", "ambiguous", ""}
FIELDS = ["time", "reviewer", "source_id", "image_id", "idx", "row", "col", "coralnet", "mermaid", "gt_code",
          "model_call", "outcome", "verdict"]


def load_reviews(data_dir: str | Path) -> pd.DataFrame:
    """Latest verdict per point (cleared ones dropped), from ``<data_dir>/reviews.csv``."""
    path = Path(data_dir) / "reviews.csv"
    if not path.exists():
        return pd.DataFrame(columns=FIELDS)
    df = pd.read_csv(path, dtype={"image_id": str}, keep_default_na=False)
    df = df.drop_duplicates(["image_id", "idx"], keep="last")
    return df[df.verdict != ""].reset_index(drop=True)


_LOCK = threading.Lock()


def review_map(data_dir: str | Path) -> dict[str, str]:
    """``{"<image_id>:<idx>": verdict}`` for the points reviewed so far."""
    return {f"{r.image_id}:{r.idx}": r.verdict for r in load_reviews(data_dir).itertuples()}


def append_review(data_dir: str | Path, body: dict, reviewer: str) -> None:
    """Append one verdict (``""`` clears) to ``<data_dir>/reviews.csv``; raises ValueError for an unknown verdict."""
    if body.get("verdict", "") not in VERDICTS:
        raise ValueError(f"unknown verdict {body.get('verdict')!r}")
    row = {k: body.get(k, "") for k in FIELDS}
    row.update(time=time.strftime("%Y-%m-%d %H:%M:%S"), reviewer=reviewer)
    path = Path(data_dir) / "reviews.csv"
    with _LOCK:
        new = not path.exists()
        with path.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if new:
                w.writeheader()
            w.writerow(row)


def create_app(data_dir: Path, reviewer: str) -> Flask:
    app = Flask(__name__)
    manifest = json.loads((data_dir / "manifest.json").read_text())

    @app.get("/")
    def index():
        return send_from_directory(APP_DIR, "index.html")

    @app.get("/api/manifest")
    def get_manifest():
        return jsonify({**manifest, "reviewer": reviewer, "data_dir": str(data_dir)})

    @app.get("/api/reviews")
    def get_reviews():
        return jsonify(review_map(data_dir))

    @app.post("/api/review")
    def post_review():
        try:
            append_review(data_dir, request.get_json(force=True), reviewer)
        except ValueError as e:
            abort(400, str(e))
        return jsonify({"ok": True})

    @app.get("/data/<path:path>")
    def data(path: str):
        return send_from_directory(data_dir, path, max_age=3600)

    return app


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, required=True, help="directory written by review_export.py")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--reviewer", default="")
    args = p.parse_args()
    data_dir = args.data.resolve()
    if not (data_dir / "manifest.json").exists():
        raise SystemExit(f"{data_dir}/manifest.json not found; run scripts/review_export.py first")
    done = load_reviews(data_dir)
    print(f"{len(json.loads((data_dir / 'manifest.json').read_text())['images'])} images, "
          f"{len(done)} points reviewed so far -> {data_dir / 'reviews.csv'}")
    create_app(data_dir, args.reviewer).run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
