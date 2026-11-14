"""Notebook widget version of the point review app: runs in a Jupyter / VS Code notebook output, no server or port.

Same page as ``scripts/review_app.py`` (``scripts/review_app/index.html``, unchanged logic and keys), with its loading
and saving routed through the kernel: the page asks for one image at a time, and every verdict is appended to the same
``<data>/reviews.csv`` as the Flask app (``review_app.load_reviews`` reads it back).

    import sys; sys.path.insert(0, "../scripts")
    from review_widget import PointReview
    w = PointReview("../eval_out/coral_cover/review/base_taxv2_live", reviewer="viktor")
    w                # click the widget once, then use the keys
    w.reviews_df()   # latest verdict per reviewed point

Requires ``anywidget``.
"""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path

import anywidget
import pandas as pd
import traitlets

from review_app import append_review, load_reviews, review_map

PAGE = Path(__file__).resolve().parent / "review_app" / "index.html"


def _scope_css(css: str, scope: str = ".review-root") -> str:
    """Prefix every selector so the page's styles only reach the widget (no @-rules in the page's CSS)."""
    def scoped(selector: str) -> str:
        selector = selector.strip()
        if selector in (":root", "html", "body"):
            return scope
        return f"{scope} {selector}"

    return re.sub(r"([^{}]+)\{", lambda m: ", ".join(scoped(s) for s in m.group(1).split(",")) + " {", css)


def _build(page: str) -> tuple[str, str]:
    css = re.search(r"<style>(.*?)</style>", page, re.S).group(1)
    markup = re.search(r"<body>\s*(.*?)\s*<script>", page, re.S).group(1)
    script = re.search(r"<script>(.*?)</script>", page, re.S).group(1)
    for old, new in (("html, body {", "html {"), ("#app { display: flex; height: 100vh; }",
                                                       "#app { display: flex; height: 100%; }")):
        if old not in css:
            raise ValueError(f"page CSS changed; update review_widget._build ({old!r})")
        css = css.replace(old, new)
    # "html" was the page's full-height root; in the widget the scoped root gets a fixed height instead.
    css = _scope_css(css) + "\n.review-root { height: var(--review-height, 820px); position: relative; outline: none;" \
        " border-radius: 8px; overflow: hidden; }\n.review-root:focus-visible { box-shadow: 0 0 0 2px #8ab4ff; }\n"
    esm = """
function makeIO(model) {
  let next = 0;
  const pending = new Map(), images = new Map();
  model.on("msg:custom", (msg) => {
    const p = pending.get(msg.req);
    if (!p) return;
    pending.delete(msg.req);
    if (msg.ok) p.resolve(msg); else p.reject(new Error(msg.error || "kernel error"));
  });
  const request = (content) => new Promise((resolve, reject) => {
    const req = ++next;
    pending.set(req, {resolve, reject});
    model.send({...content, req});
    setTimeout(() => { if (pending.delete(req)) reject(new Error("no reply from the kernel")); }, 30000);
  });
  const image = (iid) => {
    if (!images.has(iid)) {
      const p = request({type: "load", image_id: iid});
      p.catch(() => images.delete(iid));
      images.set(iid, p);
      if (images.size > 8) images.delete(images.keys().next().value);
    }
    return images.get(iid);
  };
  return {
    manifest: async () => model.get("manifest"),
    reviews: async () => ({...model.get("reviews")}),
    points: async (iid) => model.get("points")[iid],
    imageSrc: async (iid) => (await image(iid)).image,
    mapSrc: async (iid, kind) => (await image(iid)).maps[kind],
    save: async (body) => { await request({type: "review", body}); },
  };
}

export default {
  render({ model, el }) {
    el.innerHTML = '<div class="review-root" tabindex="0">' + MARKUP + "</div>";
    const root = el.querySelector(".review-root");
    root.style.setProperty("--review-height", model.get("height") + "px");
    root.addEventListener("mousedown", () => root.focus({preventScroll: true}));
    const cleanups = [];
    // The page script below is the review page's own; these stand-ins keep it inside this widget.
    const window = {
      REVIEW_IO: makeIO(model),
      get devicePixelRatio() { return globalThis.devicePixelRatio || 1; },
      addEventListener: (t, f, o) => { globalThis.addEventListener(t, f, o); cleanups.push(() => globalThis.removeEventListener(t, f, o)); },
    };
    const document = {
      getElementById: (id) => root.querySelector("#" + id),
      querySelector: (s) => root.querySelector(s),
      querySelectorAll: (s) => root.querySelectorAll(s),
      createElement: (t) => globalThis.document.createElement(t),
      // Keys stay in the widget, so notebook shortcuts (a, m, ...) don't fire while it has focus.
      addEventListener: (t, f, o) => root.addEventListener(t, t === "keydown" ? (e) => { f(e); e.stopPropagation(); } : f, o),
    };
    // ---- page script ----
    PAGE_SCRIPT
    // ---- end page script ----
    const ro = new ResizeObserver(() => { if (S.man && S.img) fit(); resize(); });
    ro.observe(root.querySelector("#stage"));
    return () => { ro.disconnect(); cleanups.forEach((f) => f()); };
  },
};
"""
    esm = esm.replace("MARKUP", json.dumps(markup)).replace("PAGE_SCRIPT", script)
    return esm, css


class PointReview(anywidget.AnyWidget):
    """Review FP / FN points of a ``review_export.py`` folder inside a notebook. See the module docstring."""

    manifest = traitlets.Dict().tag(sync=True)
    points = traitlets.Dict().tag(sync=True)
    reviews = traitlets.Dict().tag(sync=True)
    height = traitlets.Int(820).tag(sync=True)

    def __init__(self, data_dir: str | Path, reviewer: str = "", height: int = 820, **kwargs):
        self._esm, self._css = _build(PAGE.read_text())
        self.data_dir = Path(data_dir).resolve()
        self.reviewer = reviewer
        manifest = json.loads((self.data_dir / "manifest.json").read_text())
        manifest["reviewer"] = reviewer
        points = {m["image_id"]: json.loads((self.data_dir / "points" / f"{m['image_id']}.json").read_text())
                  for m in manifest["images"]}
        super().__init__(manifest=manifest, points=points, reviews=review_map(self.data_dir), height=height, **kwargs)
        self.on_msg(self._on_msg)

    def reviews_df(self) -> pd.DataFrame:
        """Latest verdict per reviewed point (as ``review_app.load_reviews``)."""
        return load_reviews(self.data_dir)

    def _data_url(self, path: Path, mime: str) -> str:
        return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()

    def _on_msg(self, _widget, content: dict, _buffers) -> None:
        req = content.get("req")
        try:
            if content.get("type") == "load":
                iid = str(content["image_id"])
                reply = {"image": self._data_url(self.data_dir / "images" / f"{iid}.jpg", "image/jpeg"),
                         "maps": {k: self._data_url(self.data_dir / "maps" / f"{iid}_{k}.png", "image/png")
                                  for k in self.manifest["maps"]}}
            elif content.get("type") == "review":
                body = content["body"]
                append_review(self.data_dir, body, self.reviewer)
                key, reviews = f"{body['image_id']}:{body['idx']}", dict(self.reviews)
                if body.get("verdict"):
                    reviews[key] = body["verdict"]
                else:
                    reviews.pop(key, None)
                self.reviews = reviews
                reply = {}
            else:
                raise ValueError(f"unknown message {content.get('type')!r}")
            self.send({"req": req, "ok": True, **reply})
        except Exception as e:  # noqa: BLE001  (report to the page instead of failing silently)
            self.send({"req": req, "ok": False, "error": str(e)})
