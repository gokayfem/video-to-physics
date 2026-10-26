"""Playable review page (CASE/review/index.html) built from CASE/review/page.json.

page.json:
    {"title": str, "status": str, "summary_md": str (paragraphs separated by blank lines),
     "items": [{"path": "relative/to/review/", "label": str, "caption": str}]}
Videos (.mp4/.webm/.mov) get a looping player; images link to the full-size file. Missing items are listed.
"""
from __future__ import annotations

import html
import json
from pathlib import Path

VIDEO = (".mp4", ".webm", ".mov", ".m4v")
E = html.escape

CSS = """
:root{--bg:#f4f1ec;--card:#fffdf9;--ink:#24211d;--muted:#6c655c;--line:#ddd5ca;--accent:#8a5a2b}
@media (prefers-color-scheme:dark){:root{--bg:#191715;--card:#23201d;--ink:#ece6dd;--muted:#a59c90;--line:#3a352f;--accent:#d29a5c}}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 -apple-system,system-ui,sans-serif}
main{max-width:1400px;margin:auto;padding:24px 16px 64px}h1{margin:0 0 6px}
.status{display:inline-block;border:1px solid var(--accent);color:var(--accent);border-radius:20px;padding:2px 12px;font-weight:600}
.summary{max-width:950px}figure{background:var(--card);border:1px solid var(--line);border-radius:10px;margin:18px 0;padding:12px}
figcaption{margin-bottom:8px;color:var(--muted)}figcaption b{color:var(--ink)}
.missing{color:#c0392b}video,img{width:100%;border-radius:6px;background:#000;display:block}
"""


def _media(item) -> str:
    p = item["path"]
    if p.lower().endswith(VIDEO):
        return f'<video controls loop muted playsinline preload="metadata" src="{E(p)}"></video>'
    return f'<a href="{E(p)}"><img loading="lazy" src="{E(p)}" alt="{E(item.get("label", p))}"></a>'


def render(spec: dict, missing=()) -> str:
    paras = "".join(f"<p>{E(x.strip())}</p>" for x in spec.get("summary_md", "").split("\n\n") if x.strip())
    cards = []
    for it in spec.get("items", []):
        flag = ' <span class="missing">(missing)</span>' if it["path"] in missing else ""
        cards.append(f'<figure><figcaption><b>{E(it.get("label", it["path"]))}</b>{flag} — {E(it.get("caption", ""))}'
                     f'</figcaption>{_media(it)}</figure>')
    title = E(spec.get("title", "Review"))
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title>'
            f'<style>{CSS}</style></head><body><main><h1>{title}</h1>'
            f'<span class="status">{E(spec.get("status", ""))}</span><div class="summary">{paras}</div>'
            f'{"".join(cards)}</main></body></html>')


def build(case) -> dict:
    """Write CASE/review/index.html; returns {"out", "items", "missing"}."""
    review = Path(case) / "review"
    spec_path = review / "page.json"
    if not spec_path.exists():
        raise FileNotFoundError(f"{spec_path} not found (see the page.json schema in v2p.review_page)")
    spec = json.loads(spec_path.read_text())
    missing = [it["path"] for it in spec.get("items", []) if not (review / it["path"]).exists()]
    out = review / "index.html"
    out.write_text(render(spec, missing))
    return {"out": str(out), "items": len(spec.get("items", [])), "missing": missing}
