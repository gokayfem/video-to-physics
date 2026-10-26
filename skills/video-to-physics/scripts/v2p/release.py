"""Config-driven release film: dark title cards, matched reference-vs-physics splits with caption pills,
count-up stats, card rows and an outro. Only real project outputs are used (paths come from the config).

Config (paths relative to the config file):
{
  "size": [1920, 1080], "fps": 24, "accent": [34, 211, 238],
  "segments": [
    {"type": "title", "frames": 96, "title": "Video 2 Physics", "accent_text": "2",
     "subtitle": "...", "tagline": "...", "hero": {"left": "ai.mp4", "right": "physics.mp4", "start_s": 6},
     "left_label": "AI video", "right_label": "Physics"},
    {"type": "full",  "clip": "ai.mp4", "start_s": 0, "frames": 96, "caption": "AI video in.", "label": "...",
     "pills": [{"start": 8, "end": 90, "text": "...", "dot": "red"}]},
    {"type": "split", "left": "ai.mp4", "right": "physics.mp4", "start_s": 3, "frames": 168, "name": "...",
     "left_label": "AI video.", "right_label": "Physics.", "pills": [...]},
    {"type": "stat",  "frames": 60, "value": 1.3576, "format": "{:.4f} N", "label": "...", "sub": "..."},
    {"type": "grid",  "frames": 168, "heading": "...", "start_s": 1,
     "items": [{"top": "a.mp4", "bottom": "b.mp4", "name": "...", "caption": "..."}]},
    {"type": "cards", "frames": 108, "heading": "...", "cards": [{"image": "x.png", "title": "...", "subtitle": "..."}]},
    {"type": "outro", "frames": 120, "clip": "physics.mp4", "start_s": 5, "title": "Video 2 Physics", "subtitle": "..."}
  ]
}
The template vocabulary (templates/release.json) is accepted too: "seconds" instead of "frames", "t"/"t0" for
start_s, split {"ai", "physics", "left"/"right" = labels}, title hero {"ai", "physics", "t"}, outro {"hero", "t",
"line"}, card {"text"}, top-level "title"/"subtitle" used by title and outro, pills as plain strings (spread evenly).
Rules: the first segment must be a title; frame 0 is the title at full opacity over the hero shot (the
thumbnail) and is checked to be non-black. Clips play at their true speed (frame = (start_s + i/fps) * clip_fps).
Pill dots: [r,g,b], "accent", "red", or omitted. Layout is designed at 1920x1080 and scaled.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .video import FrameSource, VideoWriter, even, font as _font, source_fps

BG0, BG1 = (9, 12, 20), (20, 26, 40)
RED = (255, 120, 120)


def ease(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


class Film:
    """Drawing context for one film (size, scale, accent, background, clip cache)."""

    def __init__(self, cfg, base: Path, draft=False):
        w, h = cfg.get("size", [1920, 1080])
        k = 0.5 if draft else 1.0
        self.W, self.H = even(w * k), even(h * k)
        self.S = self.W / 1920.0
        self.fps = float(cfg.get("fps", 24))
        self.accent = tuple(cfg.get("accent", [34, 211, 238]))
        self.base = base
        y = np.linspace(0, 1, self.H)[:, None, None]
        x = np.linspace(-1, 1, self.W)[None, :, None]
        g = (np.array(BG0) * (1 - y) + np.array(BG1) * y) * (1 - 0.25 * x ** 2)
        self.bg = np.clip(g, 0, 255).astype(np.uint8)
        self._clips = {}

    # ---------------------------------------------------------- assets
    def path(self, p) -> Path:
        p = Path(p)
        return p if p.is_absolute() else self.base / p

    def clip(self, p):
        key = str(self.path(p))
        if key not in self._clips:
            if not Path(key).exists():
                raise FileNotFoundError(key)
            self._clips[key] = (FrameSource(key, rgb=True), source_fps(key, self.fps))
        return self._clips[key]

    def clip_frame(self, p, t):
        src, fps = self.clip(p)
        return src.frame(round(t * fps))

    def image(self, p):
        p = self.path(p)
        if p.suffix.lower() in (".mp4", ".mov", ".webm", ".mkv", ".m4v"):
            return self.clip_frame(p, 0)
        im = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if im is None:
            raise FileNotFoundError(str(p))
        return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    # ---------------------------------------------------------- drawing primitives (design coords)
    def font(self, px, weight="Semibold"):
        return _font(max(1, int(px * self.S)), weight)

    def dot_colour(self, dot):
        if dot is None or dot is False:
            return None
        if dot == "accent":
            return self.accent
        if dot == "red":
            return RED
        return tuple(int(c) for c in dot)

    def fit(self, img, w, h):
        """Cover-crop img to w x h pixels."""
        w, h = int(w), int(h)
        ih, iw = img.shape[:2]
        s = max(w / iw, h / ih)
        r = cv2.resize(img, (max(w, round(iw * s)), max(h, round(ih * s))), interpolation=cv2.INTER_AREA)
        y0, x0 = (r.shape[0] - h) // 2, (r.shape[1] - w) // 2
        return r[y0:y0 + h, x0:x0 + w]

    def paste(self, canvas, img, x, y, radius=14, alpha=1.0, shadow=True):
        h, w = img.shape[:2]
        x, y = int(x), int(y)
        m = Image.new("L", (w, h), 0)
        ImageDraw.Draw(m).rounded_rectangle([0, 0, w - 1, h - 1], int(radius * self.S), fill=255)
        m = np.asarray(m, np.float32)[..., None] / 255 * alpha
        if shadow and alpha > 0:
            pad = int(20 * self.S) + 1
            y0, y1, x0, x1 = max(y - pad, 0), min(y + h + pad, self.H), max(x - pad, 0), min(x + w + pad, self.W)
            sh = np.zeros((y1 - y0, x1 - x0), np.float32)
            sh[y - y0:y - y0 + h, x - x0:x - x0 + w] = 0.55 * alpha
            sh = cv2.GaussianBlur(sh, (0, 0), max(1.0, 14 * self.S))[..., None]
            canvas[y0:y1, x0:x1] = (canvas[y0:y1, x0:x1].astype(np.float32) * (1 - sh)).astype(np.uint8)
        sub = canvas[y:y + h, x:x + w].astype(np.float32)
        canvas[y:y + h, x:x + w] = (sub * (1 - m) + img.astype(np.float32) * m).astype(np.uint8)

    def text(self, d, xy, s, px, weight="Semibold", fill=(255, 255, 255), anchor="la", alpha=1.0):
        if s:
            d.text((xy[0] * self.S, xy[1] * self.S), str(s), font=self.font(px, weight),
                   fill=tuple(int(c) for c in fill) + (int(255 * alpha),), anchor=anchor)

    def title_line(self, d, cx, cy, title, accent_text, px, alpha=1.0, weight="Bold"):
        """Draw `title` centred, colouring the first occurrence of accent_text with the accent colour."""
        parts = [(title, (255, 255, 255))]
        if accent_text and accent_text in title:
            a, _, b = title.partition(accent_text)
            parts = [(a, (255, 255, 255)), (accent_text, self.accent), (b, (255, 255, 255))]
        f = self.font(px, weight)
        x = cx * self.S - sum(d.textlength(t, font=f) for t, _ in parts) / 2
        for t, c in parts:
            if t:
                d.text((x, cy * self.S), t, font=f, fill=tuple(c) + (int(255 * alpha),), anchor="lm")
                x += d.textlength(t, font=f)

    def pill(self, d, cx, y, s, px=28, alpha=1.0, dot=None):
        S, f = self.S, self.font(px, "Medium")
        tw, pad, h = d.textlength(s, font=f), 22 * S, 50 * S
        x0 = cx * S - tw / 2 - pad - (18 * S if dot else 0)
        d.rounded_rectangle([x0, y * S, cx * S + tw / 2 + pad, y * S + h], radius=h / 2,
                            fill=(255, 255, 255, int(28 * alpha)), outline=(255, 255, 255, int(60 * alpha)),
                            width=max(1, int(S)))
        if dot:
            r = 6 * S
            d.ellipse([x0 + pad - r, y * S + h / 2 - r, x0 + pad + r, y * S + h / 2 + r], fill=tuple(dot) + (int(255 * alpha),))
        d.text((cx * S + (9 * S if dot else 0), y * S + h / 2), s, font=f, fill=(235, 240, 248, int(255 * alpha)), anchor="mm")

    def pills(self, d, seg, i, a, y):
        for p in seg.get("pills", []):
            pa = a * ease((i - p.get("start", 0)) / 8) * (1 - ease((i - p.get("end", 10 ** 9)) / 8))
            if pa > 0.01:
                self.pill(d, 960, y, p["text"], 28, pa, self.dot_colour(p.get("dot")))

    @staticmethod
    def overlay(canvas, fn):
        im = Image.fromarray(canvas).convert("RGBA")
        ov = Image.new("RGBA", im.size, (0, 0, 0, 0))
        fn(ImageDraw.Draw(ov))
        return np.asarray(Image.alpha_composite(im, ov).convert("RGB")).copy()


# ------------------------------------------------------------------ segments: (film, seg, i, n) -> RGB frame
def _fade(i, n, fin=10, fout=10):
    return ease(i / fin if fin else 1) * (1 - ease((i - n + fout) / fout))


def seg_title(F: Film, seg, i, n):
    """Frame 0 = thumbnail: title at full opacity over the hero (no fade-in from black)."""
    a = 1 - ease((i - n + 14) / 14)
    hero = seg["hero"]
    t = hero.get("start_s", 0) + i / F.fps
    if "left" in hero and "right" in hero:
        half = F.W // 2
        img = np.hstack([F.fit(F.clip_frame(hero["left"], t), half, F.H),
                         F.fit(F.clip_frame(hero["right"], t), F.W - half, F.H)])
    else:
        img = F.fit(F.clip_frame(hero.get("clip") or hero.get("left") or hero.get("right"), t), F.W, F.H)
    y = np.linspace(-1, 1, F.H)[:, None, None]
    x = np.linspace(-1, 1, F.W)[None, :, None]
    dim = float(seg.get("hero_brightness", 0.42)) - 0.12 * (x ** 2 + y ** 2)
    c = (F.bg * (1 - a) + img.astype(np.float32) * dim * a).astype(np.uint8)
    split = "left" in hero and "right" in hero

    def d(dr):
        F.title_line(dr, 960, 470, seg.get("title", ""), seg.get("accent_text"), seg.get("title_px", 150), a)
        F.text(dr, (960, 590), seg.get("subtitle"), 40, "Regular", (170, 180, 200), "mm", a)
        F.text(dr, (960, 650), seg.get("tagline"), 28, "Regular", (150, 160, 180), "mm", a)
        if split:
            F.text(dr, (60, 1030), seg.get("left_label", ""), 30, "Semibold", (220, 225, 235), "lm", a)
            F.text(dr, (1860, 1030), seg.get("right_label", ""), 30, "Semibold", (255, 255, 255), "rm", a)
            dr.line([(960 * F.S, 760 * F.S), (960 * F.S, 1080 * F.S)], fill=(255, 255, 255, int(70 * a)),
                    width=max(1, int(2 * F.S)))
    return F.overlay(c, d)


def seg_full(F: Film, seg, i, n):
    c, a, S = F.bg.copy(), _fade(i, n), F.S
    img = F.fit(F.clip_frame(seg["clip"], seg.get("start_s", 0) + i / F.fps), 1560 * S, 878 * S)
    F.paste(c, img, 180 * S, 150 * S, alpha=a)

    def d(dr):
        F.text(dr, (180, 92), seg.get("caption"), 46, "Semibold", (255, 255, 255), "lm", a)
        F.text(dr, (1740, 92), seg.get("label"), 26, "Medium", (150, 160, 180), "rm", a)
        F.pills(dr, seg, i, a, 950)
    return F.overlay(c, d)


def seg_split(F: Film, seg, i, n):
    c, a, S = F.bg.copy(), _fade(i, n), F.S
    t = seg.get("start_s", 0) + i / F.fps
    pw, ph = 930 * S, 523 * S
    F.paste(c, F.fit(F.clip_frame(seg["left"], t), pw, ph), 20 * S, 245 * S, alpha=a)
    F.paste(c, F.fit(F.clip_frame(seg["right"], t), pw, ph), 970 * S, 245 * S, alpha=a)

    def d(dr):
        F.text(dr, (960, 120), seg.get("name"), 54, "Semibold", (255, 255, 255), "mm", a)
        F.text(dr, (20, 212), seg.get("left_label", "AI video."), 40, "Semibold", (200, 205, 215), "lm", a)
        F.text(dr, (970, 212), seg.get("right_label", "Physics."), 40, "Semibold", (255, 255, 255), "lm", a)
        dr.ellipse([(1892 - 8) * S, (212 - 8) * S, (1892 + 8) * S, (212 + 8) * S], fill=F.accent + (int(255 * a),))
        if seg.get("time_note", True):
            F.text(dr, (960, 815), f"t = {t:4.1f} s · same frame, same time, no retiming", 22, "Regular",
                   (120, 130, 150), "mm", a)
        F.pills(dr, seg, i, a, 860)
    return F.overlay(c, d)


def seg_stat(F: Film, seg, i, n):
    c, a = F.bg.copy(), _fade(i, n, 8, 10)
    v = float(seg["value"]) * ease(i / max(n * 0.55, 1))
    fmt = seg.get("format", "{:g}")

    def d(dr):
        s = fmt.format(v)
        f = F.font(seg.get("value_px", 190), "Bold")
        dr.text((960 * F.S - dr.textlength(s, font=f) / 2, 480 * F.S), s, font=f, fill=F.accent + (int(255 * a),), anchor="lm")
        F.text(dr, (960, 640), seg.get("label"), 40, "Semibold", (255, 255, 255), "mm", a)
        F.text(dr, (960, 700), seg.get("sub"), 26, "Regular", (140, 150, 170), "mm", a)
    return F.overlay(c, d)


def seg_grid(F: Film, seg, i, n):
    c, a, S = F.bg.copy(), _fade(i, n), F.S
    items = seg["items"][:4]
    k = len(items)
    gap = 50
    tw = (1800 - gap * (k - 1)) / k
    th = min(tw * 9 / 16, 315)
    t = seg.get("start_s", 0) + i / F.fps
    for j, it in enumerate(items):
        x = 60 + j * (tw + gap)
        F.paste(c, F.fit(F.clip_frame(it["top"], t), tw * S, th * S), x * S, 230 * S, alpha=a, radius=12)
        F.paste(c, F.fit(F.clip_frame(it["bottom"], t), tw * S, th * S), x * S, (590) * S, alpha=a, radius=12)

    def d(dr):
        F.text(dr, (960, 100), seg.get("heading"), 54, "Semibold", (255, 255, 255), "mm", a)
        for j, it in enumerate(items):
            x = 60 + j * (tw + gap)
            F.text(dr, (x, 200), it.get("top_label", (it.get("name", "") + " · AI").strip(" ·")), 28, "Semibold",
                   (200, 205, 215), "lm", a)
            F.text(dr, (x, 560), it.get("bottom_label", "Physics"), 28, "Semibold", (255, 255, 255), "lm", a)
            F.text(dr, (x + tw / 2, 945), it.get("caption"), 22, "Regular", (150, 160, 180), "mm", a)
    return F.overlay(c, d)


def seg_cards(F: Film, seg, i, n):
    S = F.S
    cards = seg["cards"][:3]
    if "_thumbs" not in seg:          # loaded once per segment
        seg["_thumbs"] = [F.fit(F.image(cd["image"]), 540 * S, 360 * S) for cd in cards]
    thumbs = seg["_thumbs"]
    c, a = F.bg.copy(), 1 - ease((i - n + 10) / 10)
    x0 = 960 - (len(cards) * 600 - 60) / 2
    for j, t in enumerate(thumbs):
        aj = a * ease((i - 6 - j * 10) / 10)
        if aj > 0:
            F.paste(c, t, (x0 + j * 600) * S, (300 - 20 * (1 - aj)) * S, alpha=aj)

    def d(dr):
        F.text(dr, (960, 170), seg.get("heading"), 54, "Semibold", (255, 255, 255), "mm", a * ease(i / 8))
        for j, cd in enumerate(cards):
            aj = a * ease((i - 6 - j * 10) / 10)
            F.text(dr, (x0 + j * 600, 710), cd.get("title"), 34, "Semibold", (255, 255, 255), "lm", aj)
            F.text(dr, (x0 + j * 600, 760), cd.get("subtitle"), 24, "Regular", (150, 160, 180), "lm", aj)
    return F.overlay(c, d)


def seg_outro(F: Film, seg, i, n):
    c, S = F.bg.copy(), F.S
    a = ease(i / 12) * (1 - ease((i - n + 16) / 16))
    if seg.get("clip"):
        img = F.fit(F.clip_frame(seg["clip"], seg.get("start_s", 0) + i / F.fps), 1300 * S, 731 * S)
        F.paste(c, img, 310 * S, 90 * S, alpha=a)

    def d(dr):
        F.title_line(dr, 960, 905, seg.get("title", ""), seg.get("accent_text"), 76, a)
        F.text(dr, (960, 975), seg.get("subtitle"), 26, "Regular", (140, 150, 170), "mm", a)
    return F.overlay(c, d)


SEGMENTS = {"title": seg_title, "full": seg_full, "split": seg_split, "stat": seg_stat, "grid": seg_grid,
            "cards": seg_cards, "outro": seg_outro}
DEFAULT_FRAMES = {"title": 96, "full": 96, "split": 168, "stat": 60, "grid": 168, "cards": 108, "outro": 120}


def _pills(pills, n):
    """Plain strings are spread evenly over the segment; dicts keep their own start/end frames."""
    out, k = [], len(pills or [])
    for j, p in enumerate(pills or []):
        if isinstance(p, str):
            p = {"text": p, "start": round(j * n / k) + 6, "end": round((j + 1) * n / k) - 6}
        out.append(p)
    return out


def normalise(seg: dict, cfg: dict) -> dict:
    """Map the template vocabulary (seconds, t/t0, ai/physics, hero as a path, card text, outro line) onto
    the internal keys used by the segment renderers. Both vocabularies are accepted."""
    s = dict(seg)
    fps = float(cfg.get("fps", 24))
    if "frames" not in s:
        s["frames"] = max(1, round(float(s["seconds"]) * fps)) if "seconds" in s else DEFAULT_FRAMES[s["type"]]
    if "t0" in s:
        s.setdefault("start_s", s["t0"])
    if "t" in s and s["type"] != "title":
        s.setdefault("start_s", s["t"])
    typ = s["type"]
    if typ == "title":
        s.setdefault("title", cfg.get("title", ""))
        s.setdefault("subtitle", cfg.get("subtitle"))
        s.setdefault("accent_text", cfg.get("accent_text"))
        h = s.get("hero")
        if isinstance(h, str):
            h = {"clip": h, "start_s": s.get("t", 0)}
        if isinstance(h, dict):
            h = dict(h)
            if "ai" in h:
                h["left"] = h.pop("ai")
            if "physics" in h:
                h["right"] = h.pop("physics")
            if "t" in h:
                h.setdefault("start_s", h.pop("t"))
            s["hero"] = h
        s.setdefault("left_label", cfg.get("left_label", "AI video"))
        s.setdefault("right_label", cfg.get("right_label", "Physics"))
    elif typ == "split" and "ai" in s:
        s["left_label"] = s.get("left", s.get("left_label", "AI video."))
        s["right_label"] = s.get("right", s.get("right_label", "Physics."))
        s["left"], s["right"] = s.pop("ai"), s.pop("physics")
    elif typ == "cards":
        s["cards"] = [{**c, "title": c.get("title", c.get("text"))} for c in s.get("cards", [])]
    elif typ == "outro":
        if isinstance(s.get("hero"), str):
            s.setdefault("clip", s["hero"])
        s.setdefault("subtitle", s.get("line"))
        s.setdefault("title", cfg.get("title", ""))
        s.setdefault("accent_text", cfg.get("accent_text"))
    s["pills"] = _pills(s.get("pills"), s["frames"])
    return s


def validate(cfg) -> list[dict]:
    cfg["segments"] = [normalise(s, cfg) if s.get("type") in SEGMENTS else s for s in cfg.get("segments") or []]
    segs = cfg.get("segments") or []
    if not segs:
        raise ValueError("release config has no segments")
    for k, s in enumerate(segs):
        if s.get("type") not in SEGMENTS:
            raise ValueError(f"segment {k}: unknown type {s.get('type')!r} (one of {sorted(SEGMENTS)})")
    if segs[0]["type"] != "title":
        raise ValueError("the first segment must be a 'title' (frame 0 is the title-over-hero thumbnail)")
    title = segs[0]
    if "hero" not in title:           # default hero: the first split, else the first full clip
        for s in segs:
            if s["type"] == "split":
                title["hero"] = {"left": s["left"], "right": s["right"], "start_s": s.get("start_s", 0)}
                break
            if s["type"] == "full":
                title["hero"] = {"clip": s["clip"], "start_s": s.get("start_s", 0)}
                break
        else:
            raise ValueError("title segment needs a 'hero' (no split/full segment to borrow one from)")
    return segs


def render(config_path, out, draft=False, log=print) -> dict:
    """Render the film described by config_path to out (MP4). draft = half resolution."""
    config_path = Path(config_path)
    cfg = json.loads(config_path.read_text())
    segs = validate(cfg)
    F = Film(cfg, config_path.parent, draft)
    total = 0
    with VideoWriter(out, F.W, F.H, F.fps, crf=23 if draft else 17, pix_fmt_in="rgb24",
                     preset="veryfast" if draft else "medium") as vw:
        for k, seg in enumerate(segs):
            n = int(seg["frames"])
            fn = SEGMENTS[seg["type"]]
            for i in range(n):
                frame = fn(F, seg, i, n)
                if total == 0 and float(frame.mean()) < 8:
                    raise ValueError("frame 0 is (nearly) black; the title hero must be visible in the thumbnail")
                vw.write(frame)
                total += 1
            log(f"segment {k} {seg['type']}: {n} frames")
    return {"out": str(out), "frames": total, "seconds": round(total / F.fps, 2), "size": [F.W, F.H]}


def thumbnail(config_path, out_png, draft=False) -> Path:
    """Write frame 0 (the title-over-hero thumbnail) as a PNG."""
    config_path = Path(config_path)
    cfg = json.loads(config_path.read_text())
    segs = validate(cfg)
    F = Film(cfg, config_path.parent, draft)
    img = seg_title(F, segs[0], 0, int(segs[0]["frames"]))
    cv2.imwrite(str(out_png), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return Path(out_png)
