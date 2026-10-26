"""Release film: --draft render of a synthetic 2-segment config and of the real templates/release.json shape;
frame 0 is the title thumbnail, never black."""
import json
from pathlib import Path

import cv2
import numpy as np

from synth import gradient_video, need_ffmpeg, tmpdir
from v2p.release import render, thumbnail
from v2p.video import probe, read_frames

TEMPLATES = Path(__file__).resolve().parents[1] / "skills" / "video-to-physics" / "templates"


def _clips(d):
    gradient_video(d / "ai.mp4", n=30, w=192, h=108, colour=(60, 90, 160))
    gradient_video(d / "phys.mp4", n=30, w=192, h=108, colour=(150, 120, 60))
    img = np.full((120, 180, 3), 90, np.uint8)
    cv2.putText(img, "plot", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    cv2.imwrite(str(d / "card.png"), img)


def test_two_segment_draft():
    need_ffmpeg()
    d = tmpdir()
    _clips(d)
    cfg = {"fps": 24, "segments": [
        {"type": "title", "frames": 6, "title": "Video 2 Physics", "accent_text": "2", "subtitle": "test",
         "hero": {"left": "ai.mp4", "right": "phys.mp4", "start_s": 0.2}},
        {"type": "split", "frames": 8, "left": "ai.mp4", "right": "phys.mp4", "start_s": 0.1, "name": "split",
         "pills": [{"start": 0, "end": 8, "text": "a pill", "dot": "accent"}]}]}
    (d / "rel.json").write_text(json.dumps(cfg))
    r = render(d / "rel.json", d / "rel.mp4", draft=True, log=lambda *_: None)
    p = probe(d / "rel.mp4")
    assert p["frames"] == 14 == r["frames"] and (p["width"], p["height"]) == (960, 540)
    f0 = read_frames(d / "rel.mp4", count=1)[0]
    assert f0.mean() > 8, f0.mean()
    thumbnail(d / "rel.json", d / "thumb.png", draft=True)
    assert cv2.imread(str(d / "thumb.png")).mean() > 8


def test_first_segment_must_be_title():
    d = tmpdir()
    (d / "bad.json").write_text(json.dumps({"segments": [{"type": "stat", "frames": 3, "value": 1}]}))
    try:
        render(d / "bad.json", d / "x.mp4", draft=True)
        raise AssertionError("non-title first segment accepted")
    except ValueError:
        pass


def test_real_release_template_draft():
    need_ffmpeg()
    d = tmpdir()
    _clips(d)
    cfg = json.loads((TEMPLATES / "release.json").read_text())

    def swap(x):        # replace example asset paths with the synthetic clips, keep the shape
        if isinstance(x, dict):
            return {k: swap(v) for k, v in x.items()}
        if isinstance(x, list):
            return [swap(v) for v in x]
        if isinstance(x, str) and x.endswith(".mp4"):
            return "phys.mp4" if ("final" in x or "review" in x) else "ai.mp4"
        if isinstance(x, str) and x.endswith((".png", ".jpg")):
            return "card.png"
        return x
    cfg = swap(cfg)
    for s in cfg["segments"]:
        s["seconds"] = 0.25
    (d / "release.json").write_text(json.dumps(cfg))
    r = render(d / "release.json", d / "film.mp4", draft=True, log=lambda *_: None)
    assert r["frames"] == 6 * len(cfg["segments"])
    assert probe(d / "film.mp4")["frames"] == r["frames"]
    assert read_frames(d / "film.mp4", count=1)[0].mean() > 8
