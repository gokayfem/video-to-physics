"""probe, sbs (dims, frame count, fps), sheet, review page."""
import json

import cv2

from synth import gradient_video, need_ffmpeg, tmpdir
from v2p.compose import sbs, sheet
from v2p.review_page import build
from v2p.video import probe, sha256_file


def test_probe():
    need_ffmpeg()
    d = tmpdir()
    gradient_video(d / "a.mp4", n=10, w=160, h=96, fps=24)
    p = probe(d / "a.mp4")
    assert (p["frames"], p["width"], p["height"]) == (10, 160, 96)
    assert abs(p["fps"] - 24) < 1e-6 and abs(p["duration_s"] - 10 / 24) < 1e-6
    assert p["sha256"] == sha256_file(d / "a.mp4")


def test_sbs_video_and_png_dir():
    need_ffmpeg()
    d = tmpdir()
    gradient_video(d / "ref.mp4", n=10, w=160, h=96, fps=25)
    frames = gradient_video(d / "ours.mp4", n=8, w=320, h=192, colour=(200, 80, 40))
    r = sbs(d / "ref.mp4", d / "ours.mp4", d / "sbs.mp4", label_ours="ours")
    p = probe(d / "sbs.mp4")
    assert p["frames"] == 8 == r["frames"]
    assert [p["width"], p["height"]] == r["size"] and p["width"] == 3 * 160
    assert abs(p["fps"] - 25) < 1e-6                       # reference fps, no retiming
    (d / "png").mkdir()
    for k, f in enumerate(frames):
        cv2.imwrite(str(d / "png" / f"f{k:04d}.png"), f)
    r2 = sbs(d / "ref.mp4", d / "png", d / "sbs2.mp4", frames=5)
    assert probe(d / "sbs2.mp4")["frames"] == 5 == r2["frames"]


def test_sheet():
    need_ffmpeg()
    d = tmpdir()
    gradient_video(d / "a.mp4", n=24, w=160, h=96, fps=24)
    r = sheet(d / "a.mp4", d / "s.jpg", times=[0, 0.25, 0.5, 0.9], label="test", cols=2, thumb_w=160)
    img = cv2.imread(str(d / "s.jpg"))
    assert r["frames"] == [0, 6, 12, 22]
    assert img.shape[1] == 320 and img.shape[0] > 2 * 96


def test_review_page():
    d = tmpdir()
    (d / "review" / "visuals").mkdir(parents=True)
    (d / "review" / "visuals" / "a.png").write_bytes(b"x")
    spec = {"title": "Case <T>", "status": "ck001 preview", "summary_md": "para one\n\npara two",
            "items": [{"path": "visuals/a.png", "label": "sheet", "caption": "frame 0"},
                      {"path": "visuals/sbs.mp4", "label": "side by side", "caption": "no retiming"}]}
    (d / "review" / "page.json").write_text(json.dumps(spec))
    r = build(d)
    html = (d / "review" / "index.html").read_text()
    assert r["missing"] == ["visuals/sbs.mp4"] and r["items"] == 2
    assert "<video" in html and 'src="visuals/a.png"' in html and "Case &lt;T&gt;" in html
    assert "<p>para two</p>" in html and "(missing)" in html
