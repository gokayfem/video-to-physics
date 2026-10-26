"""generate --dry-run writes the request files without network or key; uncertain jobs are never resubmitted;
screen scores synthetic candidates."""
import json
import os

import cv2
import numpy as np

from synth import gradient_video, need_ffmpeg, tmpdir
from v2p.fal_gen import generate, run_job, screen


def _frame(d):
    img = np.full((96, 160, 3), (40, 160, 220), np.uint8)
    cv2.imwrite(str(d / "frame0.png"), img)
    return d / "frame0.png"


def test_generate_dry_run_no_network():
    d = tmpdir()
    case = d / "case"
    (case / "config").mkdir(parents=True)
    (case / "config" / "case.json").write_text(json.dumps({"generation": {"endpoint": "vendor/model/i2v",
                                                                          "settings": {"resolution": "720p"}}}))
    old = os.environ.pop("FAL_KEY", None)
    try:
        plan = generate(case, _frame(d), "a pendulum swings", n=2, seed=7, dry_run=True, round_name="r1",
                        log=lambda *_: None)
    finally:
        if old is not None:
            os.environ["FAL_KEY"] = old
    g = case / "generation" / "r1"
    assert plan["endpoint"] == "vendor/model/i2v" and plan["seeds"] == [7, 8]
    assert (g / "first_frame.png").exists() and (g / "plan.json").exists()
    for k in range(2):
        inp = json.loads((g / f"cand{k}" / "input.json").read_text())
        assert inp["seed"] == 7 + k and inp["image_url"] == "DRY-RUN" and inp["resolution"] == "720p"
        assert inp["duration"] == 10 and inp["prompt"] == "a pendulum swings"
        assert not (g / f"cand{k}" / "submission_started.json").exists()
    assert not (g / "upload.json").exists()


def test_uncertain_submission_not_resubmitted():
    d = tmpdir() / "cand0"
    d.mkdir()
    (d / "submission_started.json").write_text("{}")
    msg = run_job(d, "vendor/model", {"prompt": "x"})
    assert "UNCERTAIN" in msg and not (d / "queue.json").exists()


def test_screen_candidates():
    need_ffmpeg()
    d = tmpdir()
    g = d / "case" / "generation"
    (g / "cand0").mkdir(parents=True)
    cv2.imwrite(str(g / "first_frame.png"), np.full((96, 160, 3), (40, 160, 220), np.uint8))
    gradient_video(g / "cand0" / "video.mp4", n=12)
    (g / "cand1").mkdir()
    rep = screen(d / "case", log=lambda *_: None)
    assert rep["cand1"]["status"] == "missing" and rep["cand0"]["status"] == "ok"
    assert rep["cand0"]["frames"] == 12 and rep["cand0"]["frame0_mean_abs_diff_8bit"] < 30
    assert (g / "screen.json").exists() and (g / "screen_sheet.jpg").exists()
