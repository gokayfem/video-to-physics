"""`new` scaffolds every dir from an mp4, an image and a text idea (using the real templates); CLI help works."""
import json
import re
import subprocess
import sys

import cv2
import numpy as np

from synth import SCRIPTS, gradient_video, need_ffmpeg, tmpdir
from v2p.case import ROLES, new_case

CLI = [sys.executable, str(SCRIPTS / "v2p.py")]
COMMANDS = ["new", "probe", "generate", "screen", "stabilize", "rectify", "sbs", "sheet", "page", "checkpoint",
            "watchdog", "lock", "release"]


def _check_layout(case):
    for r in ROLES:
        assert (case / r / "visuals").is_dir() and (case / r / "PROGRESS.log").exists(), r
    for f in ("CONTRACT.md", "STATUS.json", "config/case.json", "config/acceptance.json", "config/CHANGELOG.md",
              "config/build.json", "reference/meta.json", "review/page.json"):
        assert (case / f).exists(), f
    assert (case / "checkpoints").is_dir() and (case / "review" / "visuals").is_dir()
    json.loads((case / "config" / "acceptance.json").read_text())
    json.loads((case / "review" / "page.json").read_text())
    contract = (case / "CONTRACT.md").read_text()
    for known in ("TITLE", "CASE", "DATE", "INPUT_KIND", "SHA256", "PREFIX"):
        assert "{{" + known + "}}" not in contract, known
    return contract


def test_new_from_video():
    need_ffmpeg()
    d = tmpdir()
    gradient_video(d / "clip.mp4", n=10)
    case = d / "01-test-case"
    cfg = new_case(case, str(d / "clip.mp4"), title='Test "quoted" case')
    contract = _check_layout(case)
    assert cfg["kind"] == "ai-video" and cfg["frames"] == 10 and cfg["size"] == [160, 96]
    assert (case / "reference" / "reference.mp4").exists() and (case / "reference" / "SHA256").exists()
    assert cfg["reference"]["sha256"] in contract and "160x96" in contract and "TestCase" in contract
    assert "{{" + "LAST_FRAME" + "}}" not in contract


def test_new_from_image_and_idea_and_none():
    d = tmpdir()
    cv2.imwrite(str(d / "still.png"), np.zeros((40, 60, 3), np.uint8))
    cfg = new_case(d / "img", str(d / "still.png"))
    _check_layout(d / "img")
    assert cfg["kind"] == "image" and cfg["size"] == [60, 40]
    cfg = new_case(d / "idea", "a heavy chain falls off a table edge")
    _check_layout(d / "idea")
    assert cfg["kind"] == "idea" and "chain" in (d / "idea" / "reference" / "idea.md").read_text()
    cfg = new_case(d / "none", None, kind="none")
    assert cfg["kind"] == "none" and cfg["reference"]["sha256"] is None
    try:
        new_case(d / "idea", "again")
        raise AssertionError("non-empty case overwritten")
    except FileExistsError:
        pass


def test_cli_help_and_dry_run():
    out = subprocess.run(CLI + ["-h"], capture_output=True, text=True)
    assert out.returncode == 0 and all(c in out.stdout for c in COMMANDS)
    for c in COMMANDS:
        r = subprocess.run(CLI + [c, "-h"], capture_output=True, text=True)
        assert r.returncode == 0 and "usage:" in r.stdout, c
    d = tmpdir()
    subprocess.run(CLI + ["new", str(d / "c"), "--input", "a ball rolls down a ramp"], check=True, capture_output=True)
    cv2.imwrite(str(d / "f.png"), np.zeros((40, 60, 3), np.uint8))
    r = subprocess.run(CLI + ["generate", str(d / "c"), "--first-frame", str(d / "f.png"), "--prompt", "x",
                              "--n", "1", "--seed", "3", "--param", "resolution=720p", "--dry-run"],
                       capture_output=True, text=True, env={"PATH": "/usr/bin:/bin"})
    assert r.returncode == 0, r.stderr
    inp = json.loads((d / "c" / "generation" / "cand0" / "input.json").read_text())
    assert inp["seed"] == 3 and inp["resolution"] == "720p"
    r = subprocess.run(CLI + ["lock", "--lock", str(d / "l.lock"), "--", sys.executable, "-c", "print('ran')"],
                       capture_output=True, text=True)
    assert r.returncode == 0 and "ran" in r.stdout and not (d / "l.lock").exists()
    r = subprocess.run(CLI + ["watchdog", str(d), "--lock", str(d / "l.lock")], capture_output=True, text=True)
    assert r.returncode == 0 and re.search(r"watchdog: 0 alert", r.stdout), r.stdout
