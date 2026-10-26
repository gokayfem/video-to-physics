"""Blender integration: bake_pc2 + check_bake on a tiny synthetic npz (frames and subframes, generated strip and
mapped target with a time offset) and dye_sheet frame mapping. Prints SKIP when Blender is not installed."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "skills" / "video-to-physics" / "scripts"
sys.path.insert(0, str(SCRIPTS))
from v2p import dye, elastica  # noqa: E402


def _blender():
    for c in (os.environ.get("BLENDER"), "/opt/homebrew/bin/blender", shutil.which("blender")):
        if c and Path(c).exists():
            return c
    return None


def _run(script, *args):
    b = _blender()
    p = subprocess.run([b, "-b", "--factory-startup", "--python", str(SCRIPTS / "blender" / script), "--", *map(str, args)],
                       capture_output=True, text=True, timeout=120)
    tag = {"bake_pc2.py": "V2P_BAKE ", "check_bake.py": "V2P_CHECK_BAKE ", "dye_sheet.py": "V2P_DYE_SHEET "}[script]
    lines = [ln for ln in p.stdout.splitlines() if ln.startswith(tag)]
    assert lines, f"{script} failed:\n{p.stdout[-2000:]}\n{p.stderr[-2000:]}"
    return json.loads(lines[-1][len(tag):]), p.returncode


def _synthetic_npz(path, t0=0.0, fps=24.0, spf=4, frames=3):
    s = np.linspace(0, 0.2, 21)
    t = t0 + np.arange(frames * spf + 1) / (fps * spf)
    th = np.array([0.8 * np.sin(2 * np.pi * 2 * (tt - t0)) * s / 0.2 for tt in t])
    G = elastica.centreline_to_grid(elastica.centreline_from_angles(s, th), width=0.02, cols=4, origin=(0.1, -0.2, 0.3))
    elastica.write_deformation_npz(path, G, t, thickness=0.001)


def test_blender_bake_and_dye_sheet():
    if _blender() is None:
        print("SKIP test_blender_bake_and_dye_sheet: Blender not found")
        return
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        _synthetic_npz(d / "a.npz")
        rep, _ = _run("bake_pc2.py", "--npz", d / "a.npz", "--fps", 24, "--out", d / "a.blend")
        assert rep["samples_per_frame"] == 4 and rep["vertices"] == 2 * 21 * 4
        chk, rc = _run("check_bake.py", "--blend", d / "a.blend", "--frames", "0,0.25,1.5,2.75,3")
        assert rc == 0 and chk["max_err_m"] < 1e-6, chk
        # existing target object, cache starting 2 frames later
        _synthetic_npz(d / "b.npz", t0=2.0 / 24)
        _run("bake_pc2.py", "--npz", d / "b.npz", "--fps", 24, "--blend", d / "a.blend", "--target", "V2P_Strip",
             "--out", d / "b.blend")
        chk, rc = _run("check_bake.py", "--blend", d / "b.blend", "--frames", "2,2.25,3.5,4.75,5")
        assert rc == 0 and chk["max_err_m"] < 1e-6, chk
        # dye sheet: 4 tracers (2 PNG groups), scene frame n <-> f{n - 10}
        plane = dye.Plane(0.0, 0.1, -0.02, 0.02, 0.001)
        vel = lambda t, P: np.c_[np.full(len(P), 0.02), np.zeros(len(P))]
        srcs = [dye.Source((0.01, y), tracer=k, t_start=0.0) for k, y in enumerate((-0.01, -0.003, 0.003, 0.01))]
        dye.write_dye_sequence(d / "dye", vel, plane, np.arange(5) / 24, sources=srcs, frame_numbers=range(3, 8))
        rep, rc = _run("dye_sheet.py", "--dye-dir", d / "dye", "--out", d / "dye.blend", "--z", 0.01, "--frame-offset", 10)
        assert rc == 0 and rep["mapping_ok"] and rep["groups"] == 2 and rep["resolved"]["15"] == "f0005.png", rep


if __name__ == "__main__":
    test_blender_bake_and_dye_sheet()
    print("PASS test_blender_bake_and_dye_sheet")
