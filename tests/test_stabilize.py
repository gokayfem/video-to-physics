"""Stabilize: synthetic textured video translated by a known amount per frame -> recovered within 0.2 px."""
import json

import numpy as np

from synth import moving_video, need_ffmpeg, tmpdir
from v2p.stabilize import apply_h, homographies, stabilize


def test_recovers_translation_png_dir():
    import cv2
    d = tmpdir()
    from synth import shifted, texture
    tex = texture(384, 304)
    for k in range(8):
        cv2.imwrite(str(d / f"f{k:04d}.png"), shifted(tex, 20 + 0.73 * k, 20 - 0.41 * k, 320, 240))
    Hs, cc = homographies(d, scale=1.0, motion="homography")
    for k, H in enumerate(Hs):
        p = apply_h(H, [[160, 120], [30, 30], [290, 210]])
        err = p - np.array([[160, 120], [30, 30], [290, 210]]) - [0.73 * k, -0.41 * k]
        assert np.abs(err).max() < 0.2, (k, err)


def test_stabilize_cli_outputs_mp4():
    need_ffmpeg()
    d = tmpdir()
    moving_video(d / "v.mp4", n=12, step=(0.6, -0.35))
    s = stabilize(d / "v.mp4", d / "out")
    for f in ("H_frame0_to_n.npy", "stability.csv", "stability.json", "drift.png"):
        assert (d / "out" / f).exists(), f
    tx, ty = s["final_translation_px"]
    assert abs(tx - 0.6 * 11) < 0.2 and abs(ty + 0.35 * 11) < 0.2, (tx, ty)
    assert json.loads((d / "out" / "stability.json").read_text())["frames"] == 12
