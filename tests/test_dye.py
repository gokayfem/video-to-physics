"""Tests for v2p.dye: translation accuracy, mass conservation under stretching (peak ~ 1/lam), multiple tracers,
PNG/meta round trip."""
import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "skills" / "video-to-physics" / "scripts"))
from v2p import dye  # noqa: E402

PLANE = dye.Plane(0.0, 0.2, -0.05, 0.05, 0.0005)          # 400 x 200 px


def _centroid_x(img):
    xs, _ = PLANE.centres()
    return float((img.sum(0) * xs).sum() / img.sum())


def test_uniform_translation_line_and_streak():
    U = 0.01
    xg, yg = np.linspace(-0.1, 0.3, 9), np.linspace(-0.1, 0.1, 5)
    vel = dye.GridVelocity([0.0, 10.0], xg, yg, np.full((2, 5, 9), U), np.zeros((2, 5, 9)))
    line = dye.MaterialLine(np.c_[np.full(41, 0.05), np.linspace(-0.02, 0.02, 41)], tracer=0, t0=0.0)
    src = dye.Source((0.02, 0.03), tracer=1, t_start=0.0)
    cfg = dye.RenderConfig(sigma0=0.001, sigma_aa_px=0.5)
    times = [0.0, 1.0, 2.5, 4.0]
    for fi, t, img, budget, _ in dye.advect_dye(vel, PLANE, times, [src], [line], dt=0.05, cfg=cfg):
        err_line = abs(_centroid_x(img[0]) - (0.05 + U * t)) / PLANE.px
        assert err_line <= 1.0, (t, err_line)
        if t > 0:
            err_streak = abs(_centroid_x(img[1]) - (0.02 + 0.5 * U * t)) / PLANE.px
            assert err_streak <= 1.0, (t, err_streak)
            assert abs(budget[1]["image_unclipped"] / budget[1]["injected"] - 1) < 0.02


def test_pure_shear_mass_and_peak():
    gamma = 1.0
    vel = lambda t, P: np.c_[gamma * P[:, 1], np.zeros(len(P))]
    line = dye.MaterialLine(np.c_[np.full(81, 0.10025), np.linspace(-0.02, 0.02, 81)], tracer=0, t0=0.0)
    s0_px, saa = 0.3, 2.0
    cfg = dye.RenderConfig(sigma0=s0_px * PLANE.px, sigma_aa_px=saa)
    se0 = math.hypot(s0_px, saa)
    res = []
    for fi, t, img, budget, _ in dye.advect_dye(vel, PLANE, [0.0, 1.0, 2.0], lines=[line], dt=0.02, cfg=cfg):
        lam = math.sqrt(1 + (gamma * t) ** 2)
        mid = img[0][80:120, 150:250]                              # away from the line ends
        c_pred = se0 / (lam * math.sqrt(s0_px**2 / lam**2 + saa**2))
        res.append((lam, float(img[0].sum()), float(mid.max()), c_pred))
    m0, p0, c0 = res[0][1], res[0][2], res[0][3]
    for lam, m, p, c in res[1:]:
        assert abs(m / m0 - 1) < 0.02, (lam, m / m0)
        assert abs((p / p0) / (c / c0) - 1) < 0.05, (lam, p / p0, c / c0)  # pixel sampling of a tilted line
    assert abs(res[2][2] / res[0][2] * res[2][0] - 1) < 0.05      # peak ~ 1/lam when sigma ~ sigma_aa


def test_multiple_tracers_png_meta_roundtrip():
    U = 0.02
    vel = lambda t, P: np.c_[np.full(len(P), U), np.zeros(len(P))]
    srcs = [dye.Source((0.02, y), tracer=k, t_start=0.0) for k, y in enumerate((-0.03, -0.01, 0.01, 0.03))]
    with tempfile.TemporaryDirectory() as d:
        meta = dye.write_dye_sequence(d, vel, PLANE, [0.0, 0.5, 1.0], sources=srcs, frame_numbers=[10, 11, 12],
                                      dt=0.05, keep_images=True)
        assert (Path(d) / "f0012.png").exists() and (Path(d) / "g1" / "f0012.png").exists()
        m = json.loads((Path(d) / "meta.json").read_text())
        assert m["plane"]["nx"] == 400 and m["plane"]["ny"] == 200 and "pixel_centre_rule" in m
        assert [tr["dir"] for tr in m["tracers"]] == [".", ".", ".", "g1"]
        rgb = dye.read_tracer_png(Path(d) / "f0012.png")
        assert rgb.shape == (200, 400, 3) and np.max(np.abs(rgb[..., 0] - meta["images"][2, 0])) < 1e-4
        g1 = dye.read_tracer_png(Path(d) / "g1" / "f0012.png")
        assert np.max(np.abs(g1[..., 0] - meta["images"][2, 3])) < 1e-4 and g1[..., 1:].max() == 0
        _, ys = PLANE.centres()
        for k, y in enumerate((-0.03, -0.01, 0.01, 0.03)):    # each tracer sits on its own source row
            img = meta["images"][2, k]
            yc = float((img.sum(1) * ys).sum() / img.sum())
            assert abs(yc - y) < PLANE.px, (k, yc)
            b = m["frames"][2]["budget_px2"][str(k)]
            assert abs(b["png"] / b["injected"] - 1) < 0.02


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name)
