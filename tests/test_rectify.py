"""Rectify: an obliquely viewed synthetic checkerboard plane comes back square and in place."""
import json

import cv2
import numpy as np

from synth import tmpdir
from v2p.camera import Camera
from v2p.rectify import grid, rect_maps, rectify_video, remap, valid_mask

SQ = 0.02    # checker square (m)


def checker(x, y):
    return (((np.floor(x / SQ) + np.floor(y / SQ)) % 2) * 200 + 30).astype(np.uint8)


def render_plane(cam, z, refract=None, ss=3):
    """Ground-truth image: supersampled ray casting onto the plane."""
    W, H = cam.size
    acc = np.zeros((H, W), np.float64)
    offs = (np.arange(ss) + 0.5) / ss
    u0, v0 = np.meshgrid(np.arange(W, dtype=float), np.arange(H, dtype=float))
    for oy in offs:
        for ox in offs:
            P = cam.ray_plane(np.stack([u0 + ox, v0 + oy], -1), z, refract=refract)
            acc += checker(P[..., 0], P[..., 1])
    g = (acc / ss ** 2).astype(np.uint8)
    return cv2.merge([g, g, g])


def _check(refract):
    cam = Camera.look_at(eye=[0.05, -0.35, 0.30], target=[0, 0, 0], size=(480, 360), fx=520)
    img = render_plane(cam, 0.0, refract)
    extent, px = (-0.08, 0.08, -0.06, 0.06), 0.001
    u, v = rect_maps(cam, 0.0, extent, px, refract)
    out = remap(img, u, v)[..., 0].astype(float)
    xs, ys = grid(extent, px)
    X, Y = np.meshgrid(xs, ys)
    truth = checker(X, Y).astype(float)
    # compare away from checker edges (resampling blur) and image borders
    fx, fy = (X / SQ) % 1, (Y / SQ) % 1
    interior = (np.minimum(fx, 1 - fx) > 0.2) & (np.minimum(fy, 1 - fy) > 0.2) & valid_mask(u, v, cam.size, 3)
    assert interior.mean() > 0.2
    err = np.abs(out - truth)[interior]
    assert err.mean() < 3 and np.percentile(err, 99) < 20, (err.mean(), np.percentile(err, 99))


def test_rectify_checkerboard():
    _check(None)


def test_rectify_checkerboard_refracted():
    _check((1.333, 0.03))


def test_rectify_video_writes_frames_and_meta():
    d = tmpdir()
    cam = Camera.look_at(eye=[0.0, -0.3, 0.3], target=[0, 0, 0], size=(320, 240), fx=360)
    img = render_plane(cam, 0.0, ss=1)
    (d / "frames").mkdir()
    for k in range(3):
        cv2.imwrite(str(d / "frames" / f"f{k:04d}.png"), img)
    meta = rectify_video(d / "frames", cam, 0.0, (-0.05, 0.05, -0.04, 0.04), 0.002, d / "out")
    assert meta["frames"] == 3 and meta["shape"] == [40, 50]
    assert cv2.imread(str(d / "out" / "r0002.png")).shape[:2] == (40, 50)
    assert json.loads((d / "out" / "meta.json").read_text())["px_m"] == 0.002
