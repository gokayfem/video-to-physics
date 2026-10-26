"""Camera-motion measurement by ECC image alignment on static regions.

`homographies` returns H_n (frame 0 -> frame n) in full-resolution pixel coordinates using the +0.5
pixel-centre convention: a static point at u0 in frame 0 is at  u_n ~ H_n u0  in frame n.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np

from .video import iter_frames, source_fps


def _prep(frame, scale, blur):
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    if scale != 1:
        g = cv2.resize(g, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    g = g.astype(np.float32) / 255.0
    return cv2.GaussianBlur(g, (0, 0), blur) if blur > 0 else g


def _shift(a):
    T = np.eye(3)
    T[:2, 2] = a
    return T


def index_to_centre(H_idx, scale=1.0):
    """Homography in (scaled) array-index coords -> full-res pixel-centre coords."""
    S = np.diag([scale, scale, 1.0])
    H = np.linalg.inv(S) @ _shift(0.5) @ H_idx @ _shift(-0.5) @ S
    return H / H[2, 2]


def apply_h(H, pts) -> np.ndarray:
    pts = np.asarray(pts, float)
    q = np.c_[pts, np.ones(len(pts))] @ np.asarray(H, float).T
    return q[:, :2] / q[:, 2:]


def homographies(source, mask=None, scale=0.5, blur=1.0, motion="homography", iters=100, eps=1e-6,
                 max_frames=None):
    """ECC-align every frame to frame 0. mask: uint8 image (non-zero = static, usable) at full res.

    Returns (H array (N,3,3) frame0->n pixel-centre coords, ecc correlation per frame (NaN = failed)).
    Each frame is warm-started from the previous solution, so slow drifts are tracked robustly.
    """
    mode = {"translation": cv2.MOTION_TRANSLATION, "euclidean": cv2.MOTION_EUCLIDEAN,
            "affine": cv2.MOTION_AFFINE, "homography": cv2.MOTION_HOMOGRAPHY}[motion]
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iters, eps)
    frames = iter_frames(source, count=max_frames)
    f0 = next(frames)
    T = _prep(f0, scale, blur)
    m = None
    if mask is not None:
        m = cv2.resize((np.asarray(mask) > 0).astype(np.uint8) * 255, (T.shape[1], T.shape[0]),
                       interpolation=cv2.INTER_NEAREST)
    warp = np.eye(3, dtype=np.float32) if mode == cv2.MOTION_HOMOGRAPHY else np.eye(2, 3, dtype=np.float32)
    Hs, ccs = [np.eye(3)], [1.0]
    for f in frames:
        try:
            cc, warp = cv2.findTransformECC(T, _prep(f, scale, blur), warp.copy(), mode, crit, m, 5)
        except cv2.error:
            cc = float("nan")
        W3 = warp.astype(float) if warp.shape[0] == 3 else np.vstack([warp.astype(float), [0, 0, 1]])
        Hs.append(index_to_centre(W3, scale))
        ccs.append(float(cc))
    return np.array(Hs), np.array(ccs)


def auto_static_mask(source, std_max=6.0, step=10, max_frames=None) -> np.ndarray:
    """Pixels whose grey level varies little over the clip (std < std_max, 8-bit) -> 255."""
    stack = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for k, f in enumerate(iter_frames(source, count=max_frames))
             if k % step == 0]
    return ((np.stack(stack).astype(np.float32).std(0) < std_max) * 255).astype(np.uint8)


def probe_points(size, extra=()) -> np.ndarray:
    """Four inset corners + centre (+ caller points) used to report displacement in pixels."""
    W, H = size
    mx, my = 0.1 * W, 0.1 * H
    return np.array([[mx, my], [W - mx, my], [mx, H - my], [W - mx, H - my], [W / 2, H / 2], *extra], float)


def drift_plot(t, disp, out, title="camera drift (px)") -> Path:
    """Small dependency-free line plot: max corner displacement and centre displacement vs time."""
    from PIL import Image, ImageDraw
    from .video import font
    Wp, Hp, pad = 900, 360, 50
    im = Image.new("RGB", (Wp, Hp), (250, 250, 248))
    d = ImageDraw.Draw(im)
    ymax = max(float(np.nanmax(disp)) if np.isfinite(disp).any() else 0, 0.5) * 1.1
    tmax = max(float(t[-1]), 1e-9)
    X = lambda v: pad + (Wp - 2 * pad) * v / tmax
    Y = lambda v: Hp - pad - (Hp - 2 * pad) * v / ymax
    d.rectangle([pad, pad, Wp - pad, Hp - pad], outline=(120, 120, 120))
    for frac in (0, 0.5, 1):
        d.text((5, Y(ymax * frac / 1.1) - 6), f"{ymax * frac / 1.1:.2f}", fill=(80, 80, 80), font=font(12))
    d.text((Wp - pad - 40, Hp - pad + 8), f"{tmax:.2f} s", fill=(80, 80, 80), font=font(12))
    for col, series, name in (((200, 60, 40), np.nanmax(disp[:, :4], 1), "max corner"),
                              ((40, 90, 200), disp[:, 4], "centre")):
        pts = [(X(a), Y(b)) for a, b in zip(t, series) if np.isfinite(b)]
        if len(pts) > 1:
            d.line(pts, fill=col, width=2)
        d.text((pad + 10 + (0 if name == "max corner" else 120), pad + 6), name, fill=col, font=font(13))
    d.text((pad, 12), title, fill=(30, 30, 30), font=font(16, "Semibold"))
    out = Path(out)
    im.save(out)
    return out


def stabilize(video, out_dir, mask=None, scale=0.5, motion="homography", auto_mask=False) -> dict:
    """Measure camera motion of `video`; write H_frame0_to_n.npy, stability.csv, stability.json, drift.png."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    m = None
    if mask is not None:
        m = cv2.imread(str(mask), cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise ValueError(f"cannot read mask {mask}")
    elif auto_mask:
        m = auto_static_mask(video)
        cv2.imwrite(str(out / "static_mask.png"), m)
    Hs, ccs = homographies(video, m, scale=scale, motion=motion)
    fps = source_fps(video)
    f0 = next(iter_frames(video, count=1))
    size = (f0.shape[1], f0.shape[0])
    pts = probe_points(size)
    disp = np.array([np.linalg.norm(apply_h(H, pts) - pts, axis=1) for H in Hs])
    t = np.arange(len(Hs)) / fps
    np.save(out / "H_frame0_to_n.npy", Hs)
    with open(out / "stability.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "t_s", "ecc_cc", "disp_px_corner_max", "disp_px_centre"] +
                   [f"h{i}{j}" for i in range(3) for j in range(3)])
        for i in range(len(Hs)):
            w.writerow([i, f"{t[i]:.6f}", f"{ccs[i]:.6f}", f"{disp[i, :4].max():.4f}", f"{disp[i, 4]:.4f}"] +
                       [f"{v:.10g}" for v in Hs[i].ravel()])
    summary = {"video": str(video), "frames": len(Hs), "fps": fps, "size": list(size), "motion": motion,
               "scale": scale, "mask": str(mask) if mask else ("auto" if auto_mask else None),
               "convention": "H maps frame-0 pixel-centre coords (u = col + 0.5) to frame n",
               "max_corner_disp_px": float(np.nanmax(disp[:, :4])), "max_centre_disp_px": float(np.nanmax(disp[:, 4])),
               "final_translation_px": [float(Hs[-1][0, 2]), float(Hs[-1][1, 2])],
               "ecc_min": float(np.nanmin(ccs)), "failed_frames": [int(i) for i in np.where(~np.isfinite(ccs))[0]]}
    (out / "stability.json").write_text(json.dumps(summary, indent=2))
    drift_plot(t, disp, out / "drift.png")
    return summary


def affine_drift(f0, f, mask=None, scale=0.5) -> float:
    """Max corner displacement (full-res px) of an affine ECC fit frame0 -> frame (NaN if ECC fails)."""
    g0, g = _prep(f0, scale, 0), _prep(f, scale, 0)
    m = None if mask is None else cv2.resize(mask.astype(np.uint8), (g0.shape[1], g0.shape[0]),
                                             interpolation=cv2.INTER_NEAREST)
    warp = np.eye(2, 3, dtype=np.float32)
    try:
        cv2.findTransformECC(g0, g, warp, cv2.MOTION_AFFINE,
                             (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-5), m, 5)
    except cv2.error:
        return float("nan")
    h, w = g0.shape
    pts = np.array([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1]], np.float32)
    return float(np.abs(pts @ warp.T - pts[:, :2]).max() / scale)
