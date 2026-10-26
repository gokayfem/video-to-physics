"""Top-down rectification of a world plane Z = z_plane seen by a calibrated camera.

Output grid: x right, y up; row 0 is the +y edge, column 0 the -x edge, pixel centres at +0.5 (so output
pixel (r, c) samples world x = x0 + (c + 0.5) px, y = y1 - (r + 0.5) px). Optional flat-interface refraction
(n, z_s) for planes under a liquid surface, and optional per-frame homographies from `stabilize` to follow a
moving camera (frame-0 camera of record, then H_n).
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .camera import Camera
from .video import iter_frames, source_fps


def grid(extent, px):
    """Pixel-centre world coordinates of the output grid. extent = (x0, x1, y0, y1), px = metres/pixel."""
    x0, x1, y0, y1 = extent
    nx, ny = int(round((x1 - x0) / px)), int(round((y1 - y0) / px))
    if nx <= 0 or ny <= 0:
        raise ValueError(f"empty rectification grid for extent {extent} at {px} m/px")
    xs = x0 + px * (np.arange(nx) + 0.5)
    ys = y1 - px * (np.arange(ny) + 0.5)
    return xs, ys


def rect_maps(cam: Camera, z_plane: float, extent, px, refract=None):
    """Source-image pixel-centre coords (u, v) for every output pixel: arrays (ny, nx) float64."""
    xs, ys = grid(extent, px)
    Xg, Yg = np.meshgrid(xs, ys)
    P = np.stack([Xg, Yg, np.full_like(Xg, z_plane)], -1)
    uv = cam.project(P, refract=refract)
    behind = cam.depth(P) <= 0
    uv[behind] = np.nan
    return uv[..., 0], uv[..., 1]


def remap(img, u, v, H=None, interp=cv2.INTER_LINEAR):
    """Sample `img` at pixel-centre coords (u, v), optionally first mapped by homography H (frame0 -> n)."""
    if H is not None:
        q = np.stack([u, v, np.ones_like(u)], -1) @ np.asarray(H, float).T
        u, v = q[..., 0] / q[..., 2], q[..., 1] / q[..., 2]
    mu = np.nan_to_num(u - 0.5, nan=-1e6).astype(np.float32)   # pixel centre -> array index
    mv = np.nan_to_num(v - 0.5, nan=-1e6).astype(np.float32)
    return cv2.remap(img, mu, mv, interp, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def valid_mask(u, v, size, margin=1.0) -> np.ndarray:
    W, H = size
    return np.isfinite(u) & np.isfinite(v) & (u >= margin) & (u <= W - margin) & (v >= margin) & (v <= H - margin)


def rectify_video(video, cam: Camera, z_plane, extent, px, out_dir, refract=None, homographies=None,
                  max_frames=None, ext="png") -> dict:
    """Rectify every frame of `video` (or image dir) to out_dir/r%04d.png + meta.json + valid.png."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    u, v = rect_maps(cam, z_plane, extent, px, refract)
    valid = valid_mask(u, v, cam.size)
    Hs = None if homographies is None else np.asarray(homographies, float)
    n = 0
    for i, frame in enumerate(iter_frames(video, count=max_frames)):
        if (frame.shape[1], frame.shape[0]) != tuple(cam.size):
            raise ValueError(f"frame {i} is {frame.shape[1]}x{frame.shape[0]}, camera expects {cam.size}")
        H = None if Hs is None else Hs[min(i, len(Hs) - 1)]
        im = remap(frame, u, v, H)
        im[~valid] = 0
        cv2.imwrite(str(out / f"r{i:04d}.{ext}"), im)
        n += 1
    cv2.imwrite(str(out / "valid.png"), valid.astype(np.uint8) * 255)
    xs, ys = grid(extent, px)
    meta = {"source": str(video), "frames": n, "fps": source_fps(video), "z_plane_m": z_plane,
            "extent_m": {"x0": extent[0], "x1": extent[1], "y0": extent[2], "y1": extent[3]},
            "px_m": px, "shape": [len(ys), len(xs)],
            "pixel_rule": "output (row r, col c) samples x = x0 + (c+0.5)*px, y = y1 - (r+0.5)*px; row 0 = +y edge",
            "refraction": None if refract is None else {"n": refract[0], "z_surface_m": refract[1],
                                                         "model": "flat interface, exact Snell"},
            "stabilised": Hs is not None, "valid_fraction": float(valid.mean()),
            "note": "pixels outside valid.png (off-image or behind camera) are black"}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def parse_plane(s: str) -> float:
    """'z=0.013' -> 0.013"""
    k, _, val = s.partition("=")
    if k.strip().lower() != "z" or not val:
        raise ValueError(f"--plane must look like z=H, got {s!r}")
    return float(val)


def parse_refract(s: str | None):
    """'n=1.333,z=0.023' -> (1.333, 0.023)"""
    if not s:
        return None
    kv = dict(p.split("=", 1) for p in s.split(","))
    return float(kv["n"]), float(kv["z"])
