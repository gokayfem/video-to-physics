"""Pinhole camera from the contract JSON, projection, ray-plane intersection and flat-interface refraction.

Camera JSON (OpenCV convention, x right / y down / z forward; pixel centres at +0.5, so pixel (col, row)
covers u in [col, col+1) and its centre is u = col + 0.5):

    {"image_size": [W, H],
     "intrinsics": {"fx_px": .., "fy_px": .., "cx_px": .., "cy_px": ..},
     "extrinsics": {"R_world_to_cam": [[3x3]], "t": [3]}}          # X_cam = R X_world + t

Refraction model: a flat horizontal interface z = z_s (world Z up), camera above it in a medium of index 1,
points below it in a medium of index n (e.g. water 1.333). Exact Snell geometry, no paraxial approximation.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Camera:
    K: np.ndarray            # 3x3 intrinsics
    R: np.ndarray            # world -> camera rotation
    t: np.ndarray            # world -> camera translation
    size: tuple              # (W, H) pixels

    # ---------------------------------------------------------- construction / serialisation
    @classmethod
    def from_dict(cls, d: dict) -> "Camera":
        intr, extr = d["intrinsics"], d["extrinsics"]
        if "K" in intr:
            K = np.array(intr["K"], float)
        else:
            K = np.array([[intr["fx_px"], intr.get("skew", 0.0), intr["cx_px"]],
                          [0, intr["fy_px"], intr["cy_px"]], [0, 0, 1]], float)
        R = np.array(extr.get("R_world_to_cam", extr.get("R")), float)
        t = np.array(extr.get("t", extr.get("t_world_to_cam")), float).reshape(3)
        return cls(K, R, t, tuple(int(v) for v in d["image_size"]))

    @classmethod
    def from_json(cls, path) -> "Camera":
        return cls.from_dict(json.loads(Path(path).read_text()))

    @classmethod
    def look_at(cls, eye, target, size, fx, fy=None, up=(0, 0, 1), cx=None, cy=None) -> "Camera":
        """Convenience constructor: camera at `eye` looking at `target` (world Z up)."""
        eye, target, up = (np.asarray(v, float) for v in (eye, target, up))
        z = target - eye
        z /= np.linalg.norm(z)
        x = np.cross(z, up)
        if np.linalg.norm(x) < 1e-9:            # looking straight along `up`
            x = np.cross(z, [0.0, 1.0, 0.0])
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        R = np.stack([x, y, z])
        W, H = size
        K = np.array([[fx, 0, W / 2 if cx is None else cx], [0, fy or fx, H / 2 if cy is None else cy], [0, 0, 1]])
        return cls(K, R, -R @ eye, (int(W), int(H)))

    def to_dict(self) -> dict:
        return {"image_size": list(self.size),
                "intrinsics": {"fx_px": float(self.K[0, 0]), "fy_px": float(self.K[1, 1]),
                               "cx_px": float(self.K[0, 2]), "cy_px": float(self.K[1, 2])},
                "extrinsics": {"R_world_to_cam": self.R.tolist(), "t": self.t.tolist()}}

    @property
    def center(self) -> np.ndarray:
        """Camera centre in world coordinates."""
        return -self.R.T @ self.t

    # ---------------------------------------------------------- projection
    def project(self, X, refract: tuple | None = None) -> np.ndarray:
        """World points (...,3) -> pixel coords (...,2) in the +0.5 pixel-centre convention.

        refract = (n, z_s): points below z_s are seen through a flat interface of index n.
        """
        X = np.asarray(X, float)
        if refract is not None:
            n, zs = refract
            S = surface_point(X, self.center, zs, n)
            X = np.where((X[..., 2] < zs)[..., None], S, X)
        Xc = X @ self.R.T + self.t
        uv = Xc @ self.K.T
        return uv[..., :2] / uv[..., 2:3]

    def depth(self, X) -> np.ndarray:
        return (np.asarray(X, float) @ self.R.T + self.t)[..., 2]

    def rays(self, uv) -> tuple[np.ndarray, np.ndarray]:
        """Pixel coords (...,2) -> (origin (3,), unit world directions (...,3))."""
        uv = np.asarray(uv, float)
        h = np.concatenate([uv, np.ones(uv.shape[:-1] + (1,))], -1)
        d = h @ np.linalg.inv(self.K).T @ self.R          # R^T applied on the right
        return self.center, d / np.linalg.norm(d, axis=-1, keepdims=True)

    def ray_plane(self, uv, z: float, refract: tuple | None = None) -> np.ndarray:
        """Back-project pixels onto the world plane Z = z (optionally through an interface (n, z_s))."""
        o, d = self.rays(uv)
        if refract is None:
            return intersect_z(o, d, z)
        n, zs = refract
        S = intersect_z(o, d, zs)
        if z >= zs:
            return intersect_z(o, d, z)
        d2 = snell(d, np.array([0.0, 0.0, 1.0]), 1.0 / n)
        return intersect_z(S, d2, z)


def intersect_z(o, d, z) -> np.ndarray:
    """Intersection of rays o + s d with the plane Z = z (NaN where parallel or behind)."""
    o, d = np.asarray(o, float), np.asarray(d, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        s = (z - o[..., 2]) / d[..., 2]
    s = np.where(s > 0, s, np.nan)
    return o + s[..., None] * d


def snell(d, normal, eta) -> np.ndarray:
    """Refract unit directions d at a surface with unit `normal` facing the incoming side; eta = n1/n2.

    Returns unit transmitted directions (NaN on total internal reflection).
    """
    d = np.asarray(d, float)
    nrm = np.broadcast_to(np.asarray(normal, float), d.shape)
    cos_i = -np.sum(d * nrm, -1, keepdims=True)
    k = 1 - eta ** 2 * (1 - cos_i ** 2)
    with np.errstate(invalid="ignore"):
        t = eta * d + (eta * cos_i - np.sqrt(k)) * nrm
    return np.where(k >= 0, t, np.nan)


def surface_point(X, C, zs, n, iters=60) -> np.ndarray:
    """For points X (...,3) below the interface z = zs, the interface point S where the refracted ray
    from X to the camera centre C crosses it (Fermat / Snell, solved by bisection on the radius)."""
    X, C = np.asarray(X, float), np.asarray(C, float)
    dx, dy = X[..., 0] - C[0], X[..., 1] - C[1]
    L = np.hypot(dx, dy)
    h, dep = C[2] - zs, zs - X[..., 2]
    lo, hi = np.zeros_like(L), L.copy()
    for _ in range(iters):      # f(r) = sin(theta_air) - n sin(theta_medium) increases with r
        r = 0.5 * (lo + hi)
        f = r / np.sqrt(r * r + h * h) - n * (L - r) / np.sqrt((L - r) ** 2 + dep * dep)
        hi, lo = np.where(f > 0, r, hi), np.where(f > 0, lo, r)
    r = 0.5 * (lo + hi)
    w = np.where(L > 0, r / np.maximum(L, 1e-300), 0.0)
    return np.stack([C[0] + w * dx, C[1] + w * dy, np.full_like(L, zs)], -1)


def blender_camera_params(cam: Camera, sensor_mm: float = 36.0) -> dict:
    """Blender camera settings reproducing `cam`: lens, sensor fit, shifts, pixel aspect, world matrix.

    Blender cameras look along -Z with +Y up; OpenCV looks along +Z with +Y down, hence the axis flip.
    Shifts are in units of the larger image dimension (Blender's convention).
    """
    W, H = cam.size
    fx, fy, cx, cy = cam.K[0, 0], cam.K[1, 1], cam.K[0, 2], cam.K[1, 2]
    horizontal = W >= H
    big = max(W, H)
    lens = (fx * sensor_mm / W) if horizontal else (fy * sensor_mm / H)
    R_c2w = cam.R.T @ np.diag([1.0, -1.0, -1.0])
    M = np.eye(4)
    M[:3, :3], M[:3, 3] = R_c2w, cam.center
    return {"lens_mm": float(lens), "sensor_mm": sensor_mm, "sensor_fit": "HORIZONTAL" if horizontal else "VERTICAL",
            "shift_x": float(-(cx - W / 2) / big), "shift_y": float((cy - H / 2) / big),
            "resolution": [int(W), int(H)],
            # Blender pixel aspect: x/y; fx > fy means pixels are taller than wide
            "pixel_aspect_x": float(fy / fx) if fx > fy else 1.0, "pixel_aspect_y": float(fx / fy) if fy > fx else 1.0,
            "matrix_world": M.tolist()}
