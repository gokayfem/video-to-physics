"""Per-frame camera for the measurement scripts (clip 03). The AI camera orbits: camera of record = per-frame pose
(camera/out/camera_poses.json: R_n, C_n world (origin A, unit D), constant f) written by camera/pose_bundle.py.
Before that file exists (bootstrap for camera/locate.py), cam(0) returns the frame-0 rotation with C = 0
(camera-centred world axes)."""
import numpy as np, json, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[2]
_P = ROOT / "camera/out/camera_poses.json"
if _P.exists():
    _J = json.load(open(_P)); POSES = [dict(R=np.array(r["R"]), C=np.array(r["C"]), f=_J["f_px"], cx=_J["cx"], cy=_J["cy"]) for r in _J["frames"]]; MODE = "world"
else:
    _C = json.load(open(ROOT / "camera/out/calib_per_frame_fixf.json"))
    POSES = [dict(R=np.array(r["R"]), C=np.zeros(3), f=r["f_px"], cx=960.0, cy=544.0) for r in _C]; MODE = "camcentred"
def cam(n): return POSES[n]
def proj(X, k):
    Xc = (np.asarray(X) - k["C"]) @ k["R"].T
    return np.stack([k["cx"] + k["f"] * Xc[..., 0] / Xc[..., 2], k["cy"] + k["f"] * Xc[..., 1] / Xc[..., 2]], -1)
def ray(uv, k):
    uv = np.atleast_2d(uv); d = np.c_[(uv[:, 0] - k["cx"]) / k["f"], (uv[:, 1] - k["cy"]) / k["f"], np.ones(len(uv))] @ k["R"]
    return d / np.linalg.norm(d, axis=1, keepdims=True)
