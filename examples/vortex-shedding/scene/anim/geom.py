"""Camera (config/camera_ff.json) projection helpers: pinhole + flat-surface Snell refraction (NOTES.md model, 3-D)."""
import json, os, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
C = json.load(open(os.path.join(ROOT, "config", "camera_ff.json")))
P = json.load(open(os.path.join(ROOT, "config", "scene_params.json")))
K = np.array(C["intrinsics"]["K"]); R = np.array(C["extrinsics"]["R_world_to_cv_camera"]); t = np.array(C["extrinsics"]["t_world_to_cv_camera"])
CAM = np.array(C["extrinsics"]["camera_position_world_m"])
ZS, N = P["water"]["surface_z"], P["water"]["ior"]

def proj(Xw, K=K, R=R, t=t):
    Xw = np.atleast_2d(np.asarray(Xw, float)); pc = Xw @ R.T + t
    return np.stack([K[0, 0] * pc[:, 0] / pc[:, 2] + K[0, 2], K[1, 1] * pc[:, 1] / pc[:, 2] + K[1, 2]], 1)

def surface_hit(Xw):
    """point on the flat water surface through which the sub-surface point Xw is seen (Snell, n=N); vectorised bisection."""
    p = np.atleast_2d(np.asarray(Xw, float)); dh = CAM[None, :2] - p[:, :2]; d = np.linalg.norm(dh, axis=1); e = dh / d[:, None]
    h1 = CAM[2] - ZS; h2 = ZS - p[:, 2]; lo = np.zeros_like(d); hi = d.copy()
    for _ in range(60):
        s = 0.5 * (lo + hi)   # horizontal distance from p to the surface hit
        f = (d - s) / np.hypot(d - s, h1) - N * s / np.hypot(s, h2)
        lo = np.where(f > 0, s, lo); hi = np.where(f > 0, hi, s)
    s = 0.5 * (lo + hi)
    return np.stack([p[:, 0] + e[:, 0] * s, p[:, 1] + e[:, 1] * s, np.full_like(s, ZS)], 1)

def proj_refr(Xw):
    return proj(surface_hit(Xw))

# solver dye-image map (CONTRACT): x in [-0.30,0.30], y in [-0.06,0.06], 0.25 mm/px, row 0 = +y, col 0 = -x
DX = 0.00025; NXI, NYI = 2400, 480; X0, Y1 = -0.30, 0.06
def px_to_world(col, row):  # pixel centres
    return X0 + (np.asarray(col) + 0.5) * DX, Y1 - (np.asarray(row) + 0.5) * DX
def world_to_px(x, y):
    return (np.asarray(x) - X0) / DX - 0.5, (Y1 - np.asarray(y)) / DX - 0.5
