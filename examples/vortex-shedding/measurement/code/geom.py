"""Camera + flat-surface Snell refraction: world point (below z_s) -> pixel. Pinhole from config/camera_ff.json."""
import json, numpy as np, os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
cam = json.load(open(os.path.join(ROOT, 'config/camera_ff.json')))
K = np.array(cam['intrinsics']['K'], float)
Rm = np.array(cam['extrinsics']['R_world_to_cv_camera'], float)
tv = np.array(cam['extrinsics']['t_world_to_cv_camera'], float)
C = -Rm.T @ tv            # camera centre (world)
ZS, NW = 0.023, 1.333

def pinhole(X):
    X = np.asarray(X, float); Xc = X @ Rm.T + tv
    uv = Xc @ K.T
    return uv[..., :2] / uv[..., 2:3]

def surface_point(X, zs=ZS, n=NW):
    """X (...,3) below the surface -> surface intersection point S of the refracted ray to the camera."""
    X = np.asarray(X, float)
    dx = X[..., 0] - C[0]; dy = X[..., 1] - C[1]; L = np.hypot(dx, dy)
    h = C[2] - zs; d = zs - X[..., 2]
    lo = np.zeros_like(L); hi = L.copy()
    for _ in range(60):
        r = 0.5 * (lo + hi)
        s1 = r / np.sqrt(r * r + h * h)
        s2 = (L - r) / np.sqrt((L - r) ** 2 + d * d)
        f = s1 - n * s2         # increasing in r
        big = f > 0
        hi = np.where(big, r, hi); lo = np.where(big, lo, r)
    r = 0.5 * (lo + hi); w = np.where(L > 0, r / np.maximum(L, 1e-12), 0)
    return np.stack([C[0] + w * dx, C[1] + w * dy, np.full_like(L, zs)], -1)

def project(X, refract=True):
    X = np.asarray(X, float)
    if not refract: return pinhole(X)
    S = surface_point(X)
    under = X[..., 2] < ZS
    return np.where(under[..., None], pinhole(S), pinhole(X))

# rectification grid of the dye plane
ZD = 0.013; RES = 0.00025
XR = (-0.30, 0.30); YR = (-0.06, 0.06)
def rect_maps():
    xs = XR[0] + RES * (np.arange(int(round((XR[1]-XR[0])/RES))) + 0.5)
    ys = YR[1] - RES * (np.arange(int(round((YR[1]-YR[0])/RES))) + 0.5)   # row 0 = +y
    Xg, Yg = np.meshgrid(xs, ys)
    P = np.stack([Xg, Yg, np.full_like(Xg, ZD)], -1)
    uv = project(P) - 0.5        # pixel-centre convention -> array index
    return xs, ys, uv[..., 0].astype(np.float32), uv[..., 1].astype(np.float32)
