"""Stage 3: 3D disc pose per frame from the silhouette (measurement/cache/disc_edges.npy) through the per-frame camera.
Model [asm]: rigid circular disc, top-face radius 0.5 D (declared unit D = disc diameter = 1), thickness h (rim band,
one value for the clip, fitted on frames 0-23 then fixed), silhouette = outline of the union of the top-face circle
(centre P, unit normal n) and the bottom-face circle (centre P - h n).
Unknowns per frame: P (3, camera-centred world axes), tilt (2: n = Rx/Ry small rotation of +Z).
Residual: distance of each silhouette point to its projected circle polyline (upper arc -> top face, lower arc -> bottom face, sides -> nearer) (soft-L1, f_scale 0.7 px).
sigma(P, tilt) from the Gauss-Newton covariance scaled by the residual variance (formal, per frame).
Writes measurement/cache/disc_pose.npy and measurement/disc_pose_camcentred.csv."""
import numpy as np, os, sys, csv
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
sys.path.insert(0, os.path.dirname(__file__))
from common import *
import camfr
E = np.load(os.path.join(CACHE, "disc_edges.npy")); N = int(os.environ.get("NMAX", len(E)))
TH = np.linspace(0, 2 * np.pi, 1440, endpoint=False)
def normal(a, b):
    n = np.array([np.sin(b), -np.sin(a) * np.cos(b), np.cos(a) * np.cos(b)]); return n / np.linalg.norm(n)  # a about X, b about Y
def basis(n):
    ex = np.array([1.0, 0, 0]) - n[0] * n; ex /= np.linalg.norm(ex); return ex, np.cross(n, ex)
def circles(x, h, k):
    P = x[:3]; n = normal(x[3], x[4]); ex, ey = basis(n)
    ring = 0.5 * (np.cos(TH)[:, None] * ex + np.sin(TH)[:, None] * ey)
    return camfr.proj(P + ring, k), camfr.proj(P - h * n + ring, k)
def res(x, pts, h, k, cls):
    a, b = circles(x, h, k)
    da = cKDTree(a).query(pts)[0]; db = cKDTree(b).query(pts)[0]
    # lower arc -> bottom-face circle, upper arc -> top-face circle, sides (silhouette tangents) -> nearer one
    return np.where(cls == 1, db, np.where(cls == 2, da, np.minimum(da, db)))
def fit(n, h, x0):
    k = camfr.cam(n); ok = np.isfinite(E[n, :, 0]); pts = E[n][ok]; d = pts - pts.mean(0); sn = d[:, 1] / np.hypot(d[:, 0] / 2.8, d[:, 1])  # image-space elevation of the sample
    cls = np.where(sn > 0.34, 1, np.where(sn < -0.34, 2, 0))
    s = least_squares(res, x0, args=(pts, h, k, cls), loss="soft_l1", f_scale=0.7, x_scale=[0.01, 0.01, 0.01, 0.002, 0.002], diff_step=1e-4)
    r = s.fun; J = s.jac; dof = len(r) - 5; s2 = np.sum(r ** 2) / dof
    cov = np.linalg.pinv(J.T @ J) * s2
    return s.x, np.sqrt(np.diag(cov)), float(np.sqrt(np.mean(r ** 2))), float(np.median(r)), len(r)
def init(n):
    k = camfr.cam(n); pts = E[n][np.isfinite(E[n, :, 0])]; c = pts.mean(0)
    du = pts[:, 0].max() - pts[:, 0].min(); dist = k["f"] * 1.0 / du
    return np.r_[camfr.ray(c, k)[0] * dist, 0, 0]
if __name__ == "__main__":
    # thickness from frames 0-23 (joint grid search)
    x0 = init(0); best = None
    for h in np.arange(0.01, 0.061, 0.01):
        x = x0.copy(); tot = 0
        for n in range(0, 24, 4):
            x, sd, rms, med, m = fit(n, h, x if n else x0); tot += rms ** 2
        print("h %.3f rms %.3f" % (h, np.sqrt(tot / 6)))
        if best is None or tot < best[1]: best = (h, tot)
    H = best[0]
    hs = np.arange(max(0, H - 0.005), H + 0.0051, 0.001); tots = []
    for h in hs:
        tot = 0; x = x0.copy()
        for n in range(0, 24, 4): x, sd, rms, med, m = fit(n, h, x); tot += rms ** 2
        tots.append(tot)
    H = float(hs[int(np.argmin(tots))]); print("thickness h = %.4f D" % H)
    out = []; x = x0.copy()
    for n in range(N):
        x, sd, rms, med, m = fit(n, H, x)
        nn = normal(x[3], x[4])
        out.append(np.r_[n, n / FPS, x, sd, rms, m, nn, H])
        if n % 40 == 0: print(n, np.round(x, 5), "rms %.3f" % rms)
    out = np.array(out); np.save(os.path.join(CACHE, "disc_pose.npy"), out)
    hdr = ["frame", "t_s", "Px", "Py", "Pz", "tilt_a_rad", "tilt_b_rad", "sd_Px", "sd_Py", "sd_Pz", "sd_a", "sd_b", "sil_rms_px", "n_pts", "nx", "ny", "nz", "thickness_D"]
    with open(os.path.join(MEAS, "disc_pose_camcentred.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(hdr)
        for r in out: w.writerow([int(r[0])] + ["%.7g" % v for v in r[1:]])
