"""Static camera calibration of reference frame 0 (clip 02, vertical spring) from rectilinear frame edges.

World frame (config/coordinates.md): origin A = spring top attachment (hook bearing on the crossbar top face,
crossbar mid-depth), X along the crossbar to image-right, Y horizontal away from the camera, Z up.
LENGTH UNIT (declared gauge): W_in = crossbar inner span = distance between the inner faces of the two posts = 1.
Model (all boxes axis-aligned, sharp edges):
  posts   x in [xc-0.5-wp, xc-0.5] and [xc+0.5, xc+0.5+wp], y in [yp0, yp0+dp], z in [bz1, zpt]
  crossbar x in [xc-0.5, xc+0.5], y in [-dc/2, dc/2], z in [zc0, 0]      (A = (0,0,0) on its top face)
  plate   x in [bx0, bx1], y in [by0, by1], z in [bz0, bz1]
Camera: pinhole, square pixels, principal point fixed at the image centre (960,544) (a free pp is weakly
determined from a near-frontal view; reported as a variant), focal f_px, world->camera rotation, centre C.
Observations: sub-pixel edge points from camera/edges.py (frame 0); residual = point distance to the image of
the model 3D edge line (silhouette edges chosen from the camera position where ambiguous); hook image u
(the hook hangs at A: u_A = hook apex u, sigma 1 px).
Soft priors (assumptions, stated): posts centred in Y on the crossbar (sigma 0.02), crossbar depth 0.08 +- 0.04
(its depth is barely observable), post cross-section square
(sigma 15 %), plate symmetric about the frame centre in X (sigma 0.02).
Writes camera/out/calib_result.json."""
import numpy as np, json, pathlib, sys
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot
ROOT = pathlib.Path(__file__).resolve().parents[1]
W_, H_ = 1920, 1088
FREE_PP = "--free-pp" in sys.argv
EDGEFILE = ROOT / "camera/out/edges_f000.json"
HOOK_U = 956.8; SIG_HOOK = 1.0
SIG = {"pL_front_inner": 1.5, "pR_front_inner": 1.2, "plate_front_top": 1.2}
GEO = ["xc", "wp", "dp", "yp0", "zpt", "dc", "zc0", "bx0", "bx1", "by0", "by1", "bz1", "bz0"]
USE = ["pL_front_outer", "pL_front_inner", "pL_back_inner", "pR_front_outer", "pR_front_inner", "pR_back_inner", "pL_top_front", "pR_top_front",
       "pL_bottom_front", "pR_bottom_front", "bar_top_a", "bar_top_b", "bar_bot_a", "bar_bot_b", "plate_back_a", "plate_back_b",
       "plate_front_top", "plate_front_bot", "plate_left_end", "plate_right_end"]
X, Y, Z = np.eye(3)
def edges3d(g, C):
    xL0, xL1, xR0, xR1 = g["xc"] - 0.5 - g["wp"], g["xc"] - 0.5, g["xc"] + 0.5, g["xc"] + 0.5 + g["wp"]
    yp0, yp1 = g["yp0"], g["yp0"] + g["dp"]
    below_bar = C[2] < g["zc0"]; ybb = abs(g["dc"]) / 2 if below_bar else -abs(g["dc"]) / 2
    below_top = C[2] < g["zpt"]; ypt = yp0 if below_top else yp1
    ybt = -abs(g["dc"]) / 2 if C[2] < 0 else abs(g["dc"]) / 2   # bar top silhouette: front edge if camera below the top face
    return {
        "pL_front_outer": (np.r_[xL0, yp0, 0], Z), "pL_front_inner": (np.r_[xL1, yp0, 0], Z), "pL_back_inner": (np.r_[xL1, yp1, 0], Z),
        "pR_front_outer": (np.r_[xR1, yp0, 0], Z), "pR_front_inner": (np.r_[xR0, yp0, 0], Z), "pR_back_inner": (np.r_[xR0, yp1, 0], Z),
        "pL_top_front": (np.r_[0, ypt, g["zpt"]], X), "pR_top_front": (np.r_[0, ypt, g["zpt"]], X),
        "pL_bottom_front": (np.r_[0, yp0, g["bz1"]], X), "pR_bottom_front": (np.r_[0, yp0, g["bz1"]], X),
        "bar_top_a": (np.r_[0, ybt, 0], X), "bar_top_b": (np.r_[0, ybt, 0], X),
        "bar_bot_a": (np.r_[0, ybb, g["zc0"]], X), "bar_bot_b": (np.r_[0, ybb, g["zc0"]], X),
        "plate_back_a": (np.r_[0, g["by1"], g["bz1"]], X), "plate_back_b": (np.r_[0, g["by1"], g["bz1"]], X),
        "plate_front_top": (np.r_[0, g["by0"], g["bz1"]], X), "plate_front_bot": (np.r_[0, g["by0"], g["bz0"]], X),
        "plate_left_end": (np.r_[g["bx0"], 0, g["bz1"]], Y), "plate_right_end": (np.r_[g["bx1"], 0, g["bz1"]], Y)}
def unpack(x):
    i = 0; f = x[i]; i += 1
    if FREE_PP: cx, cy = x[i:i + 2]; i += 2
    else: cx, cy = W_ / 2, H_ / 2
    rv = x[i:i + 3]; i += 3; C = x[i:i + 3]; i += 3
    g = dict(zip(GEO, x[i:i + len(GEO)]))
    return f, cx, cy, rv, C, g
def proj(P, f, cx, cy, R, C):
    Xc = (np.atleast_2d(P) - C) @ R.T; return np.c_[f * Xc[:, 0] / Xc[:, 2] + cx, f * Xc[:, 1] / Xc[:, 2] + cy]
def image_line(A, d, f, cx, cy, R, C):
    q = proj(np.array([A, A + d]), f, cx, cy, R, C); e = q[1] - q[0]; nrm = np.array([-e[1], e[0]]) / np.linalg.norm(e)
    return nrm, -nrm @ q[0]
def residuals(x, obs, detail=False):
    f, cx, cy, rv, C, g = unpack(x); R = Rot.from_rotvec(rv).as_matrix(); E = edges3d(g, C); res = []; det = {}
    for k in USE:
        A, d = E[k]; nrm, c = image_line(A, d, f, cx, cy, R, C); dist = obs[k] @ nrm + c
        det[k] = dist; res.append(dist / SIG.get(k, 0.5))
    uA = proj(np.zeros(3), f, cx, cy, R, C)[0]; det["hook_u"] = uA[0] - HOOK_U; res.append(np.atleast_1d(det["hook_u"] / SIG_HOOK))
    pri = [(g["yp0"] + g["dp"] / 2) / 0.02, (abs(g["dc"]) - 0.08) / 0.04, (g["dp"] - g["wp"]) / (0.15 * g["wp"]), ((g["bx0"] + g["bx1"]) / 2 - g["xc"]) / 0.02]
    res.append(np.array(pri))
    out = np.concatenate([np.atleast_1d(r) for r in res])
    return (out, det) if detail else out
def load_obs():
    E = json.load(open(EDGEFILE)); return {k: np.array(E[k]["points"]) for k in USE}
def initial(f=2500.0, dist=3.6, zc=-0.7):
    R = np.array([X, -Z, Y])  # camera x = world X, camera y = -Z, camera z = +Y
    C = np.array([0.0, -dist, zc])
    g = dict(xc=0.0, wp=0.18, dp=0.18, yp0=-0.09, zpt=0.01, dc=0.06, zc0=-0.08, bx0=-0.7, bx1=0.7, by0=-0.3, by1=0.3, bz1=-1.38, bz0=-1.41)
    return np.array([f] + ([960, 544] if FREE_PP else []) + list(Rot.from_matrix(R).as_rotvec()) + list(C) + [g[k] for k in GEO], float)
def solve(obs, x0=None):
    best = None
    for f0 in ([2500.0] if x0 is not None else [1800.0, 2500.0, 3500.0]):
        for zc in ([None] if x0 is not None else [-0.4, -0.7, -1.0]):
            xi = x0 if x0 is not None else initial(f0, f0 / 690.0, zc)
            s = least_squares(residuals, xi, args=(obs,), method="trf", x_scale="jac", max_nfev=2000)
            if best is None or s.cost < best.cost: best = s
    return best
def report(s, obs):
    r, det = residuals(s.x, obs, detail=True); f, cx, cy, rv, C, g = unpack(s.x); R = Rot.from_rotvec(rv).as_matrix()
    allp = np.concatenate([det[k] for k in USE]); per = {k: dict(rms=float(np.sqrt(np.mean(det[k] ** 2))), mean=float(det[k].mean()), n=len(det[k])) for k in USE}
    for k in USE: print("  %-16s rms %.3f mean %+.3f n=%d" % (k, per[k]["rms"], per[k]["mean"], per[k]["n"]))
    J = s.jac; dof = max(1, len(s.fun) - len(s.x)); s2 = 2 * s.cost / dof; cov = np.linalg.pinv(J.T @ J) * s2
    sd = np.sqrt(np.diag(cov))
    fwd = R[2]; pitch = np.degrees(np.arcsin(fwd[2])); yaw = np.degrees(np.arctan2(fwd[0], fwd[1])); roll = np.degrees(np.arctan2(-R[0, 2], R[1, 2] * -1)) if False else np.degrees(np.arctan2(R[0][2], -R[1][2]))
    out = dict(f_px=f, f_sd_px=float(sd[0]), cx=cx, cy=cy, rotvec=list(rv), R_world_to_cam=R.tolist(), C=list(C), C_sd=list(sd[(3 if FREE_PP else 1) + 3:(3 if FREE_PP else 1) + 6]),
               geometry=g, geometry_sd=dict(zip(GEO, sd[-len(GEO):])), static_rms_px=float(np.sqrt(np.mean(allp ** 2))), n_points=int(len(allp)),
               per_edge=per, hook_u_resid_px=float(det["hook_u"]), chi2nu=float(s2), view_pitch_deg=float(pitch), view_yaw_deg_from_Y=float(yaw),
               camera_below_crossbar_bottom=bool(C[2] < g["zc0"]), camera_below_post_top=bool(C[2] < g["zpt"]))
    print("  f %.1f (sd %.1f) pp (%.1f,%.1f) C %s pitch %.2f yaw %.2f  static RMS %.3f px (n=%d) chi2nu %.2f" % (f, sd[0], cx, cy, np.round(C, 4), pitch, yaw, out["static_rms_px"], len(allp), s2))
    print("  geo", {k: round(float(v), 4) for k, v in g.items()})
    return out
def clean(o):
    if isinstance(o, dict): return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, np.ndarray)): return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)): return float(o)
    if isinstance(o, (np.bool_,)): return bool(o)
    return o
if __name__ == "__main__":
    obs = load_obs(); s = solve(obs); out = report(s, obs)
    tag = "_freepp" if FREE_PP else ""
    json.dump(clean(out), open(ROOT / f"camera/out/calib_result{tag}.json", "w"), indent=1); np.save(ROOT / f"camera/out/calib_x{tag}.npy", s.x)
