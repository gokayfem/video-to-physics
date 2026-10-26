"""Per-frame camera position (clip 03; the AI camera orbits) by a small bundle adjustment.
Fixed: per-frame rotation R_n from the vanishing-point calibration (camera/out/calib_per_frame_fixf.json),
scale gauge |C_0| from camera/out/locate.json (disc gauge D = 1). Unknown: one constant f (start --f; --fix-f keeps it), C_n for n = 0..240 and the 3D positions of 7 static
edges that keep their physical identity through the orbit (crossbar front-top / front-bottom edges left and right of the clamp,
the left post front-outer, front-inner and inner-back corners; directions = world X or Z).
Observations per frame: endpoints of each tracked edge segment (measurement/cache/edges_track.json; sigma 0.3 px; distance to the
projected model line) and the wire-top pixel (measurement/wire_track.csv, sigma 0.5 px) which must be the image of A = origin.
Writes camera/out/camera_poses.json (per-frame R, C, f; residuals) and camera/out/bundle_lines.json."""
import numpy as np, json, csv, pathlib, sys
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
ROOT = pathlib.Path(__file__).resolve().parents[1]
F_PX = float([a.split("=")[1] for a in sys.argv if a.startswith("--f=")][0]) if any(a.startswith("--f=") for a in sys.argv) else 2660.0
CP = json.load(open(ROOT / "camera/out/calib_per_frame_fixf.json")); LO = json.load(open(ROOT / "camera/out/locate.json"))
T = json.load(open(ROOT / "measurement/cache/edges_track.json")); WT = list(csv.DictReader(open(ROOT / "measurement/wire_track.csv")))
N = len(T); R = [np.array(r["R"]) for r in CP]; C0 = np.array(LO["C0_world"]); cx, cy = 960.0, 544.0
STABLE = ["xbar_top_a", "xbar_top_b", "xbar_bot_a", "xbar_bot_b", "zL_outer", "zL_frontinner", "zL_back"]
# zR_front / zR_outer are excluded: the camera crosses the planes of the right post side faces, so those silhouettes change physical corner
DIR = {k: (np.r_[1.0, 0, 0] if k.startswith("x") else np.r_[0, 0, 1.0]) for k in STABLE}
FREE = {k: ([1, 2] if k.startswith("x") else [0, 1]) for k in STABLE}   # coordinates of the line's point that are unknown
L0 = {k: np.array(LO["lines_init_world"][k]) for k in STABLE}
obs = []   # (frame, kind, key, data)
for n in range(N):
    for k in STABLE:
        v = T[n][k]
        if v["n_pts"] < 10 or not np.isfinite(v["rms"]) or v["rms"] > 1.5: continue
        obs.append((n, "edge", k, np.array(v["ends"])))
    obs.append((n, "A", None, np.array([float(WT[n]["u_top_px"]), float(WT[n]["v_top_px"])])))
NL = 2 * len(STABLE); NP = 3 * N + NL + 1   # C_0..C_240, lines, f
FREE_F = "--fix-f" not in sys.argv; D0 = float(np.linalg.norm(C0))
def unpack(x):
    C = x[:3 * N].reshape(-1, 3); L = {}
    for i, k in enumerate(STABLE):
        p = L0[k].copy(); p[FREE[k]] = x[3 * N + 2 * i: 3 * N + 2 * i + 2]; L[k] = p
    return C, L
FF = [F_PX]
def proj(X, n, C):
    Xc = (np.atleast_2d(X) - C[n]) @ R[n].T; return np.c_[cx + FF[0] * Xc[:, 0] / Xc[:, 2], cy + FF[0] * Xc[:, 1] / Xc[:, 2]]
def residuals(x):
    C, L = unpack(x); FF[0] = x[-1] if FREE_F else F_PX; out = []
    for n, kind, k, d in obs:
        if kind == "edge":
            q = proj(np.array([L[k] - 0.5 * DIR[k], L[k] + 0.5 * DIR[k]]), n, C); e = q[1] - q[0]; nr = np.array([-e[1], e[0]]) / np.linalg.norm(e)
            out.extend(((d - q[0]) @ nr) / 0.3)
        else:
            out.extend((proj(np.zeros(3), n, C)[0] - d) / 0.5)
    out.append((np.linalg.norm(C[0]) - D0) / 1e-4)   # scale gauge: frame-0 distance to A from the disc (locate.py)
    return np.array(out)
S = []; row = 0
for n, kind, k, d in obs:
    m = 2
    cols = list(range(3 * n, 3 * n + 3)) + [NP - 1]
    if kind == "edge": i = STABLE.index(k); cols += [3 * N + 2 * i, 3 * N + 2 * i + 1]
    S.append((row, m, cols)); row += m
J = lil_matrix((row + 1, NP), dtype=int)
for r0, m, cols in S:
    for c in cols: J[r0:r0 + m, c] = 1
J[row, 0:3] = 1
x0 = np.r_[np.tile(C0, N), np.concatenate([L0[k][FREE[k]] for k in STABLE]), F_PX]
sol = least_squares(residuals, x0, jac_sparsity=J, x_scale="jac", method="trf", max_nfev=200, verbose=0)
C, L = unpack(sol.x); r = sol.fun[:-1]; F_PX = float(sol.x[-1]) if FREE_F else F_PX
JJ = sol.jac.toarray() if hasattr(sol.jac, "toarray") else sol.jac; s2 = 2 * sol.cost / (len(sol.fun) - NP)
try:
    cov_f = float(np.linalg.pinv(JJ.T @ JJ)[-1, -1] * s2)
except Exception: cov_f = float("nan")
# per-frame residual summaries (px)
per = []; i = 0
for (n, kind, k, d), (r0, m, cols) in zip(obs, S):
    per.append((n, kind, k, r[r0:r0 + m] * (0.3 if kind == "edge" else 0.5)))
fr = []
for n in range(N):
    e = np.concatenate([p[3] for p in per if p[0] == n and p[1] == "edge"]); a = [p[3] for p in per if p[0] == n and p[1] == "A"][0]
    fr.append(dict(frame=n, R=R[n].tolist(), C=C[n].tolist(), edge_endpoint_rms_px=float(np.sqrt(np.mean(e ** 2))), A_resid_px=a.tolist(),
                   dist_to_A=float(np.linalg.norm(C[n]))))
dist = np.array([f["dist_to_A"] for f in fr]); az = np.degrees(np.arctan2(C[:, 0], -C[:, 1])); el = np.degrees(np.arctan2(C[:, 2], np.hypot(C[:, 0], C[:, 1])))
out = dict(f_px=F_PX, f_sd_px=float(np.sqrt(cov_f)) if FREE_F else None, f_free=FREE_F, cx=cx, cy=cy, unit="D", frames=fr, summary=dict(cost=float(sol.cost), chi2nu=float(s2), n_obs=int(row), all_rms_px=float(np.sqrt(np.mean((r * 0.3) ** 2))),
           dist_to_A_range=[float(dist.min()), float(dist.max())], azimuth_deg_range=[float(az.min()), float(az.max())], elevation_deg_range=[float(el.min()), float(el.max())],
           azimuth_deg_frame0_240=[float(az[0]), float(az[-1])], dist_frame0_240=[float(dist[0]), float(dist[-1])]))
json.dump(out, open(ROOT / "camera/out/camera_poses.json", "w"), indent=0)
json.dump({k: dict(point=L[k].tolist(), direction=DIR[k].tolist()) for k in STABLE}, open(ROOT / "camera/out/bundle_lines.json", "w"), indent=1)
print(json.dumps(out["summary"], indent=1)); print({k: np.round(L[k], 4).tolist() for k in STABLE})
for n in range(0, N, 40): print(n, np.round(C[n], 4), "az %.2f el %.2f dist %.4f rms %.3f A %s" % (az[n], el[n], dist[n], fr[n]["edge_endpoint_rms_px"], np.round(fr[n]["A_resid_px"], 2)))
