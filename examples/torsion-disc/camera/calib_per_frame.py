"""Per-frame vanishing-point calibration (clip 03: the AI camera orbits). For every frame n: f_n and world->camera rotation
R_n from the tracked frame edges (measurement/cache/edges_track.json; same model as calibrate.py: principal point fixed at
the image centre, each edge passes through the vanishing point of its class direction; edges with < 10 points or
rms > 1.5 px are excluded in that frame). Init from the previous frame. Writes camera/out/calib_per_frame.json."""
import numpy as np, json, pathlib, sys
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "camera"))
import calibrate as CAL
T = json.load(open(ROOT / "measurement/cache/edges_track.json"))
FIXF = [float(a.split("=")[1]) for a in sys.argv if a.startswith("--fix-f=")]
TAG = "_fixf" if FIXF else ""
if FIXF:   # constant focal length (no zoom): optimise rotation + offsets only
    _res = CAL.residuals
    def res_fixed(x, obs, detail=False): return _res(np.r_[FIXF[0], x], obs, detail)
x0 = None; out = []
for n, rec in enumerate(T):
    obs = {}
    for k, v in rec.items():
        P = np.array(v["pts"]) if len(v["pts"]) else np.zeros((0, 2))
        if v["n_pts"] < 10 or not np.isfinite(v["rms"]) or v["rms"] > 1.5: continue
        obs[k] = dict(P=P, cls=k[0], m=P.mean(0), n=np.array(v["n"]), sig=max(0.3, v["rms"]))
    keys = list(obs)
    if x0 is None: s = CAL.solve(obs)
    elif not FIXF: s = least_squares(CAL.residuals, np.r_[x0[:4], np.zeros(len(keys))], args=(obs,), method="trf", x_scale="jac", max_nfev=3000)
    if FIXF:
        s = least_squares(res_fixed, np.r_[(s.x if x0 is None else x0)[1:4], np.zeros(len(keys))], args=(obs,), method="trf", x_scale="jac", max_nfev=3000)
        s.x = np.r_[FIXF[0], s.x]; s.jac = np.c_[np.ones(len(s.fun)) * 1e-9, s.jac]
    r, det = CAL.residuals(s.x, obs, detail=True); f, cx, cy, rv, off = CAL.unpack(s.x, keys); R = Rot.from_rotvec(rv).as_matrix()
    J = s.jac; dof = max(1, len(s.fun) - len(s.x)); s2 = 2 * s.cost / dof; cov = np.linalg.pinv(J.T @ J) * s2; sd = np.sqrt(np.diag(cov))
    p, y, ro = CAL.angles(R); allp = np.concatenate([det[k] for k in keys])
    out.append(dict(frame=n, f_px=float(f), f_sd=float(sd[0]), rotvec=rv.tolist(), R=R.tolist(), pitch=p, yaw=y, roll=ro, rot_sd_deg=np.degrees(sd[1:4]).tolist(),
                    edge_rms=float(np.sqrt(np.mean(allp ** 2))), n_edges=len(keys), edges=keys, offsets={k: float(off[k]) for k in keys}))
    x0 = s.x
    if n % 20 == 0: print(n, "f %.1f (sd %.1f) pitch %.2f yaw %.2f roll %.3f rms %.3f edges %d" % (f, sd[0], p, y, ro, out[-1]["edge_rms"], len(keys)))
json.dump(out, open(ROOT / f"camera/out/calib_per_frame{TAG}.json", "w"), indent=0)
