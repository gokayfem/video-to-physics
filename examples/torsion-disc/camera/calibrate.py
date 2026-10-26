"""Static camera calibration of reference frame 0 (clip 03, torsion disc): focal length + orientation from
three orthogonal vanishing directions of the rectilinear walnut frame (posts = world Z, crossbar = world X,
feet = world Y), measured as sub-pixel edges by camera/edges.py.

Model: pinhole, square pixels, principal point fixed at the image centre (960,544) [asm; '--free-pp' variant],
no distortion. Unknowns: f_px, world->camera rotation (rotvec), and one nuisance offset per edge line.
Each image edge must pass through the vanishing point K R d of its class direction d.
Residual = point distance to that line / sigma (sigma = max(0.3, own straight-line rms) px).
Translation is NOT determined here (it needs a length): camera/locate.py places the camera with the disc
(declared unit D = disc diameter = 1) and origin A (wire leaves the clamp).
Writes camera/out/calib_result[ _freepp].json"""
import numpy as np, json, pathlib, sys
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot
ROOT = pathlib.Path(__file__).resolve().parents[1]
W_, H_ = 1920, 1088
FREE_PP = "--free-pp" in sys.argv
DROP = [a.split("=")[1] for a in sys.argv if a.startswith("--drop=")]
DIRS = {"z": np.r_[0, 0, 1.0], "x": np.r_[1.0, 0, 0], "y": np.r_[0, 1.0, 0]}
def load_obs(path=ROOT / "camera/out/edges_f000.json"):
    E = json.load(open(path)); out = {}
    for k, v in E.items():
        if k in DROP or len(v["points"]) < 5: continue
        P = np.array(v["points"]); out[k] = dict(P=P, cls=k[0], m=P.mean(0), n=np.array(v["line_n"]), sig=max(0.3, v["rms_px"]))
    return out
def unpack(x, keys):
    f = x[0]; i = 1
    if FREE_PP: cx, cy = x[1:3]; i = 3
    else: cx, cy = W_ / 2, H_ / 2
    rv = x[i:i + 3]; s = x[i + 3:]
    return f, cx, cy, rv, dict(zip(keys, s))
def residuals(x, obs, detail=False):
    keys = list(obs); f, cx, cy, rv, s = unpack(x, keys); K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]]); R = Rot.from_rotvec(rv).as_matrix()
    res = []; det = {}
    for k in keys:
        o = obs[k]; vp = K @ R @ DIRS[o["cls"]]; m = np.r_[o["m"] + s[k] * o["n"], 1.0]
        l = np.cross(vp, m); l = l / np.linalg.norm(l[:2]); d = o["P"] @ l[:2] + l[2]
        det[k] = d; res.append(d / o["sig"])
    out = np.concatenate(res)
    return (out, det) if detail else out
def solve(obs):
    best = None
    for f0 in [1500., 2500., 4000.]:
        for pitch in [-10., -20., -30.]:
            for yaw in [-15., 0., 15.]:
                R0 = Rot.from_euler("xz", [pitch, 0], degrees=True)  # placeholder, built below
                # camera looking along +Y world, tilted down by pitch, panned by yaw
                Rb = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0.]])     # world->cam for a level camera looking +Y
                Rw = Rot.from_euler("zx", [yaw, pitch], degrees=True).as_matrix()  # cam->world extra rotation
                R = Rb @ Rw.T
                x0 = np.r_[f0, ([960, 544] if FREE_PP else []), Rot.from_matrix(R).as_rotvec(), np.zeros(len(obs))]
                s = least_squares(residuals, x0, args=(obs,), method="trf", x_scale="jac", max_nfev=3000)
                if best is None or s.cost < best.cost: best = s
    return best
def angles(R):
    fwd = R[2]  # camera z axis in world
    pitch = np.degrees(np.arcsin(fwd[2])); yaw = np.degrees(np.arctan2(fwd[0], fwd[1]))
    roll = np.degrees(np.arctan2(-R[0, 2], -R[1, 2]))  # angle of world-up projected in the image (0 = upright)
    return pitch, yaw, roll
if __name__ == "__main__":
    obs = load_obs(); s = solve(obs); keys = list(obs)
    r, det = residuals(s.x, obs, detail=True); f, cx, cy, rv, off = unpack(s.x, keys); R = Rot.from_rotvec(rv).as_matrix()
    J = s.jac; dof = max(1, len(s.fun) - len(s.x)); s2 = 2 * s.cost / dof; cov = np.linalg.pinv(J.T @ J) * s2; sd = np.sqrt(np.diag(cov))
    per = {k: dict(rms=float(np.sqrt(np.mean(det[k] ** 2))), n=int(len(det[k])), sigma=obs[k]["sig"]) for k in keys}
    for k in keys: print("  %-14s rms %.3f px (n=%d, sigma %.2f)" % (k, per[k]["rms"], per[k]["n"], obs[k]["sig"]))
    p, y, ro = angles(R); allp = np.concatenate([det[k] for k in keys])
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
    vps = {c: (K @ R @ d).tolist() for c, d in DIRS.items()}
    out = dict(f_px=float(f), f_sd_px=float(sd[0]), cx=float(cx), cy=float(cy), rotvec=rv.tolist(), R_world_to_cam=R.tolist(), view_pitch_deg=p,
               view_yaw_deg_from_Y=y, roll_deg=ro, edge_rms_px=float(np.sqrt(np.mean(allp ** 2))), chi2nu=float(s2), per_edge=per, vanishing_points_h=vps,
               free_pp=FREE_PP, dropped=DROP)
    print("f %.1f (sd %.1f) pp (%.1f,%.1f) pitch %.2f yaw %.2f roll %.3f  edge rms %.3f px chi2nu %.2f" % (f, sd[0], cx, cy, p, y, ro, out["edge_rms_px"], s2))
    tag = ("_freepp" if FREE_PP else "") + ("_drop_" + "_".join(DROP) if DROP else "")
    json.dump(out, open(ROOT / f"camera/out/calib_result{tag}.json", "w"), indent=1)
