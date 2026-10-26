"""Calibration uncertainty: re-solve camera/calibrate.py under variants (edge subsets, priors loosened, free
principal point, jackknife over edges) and a focal-length profile (f fixed on a grid, everything else refitted).
Writes camera/out/calib_variants.json."""
import numpy as np, json, pathlib, sys, importlib
HERE = pathlib.Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import calibrate as CAL
from scipy.optimize import least_squares
obs_all = CAL.load_obs(); x_ref = np.load(HERE / "out/calib_x.npy"); base_use = list(CAL.USE)
def run(use=None, sig_override=None, prior_scale=1.0, free_pp=False, fix_f=None):
    importlib.reload(CAL); CAL.FREE_PP = free_pp
    if use is not None: CAL.USE = use
    if sig_override: CAL.SIG.update(sig_override)
    if prior_scale != 1.0:
        orig = CAL.residuals
        def res2(x, obs, detail=False):
            r = orig(x, obs, detail)
            if detail: return r
            r = r.copy(); r[-4:] /= prior_scale; return r
        CAL.residuals = res2
    x0 = x_ref.copy()
    if free_pp: x0 = np.r_[x0[0], 960, 544, x0[1:]]
    if fix_f is not None:
        f_ = fix_f; fun = lambda y: CAL.residuals(np.r_[f_, y], obs_all)
        s = least_squares(fun, np.r_[x0[1:]], x_scale="jac", max_nfev=3000); x = np.r_[f_, s.x]; cost = s.cost
    else:
        s = least_squares(CAL.residuals, x0, args=(obs_all,), x_scale="jac", max_nfev=3000); x = s.x; cost = s.cost
    r, det = CAL.residuals(x, obs_all, detail=True); f, cx, cy, rv, C, g = CAL.unpack(x)
    allp = np.concatenate([det[k] for k in CAL.USE])
    from scipy.spatial.transform import Rotation as Rot
    R = Rot.from_rotvec(rv).as_matrix(); fwd = R[2]
    return dict(f=float(f), cx=float(cx), cy=float(cy), C=[float(c) for c in C], dist_to_A=float(np.linalg.norm(C)), pitch_deg=float(np.degrees(np.arcsin(fwd[2]))),
                yaw_deg=float(np.degrees(np.arctan2(fwd[0], fwd[1]))), static_rms=float(np.sqrt(np.mean(allp ** 2))), cost=float(cost),
                wp=float(g["wp"]), dp=float(g["dp"]), plate_depth=float(g["by1"] - g["by0"]), zpt=float(g["zpt"]), bz1=float(g["bz1"]))
V = {}
V["reference"] = run()
V["free_pp"] = run(free_pp=True)
V["no_internal_edges"] = run(use=[k for k in base_use if k not in ("pL_front_inner", "pR_front_inner", "plate_front_top", "pL_bottom_front", "pR_bottom_front")])
V["no_plate_ends"] = run(use=[k for k in base_use if k not in ("plate_left_end", "plate_right_end")])
V["priors_x10_looser"] = run(prior_scale=10.0)
V["internal_sigma_x3"] = run(sig_override={"pL_front_inner": 4.5, "pR_front_inner": 3.6, "plate_front_top": 3.6})
for k in base_use:
    V["jack_" + k] = run(use=[e for e in base_use if e != k])
prof = {}
for f in [2600, 3000, 3300, 3556, 3800, 4200, 4800]:
    prof[f] = run(fix_f=float(f))
keys = [k for k in V if k != "free_pp"]
fs = np.array([V[k]["f"] for k in keys]); ds = np.array([V[k]["dist_to_A"] for k in keys])
summ = dict(f_sd=float(fs.std()), f_min=float(fs.min()), f_max=float(fs.max()), dist_sd=float(ds.std()), dist_min=float(ds.min()), dist_max=float(ds.max()),
            pitch_range=[float(min(V[k]["pitch_deg"] for k in keys)), float(max(V[k]["pitch_deg"] for k in keys))])
json.dump(dict(variants=V, f_profile={str(k): v for k, v in prof.items()}, summary=summ), open(HERE / "out/calib_variants.json", "w"), indent=1)
for k, v in V.items(): print("%-28s f %7.1f dist %.3f pitch %.2f yaw %.2f rms %.3f pp (%.0f,%.0f)" % (k, v["f"], v["dist_to_A"], v["pitch_deg"], v["yaw_deg"], v["static_rms"], v["cx"], v["cy"]))
for k, v in prof.items(): print("f=%d  rms %.3f cost %.1f dist %.3f" % (k, v["static_rms"], v["cost"], v["dist_to_A"]))
print(summ)
