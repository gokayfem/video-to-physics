"""Stand proportions with the camera FIXED to the camera of record (params/camera.json via camera/project.py).

Only geometry is fitted (no camera, no theta, no bob): the same parametric box model and the same hand/auto
landmarks as fit_proportions.py, per frame projected with project.load_camera(frame=n) (includes measured drift).
Lengths are fitted in metres and reported as ratios to L_eff_m (camera.json scale gauge, = physics L_c at the time
camera.json was written). Pivot and bob landmarks are NOT fitted; their residuals are reported as diagnostics.

Run from ROOT:  .venv/bin/python scene/measure/fit_fixedcam.py [--out scene/measure/fit_fixedcam.json]
"""
import json, argparse, sys, pathlib, numpy as np
from scipy.optimize import least_squares
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "camera"))
import project as PJ  # camera role's projection utility (camera of record)

ap = argparse.ArgumentParser()
ap.add_argument("--frames", nargs="+", default=["scene/measure/landmarks_f000.json", "scene/measure/landmarks_f120.json"])
ap.add_argument("--out", default="scene/measure/fit_fixedcam.json")
ap.add_argument("--joint", default="scene/measure/fit_joint.json", help="previous joint fit, for comparison and r_bob provenance")
a = ap.parse_args()
FR = [json.load(open(f)) for f in a.frames]
CAMJ = json.load(open(ROOT / "params/camera.json"))
LE = CAMJ["scale"]["L_eff_m"]
CAMS = {fr["frame"]: PJ.load_camera(frame=fr["frame"]) for fr in FR}

GEO = ["arm_h", "arm_d", "z_pin", "x_ae", "x_tip", "post_px", "post_py", "x_p", "y_p", "z_pt",
       "W", "D", "T", "h", "x_b", "y_b"]
SKIP = {"P_wire_exit", "bob_centre", "bob_radius_px", "base_top_BL"}   # BL: hidden corner, superseded by the edge lines
LINES = {"base_back_top": ("base_top_BL", "base_top_BR"), "base_left_top": ("base_top_FL", "base_top_BL")}

def world_points(g):
    zt, zb = g["z_pin"] + g["arm_h"] / 2, g["z_pin"] - g["arm_h"] / 2
    fa, ba = -g["arm_d"] / 2, g["arm_d"] / 2
    xpl, xpr = g["x_p"] - g["post_px"] / 2, g["x_p"] + g["post_px"] / 2
    ypf, ypb = g["y_p"] - g["post_py"] / 2, g["y_p"] + g["post_py"] / 2
    xbl, xbr = g["x_b"] - g["W"] / 2, g["x_b"] + g["W"] / 2
    ybf, ybb = g["y_b"] - g["D"] / 2, g["y_b"] + g["D"] / 2
    ht, hb = -g["h"], -g["h"] - g["T"]
    return {"pin_tip": (g["x_tip"], 0, g["z_pin"]),
        "arm_end_front_top": (g["x_ae"], fa, zt), "arm_end_front_bot": (g["x_ae"], fa, zb),
        "arm_end_back_top": (g["x_ae"], ba, zt), "arm_end_back_bot": (g["x_ae"], ba, zb),
        "arm_post_front_top": (xpr, fa, zt), "arm_post_front_bot": (xpr, fa, zb),
        "post_top_FL": (xpl, ypf, g["z_pt"]), "post_top_FR": (xpr, ypf, g["z_pt"]), "post_top_BR": (xpr, ypb, g["z_pt"]),
        "post_bot_FL": (xpl, ypf, ht), "post_bot_FR": (xpr, ypf, ht), "post_bot_BR": (xpr, ypb, ht),
        "base_top_FL": (xbl, ybf, ht), "base_top_BR": (xbr, ybb, ht), "base_top_FR": (xbr, ybf, ht),
        "base_top_BL": (xbl, ybb, ht), "base_bot_FL": (xbl, ybf, hb), "base_bot_BR": (xbr, ybb, hb)}

def proj(X, cam):
    u, v = PJ.project(np.array(X, float) * LE, cam)      # model in L units -> metres
    return np.array([float(u), float(v)])

def resid(q, detail=False):
    g = dict(zip(GEO, q)); wp = world_points(g); r = []; det = {}
    for fr in FR:
        cam = CAMS[fr["frame"]]
        for k, o in fr["points"].items():
            if k in SKIP: continue
            uv = proj(wp[k], cam); e = uv - np.array(o["uv"])
            r += list(e / o["sigma_px"]); det[f"{fr['frame']}:{k}"] = [round(uv[0], 1), round(uv[1], 1), o["uv"]]
        for ln, (ka, kb) in LINES.items():
            if ln not in fr.get("lines", {}): continue
            A, B = proj(wp[ka], cam), proj(wp[kb], cam)
            nrm = np.array([-(B - A)[1], (B - A)[0]]); nrm /= np.linalg.norm(nrm)
            o = fr["lines"][ln]; dd = [(np.array(p) - A) @ nrm for p in o["pts"]]
            r += [x / o["sigma_px"] for x in dd]; det[f"{fr['frame']}:{ln}"] = {"n": len(dd), "rms_px": float(np.sqrt(np.mean(np.square(dd))))}
    return (np.array(r), det) if detail else np.array(r)

J0 = json.load(open(a.joint))["geometry"]
q0 = np.array([J0[n]["value"] for n in GEO])
sol = least_squares(resid, q0, method="lm", x_scale="jac", max_nfev=20000)
res, det = resid(sol.x, True)
dof = len(res) - len(q0); s2 = (res ** 2).sum() / dof
cov = np.linalg.pinv(sol.jac.T @ sol.jac) * max(s2, 1.0); sd = np.sqrt(np.diag(cov))
# per-landmark pixel residuals
px = []
for k, v in det.items():
    if isinstance(v, list): px.append(np.hypot(v[0] - v[2][0], v[1] - v[2][1]))
# diagnostics: pivot and bob through the fixed camera
diag = {}
for fr in FR:
    cam = CAMS[fr["frame"]]; n = fr["frame"]
    uP = proj((0, 0, 0), cam); diag[f"{n}:P_ring_residual_px"] = (uP - np.array(fr["points"]["P_wire_exit"]["uv"])).round(2).tolist()
    # bob centre: intersect its pixel ray with the swing plane y=0 -> distance from P_ring in L units
    ub, vb = fr["points"]["bob_centre"]["uv"]
    B = PJ.intersect_plane(np.array(ub), np.array(vb), cam, (0, 1, 0), 0.0) / LE
    zc = PJ.depth(B * LE, cam); rpx = fr["points"]["bob_radius_px"]["value"]
    rb = rpx * zc / cam["f_px"] / LE
    diag[f"{n}:bob_centre_on_y0_L"] = np.round(B, 4).tolist()
    diag[f"{n}:|bob-P_ring|_L"] = round(float(np.linalg.norm(B)), 4)
    diag[f"{n}:bob_radius_L"] = round(float(rb), 4)
    diag[f"{n}:r_over_|bob-P_ring|"] = round(float(rb / np.linalg.norm(B)), 4)
out = {"units": "lengths / L_eff (camera.json gauge; L_eff_m = %.6f m)" % LE,
       "camera": "FIXED: params/camera.json via camera/project.py load_camera(frame=n), sha256 of camera.json in provenance",
       "frames": [fr["frame"] for fr in FR], "rms_normalised": float(np.sqrt((res ** 2).mean())),
       "pixel_residual_rms_px_points": float(np.sqrt(np.mean(np.square(px)))), "pixel_residual_max_px_points": float(np.max(px)),
       "n_obs": len(res), "n_par": len(q0),
       "geometry": {n: {"value": float(v), "sd": float(s)} for n, v, s in zip(GEO, sol.x, sd)},
       "change_vs_joint_fit": {n: {"joint": J0[n]["value"], "fixedcam": float(v), "delta": float(v - J0[n]["value"])} for n, v in zip(GEO, sol.x)},
       "diagnostics_not_fitted": diag, "reprojection_model_vs_obs": det}
out["geometry"]["r_bob"] = dict(J0["r_bob"], note="kept from the joint fit (provenance of the L_c/r = 12.30 ratio, r over |P_ring-bob|); see diagnostics for the fixed-camera cross-check")
json.dump(out, open(a.out, "w"), indent=1)
for n in GEO: print(f"{n:8s} joint {J0[n]['value']: .4f} -> fixed {out['geometry'][n]['value']: .4f} ± {out['geometry'][n]['sd']:.4f}")
print("rms_norm", out["rms_normalised"], "px rms", out["pixel_residual_rms_px_points"], "max", out["pixel_residual_max_px_points"])
print(json.dumps(diag, indent=0)); print({k: v for k, v in det.items() if isinstance(v, dict)})
