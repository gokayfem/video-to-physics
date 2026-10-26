"""Joint fit of stand proportions + one approximate pinhole camera per frame to hand-read landmarks.

Geometry-role tool, NOT the calibrated camera (camera role owns params/camera.json). Purpose: turn
perspective-distorted pixel landmarks into ratio-to-L_c proportions. Lengths are in units of L_c (L_c = 1).
Shared geometry across frames; each frame has its own camera (the reference camera drifts/dollies) and theta.
Principal point fixed at image centre, square pixels, no distortion.

Run (from ROOT):  .venv/bin/python scene/measure/fit_proportions.py \
    --frames scene/measure/landmarks_f000.json scene/measure/landmarks_f120.json --out scene/measure/fit_joint.json
"""
import json, argparse, numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot

ap = argparse.ArgumentParser()
ap.add_argument("--frames", nargs="+", default=["scene/measure/landmarks_f000.json", "scene/measure/landmarks_f120.json"])
ap.add_argument("--out", default="scene/measure/fit_joint.json")
a = ap.parse_args()
FR = [json.load(open(f)) for f in a.frames]
W_IMG, H_IMG = 1920, 1088
cx0, cy0 = W_IMG / 2, H_IMG / 2

GEO = ["r_bob", "arm_h", "arm_d", "z_pin", "x_ae", "x_tip", "post_px", "post_py", "x_p", "y_p", "z_pt",
       "W", "D", "T", "h", "x_b", "y_b"]
CAM = ["cx", "cy", "cz", "r0", "r1", "r2", "logf", "theta"]
names = GEO + [f"{c}@{fr['frame']}" for fr in FR for c in CAM]

def world_points(g, th):
    zt, zb = g["z_pin"] + g["arm_h"] / 2, g["z_pin"] - g["arm_h"] / 2
    fa, ba = -g["arm_d"] / 2, g["arm_d"] / 2
    xpl, xpr = g["x_p"] - g["post_px"] / 2, g["x_p"] + g["post_px"] / 2
    ypf, ypb = g["y_p"] - g["post_py"] / 2, g["y_p"] + g["post_py"] / 2
    xbl, xbr = g["x_b"] - g["W"] / 2, g["x_b"] + g["W"] / 2
    ybf, ybb = g["y_b"] - g["D"] / 2, g["y_b"] + g["D"] / 2
    ht, hb = -g["h"], -g["h"] - g["T"]
    return {
        "P_wire_exit": (0, 0, 0), "bob_centre": (np.sin(th), 0, -np.cos(th)), "pin_tip": (g["x_tip"], 0, g["z_pin"]),
        "arm_end_front_top": (g["x_ae"], fa, zt), "arm_end_front_bot": (g["x_ae"], fa, zb),
        "arm_end_back_top": (g["x_ae"], ba, zt), "arm_end_back_bot": (g["x_ae"], ba, zb),
        "arm_post_front_top": (xpr, fa, zt), "arm_post_front_bot": (xpr, fa, zb),
        "post_top_FL": (xpl, ypf, g["z_pt"]), "post_top_FR": (xpr, ypf, g["z_pt"]), "post_top_BR": (xpr, ypb, g["z_pt"]),
        "post_bot_FL": (xpl, ypf, ht), "post_bot_FR": (xpr, ypf, ht), "post_bot_BR": (xpr, ypb, ht),
        "base_top_FL": (xbl, ybf, ht), "base_top_BR": (xbr, ybb, ht), "base_top_FR": (xbr, ybf, ht),
        "base_top_BL": (xbl, ybb, ht), "base_bot_FL": (xbl, ybf, hb), "base_bot_BR": (xbr, ybb, hb)}

# line observations: image points that must lie on the projected (infinite) line through two model points
LINES = {"base_back_top": ("base_top_BL", "base_top_BR"), "base_left_top": ("base_top_FL", "base_top_BL")}


def project(c, X):
    C = np.array([c["cx"], c["cy"], c["cz"]])
    R = Rot.from_rotvec([c["r0"], c["r1"], c["r2"]]).as_matrix()   # camera->world; camera looks -Z, up +Y
    Xc = (np.asarray(X, float) - C) @ R
    f = np.exp(c["logf"])
    return np.array([cx0 + f * Xc[0] / -Xc[2], cy0 - f * Xc[1] / -Xc[2]]), Xc

def split(q):
    g = dict(zip(GEO, q[:len(GEO)]))
    cams = [dict(zip(CAM, q[len(GEO) + i * len(CAM): len(GEO) + (i + 1) * len(CAM)])) for i in range(len(FR))]
    return g, cams

def resid(q, detail=False):
    g, cams = split(q)
    r, det = [], {}
    for fr, c in zip(FR, cams):
        wp = world_points(g, c["theta"])
        for k, o in fr["points"].items():
            if k == "bob_radius_px":
                _, Xc = project(c, wp["bob_centre"])
                v = np.exp(c["logf"]) * g["r_bob"] / -Xc[2]
                r.append((v - o["value"]) / o["sigma_px"]); det[f"{fr['frame']}:{k}"] = [round(float(v), 2), o["value"]]
                continue
            uv, _ = project(c, wp[k])
            r += list((uv - np.array(o["uv"])) / o["sigma_px"])
            det[f"{fr['frame']}:{k}"] = [round(float(uv[0]), 1), round(float(uv[1]), 1), o["uv"]]
        for ln, (ka, kb) in LINES.items():
            if ln not in fr.get("lines", {}):
                continue
            A, _ = project(c, wp[ka]); B, _ = project(c, wp[kb])
            nrm = np.array([-(B - A)[1], (B - A)[0]]); nrm /= np.linalg.norm(nrm)
            o = fr["lines"][ln]
            dd = [(np.array(q) - A) @ nrm for q in o["pts"]]
            r += [x / o["sigma_px"] for x in dd]
            det[f"{fr['frame']}:{ln}"] = {"n": len(dd), "rms_px": float(np.sqrt(np.mean(np.square(dd))))}
    return (np.array(r), det) if detail else np.array(r)

def look_rot(C, T, up=(0, 0, 1)):
    fwd = np.array(T) - C; fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, up); right /= np.linalg.norm(right)
    return Rot.from_matrix(np.c_[right, np.cross(right, fwd), -fwd]).as_rotvec()

C0, T0 = np.array([2.3, -3.4, -0.6]), np.array([-0.1, 0.0, -0.5])
rv = look_rot(C0, T0)
g0 = dict(r_bob=0.08, arm_h=0.12, arm_d=0.1, z_pin=0.03, x_ae=-0.07, x_tip=0.06, post_px=0.17, post_py=0.15,
          x_p=-0.6, y_p=0.0, z_pt=0.13, W=1.3, D=0.6, T=0.17, h=1.12, x_b=-0.15, y_b=-0.1)
q0 = [g0[n] for n in GEO]
for fr in FR:
    q0 += [*C0, *rv, np.log(2900.0), 0.0]
q0 = np.array(q0)
sol = least_squares(resid, q0, method="lm", max_nfev=50000, x_scale="jac")
res, det = resid(sol.x, True)
dof = max(1, len(res) - len(q0)); s2 = (res ** 2).sum() / dof
J = sol.jac
cov = np.linalg.pinv(J.T @ J) * max(s2, 1.0)    # never shrink below the stated pixel sigmas
sd = np.sqrt(np.diag(cov))
g, cams = split(sol.x)
out = {"units": "lengths / L_c (L_c := 1)", "frames": [fr["frame"] for fr in FR],
       "rms_normalised": float(np.sqrt((res ** 2).mean())), "n_obs": len(res), "n_par": len(q0),
       "geometry": {n: {"value": float(sol.x[i]), "sd": float(sd[i])} for i, n in enumerate(GEO)},
       "cameras": {}, "reprojection_model_vs_obs": det}
for i, (fr, c) in enumerate(zip(FR, cams)):
    f = float(np.exp(c["logf"]))
    out["cameras"][str(fr["frame"])] = {
        "focal_px": f, "focal_mm_36mm_sensor_width": f * 36 / W_IMG,
        "position_over_Lc": [c["cx"], c["cy"], c["cz"]],
        "euler_xyz_deg_blender": list(np.degrees(Rot.from_rotvec([c["r0"], c["r1"], c["r2"]]).as_euler("xyz"))),
        "theta_deg": float(np.degrees(c["theta"]))}
json.dump(out, open(a.out, "w"), indent=1)
for i, n in enumerate(names):
    print(f"{n:10s} {sol.x[i]: .4f} ± {sd[i]:.4f}")
print("rms(normalised)", out["rms_normalised"], "obs", len(res), "par", len(q0))
for k, v in det.items(): print(k, v)
print(json.dumps(out["cameras"], indent=1))
