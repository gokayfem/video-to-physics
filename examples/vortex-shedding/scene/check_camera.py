"""Projection test: measure grid lines in the rendered first frame vs camera_ff.json.
usage: python scene/check_camera.py firstframe/frame0.png  -> updates config/camera_ff.json["projection_test_measured"]
Rows outside the channel: pure pinhole. Rows inside the channel: pinhole + flat water-surface refraction (Snell).
"""
import json, os, sys
import cv2, numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = json.load(open(os.path.join(ROOT, "config", "scene_params.json")))
CP = os.path.join(ROOT, "config", "camera_ff.json"); C = json.load(open(CP))
img = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE).astype(float)
K = C["intrinsics"]; f, cx, cy = K["fx_px"], K["cx_px"], K["cy_px"]; Zc = C["extrinsics"]["camera_position_world_m"][2]
n = P["water"]["ior"]; zs = P["water"]["surface_z"]

def u_pinhole(X): return cx + f * X / Zc
def v_pinhole(Y): return cy - f * Y / Zc
def r_refr(R):
    """radial image distance (px) of a plate point at radial distance R (m) seen through water surface z=zs."""
    lo, hi = 0.0, R
    for _ in range(60):  # find surface hit radius s
        s = 0.5 * (lo + hi); t1 = s / (Zc - zs); st = t1 / np.sqrt(1 + t1 * t1) / n
        Rh = s + zs * st / np.sqrt(1 - st * st)
        lo, hi = (s, hi) if Rh < R else (lo, s)
    return f * s / (Zc - zs)

def line_centres(profile, approx, half=6):
    out = []
    for a in approx:
        i0 = int(round(a)); w = profile[i0 - half:i0 + half + 1]
        d = np.clip(w.max() - w, 0, None)
        if d.sum() < 50: out.append(np.nan); continue
        out.append(i0 - half + (d * np.arange(len(w))).sum() / d.sum() + 0.5)  # pixel-centre convention
    return np.array(out)

Xs = np.round(np.arange(-0.12, 0.1201, 0.01), 4); res = {}
for name, Y_row, Y_line in [("outside_top", 0.0725, 0.07), ("outside_bottom", -0.0725, -0.07)]:
    v = int(v_pinhole(Y_row)); pred = u_pinhole(Xs)
    meas = line_centres(img[v - 1:v + 2].mean(0), pred)
    col_bands = [int((u_pinhole(x) + u_pinhole(x + 0.01)) / 2) for x in Xs[:-1]]
    vpred = v_pinhole(Y_line)
    vmeas = np.nanmean([line_centres(img[:, c - 2:c + 3].mean(1), [vpred])[0] for c in col_bands])
    du = meas - pred
    res[name] = {"row_Y_m": Y_row, "n_lines": int(np.isfinite(du).sum()), "du_px_mean": float(np.nanmean(du)),
                 "du_px_rms": float(np.sqrt(np.nanmean(du ** 2))), "du_px_maxabs": float(np.nanmax(np.abs(du))),
                 "h_line_Y_m": Y_line, "v_pred_px": float(vpred), "v_meas_px": float(vmeas), "dv_px": float(vmeas - vpred)}
# inside channel (through water), row y=+0.035 between dye and wall
Yr = 0.035; v = int(v_pinhole(Yr)); pin = u_pinhole(Xs)
# radial model: apparent radius r_refr(R) along direction (x, Yr)
refr = np.array([cx + r_refr(np.hypot(x, Yr)) * x / np.hypot(x, Yr) for x in Xs])
vr = np.array([cy - r_refr(np.hypot(x, Yr)) * Yr / np.hypot(x, Yr) for x in Xs])
meas = np.array([line_centres(img[int(round(vv)) - 1:int(round(vv)) + 2].mean(0), [u])[0] for vv, u in zip(vr, refr)])
ok = (Xs > -0.095) | (Xs < -0.125)   # skip rake-bar shadow/needle area
res["through_water_row_Y0.035"] = {
    "note": "skips x<=-0.095 (rake bar)", "du_vs_refraction_model_px_rms": float(np.sqrt(np.nanmean((meas - refr)[ok] ** 2))),
    "du_vs_pinhole_px_rms": float(np.sqrt(np.nanmean((meas - pin)[ok] ** 2))),
    "max_refraction_shift_px": float(np.nanmax(np.abs(refr - pin))),
    "per_line": [{"X": float(x), "u_pinhole": round(float(p), 2), "u_refr": round(float(r), 2), "u_meas": round(float(m), 2)}
                 for x, p, r, m in zip(Xs, pin, refr, meas) if np.isfinite(m)]}
C["projection_test_measured"] = res
json.dump(C, open(CP, "w"), indent=1)
for k, v in res.items():
    print(k, {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items() if kk != "per_line"})
