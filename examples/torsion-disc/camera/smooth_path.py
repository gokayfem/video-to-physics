"""Smooth the per-frame camera path (clip 03, VR03-04). Input camera/out/camera_poses_raw.json (bundle output, per-frame pose).
Path parameters: camera-centre azimuth / elevation / distance about A, and the rotation relative to frame 0 as a rotation vector
(3 small angles: carries pitch/yaw/roll); f fixed. Each parameter is smoothed with a cubic penalised smoothing spline
(scipy make_smoothing_spline) with a common time-scale; the time-scale is chosen from a grid by the data: minimise the rms
difference between the reference's own static-scene image motion (phase correlation of fixed windows on the right post, left
post and crossbar, frame to frame; same windows as visual_review/scripts/camjitter.py) and the camera's projected motion,
subject to R4 (static edge reprojection, all frames) <= 2 px; the least smoothing within 0.002 px of the best match is taken.
Writes camera/out/camera_poses.json (smoothed, camera of record), camera/out/camera_smoothing.json, measurement/cache/ref_window_motion.npy."""
import numpy as np, json, pathlib, sys, cv2, os
from scipy.interpolate import make_smoothing_spline
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "measurement/scripts"))
from common import load_frames, luma, CACHE
RAW = json.load(open(ROOT / "camera/out/camera_poses_raw.json")); N = len(RAW["frames"]); t = np.arange(N) / 24.0
f, cx, cy = RAW["f_px"], RAW["cx"], RAW["cy"]
Rr = np.array([fr["R"] for fr in RAW["frames"]]); Cr = np.array([fr["C"] for fr in RAW["frames"]])
def rv_of(R):
    c = np.clip((np.trace(R) - 1) / 2, -1, 1); th = np.arccos(c)
    if th < 1e-12: return np.zeros(3)
    return np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * np.sin(th)) * th
def R_of(v):
    th = np.linalg.norm(v)
    if th < 1e-15: return np.eye(3)
    k = v / th; K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]]); return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K
az = np.arctan2(Cr[:, 0], -Cr[:, 1]); dist = np.linalg.norm(Cr, axis=1); el = np.arcsin(Cr[:, 2] / dist)
dR = np.array([rv_of(Rr[n] @ Rr[0].T) for n in range(N)])
P = np.c_[az, el, dist, dR]   # 6 path parameters
def build(Ps):
    a, e, d = Ps[:, 0], Ps[:, 1], Ps[:, 2]
    C = np.c_[d * np.cos(e) * np.sin(a), -d * np.cos(e) * np.cos(a), d * np.sin(e)]
    R = np.array([R_of(Ps[n, 3:]) @ Rr[0] for n in range(N)]); return C, R
def proj(X, R, C): Xc = (np.atleast_2d(X) - C) @ R.T; return np.c_[cx + f * Xc[:, 0] / Xc[:, 2], cy + f * Xc[:, 1] / Xc[:, 2]]
WIN = {"right_post": ((0.926, -0.109, -0.6), 64, 128), "left_post": ((-0.926, -0.109, -0.6), 64, 128), "crossbar_left": ((-0.45, -0.109, 0.145), 128, 48)}
# reference image motion of the windows (positions from the raw camera; phase correlation frame n -> n+1)
p_ref = CACHE + "/ref_window_motion.npy"
if os.path.exists(p_ref): REF = np.load(p_ref, allow_pickle=True).item()
else:
    F = load_frames(); REF = {}
    for nm, (X, hw, hh) in WIN.items():
        win = cv2.createHanningWindow((2 * hw, 2 * hh), cv2.CV_32F); d = []
        for n in range(N - 1):
            u, v = proj(np.array(X), Rr[n], Cr[n])[0]; a = luma(F[n])[int(v - hh):int(v + hh), int(u - hw):int(u + hw)]; b = luma(F[n + 1])[int(v - hh):int(v + hh), int(u - hw):int(u + hw)]
            (dx, dy), _ = cv2.phaseCorrelate(a, b, win); d.append((dx, dy))
        REF[nm] = np.array(d)
    np.save(p_ref, REF)
T = json.load(open(ROOT / "measurement/cache/edges_track.json")); BL = json.load(open(ROOT / "camera/out/bundle_lines.json"))
import csv; WT = list(csv.DictReader(open(ROOT / "measurement/wire_track.csv")))
def r4(C, R):
    out = []
    for n in range(N):
        allr = []
        for k, L in BL.items():
            v = T[n][k]
            if v["n_pts"] < 10 or v["rms"] > 1.5: continue
            X0 = np.array(L["point"]); d = np.array(L["direction"]); q = proj(np.array([X0 - 0.8 * d, X0 + 0.8 * d]), R[n], C[n]); e = q[1] - q[0]; nr = np.array([-e[1], e[0]]) / np.linalg.norm(e)
            Pp = np.array(v["pts"]); allr.append(Pp @ nr - nr @ q[0])
        allr = np.concatenate(allr); out.append(np.sqrt(np.mean(allr ** 2)))
    return np.array(out)
def metrics(C, R):
    m = {}
    for nm, (X, hw, hh) in WIN.items():
        pr = np.array([proj(np.array(X), R[n], C[n])[0] for n in range(N)]); vel = np.diff(pr, axis=0)
        jc = np.hypot(*np.diff(vel, axis=0).T); jr = np.hypot(*np.diff(REF[nm], axis=0).T)
        m[nm] = dict(cam_jerk_median=float(np.median(jc)), cam_jerk_p95=float(np.percentile(jc, 95)), cam_jerk_max=float(jc.max()), cam_jerk_max_frame=int(1 + jc.argmax()),
                     ref_jerk_median=float(np.median(jr)), ref_jerk_p95=float(np.percentile(jr, 95)), ref_jerk_max=float(jr.max()),
                     vel_rms_vs_ref=float(np.sqrt(np.mean(np.sum((vel - REF[nm]) ** 2, 1)))))
    return m
def smooth(tau):
    if tau == 0: return P.copy()
    lam = tau ** 3   # penalty ~ (time-scale)^3 for a cubic smoothing spline (t in s)
    return np.c_[[make_smoothing_spline(t, P[:, i], lam=lam)(t) for i in range(6)]].T
grid = [0, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.2, 2.0, 3.0, 5.0]; rows = []
for tau in grid:
    C, R = build(smooth(tau)); m = metrics(C, R); rr = r4(C, R)
    score = np.mean([m[k]["vel_rms_vs_ref"] for k in m]); rows.append(dict(tau_s=tau, mean_vel_rms_vs_ref_px=float(score), r4_median=float(np.median(rr)), r4_max=float(rr.max()), metrics=m))
    print("tau %.2f  vel-rms vs ref %.3f px  right-post jerk med %.3f max %.2f (ref %.3f/%.2f)  R4 med %.3f max %.3f" % (tau, score, m["right_post"]["cam_jerk_median"], m["right_post"]["cam_jerk_max"], m["right_post"]["ref_jerk_median"], m["right_post"]["ref_jerk_max"], np.median(rr), rr.max()))
ok = [r for r in rows if r["r4_max"] <= 2.0 and r["tau_s"] > 0]; vmin = min(r["mean_vel_rms_vs_ref_px"] for r in ok)
# plateau: the least smoothing within 0.002 px of the best match (avoid over-smoothing the orbit)
best = min([r for r in ok if r["mean_vel_rms_vs_ref_px"] <= vmin + 0.002], key=lambda r: r["tau_s"]); tau = best["tau_s"]
Ps = smooth(tau); C, R = build(Ps); rr = r4(C, R)
dev = {"azimuth_deg": float(np.degrees(np.abs(Ps[:, 0] - P[:, 0]).max())), "elevation_deg": float(np.degrees(np.abs(Ps[:, 1] - P[:, 1]).max())), "distance_D": float(np.abs(Ps[:, 2] - P[:, 2]).max()),
       "rotation_deg": float(np.degrees(np.linalg.norm(Ps[:, 3:] - P[:, 3:], axis=1).max()))}
fr = []
for n in range(N):
    a = proj(np.zeros(3), R[n], C[n])[0] - [float(WT[n]["u_top_px"]), float(WT[n]["v_top_px"])]
    fr.append(dict(frame=n, R=R[n].tolist(), C=C[n].tolist(), edge_endpoint_rms_px=float(rr[n]), A_resid_px=a.tolist(), dist_to_A=float(np.linalg.norm(C[n]))))
out = dict(RAW); out["frames"] = fr; out["smoothing"] = dict(method="cubic penalised smoothing spline per path parameter (azimuth, elevation, distance, rotation vector rel. frame 0), lam = tau^3", tau_s=tau,
           selection="least smoothing whose rms(reference window motion - projected camera motion) is within 0.002 px of the grid minimum, subject to R4 max <= 2 px", max_deviation_from_raw=dev)
out["summary"] = dict(RAW["summary"], smoothed=True, r4_median_px=float(np.median(rr)), r4_max_px=float(rr.max()))
json.dump(out, open(ROOT / "camera/out/camera_poses.json", "w"), indent=0)
json.dump(dict(grid=rows, chosen_tau_s=tau, chosen=best, raw=rows[0], max_deviation_from_raw=dev, r4_per_frame_smoothed=rr.tolist(), r4_per_frame_raw=r4(*build(P)).tolist()), open(ROOT / "camera/out/camera_smoothing.json", "w"), indent=1)
print("chosen tau", tau, "dev", dev)
