"""Stage 5: assemble deliverable CSVs, image-space oscillation analysis, summary.json, plots."""
import cv2, numpy as np, os, sys, csv, json, pickle, collections, hashlib, datetime
sys.path.insert(0, os.path.dirname(__file__))
from common import *
from scipy.optimize import least_squares
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
N = 241; t = np.arange(N) / FPS
H = np.load(os.path.join(CACHE, "H_ecc.npy")); Hch = np.load(os.path.join(CACHE, "H_chain.npy"))
dd = json.load(open(os.path.join(CACHE, "drift_diag.json")))
B = np.load(os.path.join(CACHE, "bob_raw.npz")); W = pickle.load(open(os.path.join(CACHE, "wire_rows.pkl"), "rb"))
cA, cB, radA, covA, ellA = B["cA"], B["cB"], B["radA"], B["covA"], B["ellA"]
LMD = collections.defaultdict(list)
for r in csv.DictReader(open(os.path.join(MEAS, "static_landmarks.csv"))): LMD[r["landmark"]].append(r)
col = lambda name, k: np.array([float(r[k]) for r in LMD[name]])
def apply(Hm, p):
    q = Hm @ np.array([p[0], p[1], 1.0]); return q[:2] / q[2]
# ------------------------------------------------------------------ bob uncertainty
off = np.median(cB[:, :2] - cA, axis=0)
dAB = cB[:, :2] - off - cA
def local_rms(x, h=3):
    return np.array([np.sqrt(np.mean(x[max(0, i - h):i + h + 1] ** 2)) for i in range(len(x))])
sig_fit = np.sqrt(np.stack([covA[:, 0, 0], covA[:, 1, 1]], 1))
sig_ab = np.stack([local_rms(dAB[:, 0]), local_rms(dAB[:, 1])], 1) / np.sqrt(2)
sig = np.sqrt(sig_fit ** 2 + sig_ab ** 2)
# empirical jitter: residual of a local cubic (9-frame window) -- includes true high-frequency non-smoothness
from scipy.signal import savgol_filter
jit = np.array([np.std(cA[:, k] - savgol_filter(cA[:, k], 9, 3)) for k in range(2)])
conf = np.clip(1.0 - np.hypot(*sig.T) / 3.0, 0, 1)
# stabilized (frame-0 geometry) bob
cS = np.array([apply(np.linalg.inv(H[n]), cA[n]) for n in range(N)])
with open(os.path.join(MEAS, "bob_track.csv"), "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["frame", "t_s", "u_px", "v_px", "sigma_u_px", "sigma_v_px", "radius_px", "method", "confidence",
                "u_methodB_px", "v_methodB_px", "zncc_methodB", "scale_methodB", "sigma_fit_u_px", "sigma_fit_v_px",
                "ellipse_major_px", "ellipse_minor_px", "ellipse_angle_deg", "u_stab_frame0_px", "v_stab_frame0_px"])
    for n in range(N):
        el = ellA[n]; ax = sorted([el[2], el[3]])
        w.writerow([n, "%.6f" % t[n], "%.3f" % cA[n, 0], "%.3f" % cA[n, 1], "%.3f" % sig[n, 0], "%.3f" % sig[n, 1], "%.3f" % radA[n],
                    "radial_edge_circle_fit(A); B=annulus_zncc_multiscale offset-corrected", "%.3f" % conf[n],
                    "%.3f" % (cB[n, 0] - off[0]), "%.3f" % (cB[n, 1] - off[1]), "%.4f" % cB[n, 2], "%.4f" % cB[n, 3],
                    "%.3f" % sig_fit[n, 0], "%.3f" % sig_fit[n, 1], "%.2f" % ax[1], "%.2f" % ax[0], "%.1f" % el[4],
                    "%.3f" % cS[n, 0], "%.3f" % cS[n, 1]])
# ------------------------------------------------------------------ camera drift
def sim_params(Hm):
    # least-squares similarity approximating Hm over the image grid
    xs, ys = np.meshgrid(np.linspace(0, 1920, 9), np.linspace(0, 1088, 6)); P = np.stack([xs.ravel(), ys.ravel()], 1)
    Q = np.array([apply(Hm, p) for p in P])
    A = []; b = []
    for (x, y), (qx, qy) in zip(P, Q):
        A += [[x, -y, 1, 0], [y, x, 0, 1]]; b += [qx, qy]
    a, bb, tx, ty = np.linalg.lstsq(np.array(A), np.array(b), rcond=None)[0]
    Qs = np.stack([a * P[:, 0] - bb * P[:, 1] + tx, bb * P[:, 0] + a * P[:, 1] + ty], 1)
    return np.hypot(a, bb), np.degrees(np.arctan2(bb, a)), tx, ty, np.abs(Qs - Q).max()
piv_drift = np.array([apply(H[n], (1086.7, 125.0)) for n in range(N)])
with open(os.path.join(MEAS, "camera_drift.csv"), "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["frame", "t_s"] + ["H%d%d" % (i, j) for i in range(3) for j in range(3)] +
               ["sim_scale", "sim_rot_deg", "sim_tx_px", "sim_ty_px", "sim_vs_H_maxdev_px", "ecc_cc",
                "disp_u_at_pivot_px", "disp_v_at_pivot_px", "disp_u_at_bob_rest_px", "disp_v_at_bob_rest_px",
                "disp_u_at_base_front_left_px", "disp_v_at_base_front_left_px", "chain_vs_ecc_maxdiff_px", "landmark_resid_rms_px"])
    for n in range(N):
        s, r, tx, ty, md = sim_params(H[n])
        p1 = apply(H[n], (1086.7, 125.0)) - (1086.7, 125.0); p2 = apply(H[n], (1090.0, 835.0)) - (1090.0, 835.0)
        p3 = apply(H[n], (502.0, 903.0)) - (502.0, 903.0)
        cd = max(np.abs(apply(H[n], p) - apply(Hch[n], p)).max() for p in [(0, 0), (1920, 0), (0, 1088), (1920, 1088), (960, 544)])
        lr = np.sqrt(np.mean([float(LMD[k][n]["resid_u_after_drift_px"]) ** 2 + float(LMD[k][n]["resid_v_after_drift_px"]) ** 2
                              for k in LMD if k not in ("wire_departure", "wire_loop_ring", "pin_shaft_mid")]))
        w.writerow([n, "%.6f" % t[n]] + ["%.10g" % x for x in H[n].ravel()] +
                   ["%.6f" % s, "%.4f" % r, "%.3f" % tx, "%.3f" % ty, "%.3f" % md, "%.5f" % dd["ecc_cc"][n],
                    "%.3f" % p1[0], "%.3f" % p1[1], "%.3f" % p2[0], "%.3f" % p2[1], "%.3f" % p3[0], "%.3f" % p3[1], "%.3f" % cd, "%.3f" % lr])
simp = np.array([sim_params(H[n]) for n in range(N)])
# ------------------------------------------------------------------ pivot per frame (tracked wire departure point)
pu, pvv = col("wire_departure", "u_px"), col("wire_departure", "v_px")
pres = np.hypot(col("wire_departure", "resid_u_after_drift_px"), col("wire_departure", "resid_v_after_drift_px"))
sig_piv = np.full(N, max(0.5, float(np.std(col("wire_departure", "resid_u_after_drift_px")))))
# alternative pivot estimate: least-squares intersection of the UPPER parts of the wire lines (stabilized coordinates)
D = pickle.load(open(os.path.join(CACHE, "wire_diag.pkl"), "rb"))
Arows = []; brows = []
upper_lines = []
for n in range(N):
    vv, uu, _ = D[n]
    m = vv < 330  # upper ~200 rows of wire
    if m.sum() < 60: upper_lines.append(None); continue
    pts = np.array([apply(np.linalg.inv(H[n]), (a, b)) for a, b in zip(uu[m], vv[m])])
    k, c0 = np.polyfit(pts[:, 1], pts[:, 0], 1)  # u = k v + c0
    upper_lines.append((k, c0))
    nv = np.array([1, -k]) / np.hypot(1, k); Arows.append(nv); brows.append(nv @ np.array([c0, 0.0]))
Arows = np.array(Arows); brows = np.array(brows)
pint, *_ = np.linalg.lstsq(Arows, brows, rcond=None)
rint = Arows @ pint - brows
# full-wire line intersection (shows effect of wire bowing)
Af = []; bf = []
for r in W:
    n = r["frame"]
    a0, vm, a1 = r["slope"], r["vm"], r["icpt"]
    p1 = apply(np.linalg.inv(H[n]), (a0 * (300 - vm) + a1, 300)); p2 = apply(np.linalg.inv(H[n]), (a0 * (700 - vm) + a1, 700))
    d = p2 - p1; nv = np.array([d[1], -d[0]]) / np.linalg.norm(d); Af.append(nv); bf.append(nv @ p1)
pfull, *_ = np.linalg.lstsq(np.array(Af), np.array(bf), rcond=None)
# ------------------------------------------------------------------ angle series
du = cA[:, 0] - pu; dv = cA[:, 1] - pvv
lenp = np.hypot(du, dv)
theta = np.degrees(np.arctan2(du, dv))  # 0 = bob straight below pivot in the image, + = bob toward +u (image right)
# uncertainty: perpendicular components of bob + pivot uncertainty
perp = np.stack([dv, -du], 1) / lenp[:, None]
sb = np.sqrt((perp[:, 0] * sig[:, 0]) ** 2 + (perp[:, 1] * sig[:, 1]) ** 2)
sth = np.degrees(np.sqrt(sb ** 2 + sig_piv ** 2) / lenp)
theta_alt = np.degrees(np.arctan2(cS[:, 0] - pint[0], cS[:, 1] - pint[1]))  # stabilized, vs upper-wire intersection pivot
with open(os.path.join(MEAS, "angle_series.csv"), "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["frame", "t_s", "angle_img_deg", "sigma_angle_img_deg", "angle_img_rad", "sigma_angle_img_rad", "pivot_u_px", "pivot_v_px",
                "bob_u_px", "bob_v_px", "pivot_to_bob_px", "angle_img_alt_deg_stab_upperwire_pivot", "definition"])
    for n in range(N):
        w.writerow([n, "%.6f" % t[n], "%.4f" % theta[n], "%.4f" % sth[n], "%.6f" % np.radians(theta[n]), "%.6f" % np.radians(sth[n]),
                    "%.3f" % pu[n], "%.3f" % pvv[n], "%.3f" % cA[n, 0], "%.3f" % cA[n, 1], "%.3f" % lenp[n], "%.4f" % theta_alt[n],
                    "atan2(u_bob-u_piv, v_bob-v_piv), image plane, same frame; 0 = image-vertical below pivot; + toward image right" if n == 0 else ""])
# ------------------------------------------------------------------ suspension.csv
with open(os.path.join(MEAS, "suspension.csv"), "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["frame", "t_s", "top_u_px", "top_v_px", "bottom_u_px", "bottom_v_px", "wire_length_px", "max_dev_px",
                "sagitta_quadfit_px", "sigma_sagitta_px", "rms_resid_straight_px", "wire_rows_used", "wire_detected_v_min", "wire_detected_v_max",
                "wire_line_angle_img_deg", "bob_centre_offset_from_wire_line_px", "pivot_to_bobcentre_px", "top_definition", "bottom_definition"])
    for r in W:
        n = r["frame"]
        # top attachment = tracked wire-departure point (loop bottom). bottom = wire line ∩ bob circle (upper)
        Lw = np.hypot(r["bot_u"] - pu[n], r["bot_v"] - pvv[n])
        w.writerow([n, "%.6f" % t[n], "%.3f" % pu[n], "%.3f" % pvv[n], "%.3f" % r["bot_u"], "%.3f" % r["bot_v"], "%.3f" % Lw,
                    "%.3f" % r["max_dev_smoothed_px"], "%.3f" % r["sagitta_quad_px"], "%.3f" % r["sagitta_sd_px"], "%.3f" % r["rms_resid_line_px"],
                    r["n_rows"], "%.1f" % r["detected_v_min"], "%.1f" % r["detected_v_max"], "%.4f" % r["wire_angle_deg"],
                    "%.3f" % r["bobcentre_offset_from_wireline_px"], "%.3f" % lenp[n],
                    "tracked wire-departure point at bottom of pin loop (landmark wire_departure)" if n == 0 else "",
                    "upper intersection of fitted wire line with fitted bob circle" if n == 0 else ""])
sag = np.array([r["sagitta_quad_px"] for r in W]); sagsd = np.array([r["sagitta_sd_px"] for r in W]); mdev = np.array([r["max_dev_smoothed_px"] for r in W])
wlen = np.array([np.hypot(r["bot_u"] - pu[r["frame"]], r["bot_v"] - pvv[r["frame"]]) for r in W])
# ------------------------------------------------------------------ oscillation analysis (image space)
def damped(p, tt):
    c0, c1, A, g, om, ph = p
    return c0 + c1 * tt + A * np.exp(-g * tt) * np.cos(om * tt + ph)
def fit_series(y, sy, tt, p0, fix_g=False):
    f = (lambda q: (damped(np.r_[q[:3], 0.0, q[3:]], tt) - y) / sy) if fix_g else (lambda q: (damped(q, tt) - y) / sy)
    q0 = np.r_[p0[:3], p0[4:]] if fix_g else p0
    r = least_squares(f, q0, x_scale="jac")
    J = r.jac; chi2 = (r.fun ** 2).sum(); dof = len(y) - len(q0)
    cov = np.linalg.inv(J.T @ J) * max(chi2 / dof, 1.0)
    q = r.x if not fix_g else np.r_[r.x[:3], 0.0, r.x[3:]]
    e = np.sqrt(np.diag(cov)); e = e if not fix_g else np.r_[e[:3], 0.0, e[3:]]
    res = y - damped(q, tt)
    return q, e, res, chi2 / dof
# initial guess from FFT
y = theta - theta.mean()
fr = np.fft.rfftfreq(4096, 1 / FPS); sp = np.abs(np.fft.rfft(y * np.hanning(N), 4096)); f0 = fr[np.argmax(sp)]
p0 = [theta.mean(), 0.0, (theta.max() - theta.min()) / 2, 0.0, 2 * np.pi * f0, 0.0]
best = None
for ph in np.linspace(-np.pi, np.pi, 12, endpoint=False):
    p0[5] = ph
    q, e, res, rc = fit_series(theta, sth, t, np.array(p0))
    if best is None or np.sum(res ** 2) < best[3]: best = (q, e, res, np.sum(res ** 2), rc)
qD, eD, resD, _, rcD = best
qU, eU, resU, rcU = fit_series(theta, sth, t, qD, fix_g=True)
# fit only t<7.5 and predict 7.5..10 (held-out consistency of image-space model)
m75 = t < 7.5
q75, e75, _, _ = fit_series(theta[m75], sth[m75], t[m75], qD)
pred = damped(q75, t); ho_rms = np.sqrt(np.mean((theta[~m75] - pred[~m75]) ** 2))
# zero crossings of (theta - offset - trend)
yc = theta - (qD[0] + qD[1] * t)
zc = []
for n in range(N - 1):
    if yc[n] == 0 or yc[n] * yc[n + 1] < 0:
        fr_ = yc[n] / (yc[n] - yc[n + 1]); tz = (n + fr_) / FPS
        slope = (yc[n + 1] - yc[n]) * FPS
        sz = np.hypot(sth[n], sth[n + 1]) / np.sqrt(2) / abs(slope)
        zc.append((tz, sz, np.sign(slope)))
zc = np.array(zc)
k = np.arange(len(zc)); wz = 1 / zc[:, 1] ** 2
Aw = np.stack([np.ones_like(k), k], 1).astype(float)
covz = np.linalg.inv(Aw.T @ (wz[:, None] * Aw)); bz = covz @ Aw.T @ (wz * zc[:, 0])
rz = zc[:, 0] - Aw @ bz; s2z = (wz * rz ** 2).sum() / (len(zc) - 2); covz = covz * max(s2z, 1.0)
T_zc = 2 * bz[1]; sT_zc = 2 * np.sqrt(covz[1, 1])
full_periods = np.array([zc[i + 2, 0] - zc[i, 0] for i in range(len(zc) - 2)])
# turning points: local fit c + a cos(om (t - t0)) with offset fixed to global (offset+trend), +-3 frames
omg = qD[4]
peaks = []
ext = [n for n in range(1, N - 1) if (yc[n] - yc[n - 1]) * (yc[n + 1] - yc[n]) <= 0 and abs(yc[n]) > 0.5 * abs(qD[2]) * np.exp(-qD[3] * t[n])]
# collapse adjacent
ext2 = []
for n in ext:
    if ext2 and n - ext2[-1] < 5: 
        if abs(yc[n]) > abs(yc[ext2[-1]]): ext2[-1] = n
        continue
    ext2.append(n)
for n in ext2:
    lo, hi = max(0, n - 3), min(N, n + 4)
    tt, yy, ss = t[lo:hi], yc[lo:hi], sth[lo:hi]
    f = lambda q: (q[0] * np.cos(omg * (tt - q[1])) - yy) / ss
    r = least_squares(f, [yc[n], t[n]])
    J = r.jac; dof = max(len(tt) - 2, 1); cov = np.linalg.inv(J.T @ J) * max((r.fun ** 2).sum() / dof, 1.0)
    peaks.append((n, r.x[1], np.sqrt(cov[1, 1]), r.x[0], np.sqrt(cov[0, 0])))
peaks = np.array(peaks)
pos = peaks[peaks[:, 3] > 0]; neg = peaks[peaks[:, 3] < 0]
def logfit(P):
    a = np.abs(P[:, 3]); s = P[:, 4] / a
    Aw = np.stack([np.ones(len(P)), P[:, 1]], 1); wv = 1 / s ** 2
    cv = np.linalg.inv(Aw.T @ (wv[:, None] * Aw)); b = cv @ Aw.T @ (wv * np.log(a))
    rr = np.log(a) - Aw @ b; s2 = (wv * rr ** 2).sum() / max(len(P) - 2, 1); cv = cv * max(s2, 1.0)
    return b[1], np.sqrt(cv[1, 1]), np.exp(b[0])
gpos = logfit(pos); gneg = logfit(neg)
# peak-to-peak amplitude per half cycle (successive extremes)
p2p = [(0.5 * (peaks[i, 1] + peaks[i + 1, 1]), 0.5 * abs(peaks[i, 3] - peaks[i + 1, 3]), 0.5 * np.hypot(peaks[i, 4], peaks[i + 1, 4])) for i in range(len(peaks) - 1)]
p2p = np.array(p2p)
Aw = np.stack([np.ones(len(p2p)), p2p[:, 0]], 1); wv = 1 / p2p[:, 2] ** 2
cv = np.linalg.inv(Aw.T @ (wv[:, None] * Aw)); bp = cv @ Aw.T @ (wv * p2p[:, 1]); rr = p2p[:, 1] - Aw @ bp
cv = cv * max((wv * rr ** 2).sum() / (len(p2p) - 2), 1.0)
p2p_slope, p2p_slope_sd = bp[1], np.sqrt(cv[1, 1])
# same analysis in pixels on stabilized u (frame-0 geometry)
qUu, eUu, resUu, rcUu = None, None, None, None
best = None
for ph in np.linspace(-np.pi, np.pi, 12, endpoint=False):
    p0u = np.array([cS[:, 0].mean(), 0.0, (cS[:, 0].max() - cS[:, 0].min()) / 2, 0.0, qD[4], ph])
    q, e, res, rc = fit_series(cS[:, 0], np.maximum(sig[:, 0], 0.2), t, p0u)
    if best is None or np.sum(res ** 2) < best[3]: best = (q, e, res, np.sum(res ** 2), rc)
qSu, eSu, resSu, _, rcSu = best
# raw u
best = None
for ph in np.linspace(-np.pi, np.pi, 12, endpoint=False):
    p0u = np.array([cA[:, 0].mean(), 0.0, (cA[:, 0].max() - cA[:, 0].min()) / 2, 0.0, qD[4], ph])
    q, e, res, rc = fit_series(cA[:, 0], np.maximum(sig[:, 0], 0.2), t, p0u)
    if best is None or np.sum(res ** 2) < best[3]: best = (q, e, res, np.sum(res ** 2), rc)
qRu, eRu, resRu, _, rcRu = best
# phase at t=0
th0 = damped(qD, 0.0) - qD[0]; om0 = -qD[2] * (qD[4] * np.sin(qD[5]) + qD[3] * np.cos(qD[5]))
phase0 = (qD[5] / (2 * np.pi)) % 1.0
vel_u0 = (-3 * cA[0, 0] + 4 * cA[1, 0] - cA[2, 0]) / 2 * FPS
vmax_u = np.abs(np.gradient(cA[:, 0]) * FPS).max()
# radius vs swing position
rcoef = np.polyfit(cS[:, 0] - cS[:, 0].mean(), radA / simp[:, 0], 1)
# vertical arc: stabilized v vs stabilized u (quadratic), apparent radius of curvature
qv = np.polyfit(cS[:, 0] - pint[0], cS[:, 1], 2)
Rcurv = 1 / (2 * qv[0])
# frame-to-frame jumps
ju = np.diff(cS[:, 0]); acc = np.diff(cS[:, 0], 2)
# ------------------------------------------------------------------ summary
def f(x, nd=4): return float(np.round(x, nd))
summary = {
  "role": "reference measurement", "created_utc": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
  "reference": {"path": VIDEO, "sha256": VIDEO_SHA, "frames_decoded": N, "fps": FPS, "time_convention": "t = n/24 s, n 0-based; verified 241 frames decoded (OpenCV), ffprobe count 241, GOP: 8 I / 64 P / 169 B"},
  "units": "pixels in the contract pixel convention; angles in degrees (image-plane only) unless *_rad",
  "camera_drift": {
    "verdict": "GENUINE FRAMING DRIFT (camera move): smooth, monotonic zoom-in/dolly-in over the whole clip; not a static camera",
    "similarity_scale_end": f(simp[-1, 0], 5), "similarity_rot_end_deg": f(simp[-1, 1], 4),
    "displacement_end_px": {"pivot": [f(piv_drift[-1, 0] - 1086.7, 2), f(piv_drift[-1, 1] - 125.0, 2)],
                            "bob_rest_region_(1090,835)": [f(apply(H[-1], (1090, 835))[0] - 1090, 2), f(apply(H[-1], (1090, 835))[1] - 835, 2)],
                            "base_front_left_corner": [f(apply(H[-1], (502, 903))[0] - 502, 2), f(apply(H[-1], (502, 903))[1] - 903, 2)]},
    "model": "per-frame 8-dof homography frame0->frame n, ECC on static mask, init chained KLT; camera_drift.csv",
    "landmark_residual_after_homography_rms_px": f(np.sqrt(np.mean([float(LMD[k][n]["resid_u_after_drift_px"]) ** 2 + float(LMD[k][n]["resid_v_after_drift_px"]) ** 2 for k in LMD if k not in ("wire_departure", "wire_loop_ring", "pin_shaft_mid") for n in range(N)])), 3),
    "ecc_cc_min": f(min(dd["ecc_cc"]), 4)},
  "pivot": {"definition": "wire departure point at the bottom of the wire loop on the steel pin (tracked landmark wire_departure)",
            "frame0_px": [f(pu[0], 2), f(pvv[0], 2)], "frame240_px": [f(pu[-1], 2), f(pvv[-1], 2)],
            "residual_motion_after_drift_rms_px": f(np.sqrt(np.mean(pres ** 2)), 3),
            "swing_correlated_motion_pp_px": f(0.0023 * (cA[:, 0].max() - cA[:, 0].min()), 2),
            "alt_upper_wire_lines_intersection_frame0_px": [f(pint[0], 2), f(pint[1], 2)], "alt_intersection_line_rms_px": f(np.sqrt(np.mean(rint ** 2)), 3),
            "alt_full_wire_lines_intersection_frame0_px": [f(pfull[0], 2), f(pfull[1], 2)],
            "loop_ring_rocks_with_swing": False},
  "bob": {"method": "A radial luma edge + robust geometric circle fit (free radius); B annulus multi-scale ZNCC; per-frame sigma = sqrt(fit_cov + local-RMS(A-B)^2/2)",
          "A_minus_B_rms_px": [f(np.sqrt(np.mean(dAB[:, 0] ** 2)), 3), f(np.sqrt(np.mean(dAB[:, 1] ** 2)), 3)],
          "A_minus_B_max_px": [f(np.abs(dAB[:, 0]).max(), 2), f(np.abs(dAB[:, 1]).max(), 2)],
          "sigma_median_px": [f(np.median(sig[:, 0]), 3), f(np.median(sig[:, 1]), 3)], "sigma_max_px": [f(sig[:, 0].max(), 3), f(sig[:, 1].max(), 3)],
          "jitter_vs_local_cubic_px": [f(jit[0], 3), f(jit[1], 3)],
          "radius_px_range": [f(radA.min(), 2), f(radA.max(), 2)],
          "radius_over_drift_scale_vs_u_stab_slope_px_per_px": f(rcoef[0], 5), "radius_over_drift_scale_at_mean_u": f(rcoef[1], 3),
          "radius_note": "apparent radius grows ~5% from the left to the right turning point (both methods: A circle radius, B best template scale); consistent with the swing having a depth component toward the camera on the image-right side (inference) - the camera specialist should use it",
          "ellipse_minor_over_major_range": [f((np.minimum(ellA[:, 2], ellA[:, 3]) / np.maximum(ellA[:, 2], ellA[:, 3])).min(), 3), f((np.minimum(ellA[:, 2], ellA[:, 3]) / np.maximum(ellA[:, 2], ellA[:, 3])).max(), 3)],
          "max_framediff_u_stab_px": f(np.abs(ju).max(), 2), "max_abs_second_diff_u_stab_px": f(np.abs(acc).max(), 2),
          "u_raw_range": [f(cA[:, 0].min(), 2), f(cA[:, 0].max(), 2)], "v_raw_range": [f(cA[:, 1].min(), 2), f(cA[:, 1].max(), 2)],
          "stab_v_vs_u_quadratic": [f(x, 7) for x in qv], "stab_arc_apparent_radius_of_curvature_px": f(Rcurv, 1)},
  "angle_image": {"definition": "atan2(u_bob-u_pivot, v_bob-v_pivot) in the same raw frame; NOT a world angle",
                  "sigma_median_deg": f(np.median(sth), 4), "pivot_to_bob_px_range": [f(lenp.min(), 2), f(lenp.max(), 2)]},
  "period": {
    "damped_sine_fit_angle": {"model": "theta = c0 + c1 t + A exp(-g t) cos(w t + phi), weighted by sigma, all 241 frames",
        "c0_deg": f(qD[0]), "c0_sd": f(eD[0]), "c1_deg_per_s": f(qD[1], 5), "c1_sd": f(eD[1], 5), "A_deg": f(qD[2]), "A_sd": f(eD[2]),
        "g_per_s": f(qD[3], 5), "g_sd": f(eD[3], 5), "omega_rad_s": f(qD[4], 5), "omega_sd": f(eD[4], 5),
        "period_s": f(2 * np.pi / qD[4], 5), "period_sd_s": f(2 * np.pi / qD[4] ** 2 * eD[4], 5), "phi_rad": f(qD[5]), "phi_sd": f(eD[5]),
        "reduced_chi2": f(rcD, 3), "rms_resid_deg": f(np.sqrt(np.mean(resD ** 2)), 4)},
    "undamped_sine_fit_angle": {"A_deg": f(qU[2]), "A_sd": f(eU[2]), "period_s": f(2 * np.pi / qU[4], 5), "period_sd_s": f(2 * np.pi / qU[4] ** 2 * eU[4], 5), "rms_resid_deg": f(np.sqrt(np.mean(resU ** 2)), 4), "reduced_chi2": f(rcU, 3)},
    "fit_t_lt_7p5_predict_rest": {"period_s": f(2 * np.pi / q75[4], 5), "g_per_s": f(q75[3], 5), "heldout_rms_deg": f(ho_rms, 4)},
    "zero_crossings": {"n": int(len(zc)), "half_period_s": f(bz[1], 5), "period_s": f(T_zc, 5), "period_sd_s": f(sT_zc, 5),
                       "times_s": [f(x, 4) for x in zc[:, 0]], "sigma_s": [f(x, 4) for x in zc[:, 1]],
                       "full_periods_s": [f(x, 4) for x in full_periods], "full_period_std_s": f(full_periods.std(ddof=1), 4),
                       "resid_rms_s": f(np.sqrt(np.mean(rz ** 2)), 4)},
    "stab_u_px_fit": {"A_px": f(qSu[2], 3), "A_sd": f(eSu[2], 3), "g_per_s": f(qSu[3], 5), "g_sd": f(eSu[3], 5), "period_s": f(2 * np.pi / qSu[4], 5), "period_sd_s": f(2 * np.pi / qSu[4] ** 2 * eSu[4], 5), "rms_px": f(np.sqrt(np.mean(resSu ** 2)), 3)},
    "raw_u_px_fit": {"A_px": f(qRu[2], 3), "A_sd": f(eRu[2], 3), "g_per_s": f(qRu[3], 5), "g_sd": f(eRu[3], 5), "period_s": f(2 * np.pi / qRu[4], 5), "period_sd_s": f(2 * np.pi / qRu[4] ** 2 * eRu[4], 5), "rms_px": f(np.sqrt(np.mean(resRu ** 2)), 3)}},
  "amplitude": {
    "turning_points": [{"frame": int(p[0]), "t_s": f(p[1]), "t_sd": f(p[2]), "angle_minus_offset_deg": f(p[3]), "sd_deg": f(p[4])} for p in peaks],
    "half_cycle_peak_to_peak_half_deg": [{"t_s": f(a), "amp_deg": f(b), "sd_deg": f(c)} for a, b, c in p2p],
    "p2p_amp_linear_trend_deg_per_s": f(p2p_slope, 5), "p2p_amp_linear_trend_sd": f(p2p_slope_sd, 5),
    "log_decay_rate_positive_peaks_per_s": [f(gpos[0], 5), f(gpos[1], 5)], "log_decay_rate_negative_peaks_per_s": [f(gneg[0], 5), f(gneg[1], 5)],
  },
  "phase_t0": {"angle_minus_offset_deg_fit": f(th0), "angular_velocity_deg_s_fit": f(om0, 3), "phase_cycles_cos": f(phase0, 4),
               "u_velocity_t0_px_s": f(vel_u0, 1), "max_u_speed_px_s": f(vmax_u, 1)},
  "suspension": {"sagitta_px_range": [f(sag.min(), 3), f(sag.max(), 3)], "sagitta_sd_median_px": f(np.median(sagsd), 3),
                 "max_dev_px_range": [f(mdev.min(), 3), f(mdev.max(), 3)], "corr_sagitta_vs_angle": f(np.corrcoef(sag, theta)[0, 1], 3),
                 "sagitta_vs_angle_slope_px_per_deg": f(np.polyfit(theta - qD[0], sag, 1)[0], 4),
                 "wire_length_px_range": [f(wlen.min(), 2), f(wlen.max(), 2)],
                 "pivot_to_bobcentre_over_drift_scale_range": [f((lenp / simp[:, 0]).min(), 2), f((lenp / simp[:, 0]).max(), 2)]},
}
json.dump(summary, open(os.path.join(MEAS, "summary.json"), "w"), indent=1)
np.savez(os.path.join(CACHE, "analysis.npz"), theta=theta, sth=sth, qD=qD, peaks=peaks, zc=zc, cS=cS, sig=sig, pu=pu, pv=pvv, sag=sag, lenp=lenp, simp=simp, pint=pint, resD=resD, dAB=dAB)
print(json.dumps({k: summary[k] for k in ["period", "amplitude", "phase_t0"]}, indent=0)[:6000])
print(json.dumps({k: summary[k] for k in ["camera_drift", "pivot", "bob", "suspension"]}, indent=0))
