"""Stage 6: final tracks + analysis (clip 03).
psi = mean of methods A and B (twist_raw.csv); per-frame sigma = max(0.1 deg, rolling (9-frame) rms(A-B)/2) (+ formal B).
Disc rotation centre Q = silhouette top-face centre P + apex offset (in-plane), world (origin A, unit D), per frame;
tilt = angle of the disc normal from world +Z, and its azimuth.
Models for psi(t) over 0-10 s (weighted LSQ): constant spin, constant angular acceleration, cubic, and a slow torsional
oscillation c + A sin(w t + phi) (+ damping). Sway: sinusoid fits to Q_x, Q_y (and wire image tilt).
Cadence: residual of a smooth (7th-order) psi model grouped by n mod 4, converted to timing offsets with the local rate.
Discontinuity screens: psi residual outliers, kink and line-offset jumps, silhouette rms, static-highlight (lobe) rotation.
Writes measurement/twist_track.csv, disc_track.csv, analysis.json, cache/analysis.npz."""
import numpy as np, csv, json, os, sys
from scipy.optimize import least_squares
from scipy.ndimage import median_filter
sys.path.insert(0, os.path.dirname(__file__))
from common import *
import camfr
def rd(p): return list(csv.DictReader(open(os.path.join(MEAS, p))))
TW = rd("twist_raw.csv"); WT = rd("wire_track.csv"); N = len(TW); n = np.arange(N); t = n / FPS
g = lambda R, k: np.array([float(r[k]) for r in R])
pA, pB = g(TW, "psi_A_deg"), g(TW, "psi_B_deg"); dAB = pA - pB; psi = 0.5 * (pA + pB)
roll = np.array([np.sqrt(np.mean(dAB[max(0, i - 4):i + 5] ** 2)) for i in range(N)])
sig = np.sqrt(np.maximum(0.1, roll / 2) ** 2 + (g(TW, "sigma_B_formal_deg") / 2) ** 2)
POSE = np.load(os.path.join(CACHE, "disc_pose.npy")); apex = np.load(os.path.join(CACHE, "apex_offset.npy"))
def normal(a, b):
    v = np.array([np.sin(b), -np.sin(a) * np.cos(b), np.cos(a) * np.cos(b)]); return v / np.linalg.norm(v)
def basis(v):
    ex = np.array([1.0, 0, 0]) - v[0] * v; ex /= np.linalg.norm(ex); return ex, np.cross(v, ex)
Q = []; NRM = []
for r in POSE:
    v = normal(r[5], r[6]); ex, ey = basis(v); Q.append(r[2:5] + apex[0] * ex + apex[1] * ey); NRM.append(v)
Q = np.array(Q); NRM = np.array(NRM); sQ = POSE[:, 7:10]
tilt = np.degrees(np.arccos(NRM[:, 2])); tilt_az = np.degrees(np.arctan2(NRM[:, 1], NRM[:, 0]))
tx, ty = np.degrees(POSE[:, 5]), np.degrees(POSE[:, 6])   # small rotations about world X, Y (deg)
silrms = POSE[:, 12]
# ---------------- psi models
def wfit(f, p0):
    r = least_squares(lambda p: (f(p, t) - psi) / sig, p0); J = r.jac; s2 = 2 * r.cost / (N - len(p0)); cov = np.linalg.pinv(J.T @ J) * max(s2, 1)
    res = f(r.x, t) - psi; return dict(p=r.x.tolist(), sd=np.sqrt(np.diag(cov)).tolist(), rms_deg=float(np.sqrt(np.mean(res ** 2))), chi2nu=float(s2), bic=float(2 * r.cost + len(p0) * np.log(N))), res
models = {}
models["constant_spin"], _ = wfit(lambda p, t: p[0] + p[1] * t, [psi[0], -38])
models["constant_accel"], _ = wfit(lambda p, t: p[0] + p[1] * t + 0.5 * p[2] * t ** 2, [psi[0], -38, 0])
models["cubic"], _ = wfit(lambda p, t: p[0] + p[1] * t + p[2] * t ** 2 + p[3] * t ** 3, [psi[0], -38, 0, 0])
best = None
for T0 in (20, 30, 40, 60):
    for ph in np.linspace(-np.pi, np.pi, 8, endpoint=False):
        m, r = wfit(lambda p, t: p[0] + p[1] * np.sin(2 * np.pi / p[2] * t + p[3]), [psi.mean(), 200, T0, ph])
        if best is None or m["rms_deg"] < best[0]["rms_deg"]: best = (m, r)
models["slow_oscillation_c+A_sin(2pi t/T+phi)"] = best[0]
vel = np.gradient(psi, 1 / FPS); vs = np.convolve(vel, np.ones(8) / 8, "same"); vs[:4] = np.nan; vs[-4:] = np.nan
# smooth reference (7th-order poly) for cadence / outliers
cp = np.polyfit(t, psi, 7, w=1 / sig); sm = np.polyval(cp, t); res7 = psi - sm; dps = np.polyval(np.polyder(cp), t)
cad = {}
for j in range(4):
    m = (n % 4) == j; off = -res7[m] / dps[m] * FPS   # frames: positive = frame content later than its timestamp
    cad[str(j)] = dict(mean_resid_deg=float(res7[m].mean()), timing_offset_frames=float(np.mean(off)), sd=float(np.std(off) / np.sqrt(m.sum())))
step = np.abs(np.diff(psi)); stepr = {str(j): float(np.median(step[(n[1:] % 4) == j] / np.median(step))) for j in range(4)}
# outliers / discontinuities
mad = 1.4826 * np.median(np.abs(res7 - np.median(res7))); outl = [int(i) for i in np.where(np.abs(res7) > 5 * max(mad, 0.1))[0]]
kinkA, kinkB = g(TW, "kink_A_deg"), g(TW, "kink_B_deg"); lor = g(TW, "line_offset_resid_D")
def jumps(x, k=6.0):
    d = np.diff(x); s = 1.4826 * np.median(np.abs(d - np.median(d))); return [int(i + 1) for i in np.where(np.abs(d) > k * max(s, 1e-9))[0]], float(s)
# highlight lobes: low-pass angular profile (line masked) vs frame 0
pol = np.load(os.path.join(CACHE, "rect_polar.npy"), mmap_mode="r"); RR = np.linspace(0.06, 0.48, 85); band = (RR > 0.2) & (RR < 0.45)
th = np.arange(720) * 0.5; lobe = []
prof0 = None
for i in range(N):
    p = np.asarray(pol[i])[band].mean(0); bad = (np.abs((th - psi[i] + 90) % 180 - 90) < 6) | (np.abs(((th - g(TW, "wire_azimuth_deg")[i]) + 180) % 360 - 180) < 4)
    p = p.copy(); p[bad] = np.interp(th[bad], th[~bad], p[~bad]); p = median_filter(p, 21, mode="wrap"); p -= p.mean()
    if prof0 is None: prof0 = p
    cc = [np.dot(np.roll(prof0, s), p) for s in range(-120, 121)]; k = int(np.argmax(cc)); lobe.append((k - 120) * 0.5)
lobe = np.array(lobe)
# camera azimuth (for comparison with the lobe drift)
caz = np.degrees(np.array([np.arctan2(camfr.cam(i)["C"][0], -camfr.cam(i)["C"][1]) for i in range(N)]))
# ---------------- sway
def sinfit(y, s):
    bestm = None
    for T0 in (0.8, 1.2, 1.6, 2.0, 2.5, 3.0, 4.0, 6.0):
        for ph in (0, 1.5, 3, 4.5):
            r = least_squares(lambda p: (p[0] + p[1] * t + p[2] * np.sin(2 * np.pi * t / p[3] + p[4]) - y) / s, [y.mean(), 0, np.std(y), T0, ph])
            if bestm is None or r.cost < bestm.cost: bestm = r
    r = bestm; J = r.jac; s2 = 2 * r.cost / (N - 5); cov = np.linalg.pinv(J.T @ J) * max(s2, 1); sd = np.sqrt(np.diag(cov))
    return dict(mean=float(r.x[0]), trend_per_s=float(r.x[1]), amp=float(abs(r.x[2])), amp_sd=float(sd[2]), period_s=float(r.x[3]), period_sd=float(sd[3]),
                rms_resid=float(np.sqrt(np.mean((r.fun * s) ** 2))), chi2nu=float(s2))
def spec(y):
    yy = (y - np.polyval(np.polyfit(t, y, 2), t)) * np.hanning(N); F_ = np.abs(np.fft.rfft(yy, 8 * N)); fr = np.fft.rfftfreq(8 * N, 1 / FPS)
    m = fr > 0.15; i = np.argmax(F_ * m); return float(fr[i])
wtilt = g(WT, "image_tilt_deg_(+=bottom_toward_+u_is_negative)")
sway = {"Qx_D": sinfit(Q[:, 0], np.maximum(sQ[:, 0], 0.001)), "Qy_D": sinfit(Q[:, 1], np.maximum(sQ[:, 1], 0.002)), "Qz_D": sinfit(Q[:, 2], np.maximum(sQ[:, 2], 0.001)),
        "tilt_about_X_deg": sinfit(tx, np.full(N, 0.1)), "tilt_about_Y_deg": sinfit(ty, np.full(N, 0.1)), "wire_image_tilt_deg": sinfit(wtilt, np.full(N, 0.1))}
peaks = {k: spec(y) for k, y in (("Qx", Q[:, 0]), ("Qy", Q[:, 1]), ("Qz", Q[:, 2]), ("tilt_X", tx), ("tilt_Y", ty), ("wire_tilt", wtilt))}
# swing angles of the wire (A -> Q) from vertical
sw_x = np.degrees(np.arctan2(Q[:, 0], -Q[:, 2])); sw_y = np.degrees(np.arctan2(Q[:, 1], -Q[:, 2]))
# tilt vs twist (non-circular-disc / wobble test): regress tilt components on cos/sin 2psi and on the swing angles
X = np.c_[np.ones(N), np.cos(2 * np.radians(psi)), np.sin(2 * np.radians(psi))]
reg = {k: np.linalg.lstsq(X, y, rcond=None)[0].tolist() for k, y in (("tilt_X", tx), ("tilt_Y", ty))}
Xs = np.c_[np.ones(N), sw_x, sw_y]; regs = {k: np.linalg.lstsq(Xs, y, rcond=None)[0].tolist() for k, y in (("tilt_X", tx), ("tilt_Y", ty))}
cor = {"tilt_Y_vs_swing_x": float(np.corrcoef(ty, sw_x)[0, 1]), "tilt_X_vs_swing_y": float(np.corrcoef(tx, sw_y)[0, 1])}
# ---------------- anchor-window rule (CHANGELOG 2026-10-26T03:09:32Z)
tps = []   # turning points of psi (velocity sign changes of the 8-frame-smoothed rate)
sgn = np.sign(vs); ch = np.where((sgn[:-1] * sgn[1:]) < 0)[0]
tps = [int(c) for c in ch]
kj, ks = jumps(kinkA, 6); oj, os_ = jumps(lor, 6)
anchor = {"turning_points_frames": tps,
          "a_halfcycle_amplitude_jump": "not applicable: psi has no turning point in 0-10 s (monotonic clockwise rotation, rate %.0f..%.0f deg/s), so no half-cycle amplitudes exist" % (np.nanmin(vs), np.nanmax(vs)),
          "b_cumulative_drift": "not applicable: no full cycle exists to fit the passive-decay reference model (needs two full cycles); sway mode checked separately below",
          "c_visible_discontinuity": {"psi_residual_outliers_frames(>5 MAD)": outl, "kink_jumps_frames(>6 MAD)": kj, "line_offset_jumps_frames(>6 MAD)": oj,
                                      "silhouette_rms_max_px": float(silrms.max()), "verdict": "none visible (see notes): line stays one full diameter, slight kink varies smoothly"},
          "T_a_s": 10.0, "held_out_s": [7.5, 10.0],
          "decision": "rule (a)/(b) have no cycles to act on and (c) finds no discontinuity -> default T_a = 10 s with the last 2.5 s held out. FLAG to lead: the anchor rule presumes an oscillation; the AI twist is a one-directional spin, so the physics fit is effectively unconstrained by the rule."}
# sway mode anchor checks (a) on the Qx oscillation half-cycles
def halfcycles(y):
    ys = np.convolve(y, np.ones(5) / 5, "same"); d = np.sign(np.diff(ys)); idx = [i for i in range(3, N - 4) if d[i - 1] != d[i] and d[i] != 0]
    out = []
    for i in idx:
        a, b = max(0, i - 4), min(N, i + 5); c = np.polyfit(t[a:b] - t[i], y[a:b], 2); tm = -c[1] / (2 * c[0]) if c[0] != 0 else 0
        if abs(tm) < 4 / FPS: out.append((t[i] + tm, np.polyval(c, tm)))
    out = np.array(out); amp = np.abs(np.diff(out[:, 1])) / 2 if len(out) > 1 else np.array([])
    return out, amp
tpx, ampx = halfcycles(Q[:, 0])
anchor["sway_x_turning_points_s"] = tpx[:, 0].tolist() if len(tpx) else []; anchor["sway_x_halfcycle_amplitude_D"] = ampx.tolist()
# ---------------- write tracks
with open(os.path.join(MEAS, "twist_track.csv"), "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["frame", "t_s", "psi_deg", "sigma_psi_deg", "psi_A_deg", "psi_B_deg", "rate_deg_per_s_8frame", "kink_A_deg", "kink_B_deg", "highlight_lobe_rotation_deg", "camera_azimuth_deg"])
    for i in range(N): w.writerow([i, "%.5f" % t[i]] + ["%.4f" % x for x in (psi[i], sig[i], pA[i], pB[i], vs[i], kinkA[i], kinkB[i], lobe[i], caz[i])])
with open(os.path.join(MEAS, "disc_track.csv"), "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["frame", "t_s", "Qx_D", "Qy_D", "Qz_D", "sd_x", "sd_y", "sd_z", "tilt_deg", "tilt_azimuth_deg", "tilt_about_X_deg", "tilt_about_Y_deg", "swing_x_deg", "swing_y_deg",
                                    "u_Q_px", "v_Q_px", "silhouette_rms_px"])
    for i in range(N):
        uv = camfr.proj(Q[i], camfr.cam(i))
        w.writerow([i, "%.5f" % t[i]] + ["%.6f" % x for x in (*Q[i], *sQ[i], tilt[i], tilt_az[i], tx[i], ty[i], sw_x[i], sw_y[i], *uv, silrms[i])])
an = dict(n_frames=N, psi_methods={"A_minus_B_rms_deg": float(np.sqrt(np.mean(dAB ** 2))), "A_minus_B_mean_deg": float(dAB.mean()), "A_minus_B_max_deg": float(np.abs(dAB).max()),
          "sigma_psi_median_deg": float(np.median(sig))},
          psi={"psi0_deg": float(psi[0]), "psi240_deg": float(psi[-1]), "total_rotation_deg": float(psi[-1] - psi[0]), "turns": float((psi[-1] - psi[0]) / 360),
               "rate_8frame_min_max_deg_s": [float(np.nanmin(vs)), float(np.nanmax(vs))], "rate_t0_deg_s": float(np.polyval(np.polyder(cp), 0)),
               "rate_by_second_deg_s": [float((psi[min(N - 1, 24 * (k + 1))] - psi[24 * k]) / (min(N - 1, 24 * (k + 1)) - 24 * k) * FPS) for k in range(10)],
               "direction": "clockwise seen from above (psi decreasing; psi = azimuth of the line from world +X toward +Y)", "turning_points": tps, "models": models},
          cadence={"timing_offsets_by_n_mod_4": cad, "step_ratio_by_n_mod_4_of_destination": stepr, "resid_rms_7th_order_deg": float(np.sqrt(np.mean(res7 ** 2)))},
          disc_centre={"mean_D": Q.mean(0).tolist(), "ptp_D": np.ptp(Q, 0).tolist(), "std_D": Q.std(0).tolist(), "sd_formal_median_D": np.median(sQ, 0).tolist(),
                       "horizontal_offset_of_mean_from_below_A_D": Q.mean(0)[:2].tolist(), "wire_length_A_to_top_face_mean_D": float(-Q[:, 2].mean())},
          tilt={"mean_deg": float(tilt.mean()), "max_deg": float(tilt.max()), "about_X_mean_ptp": [float(tx.mean()), float(np.ptp(tx))], "about_Y_mean_ptp": [float(ty.mean()), float(np.ptp(ty))],
                "regression_on_cos2psi_sin2psi": reg, "regression_on_swing_angles": regs, "correlations": cor},
          sway=sway, spectral_peak_hz=peaks, swing_angles_deg={"x_mean_ptp": [float(sw_x.mean()), float(np.ptp(sw_x))], "y_mean_ptp": [float(sw_y.mean()), float(np.ptp(sw_y))]},
          thickness_D=float(POSE[0, 17]), apex_offset_D=apex.tolist(), silhouette_rms_px_median=float(np.median(silrms)),
          kink={"A_mean_sd_range": [float(kinkA.mean()), float(kinkA.std()), float(kinkA.min()), float(kinkA.max())], "B_mean_sd_range": [float(kinkB.mean()), float(kinkB.std()), float(kinkB.min()), float(kinkB.max())]},
          highlight_lobes={"rotation_deg_range": [float(lobe.min()), float(lobe.max())], "lobe_vs_camera_azimuth_slope": float(np.polyfit(caz - caz[0], lobe, 1)[0]),
                           "note": "lighting-locked anisotropic highlights; should not follow the disc spin"},
          anchor_window=anchor)
json.dump(an, open(os.path.join(MEAS, "analysis.json"), "w"), indent=1)
np.savez(os.path.join(CACHE, "analysis.npz"), t=t, psi=psi, sig=sig, pA=pA, pB=pB, vs=vs, Q=Q, tx=tx, ty=ty, sw_x=sw_x, sw_y=sw_y, lobe=lobe, caz=caz, kinkA=kinkA, kinkB=kinkB, wtilt=wtilt, res7=res7)
print(json.dumps({k: an[k] for k in ("psi_methods", "psi", "cadence", "disc_centre", "tilt", "sway", "spectral_peak_hz", "swing_angles_deg", "kink", "highlight_lobes")}, indent=1, default=float)[:6000])
print(json.dumps(anchor, indent=1)[:2000])
