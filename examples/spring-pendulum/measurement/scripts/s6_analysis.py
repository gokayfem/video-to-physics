"""Stage 6: assemble tracks, map to frame-0 geometry (drift removed with H_n^-1), per-frame sigmas, pivot,
mode decomposition and per-half-cycle analysis. Writes weight_track.csv, ring_track.csv, modes_series.csv,
cache/analysis.npz, and analysis.json (consumed by s8_summary / plots)."""
import numpy as np, os, sys, json, csv, pickle
sys.path.insert(0, os.path.dirname(__file__))
from common import *
from scipy.optimize import least_squares
from scipy.ndimage import uniform_filter1d
H = np.load(os.path.join(CACHE, "H_ecc.npy")); Hi = np.array([np.linalg.inv(h) for h in H])
n = np.arange(N); t = n / FPS
W = np.load(os.path.join(CACHE, "weight_raw.npz")); Rg = np.load(os.path.join(CACHE, "ring_raw.npz"))
SP = list(csv.DictReader(open(os.path.join(MEAS, "spring_track.csv")))); spf = lambda k: np.array([float(r[k]) for r in SP])
def stab(P):  # raw frame-n points (N,2) -> frame-0 geometry
    return np.array([Happly(Hi[k], P[k])[0] for k in range(N)])
def stab_dir_angle(P, th):  # angle (from image vertical, + bottom toward +u) of a direction at point P, mapped to frame 0
    out = []
    for k in range(N):
        a = np.array([np.sin(th[k]), np.cos(th[k])]); q = Happly(Hi[k], np.array([P[k], P[k] + 50 * a]))
        d = q[1] - q[0]; out.append(np.arctan2(d[0], d[1]))
    return np.array(out)
def local_rms(x, w=7): return np.sqrt(uniform_filter1d(x ** 2, w, mode="nearest"))
# ---- weight: B primary (silhouette profiles), C check (Canny/RANSAC); constant offset removed (median)
cB, thB, cC, thC = W["cB"], W["thB"], W["cC"], W["thC"]
offC = np.median(cC - cB, 0); dthC = np.median(thC - thB)
dW = cC - offC - cB; dTh = thC - dthC - thB
sigW = np.sqrt(np.c_[np.array(W["sigperpB"]) ** 2, np.array(W["sigparB"]) ** 2].max(1)[:, None] + local_rms(dW) ** 2 / 2)  # (N,2)
sigTh = np.sqrt(np.array(W["sigthB"]) ** 2 + local_rms(dTh) ** 2 / 2)
cW0 = stab(cB); thW0 = stab_dir_angle(cB, thB); cW0_C = stab(cC - offC); thW0_C = stab_dir_angle(cC, thC - dthC)
sdrift = np.array([np.sqrt(abs(np.linalg.det(h[:2, :2]))) for h in H])
# ---- ring: A primary (radial edges + circle fit), B check (annulus ZNCC); offset removed
cRA, cRB = Rg["cA"], Rg["cB"][:, :2]; offR = np.median(cRB - cRA, 0); dR = cRB - offR - cRA
covR = Rg["covA"]; sigR = np.sqrt(np.c_[covR[:, 0, 0], covR[:, 1, 1]] + local_rms(dR) ** 2 / 2)
cR0 = stab(cRA); cR0_B = stab(cRB - offR); radR0 = Rg["radA"] / sdrift
# ---- spring axis (extent method primary, centroid check) and ends
Ptop = np.c_[spf("axis_u_at_top"), spf("v_top")]; Pbot = np.c_[spf("axis_u_at_bot"), spf("v_bot")]
thS = np.radians(spf("tilt_extent_deg")); thS_c = np.radians(spf("tilt_centroid_deg"))
Ptop0 = stab(Ptop); Pbot0 = stab(Pbot); thS0 = stab_dir_angle(Pbot, thS); thS0_c = stab_dir_angle(Pbot, thS_c)
sigThS = np.sqrt(np.radians(spf("sig_tilt_deg")) ** 2 + local_rms(thS - thS_c) ** 2 / 2)
Lbody0 = np.linalg.norm(Pbot0 - Ptop0, axis=1)
# ---- pivot P_eff: least-squares intersection of all stabilized spring-axis lines (frame-0 px)
A_ = []; b_ = []
for k in range(N):
    d = np.array([np.sin(thS0[k]), np.cos(thS0[k])]); nn = np.array([d[1], -d[0]]); A_.append(nn); b_.append(nn @ Pbot0[k])
A_ = np.array(A_); b_ = np.array(b_); Peff = np.linalg.lstsq(A_, b_, rcond=None)[0]
rP = A_ @ Peff - b_; covP = np.linalg.inv(A_.T @ A_) * np.var(rP)
# ring-based pivot: centre of the circle best fitting the stabilized ring path (radius free) - check
def circ(p): return np.linalg.norm(cR0 - p[:2], axis=1) - p[2]
cfit = least_squares(circ, [Peff[0], Peff[1], 700.0]); Pring_circle = cfit.x
# static hook (landmark track, frame-0 geometry)
hook = [r for r in csv.DictReader(open(os.path.join(MEAS, "static_landmarks.csv"))) if r["landmark"] == "hook_top"]
hook_res = np.array([[float(r["resid_u_after_drift_px"]), float(r["resid_v_after_drift_px"])] for r in hook])
# ---- modes (image, frame-0 geometry) using the ring centre (spring lower attachment) about P_eff
dv = cR0 - Peff; phi = np.arctan2(dv[:, 0], dv[:, 1]); rr = np.linalg.norm(dv, axis=1)
sig_phi = np.hypot(sigR[:, 0], sigR[:, 1]) / rr
dvW = cW0 - Peff; phiW = np.arctan2(dvW[:, 0], dvW[:, 1]); rW = np.linalg.norm(dvW, axis=1)
psi = thW0 - phi            # weight tilt relative to the pivot->ring line (rocking of the weight)
psiS = thW0 - thS0          # weight tilt relative to the spring axis
chi = thS0 - phi            # spring axis vs pivot->ring line (spring bending / hook kink)
# ---- turning points & zero crossings of a signal (local quadratic fits +-3 frames)
def turning_points(y, sy, win=7):
    """extrema that are the max/min of y within +-win frames (rejects noise wiggles), refined by a weighted quadratic
    over +-3 frames."""
    out = []
    for k in range(3, N - 3):
        seg = y[max(k - win, 0):k + win + 1]
        if y[k] == seg.max() or y[k] == seg.min():
            sl = slice(k - 3, k + 4); tt = t[sl] - t[k]; A = np.c_[np.ones(7), tt, tt ** 2]
            wgt = 1 / np.maximum(sy[sl], 1e-6)
            co, *_ = np.linalg.lstsq(A * wgt[:, None], y[sl] * wgt, rcond=None)
            if co[2] == 0: continue
            te = -co[1] / (2 * co[2]); ye = co[0] + co[1] * te + co[2] * te ** 2
            if abs(te) > 1.5 / FPS: continue
            cov = np.linalg.inv((A * wgt[:, None]).T @ (A * wgt[:, None])); res_ = y[sl] - A @ co
            s2 = max((res_ ** 2 * wgt ** 2).sum() / 4, 1.0)
            g = np.array([1, te, te ** 2]); sye = np.sqrt(g @ cov @ g * s2)
            # time sd: dt = -c1/(2c2)
            J = np.array([0, -1 / (2 * co[2]), co[1] / (2 * co[2] ** 2)]); ste = np.sqrt(J @ cov @ J * s2)
            out.append(dict(t=t[k] + te, y=ye, sy=sye, st=ste, kind="max" if co[2] < 0 else "min", frame=k))
    # merge duplicates
    res = []
    for o in out:
        if res and abs(o["t"] - res[-1]["t"]) < 0.1 and o["kind"] == res[-1]["kind"]: continue
        res.append(o)
    return res
def zero_crossings(y, sy, c=0.0):
    z = []
    for k in range(N - 1):
        if (y[k] - c) * (y[k + 1] - c) < 0:
            sl = slice(max(k - 2, 0), min(k + 4, N)); co = np.polyfit(t[sl], y[sl] - c, 2); rts = np.roots(co)
            rts = rts[np.isreal(rts)].real; rts = rts[(rts >= t[k] - 1e-9) & (rts <= t[k + 1] + 1e-9)]
            tz = rts[0] if len(rts) else t[k] + (c - y[k]) / (y[k + 1] - y[k]) / FPS
            slope = np.polyval(np.polyder(co), tz); z.append(dict(t=tz, st=np.mean(sy[sl]) / abs(slope), dir="up" if slope > 0 else "down"))
    return z
def half_cycles(tp):
    hc = []
    for a, b in zip(tp[:-1], tp[1:]):
        if a["kind"] == b["kind"]: continue
        amp = abs(b["y"] - a["y"]) / 2; samp = 0.5 * np.hypot(a["sy"], b["sy"])
        hc.append(dict(t_start=a["t"], t_end=b["t"], t_mid=0.5 * (a["t"] + b["t"]), half_period=b["t"] - a["t"], s_half_period=np.hypot(a["st"], b["st"]),
                       amp=amp, s_amp=samp, mid=0.5 * (a["y"] + b["y"])))
    return hc
phid = np.degrees(phi); sphid = np.degrees(sig_phi)
# smoothing-free signals; mean level for lateral swing = median (rest direction of the swing in the image)
phi_c = float(np.mean(phid))  # mean level (a median would coincide with a sample)
tp_phi = turning_points(phid, sphid); zc_phi = zero_crossings(phid, sphid, phi_c); hc_phi = half_cycles(tp_phi)
srr = np.hypot(sigR[:, 0], sigR[:, 1]) / np.sqrt(2)
# radial (spring-stretch) signal: remove the pendulum's geometric 2w content? -> report raw r and its spectrum
tp_r = turning_points(rr, srr, win=3); hc_r = half_cycles(tp_r)
psid = np.degrees(psi); spsi = np.degrees(np.sqrt(sigTh ** 2 + sig_phi ** 2))
tp_psi = turning_points(psid, spsi); hc_psi = half_cycles(tp_psi)
# ---- sinusoid fits (global) for periods / phases at t=0
def sine_fit(y, sy, T0, damped=False):
    best = None
    for ph in np.linspace(-np.pi, np.pi, 12, endpoint=False):
        p0 = [np.mean(y), np.ptp(y) / 2, 2 * np.pi / T0, ph] + ([0.0] if damped else [])
        f = lambda p: ((p[0] + p[1] * np.exp(-(p[4] if damped else 0) * t) * np.cos(p[2] * t + p[3])) - y) / sy
        r = least_squares(f, p0)
        if best is None or r.cost < best.cost: best = r
    J = best.jac; chi2 = 2 * best.cost / (N - len(best.x)); cov = np.linalg.inv(J.T @ J) * max(chi2, 1)
    return best.x, np.sqrt(np.diag(cov)), chi2
def spectrum(y, fmin=0.2, fmax=4.0):
    yy = (y - y.mean()) * np.hanning(N); fr = np.linspace(fmin, fmax, 3801)
    P = np.abs(np.exp(-2j * np.pi * fr[:, None] * t[None]) @ yy) ** 2
    from scipy.signal import find_peaks
    pk, _ = find_peaks(P, distance=60); pk = pk[np.argsort(P[pk])[::-1][:4]]
    return [(round(float(fr[k]), 4), round(float(P[k] / P.max()), 4)) for k in pk]
zc_up = [z["t"] for z in zc_phi]
# full-period estimate from zero crossings: linear fit of crossing times vs index (half period slope)
hp0 = np.median(np.diff(zc_up)); k_ = np.round((np.array(zc_up) - zc_up[0]) / hp0); co_zc, cov_zc = np.polyfit(k_, zc_up, 1, cov=True)
T_phi = 2 * co_zc[0]; sT_phi = 2 * np.sqrt(cov_zc[0, 0])
xphi, sxphi, chi_phi = sine_fit(phid, np.maximum(sphid, 0.05), T_phi, damped=True)
spec_phi = spectrum(phid); spec_r = spectrum(rr); spec_psi = spectrum(psid); spec_L = spectrum(Lbody0)
# radial model: r = r0 + a2 cos(2 w t + p2) + az cos(wz t + pz) with w from the swing fit; wz free (init from spectrum)
w_s = xphi[2]
def rmodel(p, tt=t): return p[0] + p[1] * np.cos(2 * w_s * tt + p[2]) + p[3] * np.cos(p[4] * tt + p[5])
best = None
for fz in [f for f, _ in spec_r[:3]]:
    for ph in np.linspace(-np.pi, np.pi, 6, endpoint=False):
        r_ = least_squares(lambda p: (rmodel(p) - rr) / np.maximum(srr, 0.2), [rr.mean(), 1, 0, 1, 2 * np.pi * fz, ph])
        if best is None or r_.cost < best.cost: best = r_
xr_fit = best.x; J = best.jac; chi_r = 2 * best.cost / (N - 6); sxr = np.sqrt(np.diag(np.linalg.inv(J.T @ J) * max(chi_r, 1)))
# ---- rocking (weight tilt) amplitude per swing half-cycle: weight tilt (absolute image tilt, and relative psi)
# evaluated at each swing turning time by a weighted quadratic over +-3 frames; half-cycle amplitude = half the
# difference between consecutive turning-time values. Avoids noise-driven turning points of the noisy tilt signal.
def value_at(y, sy, tq):
    k = int(round(tq * FPS)); sl = slice(max(k - 3, 0), min(k + 4, N)); tt = t[sl] - tq; A = np.c_[np.ones(len(tt)), tt, tt ** 2]; w = 1 / sy[sl]
    co, *_ = np.linalg.lstsq(A * w[:, None], y[sl] * w, rcond=None); res_ = y[sl] - A @ co; s2 = max((res_ ** 2 * w ** 2).sum() / max(len(tt) - 3, 1), 1.0)
    return co[0], np.sqrt(np.linalg.inv((A * w[:, None]).T @ (A * w[:, None]))[0, 0] * s2)
thWd = np.degrees(thW0); sthW = np.degrees(sigTh)
def amp_at_swing_tps(y, sy):
    vals = [value_at(y, sy, tp["t"]) for tp in tp_phi]; out = []
    for i in range(len(vals) - 1):
        out.append(dict(t_start=tp_phi[i]["t"], t_end=tp_phi[i + 1]["t"], amp=abs(vals[i + 1][0] - vals[i][0]) / 2, s_amp=0.5 * np.hypot(vals[i][1], vals[i + 1][1]),
                        mid=0.5 * (vals[i + 1][0] + vals[i][0])))
    return out
hc_tilt = amp_at_swing_tps(thWd, sthW); hc_psi_tp = amp_at_swing_tps(psid, spsi)
# coupling gain weight tilt vs swing angle per half-cycle
# ---- independent vertical-bob search: regress r on the swing-locked terms [1, phi, phi^2, phidot^2] and inspect
# the residual (an independent axial mode would remain in it)
phir = np.radians(phid - phi_c); phidot = np.gradient(phir, 1 / FPS)
Xr_ = np.c_[np.ones(N), phir, phir ** 2, phidot ** 2]; cr_, *_ = np.linalg.lstsq(Xr_ / srr[:, None], rr / srr, rcond=None)
r_res = rr - Xr_ @ cr_; spec_rres = spectrum(r_res)
bob = dict(regression_coef_px=dict(const=cr_[0], per_rad=cr_[1], per_rad2=cr_[2], per_rad2_s2=cr_[3]), residual_rms_px=float(np.sqrt(np.mean(r_res ** 2))),
           median_sigma_r_px=float(np.median(srr)), residual_chi2nu=float(np.mean((r_res / srr) ** 2)), residual_spectrum_peaks_hz=spec_rres,
           raw_r_rms_about_mean_px=float(rr.std()))
# ---- discontinuity screen: frame-to-frame second differences (acceleration) of stabilized tracks vs a robust scale,
# reported separately for n = 0 mod 4 (generator cadence) and other frames; jumps of shape metrics.
def accel_flags(P, name, thr=6.0):
    a = P[2:] - 2 * P[1:-1] + P[:-2]; a = np.linalg.norm(a, axis=1) if a.ndim > 1 else np.abs(a)
    sc = 1.4826 * np.median(np.abs(a - np.median(a))) + 1e-9; z = (a - np.median(a)) / sc
    idx = np.nonzero(z > thr)[0] + 1
    return dict(signal=name, robust_scale=float(sc), n_flagged=int(len(idx)), frames=[int(i) for i in idx], z=[round(float(z[i - 1]), 1) for i in idx],
                frac_flagged_mod4_0=float(np.mean(np.array(idx) % 4 == 0)) if len(idx) else None)
disc = [accel_flags(cR0, "ring_centre_stab"), accel_flags(cW0, "weight_centre_stab"), accel_flags(np.degrees(thW0), "weight_tilt_stab"),
        accel_flags(np.degrees(thS0), "spring_axis_tilt_stab"), accel_flags(Lbody0, "spring_body_length_stab"),
        accel_flags(np.array(W["lengthB"]) / sdrift, "weight_length_stab"), accel_flags(np.array(W["widthB"]) / sdrift, "weight_width_stab"),
        accel_flags(radR0, "ring_radius_stab")]
jumps = {}
for nm, y in (("spring_top_stab_u", Ptop0[:, 0]), ("spring_top_stab_v", Ptop0[:, 1]), ("weight_length_stab", np.array(W["lengthB"]) / sdrift),
              ("weight_width_stab", np.array(W["widthB"]) / sdrift), ("spring_body_len_stab", Lbody0), ("ring_radius_stab", radR0)):
    d1 = np.diff(y); jumps[nm] = dict(max_abs_step=float(np.abs(d1).max()), at_frame=int(np.argmax(np.abs(d1)) + 1), step_sd=float(d1.std()),
                                       first_second_half_mean=[float(y[:120].mean()), float(y[120:].mean())])
gaps_ = spf("n_open_gaps")
coil = dict(open_gaps_values=sorted(set(int(x) for x in gaps_)), frames_with_12=int((gaps_ == 12).sum()),
            mean_swing_when_12_deg=float(phid[gaps_ == 12].mean()), mean_swing_when_11_deg=float(phid[gaps_ == 11].mean()),
            n_transitions=int((np.diff(gaps_) != 0).sum()), dense_scallops_R_median=float(np.nanmedian(spf("n_dense_scallops_R"))),
            open_pitch_px_median=float(np.nanmedian(spf("open_pitch_px"))), dense_len_px_median=float(np.nanmedian(spf("dense_len_px"))))
# ---- write tracks
def wcsv(name, header, cols):
    with open(os.path.join(MEAS, name), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(header)
        for k in range(N): w.writerow([k, "%.6f" % t[k]] + ["%.4f" % c[k] for c in cols])
wcsv("weight_track.csv", ["frame", "t_s", "u_px", "v_px", "tilt_deg", "sigma_u_px", "sigma_v_px", "sigma_tilt_deg", "u_methodC_px", "v_methodC_px", "tilt_methodC_deg",
     "u_stab_frame0_px", "v_stab_frame0_px", "tilt_stab_frame0_deg", "width_px", "length_px", "width_stab_px", "length_stab_px", "ecc_scale_diag", "ecc_tilt_diag_deg"],
     [cB[:, 0], cB[:, 1], np.degrees(thB), sigW[:, 0], sigW[:, 1], np.degrees(sigTh), cC[:, 0] - offC[0], cC[:, 1] - offC[1], np.degrees(thC - dthC),
      cW0[:, 0], cW0[:, 1], np.degrees(thW0), np.array(W["widthB"]), np.array(W["lengthB"]), np.array(W["widthB"]) / sdrift, np.array(W["lengthB"]) / sdrift,
      np.array(W["scaleA"]), np.degrees(W["thA"])])
wcsv("ring_track.csv", ["frame", "t_s", "u_px", "v_px", "sigma_u_px", "sigma_v_px", "u_methodB_px", "v_methodB_px", "radius_px", "u_stab_frame0_px", "v_stab_frame0_px", "radius_stab_px"],
     [cRA[:, 0], cRA[:, 1], sigR[:, 0], sigR[:, 1], cRB[:, 0] - offR[0], cRB[:, 1] - offR[1], Rg["radA"], cR0[:, 0], cR0[:, 1], radR0])
wcsv("modes_series.csv", ["frame", "t_s", "swing_angle_deg", "sigma_swing_deg", "radius_pivot_ring_px", "sigma_radius_px", "weight_rel_tilt_deg", "sigma_weight_rel_deg",
     "spring_axis_tilt_deg", "sigma_spring_tilt_deg", "spring_minus_swing_deg", "weight_minus_spring_deg", "spring_body_len_stab_px", "swing_angle_weight_centre_deg", "radius_pivot_weight_px"],
     [phid, sphid, rr, srr, psid, spsi, np.degrees(thS0), np.degrees(sigThS), np.degrees(chi), np.degrees(psiS), Lbody0, np.degrees(phiW), rW])
np.savez(os.path.join(CACHE, "analysis.npz"), t=t, phid=phid, sphid=sphid, rr=rr, srr=srr, psid=psid, spsi=spsi, cW0=cW0, cR0=cR0, Peff=Peff, thS0=thS0, Lbody0=Lbody0,
         thW0=thW0, cB=cB, cRA=cRA, Ptop=Ptop, Pbot=Pbot, Ptop0=Ptop0, Pbot0=Pbot0, sigW=sigW, sigR=sigR, dW=dW, dR=dR, dTh=dTh, dThS=thS - thS_c, sdrift=sdrift, chi=chi)
np.savez(os.path.join(CACHE, 'analysis_extra.npz'), r_res=r_res, thWd=thWd, sthW=sthW)
A = dict(
    pivot_Peff_frame0_px=[round(float(x), 2) for x in Peff], pivot_Peff_sd_px=[round(float(x), 2) for x in np.sqrt(np.diag(covP))], pivot_line_resid_rms_px=round(float(np.sqrt(np.mean(rP ** 2))), 2),
    pivot_ring_path_circle_frame0_px=[round(float(x), 2) for x in Pring_circle],
    hook_top_resid_after_drift_rms_px=round(float(np.sqrt(np.mean(np.sum(hook_res ** 2, 1)))), 3), hook_top_resid_max_px=round(float(np.max(np.hypot(*hook_res.T))), 3),
    weight_B_vs_C_rms_px=[round(float(x), 3) for x in np.sqrt(np.mean(dW ** 2, 0))], weight_B_vs_C_max_px=[round(float(x), 3) for x in np.abs(dW).max(0)],
    weight_tilt_B_vs_C_rms_deg=round(float(np.degrees(np.sqrt(np.mean(dTh ** 2)))), 3), weight_tilt_B_vs_C_max_deg=round(float(np.degrees(np.abs(dTh).max())), 3),
    weight_tilt_B_vs_C_const_offset_deg=round(float(np.degrees(dthC)), 3), weight_centre_B_vs_C_const_offset_px=[round(float(x), 3) for x in offC],
    weight_sigma_median_px=[round(float(x), 3) for x in np.median(sigW, 0)], weight_sigma_max_px=[round(float(x), 3) for x in sigW.max(0)],
    weight_tilt_sigma_median_deg=round(float(np.degrees(np.median(sigTh))), 3),
    ring_A_vs_B_rms_px=[round(float(x), 3) for x in np.sqrt(np.mean(dR ** 2, 0))], ring_A_vs_B_max_px=[round(float(x), 3) for x in np.abs(dR).max(0)],
    ring_sigma_median_px=[round(float(x), 3) for x in np.median(sigR, 0)],
    spring_tilt_extent_vs_centroid_rms_deg=round(float(np.degrees(np.sqrt(np.mean((thS - thS_c) ** 2)))), 3),
    swing_center_deg=round(float(phi_c), 3),
    swing_period_from_zero_crossings_s=[round(float(T_phi), 5), round(float(sT_phi), 5)],
    swing_damped_sine=dict(offset_deg=xphi[0], amp_deg=xphi[1], omega=xphi[2], period_s=2 * np.pi / xphi[2], s_period_s=2 * np.pi / xphi[2] ** 2 * sxphi[2],
                           phase_rad=xphi[3], decay_per_s=xphi[4], s_decay_per_s=sxphi[4], chi2nu=chi_phi),
    radial_fit=dict(r0_px=xr_fit[0], a_2w_px=xr_fit[1], phase_2w=xr_fit[2], a_z_px=xr_fit[3], omega_z=xr_fit[4], period_z_s=2 * np.pi / xr_fit[4],
                    s_period_z_s=2 * np.pi / xr_fit[4] ** 2 * sxr[4], phase_z=xr_fit[5], s=list(sxr), chi2nu=chi_r),
    spectra_peaks_hz=dict(swing=spec_phi, radius=spec_r, weight_rel_tilt=spec_psi, spring_body_len=spec_L),
    turning_points_swing=tp_phi, zero_crossings_swing=zc_phi, half_cycles_swing=hc_phi, turning_points_radius=tp_r, half_cycles_radius=hc_r,
    turning_points_weight_rel_tilt=tp_psi, half_cycles_weight_rel_tilt=hc_psi,
    spring_body_len_stab_px=dict(mean=float(Lbody0.mean()), sd=float(Lbody0.std()), min=float(Lbody0.min()), max=float(Lbody0.max())),
    half_cycles_weight_tilt_at_swing_tps=hc_tilt, half_cycles_weight_rel_tilt_at_swing_tps=hc_psi_tp, vertical_bob_search=bob,
    discontinuity_screen=disc, shape_jumps=jumps, coils=coil,
    ring_radius_stab_px=dict(mean=float(radR0.mean()), sd=float(radR0.std()), min=float(radR0.min()), max=float(radR0.max())),
)
def clean(o):
    if isinstance(o, dict): return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)): return round(float(o), 6)
    if isinstance(o, np.integer): return int(o)
    return o
json.dump(clean(A), open(os.path.join(MEAS, "analysis.json"), "w"), indent=1)
print(json.dumps(clean({k: v for k, v in A.items() if not k.startswith(("turning", "zero", "half"))}), indent=0)[:4000])
print("swing half-cycles:"); [print(" %.3f-%.3f amp %.3f+-%.3f  hp %.3f" % (h["t_start"], h["t_end"], h["amp"], h["s_amp"], h["half_period"])) for h in hc_phi]
print(json.dumps(clean(disc))[:3000]); print(json.dumps(clean(jumps))); print(json.dumps(clean(coil)))
print("tilt@swingTP half-cycles:"); [print(" %.3f-%.3f amp %.3f+-%.3f mid %.2f" % (h["t_start"], h["t_end"], h["amp"], h["s_amp"], h["mid"])) for h in hc_tilt]
print("psi@swingTP:"); [print(" %.3f-%.3f amp %.3f+-%.3f mid %.2f" % (h["t_start"], h["t_end"], h["amp"], h["s_amp"], h["mid"])) for h in hc_psi_tp]
print("bob", json.dumps(clean(bob)))
print("radius half-cycles:"); [print(" %.3f-%.3f amp %.3f+-%.3f  hp %.3f mid %.2f" % (h["t_start"], h["t_end"], h["amp"], h["s_amp"], h["half_period"], h["mid"])) for h in hc_r]
