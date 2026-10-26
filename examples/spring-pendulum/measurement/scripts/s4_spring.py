"""Stage 4: spring geometry per frame (helical coil body between the upper S-hook and the lower hook).
Coil mask = dark wire pixels vs the bright smooth backdrop (luma < local backdrop level - 28, backdrop level = 95th
percentile of a 61-px row neighbourhood) OR strong gradient (> 18 /px). Per image row inside the coil body:
left/right extent of the mask (outermost runs) -> row midpoint (method 'extent'); and gradient-energy centroid
(method 'centroid'). Robust line fits (Tukey) of midpoints vs v give two independent spring axes; axis angle =
image angle of the downward axis from image vertical (+ = bottom toward +u). Along the 'extent' axis: coil-body
ends (first/last row with mask width > 40 % of the median), apparent body length, and the luma profile along
the axis (+-30 % of the half-width) -> dark troughs = wire crossings -> coil count, pitch per region (open / close-
wound). Coil-count changes between frames are flagged."""
import cv2, numpy as np, os, sys, json, csv
sys.path.insert(0, os.path.dirname(__file__))
from common import *
from scipy.signal import find_peaks
from scipy.ndimage import maximum_filter1d
F = load_frames(); H = np.load(os.path.join(CACHE, "H_ecc.npy"))
def tukey_fit(x, y, it=20):
    w = np.ones(len(x))
    for _ in range(it):
        A = np.c_[np.ones_like(x), x]; coef = np.linalg.lstsq(A * w[:, None] ** .5, y * w ** .5, rcond=None)[0]
        r = y - A @ coef; sc = 1.4826 * np.median(np.abs(r[w > 0])) + 0.2
        w = np.where(np.abs(r) < 4.685 * sc, (1 - (r / (4.685 * sc)) ** 2) ** 2, 0)
    cov = np.linalg.inv((A * w[:, None]).T @ A) * (w * r ** 2).sum() / max(w.sum() - 2, 1)
    return coef, r, w, cov
rows = []; prof_store = []
prev = None
for n in range(N):
    L = cv2.GaussianBlur(luma(F[n]), (0, 0), 0.8)
    gx = cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3) / 8; gy = cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3) / 8; G = np.hypot(gx, gy)
    # geometry prior from frame-0 positions mapped by the drift (static hook) and the previous frame's axis
    top0 = Happly(H[n], [[957.0, 212.0]])[0]          # top of coil body (frame-0 px), under the S-hook
    if prev is None: a, b = 954.0 - 0.13 * 212, -0.13  # u = a + b v ; frame-0 rough axis
    else: a, b = prev
    v_lo, v_hi = int(top0[1]) - 12, int(top0[1]) + 520
    mids_e, mids_c, vv, widths = [], [], [], []
    for v in range(v_lo, min(v_hi, 1080)):
        uc = a + b * (v + 0.5); u0, u1 = int(uc - 75), int(uc + 75)
        row = L[v, u0:u1]; bg = np.percentile(row, 95)
        m = (row < bg - 28) | (G[v, u0:u1] > 18)
        idx = np.nonzero(m)[0]
        if len(idx) < 6: continue
        l, r = idx[0], idx[-1]
        if r - l < 12: continue
        mids_e.append(u0 + 0.5 * (l + r) + 0.5); widths.append(r - l)
        gg = G[v, u0:u1] ** 2; mids_c.append(u0 + 0.5 + (gg * np.arange(len(gg))).sum() / gg.sum()); vv.append(v + 0.5)
    vv = np.array(vv); widths = np.array(widths, float); mids_e = np.array(mids_e); mids_c = np.array(mids_c)
    # coil body = rows with width > 0.55 * robust max (excludes the thin hooks)
    wref = np.percentile(widths, 80); body = widths > 0.55 * wref
    # longest contiguous body run (allow 24-row gaps: open coils)
    ib = np.nonzero(body)[0]; runs = np.split(ib, np.nonzero(np.diff(vv[ib]) > 24)[0] + 1); run = max(runs, key=len)
    sel = np.zeros(len(vv), bool); sel[run[0]:run[-1] + 1] = True; sel &= body
    ce, re_, we, cove = tukey_fit(vv[sel], mids_e[sel]); cc_, rc, wc, covc = tukey_fit(vv[sel], mids_c[sel])
    prev = (ce[0], ce[1])
    th_e = np.arctan(ce[1]); th_c = np.arctan(cc_[1])
    v_top, v_bot = vv[run[0]] - 0.5, vv[run[-1]] + 0.5
    P_top = np.array([ce[0] + ce[1] * v_top, v_top]); P_bot = np.array([ce[0] + ce[1] * v_bot, v_bot])
    # along-axis luma profile (0.25 px) between the ends +-8 px
    d = (P_bot - P_top); Lax = np.linalg.norm(d); ax = d / Lax; pp = np.array([ax[1], -ax[0]])
    s = np.arange(-8, Lax + 8, 0.25); hw = 0.5 * np.median(widths[sel])
    X = np.linspace(-0.3 * hw, 0.3 * hw, 9)
    U = P_top[0] + s[None] * ax[0] + X[:, None] * pp[0]; V = P_top[1] + s[None] * ax[1] + X[:, None] * pp[1]
    pr = bilinear(L, U, V).mean(0)
    pk, _ = find_peaks(-pr, distance=int(4.5 / 0.25), prominence=6)
    sp = s[pk]; sp = sp[(sp > -2) & (sp < Lax + 2)]
    pitch = np.diff(sp)
    # split open / close-wound by pitch > 1.6 * min-region pitch
    pmed_small = np.median(np.sort(pitch)[:max(3, len(pitch) // 3)]) if len(pitch) else np.nan
    dense = pitch < 1.5 * pmed_small
    first_dense = int(np.argmax(dense)) if dense.any() else len(pitch)
    open_p = pitch[:first_dense]; dense_p = pitch[first_dense:]
    # Coil turns from the rectified silhouette: strip along the axis (0.25 px) x across (+-60 px, 0.5 px);
    # dark = luma < backdrop(s) - 30 where backdrop(s) = median of the 8 outermost columns each side.
    # Open section: rows with NO dark pixel are backdrop gaps between turns -> gap runs >= 2 px counted.
    # Close-wound section (after the last gap): the left/right silhouette edges are scalloped, one scallop per
    # turn -> local minima of the outer edge (prominence >= 0.8 px, distance >= 4 px) counted per side.
    Xr = np.arange(-60, 60.01, 0.5)
    U = P_top[0] + s[None] * ax[0] + Xr[:, None] * pp[0]; V = P_top[1] + s[None] * ax[1] + Xr[:, None] * pp[1]
    I = cv2.GaussianBlur(bilinear(L, U, V), (0, 0), 1.0); bgs = np.median(np.r_[I[:8], I[-8:]], 0); dark = I < bgs[None] - 30
    anyd = dark.any(0); xr = np.array([Xr[np.nonzero(c)[0][-1]] if c.any() else np.nan for c in dark.T]); xl = np.array([Xr[np.nonzero(c)[0][0]] if c.any() else np.nan for c in dark.T])
    inside = (s > 2) & (s < Lax - 2); gaps = []; k = 0
    while k < len(s):
        if inside[k] and not anyd[k]:
            j = k
            while j < len(s) and inside[j] and not anyd[j]: j += 1
            if (j - k) * 0.25 >= 2: gaps.append((s[k], s[j - 1]))
            k = j
        else: k += 1
    n_open_turns = len(gaps)  # number of backdrop gaps = open turns below the top turn group
    s_dense0 = gaps[-1][1] if gaps else np.nan
    dsel = (s > s_dense0 + 3) & (s < Lax - 3)
    nd = {}
    for nm, e in (("R", xr), ("L", -xl)):
        ee = e[dsel]
        if len(ee) < 40 or np.isnan(ee).any(): nd[nm] = (np.nan, np.nan); continue
        ee = cv2.GaussianBlur(ee.reshape(1, -1).astype(np.float32), (0, 0), 1.5).ravel()
        mins, _ = find_peaks(-ee, distance=int(4 / 0.25), prominence=0.8)
        nd[nm] = (len(mins) + 1, float(np.median(np.diff(s[dsel][mins]))) if len(mins) > 2 else np.nan)
    gc = np.array([0.5 * (g0 + g1) for g0, g1 in gaps])
    open_pitch = float(np.median(np.diff(gc))) if len(gc) > 2 else np.nan
    dense_len = Lax - s_dense0
    rows.append(dict(frame=n, t=n / FPS, axis_u_at_top=P_top[0], v_top=P_top[1], axis_u_at_bot=P_bot[0], v_bot=P_bot[1],
                     tilt_extent_deg=np.degrees(th_e), tilt_centroid_deg=np.degrees(th_c),
                     sig_tilt_deg=np.degrees(np.sqrt(cove[1, 1])) if np.isfinite(cove[1, 1]) else np.nan,
                     axis_resid_rms_px=float(np.sqrt(np.mean(re_[we > 0] ** 2))),
                     body_length_px=Lax, body_width_px=float(np.median(widths[sel])), n_axis_troughs=len(sp),
                     n_open_gaps=n_open_turns, open_pitch_px=open_pitch, s_dense_start_px=s_dense0, dense_len_px=dense_len,
                     n_dense_scallops_R=nd['R'][0], n_dense_scallops_L=nd['L'][0], dense_pitch_R_px=nd['R'][1], dense_pitch_L_px=nd['L'][1],
                     u_mid_dev_vs_centroid_axis_px=float(np.median(mids_e[sel] - (cc_[0] + cc_[1] * vv[sel])))))
    prof_store.append(dict(s=s.astype(np.float32), pr=pr.astype(np.float32), sp=sp, gaps=gaps, s_dense0=s_dense0, xr=xr.astype(np.float32), xl=xl.astype(np.float32), P_top=P_top, ax=ax))
    if n % 24 == 0:
        r_ = rows[-1]; print(n, "top(%.1f,%.1f) bot(%.1f,%.1f) tilt %.2f/%.2f len %.1f w %.1f gaps %d dense len %.1f scallops %s/%s" % (
            r_["axis_u_at_top"], r_["v_top"], r_["axis_u_at_bot"], r_["v_bot"], r_["tilt_extent_deg"], r_["tilt_centroid_deg"], r_["body_length_px"],
            r_["body_width_px"], r_["n_open_gaps"], r_["dense_len_px"], r_["n_dense_scallops_R"], r_["n_dense_scallops_L"]))
import pickle; pickle.dump(prof_store, open(os.path.join(CACHE, "spring_profiles.pkl"), "wb"))
with open(os.path.join(MEAS, "spring_track.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader()
    for r_ in rows: w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r_.items()})
for k_ in ("n_open_gaps", "open_pitch_px", "dense_len_px", "n_dense_scallops_R", "n_dense_scallops_L", "dense_pitch_R_px", "dense_pitch_L_px"):
    v_ = np.array([r_[k_] for r_ in rows], float); print(k_, "min %g median %g max %g" % (np.nanmin(v_), np.nanmedian(v_), np.nanmax(v_)), np.unique(np.round(v_[np.isfinite(v_)]), return_counts=True))
nc = np.array([r_["n_axis_troughs"] for r_ in rows]); print("crossings min/median/max", nc.min(), np.median(nc), nc.max(), "hist", np.bincount(nc)[nc.min():])
