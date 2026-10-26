"""Stage 4: wire (suspension) per frame.
Centreline: for every image row between the loop and the bob top, the wire's u is the centroid of the
dark band (background-subtracted inverted luma above 30% of the dip depth; the wire is a ~5 px twisted
two-strand band, so the single darkest pixel jumps between strands) inside a +-6 px window around the
current line estimate (two passes: prior line pivot->bob, then the fitted line). Rows with dip depth < 25
grey levels are rejected. Straight line u = a + b (v - vm) fitted robustly (IRLS Huber).
Straightness: (i) max |deviation| of the 15-row running-mean centreline from the chord between the wire
ends, (ii) quadratic term -> sagitta s = c (dv/2)^2 with its 1-sigma, (iii) per-row RMS residual.
Top attachment: topmost wire row still detected continuously from below (row-to-row continuity < 1.5 px).
Bottom attachment: intersection of the wire line with the fitted bob circle (upper intersection)."""
import cv2, numpy as np, os, sys, csv, json
sys.path.insert(0, os.path.dirname(__file__))
from common import *
F = load_frames(); N = len(F)
H = np.load(os.path.join(CACHE, "H_ecc.npy")); B = np.load(os.path.join(CACHE, "bob_raw.npz"))
cA, rad = B["cA"], B["radA"]
def pv0(n, p=(1086.7, 125.0)):
    q = H[n] @ np.array([p[0], p[1], 1.0]); return q[:2] / q[2]
def centreline(L, line, v_rows, half=6):
    us = np.full(len(v_rows), np.nan); depth = np.zeros(len(v_rows))
    for i, v in enumerate(v_rows):
        uc = line(v + 0.0)
        x0 = int(np.floor(uc - 0.5)) - half
        row = L[int(v), x0:x0 + 2 * half + 1].astype(np.float64)
        if row.size < 2 * half + 1: continue
        bg = np.median(np.r_[L[int(v), x0 - 6:x0], L[int(v), x0 + 2 * half + 1:x0 + 2 * half + 7]])
        w = bg - row
        k = int(np.argmax(w))
        if k == 0 or k == len(w) - 1: continue
        depth[i] = w[k]
        if w[k] < 25: continue
        # the wire is a ~5 px two-strand (twisted) band: use the centroid of the dark band (pixels darker
        # than 30% of the dip depth, contiguous with the minimum), not the single darkest pixel
        thr = 0.3 * w[k]; lo = k; hi = k
        while lo - 1 >= 0 and w[lo - 1] > thr: lo -= 1
        while hi + 1 < len(w) and w[hi + 1] > thr: hi += 1
        ww = w[lo:hi + 1] - thr; xx = np.arange(lo, hi + 1)
        us[i] = x0 + (ww * xx).sum() / ww.sum() + 0.5
    return us, depth
def robust_poly(v, u, deg):
    vm = v.mean(); X = np.vander(v - vm, deg + 1)
    w = np.ones(len(v))
    for it in range(20):
        Wm = np.sqrt(w)[:, None]
        coef, *_ = np.linalg.lstsq(X * Wm, u * Wm.ravel(), rcond=None)
        r = u - X @ coef; s = 1.4826 * np.median(np.abs(r)) + 1e-3
        w = np.where(np.abs(r) < 1.345 * s, 1.0, 1.345 * s / np.abs(r))
    dof = len(v) - deg - 1
    s2 = (w * r ** 2).sum() / dof
    cov = np.linalg.inv((X * w[:, None]).T @ X) * s2
    return coef, cov, r, vm, w
rows = []; lines0 = []; diag = {}
for n in range(N):
    L = luma(F[n]); P0 = pv0(n); c = cA[n]; R = rad[n]
    d = (P0 - c) / np.linalg.norm(P0 - c); btop = c + R * d
    lin = lambda v, P0=P0, btop=btop: P0[0] + (btop[0] - P0[0]) * (v - P0[1]) / (btop[1] - P0[1])
    vr = np.arange(int(P0[1]) + 12, int(btop[1]) - 3)
    us, dep = centreline(L, lin, vr, half=9)
    ok = np.isfinite(us)
    coef, cov, r, vm, w = robust_poly(vr[ok] + 0.5, us[ok], 1)
    lin2 = lambda v: coef[0] * (v - vm) + coef[1]
    vr = np.arange(int(P0[1]) - 6, int(btop[1]) + 2)
    us, dep = centreline(L, lin2, vr, half=5)
    ok = np.isfinite(us) & (np.abs(us - lin2(vr + 0.5)) < 3.0)
    # continuity from the middle upward/downward to find detected ends
    # longest run of detected rows allowing gaps <= 4 rows and steps < 1.5 px between successive detections
    idx = np.where(ok)[0]; runs = [[idx[0]]]
    for i in idx[1:]:
        j = runs[-1][-1]
        if i - j <= 5 and abs(us[i] - us[j]) < 1.5 + 0.3 * (i - j): runs[-1].append(i)
        else: runs.append([i])
    run = max(runs, key=len)
    sel = np.zeros_like(ok); sel[run] = True
    vv = vr[sel] + 0.5; uu = us[sel]
    coef, cov, r, vm, w = robust_poly(vv, uu, 1)
    q, qcov, rq, vmq, wq = robust_poly(vv, uu, 2)
    span = vv.max() - vv.min()
    sag = q[0] * (span / 2) ** 2; sag_sd = np.sqrt(qcov[0, 0]) * (span / 2) ** 2
    # chord between ends of smoothed centreline
    k = 15
    sm = np.convolve(uu, np.ones(k) / k, mode="valid"); vs = np.convolve(vv, np.ones(k) / k, mode="valid")
    chord = sm[0] + (sm[-1] - sm[0]) * (vs - vs[0]) / (vs[-1] - vs[0])
    maxdev_s = np.abs(sm - chord).max()
    maxdev_line = np.abs(rq + (np.vander(vv - vmq, 3) @ q) - (np.vander(vv - vm, 2) @ coef)).max() if False else np.abs(r).max()
    top_pt = np.array([uu[0], vv[0]])
    # bottom attachment: line u = coef0 (v-vm) + coef1 intersect circle, upper root
    # param v: (coef0(v-vm)+coef1 - cu)^2 + (v - cv)^2 = R^2
    a0, a1 = coef
    A_ = a0 ** 2 + 1; B_ = 2 * a0 * (a1 - a0 * vm - c[0]) - 2 * c[1]; C_ = (a1 - a0 * vm - c[0]) ** 2 + c[1] ** 2 - R ** 2
    disc = B_ ** 2 - 4 * A_ * C_
    vb = (-B_ - np.sqrt(max(disc, 0))) / (2 * A_); ub = a0 * (vb - vm) + a1
    # perpendicular distance of bob centre from wire line
    nvec = np.array([1, -a0]) / np.hypot(1, a0)
    dist_c = (c[0] - (a0 * (c[1] - vm) + a1)) / np.hypot(1, a0)
    ang_wire = np.degrees(np.arctan(a0))  # angle of wire from image vertical, + = bottom toward +u... u increases with v
    rows.append(dict(frame=n, t_s=n / FPS, top_u=top_pt[0], top_v=top_pt[1], top_u_line=a0 * (vv[0] - vm) + a1,
                     bot_u=ub, bot_v=vb, length_px=float(np.hypot(ub - top_pt[0], vb - top_pt[1])),
                     n_rows=int(sel.sum()), detected_v_min=float(vv.min()), detected_v_max=float(vv.max()),
                     max_dev_smoothed_px=float(maxdev_s), max_abs_resid_line_px=float(np.abs(r).max()),
                     rms_resid_line_px=float(np.sqrt((r ** 2).mean())), sagitta_quad_px=float(sag), sagitta_sd_px=float(sag_sd),
                     wire_angle_deg=float(ang_wire), wire_angle_sd_deg=float(np.degrees(np.sqrt(cov[0, 0]))),
                     bobcentre_offset_from_wireline_px=float(dist_c), slope=a0, vm=vm, icpt=a1, cov=cov.tolist()))
    diag[n] = (vv, uu, r)
import pickle; pickle.dump(diag, open(os.path.join(CACHE, "wire_diag.pkl"), "wb"))
pickle.dump(rows, open(os.path.join(CACHE, "wire_rows.pkl"), "wb"))
A = lambda k: np.array([r[k] for r in rows])
for k in ["top_u", "top_v", "bot_u", "bot_v", "length_px", "n_rows", "max_dev_smoothed_px", "max_abs_resid_line_px", "rms_resid_line_px", "sagitta_quad_px", "sagitta_sd_px", "wire_angle_deg", "bobcentre_offset_from_wireline_px"]:
    x = A(k); print("%-36s min %9.3f med %9.3f max %9.3f" % (k, x.min(), np.median(x), x.max()))
