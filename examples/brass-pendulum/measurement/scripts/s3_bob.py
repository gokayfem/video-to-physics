"""Stage 3: bob centre per frame, two independent methods.
A (primary) 'radial_edge_circle': luma profiles along 180 rays from the current centre estimate; the
   silhouette edge on each ray = outermost strong dark->bright luma step in [0.72R, 1.30R] (bob rim is dark,
   wall/wood are brighter), subpixel by parabola on the derivative peak. Rays near the wire direction are
   excluded. Robust geometric circle fit (IRLS Tukey), 3 outer iterations re-centring the rays.
   Radius is FREE per frame (the apparent radius was found to vary with swing phase, so a fixed-radius
   constraint biases v). Pass 2 re-runs the rays from the pass-1 centre. An algebraic ellipse fit on the
   inlier edge points is stored as a shape diagnostic (axes, orientation).
B (check) 'annulus_zncc': ZNCC template match of a frame-0 bob template restricted to an annulus
   0.80R-1.15R (silhouette band, excludes the interior highlight), template rescaled by the drift scale,
   parabolic subpixel peak. Offset of B vs A is a constant (template reference point) and is removed by
   the median difference."""
import cv2, numpy as np, os, sys, json
sys.path.insert(0, os.path.dirname(__file__))
from common import *
from scipy.signal import savgol_filter
F = load_frames(); N = len(F)
H = np.load(os.path.join(CACHE, "H_ecc.npy"))
NR = 180
PHI = np.arange(NR) * 2 * np.pi / NR  # image angle, 0 = +u, pi/2 = +v (down)
def ray_edges(L, c, R, wire_dir):
    rr = np.arange(0.72 * R, 1.30 * R, 0.25)
    U = c[0] + np.outer(np.cos(PHI), rr); V = c[1] + np.outer(np.sin(PHI), rr)
    prof = bilinear(L, U, V)
    prof = cv2.GaussianBlur(prof.astype(np.float32), (9, 1), 1.5)  # along r only (0.375 px sigma... in samples)
    d = np.gradient(prof, axis=1) / 0.25
    edges = np.full(NR, np.nan); strength = np.zeros(NR)
    for i in range(NR):
        ang = np.angle(np.exp(1j * (PHI[i] - wire_dir)))
        if abs(ang) < np.radians(9):  # wire occludes / attaches here
            continue
        di = d[i]; m = di.max()
        if m < 4.0:
            continue
        # outermost local maximum with at least 50% of the max strength
        cand = [k for k in range(1, len(di) - 1) if di[k] >= di[k - 1] and di[k] >= di[k + 1] and di[k] > 0.5 * m]
        k = cand[-1]
        a, b, cc = di[k - 1], di[k], di[k + 1]; den = a - 2 * b + cc
        off = 0.5 * (a - cc) / den if den < 0 else 0.0
        edges[i] = rr[k] + off * 0.25; strength[i] = b
    return edges, strength
def fit_circle(P, w0, c0, R0=None, fixR=None):
    c = np.array(c0, float); R = R0 if fixR is None else fixR
    for it in range(30):
        dvec = P - c; dist = np.linalg.norm(dvec, axis=1); res = dist - R
        s = 1.4826 * np.median(np.abs(res - np.median(res))) + 0.05
        wt = w0 * np.where(np.abs(res) < 4.685 * s, (1 - (res / (4.685 * s)) ** 2) ** 2, 0)
        J = np.column_stack([-dvec[:, 0] / dist, -dvec[:, 1] / dist] + ([] if fixR else [-np.ones(len(P))]))
        A = J.T @ (wt[:, None] * J); b = -J.T @ (wt * res)
        dx = np.linalg.solve(A, b)
        c += dx[:2]
        if not fixR: R += dx[2]
        if np.abs(dx).max() < 1e-5: break
    dvec = P - c; dist = np.linalg.norm(dvec, axis=1); res = dist - R
    use = wt > 0
    dof = max(use.sum() - J.shape[1], 1)
    s2 = (wt * res ** 2).sum() / max(wt[use].mean(), 1e-9) / dof
    cov = np.linalg.inv(A) * s2 * wt[use].mean()
    return c, R, res, wt, cov
def pivot_guess(n):
    p = H[n] @ np.array([1086.7, 125.0, 1.0]); return p[:2] / p[2]
L = [luma(F[n]) for n in range(N)]
# ---- Method A, pass 1 (free radius)
c = np.array([1038.0, 840.0]); R = 55.0
A1 = []
for n in range(N):
    cn = c.copy()
    if n >= 2: cn = 2 * np.array(A1[-1][0]) - np.array(A1[-2][0])  # constant-velocity prediction
    Rn = R
    for outer in range(3):
        pv = pivot_guess(n); wd = np.arctan2(pv[1] - cn[1], pv[0] - cn[0])
        e, st = ray_edges(L[n], cn, Rn, wd)
        ok = np.isfinite(e)
        P = np.column_stack([cn[0] + e[ok] * np.cos(PHI[ok]), cn[1] + e[ok] * np.sin(PHI[ok])])
        cn, Rn, res, wt, cov = fit_circle(P, np.ones(ok.sum()), cn, Rn)
    A1.append((cn.copy(), Rn, res, wt, cov, ok.sum(), (wt > 0).sum(), P, e, st))
    R = Rn
radA = np.array([a[1] for a in A1])
# smoothed radius: drift scale * constant is the physical expectation; use robust smoothing of measured radius
radS = savgol_filter(radA, 49, 2)
# ---- Method A, pass 2 (free radius, re-centred rays)
outA = []
for n in range(N):
    cn = A1[n][0].copy(); Rn = A1[n][1]
    for outer in range(2):
        pv = pivot_guess(n); wd = np.arctan2(pv[1] - cn[1], pv[0] - cn[0])
        e, st = ray_edges(L[n], cn, Rn, wd)
        ok = np.isfinite(e)
        P = np.column_stack([cn[0] + e[ok] * np.cos(PHI[ok]), cn[1] + e[ok] * np.sin(PHI[ok])])
        cn, Rn, res, wt, cov = fit_circle(P, np.ones(ok.sum()), cn, Rn)
    el = cv2.fitEllipseDirect(P[wt > 0].astype(np.float32))
    outA.append(dict(c=cn, R=Rn, cov=cov, res_rms=float(np.sqrt((wt * res ** 2).sum() / wt.sum())), n_used=int((wt > 0).sum()), n_rays=int(ok.sum()),
                     P=P, wt=wt, ell=[el[0][0], el[0][1], el[1][0], el[1][1], el[2]]))
radA = np.array([o["R"] for o in outA]); radS = radA
cA = np.array([o["c"] for o in outA])
# ---- Method B: annulus ZNCC
s_drift = np.array([np.sqrt(abs(np.linalg.det(H[n][:2, :2]))) for n in range(N)])
c0 = cA[0]; R0 = radS[0]
h = int(np.ceil(1.2 * R0)) + 2
x0, y0 = int(np.floor(c0[0])), int(np.floor(c0[1]))
tplF = L[0][y0 - h:y0 + h + 1, x0 - h:x0 + h + 1].copy()
cref = np.array([x0 + 0.5, y0 + 0.5])  # template centre in contract coords
outB = []
predB = cref.copy()
def zncc_at(n, s, pc, SRB=14):
    tpl = cv2.resize(tplF, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    th, tw = tpl.shape
    yy, xx = np.mgrid[0:th, 0:tw]; rc = (tw - 1) / 2.0
    rr = np.hypot(xx - rc, yy - rc) / s
    # offset of template centre after resizing: map tplF centre index h -> h*s approximately; account via rc
    msk = ((rr > 0.80 * R0) & (rr < 1.15 * R0)).astype(np.uint8)
    # exclude wire band above (angle within 12 deg of up) to keep wire out of the score
    ang = np.degrees(np.arctan2(yy - rc, xx - rc))
    msk[(np.abs(ang + 90) < 12)] = 0
    sx, sy = int(round(pc[0] - 0.5 - rc - SRB)), int(round(pc[1] - 0.5 - rc - SRB))
    srch = L[n][sy:sy + th + 2 * SRB, sx:sx + tw + 2 * SRB]
    r = cv2.matchTemplate(srch, tpl, cv2.TM_CCORR_NORMED, mask=msk.astype(np.float32))
    # use ZNCC-like: TM_CCOEFF_NORMED with mask
    r = cv2.matchTemplate(srch, tpl, cv2.TM_CCOEFF_NORMED, mask=msk.astype(np.float32))
    r = np.nan_to_num(r, nan=-1)
    _, mx, _, (x, y) = cv2.minMaxLoc(r)
    dx = dy = 0.0
    if 0 < x < r.shape[1] - 1:
        a, b, cc_ = r[y, x - 1], r[y, x], r[y, x + 1]; den = a - 2 * b + cc_; dx = 0.5 * (a - cc_) / den if den < 0 else 0
    if 0 < y < r.shape[0] - 1:
        a, b, cc_ = r[y - 1, x], r[y, x], r[y + 1, x]; den = a - 2 * b + cc_; dy = 0.5 * (a - cc_) / den if den < 0 else 0
    return [sx + x + dx + rc + 0.5, sy + y + dy + rc + 0.5, mx, s]
SC = np.arange(0.94, 1.121, 0.01)
for n in range(N):
    if n >= 2: predB = 2 * np.array(outB[-1][:2]) - np.array(outB[-2][:2])
    elif n == 1: predB = np.array(outB[-1][:2])
    cand = [zncc_at(n, s, predB) for s in SC]
    k = int(np.argmax([c_[2] for c_ in cand]))
    # parabolic refinement of scale, then re-evaluate at refined scale
    if 0 < k < len(SC) - 1:
        a, b, c_ = cand[k - 1][2], cand[k][2], cand[k + 1][2]; den = a - 2 * b + c_
        sk = SC[k] + (0.5 * (a - c_) / den * 0.01 if den < 0 else 0)
    else:
        sk = SC[k]
    outB.append(zncc_at(n, sk, predB))
cB = np.array(outB)
ellA = np.array([o["ell"] for o in outA])
np.savez(os.path.join(CACHE, "bob_raw.npz"), cA=cA, radA=radA, radS=radS, cB=cB, s_drift=s_drift, ellA=ellA,
         covA=np.array([o["cov"] for o in outA]), resA=np.array([o["res_rms"] for o in outA]),
         nusedA=np.array([o["n_used"] for o in outA]), nraysA=np.array([o["n_rays"] for o in outA]))
import pickle; pickle.dump([(o["P"], o["wt"]) for o in outA], open(os.path.join(CACHE, "bob_edges.pkl"), "wb"))
dAB = cB[:, :2] - cA
off = np.median(dAB, axis=0)
print("A-B median offset", off, "rms after offset", np.sqrt(((dAB - off) ** 2).mean(axis=0)), "max", np.abs(dAB - off).max(axis=0))
print("radius free: min %.2f max %.2f ; B scale min %.3f max %.3f ; drift scale end %.4f" % (radA.min(), radA.max(), cB[:,3].min(), cB[:,3].max(), s_drift[-1]))
print("ellipse axes ratio min/max", (ellA[:,2]/ellA[:,3]).min(), (ellA[:,2]/ellA[:,3]).max())
print("resA rms median", np.median([o["res_rms"] for o in outA]), "max", max(o["res_rms"] for o in outA), "nused min", min(o["n_used"] for o in outA))
print("sigma fit (median)", np.median(np.sqrt(np.array([np.diag(o["cov"]) for o in outA])), axis=0))
print("B zncc min", cB[:, 2].min(), "median", np.median(cB[:, 2]))
for n in list(range(0, 40, 3))+[120,200,240]: print(n, np.round(cA[n], 2), round(radA[n], 2), np.round(cB[n, :2] - off, 2), round(cB[n,3],3), np.round(ellA[n],2))
