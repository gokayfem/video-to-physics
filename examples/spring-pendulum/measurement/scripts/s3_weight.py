"""Stage 3: weight (brushed-steel cylinder) pose per frame, two independent methods.
B 'silhouette_lines' (geometric): profiles perpendicular to the predicted axis at 41 stations; outermost strong
  |d luma| on each side -> robust (Tukey) line fits of the left and right silhouette generators; axis = their mean
  line, tilt = axis direction (image angle of the downward axis from image vertical, + = bottom toward +u),
  apparent width = generator separation. Top/bottom silhouette extrema along the axis from axis-parallel
  profiles (+-0.4 width); centre = midpoint. Own constant-velocity prediction.
A 'ecc_affine' (photometric): frame-0 luma template of the body (mask = frame-0 silhouette quad eroded 3 px),
  6-dof affine ECC of frame n vs frame 0, warm-started from frame n-1; tracked point = W(c0_B); tilt = rotation
  from the polar decomposition; isotropic scale sqrt(det). Independent of B except for the frame-0 reference point.
Constant A-B offsets are removed by the median (reference-point definition), as in clip 01."""
import cv2, numpy as np, os, sys, json
sys.path.insert(0, os.path.dirname(__file__))
from common import *
F = load_frames(); Lm = [cv2.GaussianBlur(luma(F[n]), (0, 0), 0.8) for n in range(N)]
def tukey_line(P, it=20):
    """fit u = a + b*s  (P columns s, x) robustly; returns a,b,resid,weights"""
    w = np.ones(len(P)); s, x = P[:, 0], P[:, 1]
    for _ in range(it):
        A = np.c_[np.ones_like(s), s]; W = w
        coef = np.linalg.lstsq(A * W[:, None] ** .5, x * W ** .5, rcond=None)[0]
        r = x - A @ coef; sc = 1.4826 * np.median(np.abs(r)) + 0.15
        w = np.where(np.abs(r) < 4.685 * sc, (1 - (r / (4.685 * sc)) ** 2) ** 2, 0)
    return coef, r, w
def method_B(L, c, th, hl, hw):
    a = np.array([np.sin(th), np.cos(th)]); p = np.array([np.cos(th), -np.sin(th)])
    S = np.linspace(-0.62 * hl, 0.62 * hl, 41); X = np.arange(-hw - 26, hw + 26.01, 0.25)
    U = c[0] + S[:, None] * a[0] + X[None] * p[0]; V = c[1] + S[:, None] * a[1] + X[None] * p[1]
    prof = bilinear(L, U, V); d = np.abs(np.gradient(prof, axis=1)) / 0.25
    edges = {}
    for side, sel in (("L", X < -0.55 * hw), ("R", X > 0.55 * hw)):
        pts = []
        for i in range(len(S)):
            di = np.where(sel, d[i], 0); m = di.max()
            if m < 3: continue
            ks = [k for k in range(1, len(X) - 1) if sel[k] and di[k] >= di[k - 1] and di[k] >= di[k + 1] and di[k] > 0.45 * m]
            if not ks: continue
            k = ks[0] if side == "L" else ks[-1]
            aa, bb, cc = di[k - 1], di[k], di[k + 1]; den = aa - 2 * bb + cc
            pts.append([S[i], X[k] + (0.5 * (aa - cc) / den * 0.25 if den < 0 else 0)])
        coef, r, w = tukey_line(np.array(pts)); edges[side] = (coef, r, w, np.array(pts))
    (aL, bL), (aR, bR) = edges["L"][0], edges["R"][0]
    off = (aL + aR) / 2; slope = (bL + bR) / 2; width = (aR - aL) / np.sqrt(1 + slope ** 2)
    c2 = c + off * p; th2 = th + np.arctan(slope)
    # along-axis extrema
    a2 = np.array([np.sin(th2), np.cos(th2)]); p2 = np.array([np.cos(th2), -np.sin(th2)])
    T = np.arange(-hl - 16, hl + 16.01, 0.25); Xs = np.r_[np.linspace(-0.8, -0.45, 5), np.linspace(0.45, 0.8, 5)] * width / 2
    U = c2[0] + T[None] * a2[0] + Xs[:, None] * p2[0]; V = c2[1] + T[None] * a2[1] + Xs[:, None] * p2[1]
    pr = bilinear(L, U, V); dd = np.abs(np.gradient(pr, axis=1)) / 0.25; tops = []; bots = []
    for row in dd:
        for lst, sel in ((tops, np.abs(T + hl) < 14), (bots, np.abs(T - hl) < 14)):
            di = np.where(sel, row, 0); k = int(np.argmax(di))
            if 0 < k < len(T) - 1:
                aa, bb, cc = di[k - 1], di[k], di[k + 1]; den = aa - 2 * bb + cc
                lst.append(T[k] + (0.5 * (aa - cc) / den * 0.25 if den < 0 else 0))
    tt, tb = np.median(tops), np.median(bots)
    c3 = c2 + 0.5 * (tt + tb) * a2
    rL = edges["L"][1][edges["L"][2] > 0]; rR = edges["R"][1][edges["R"][2] > 0]
    sig_line = np.sqrt((np.mean(rL ** 2) + np.mean(rR ** 2)) / 2 / max(min(len(rL), len(rR)) - 2, 1))
    sig_th = sig_line / (0.62 * hl) * np.sqrt(3 / len(S)) * 2
    return dict(c=c3, th=th2, width=width, length=tb - tt, sig_c_perp=sig_line, sig_th=sig_th,
                sig_c_par=0.5 * np.hypot(1.4826 * np.median(np.abs(np.array(tops) - tt)), 1.4826 * np.median(np.abs(np.array(bots) - tb))) / 3 + 0.1,
                edgeL=edges["L"][3], edgeR=edges["R"][3], wL=edges["L"][2], wR=edges["R"][2], top=tt, bot=tb)
# --- B over all frames
c = np.array([864.0, 876.0]); th = np.radians(-9.2); hl, hw = 93.0, 68.0
outB = []
for n in range(N):
    if n >= 2:
        c = 2 * outB[-1]["c"] - outB[-2]["c"]; th = 2 * outB[-1]["th"] - outB[-2]["th"]
    for it in range(3):
        r = method_B(Lm[n], c, th, hl, hw); c, th = r["c"], r["th"]
    hl, hw = r["length"] / 2, r["width"] / 2
    outB.append(r)
cB = np.array([o["c"] for o in outB]); thB = np.array([o["th"] for o in outB])
# --- A: ECC affine vs frame 0
c0 = cB[0]; th0 = thB[0]; hl0 = outB[0]["length"] / 2; hw0 = outB[0]["width"] / 2
a0 = np.array([np.sin(th0), np.cos(th0)]); p0 = np.array([np.cos(th0), -np.sin(th0)])
quad = np.array([c0 + sx * (hl0 - 4) * a0 + sy * (hw0 - 3) * p0 for sx, sy in ((-1, -1), (-1, 1), (1, 1), (1, -1))])
mask = np.zeros((1088, 1920), np.uint8); cv2.fillConvexPoly(mask, np.round(quad - 0.5).astype(np.int32), 255)
tpl = (Lm[0] / 255).astype(np.float32)
T = np.array([[1, 0, 0.5], [0, 1, 0.5], [0, 0, 1.0]]); Ti = np.linalg.inv(T)
# ROI version: template canvas 400x400 around the body of frame 0; input crop of frame n (+-230 px around the
# predicted centre). Warps are converted to full-frame OpenCV coordinates for storage.
tx0, ty0 = int(c0[0] - 200), int(c0[1] - 200); tx1, ty1 = tx0 + 400, ty0 + 400  # same-size canvases (ECC needs equal sizes)
tplC = tpl[ty0:ty1, tx0:tx1].copy(); mC = mask[ty0:ty1, tx0:tx1].copy()
def full_from_crop(Wc_, ix0, iy0):  # crop warp (tpl-crop -> input-crop) -> full warp (frame0 -> frame n), opencv coords
    A = np.vstack([Wc_, [0, 0, 1]]); return (np.array([[1, 0, ix0], [0, 1, iy0], [0, 0, 1.0]]) @ A @ np.array([[1, 0, -tx0], [0, 1, -ty0], [0, 0, 1.0]]))[:2]
def crop_from_full(Wf, ix0, iy0):
    A = np.vstack([Wf, [0, 0, 1]]); return (np.array([[1, 0, -ix0], [0, 1, -iy0], [0, 0, 1.0]]) @ A @ np.array([[1, 0, tx0], [0, 1, ty0], [0, 0, 1.0]]))[:2]
W = np.eye(2, 3); outA = []
for n in range(N):
    imgF = (Lm[n] / 255).astype(np.float32)
    if n > 0:
        Wp = outA[-1]["W"]; vel = (outA[-1]["c"] - outA[-2]["c"]) if n >= 2 else np.zeros(2)
        cpred = outA[-1]["c"] + vel
    else:
        Wp = np.eye(2, 3); vel = np.zeros(2); cpred = c0
    ix0, iy0 = int(cpred[0] - 200), int(cpred[1] - 200); ix0 = min(max(ix0, 0), 1920 - 400); iy0 = min(max(iy0, 0), 1088 - 400)
    img = imgF[iy0:iy0 + 400, ix0:ix0 + 400]
    # coarse translation: template warped by the previous affine (+ velocity), masked ZNCC +-40 px (own prediction only)
    Wcp = crop_from_full(Wp, ix0, iy0); Wcp[:, 2] += vel
    tw = cv2.warpAffine(tplC, Wcp, (img.shape[1], img.shape[0])); mw = cv2.warpAffine(mC, Wcp, (img.shape[1], img.shape[0]), flags=cv2.INTER_NEAREST)
    yy, xx = np.nonzero(mw); x0_, y0_, x1_, y1_ = xx.min(), yy.min(), xx.max() + 1, yy.max() + 1
    t_ = tw[y0_:y1_, x0_:x1_]; m_ = (mw[y0_:y1_, x0_:x1_] > 0).astype(np.float32); SR = 40
    sx0, sy0 = max(x0_ - SR, 0), max(y0_ - SR, 0)
    srch = img[sy0:y1_ + SR, sx0:x1_ + SR]
    r = np.nan_to_num(cv2.matchTemplate(srch, t_, cv2.TM_CCOEFF_NORMED, mask=m_), nan=-1)
    _, mx, _, (bx, by) = cv2.minMaxLoc(r)
    W = Wcp.copy(); W[0, 2] += bx + sx0 - x0_; W[1, 2] += by + sy0 - y0_
    W0 = W.astype(np.float32).copy()
    def valid(Wt):
        M2 = Wt[:, :2].astype(float); d = np.linalg.det(M2); sv = np.linalg.svd(M2, compute_uv=False)
        ok = 0.85 < np.sqrt(abs(d)) < 1.2 and sv[0] / sv[1] < 1.15 and d > 0
        cq = Wt[:, :2] @ (c0 - 0.5 - [tx0, ty0]) + Wt[:, 2]; cq0 = W0[:, :2] @ (c0 - 0.5 - [tx0, ty0]) + W0[:, 2]
        return ok and np.hypot(*(cq - cq0)) < 20
    cc = np.nan; stage = "coarse_only"
    for blur, mode in ((2.5, cv2.MOTION_EUCLIDEAN), (2.5, cv2.MOTION_AFFINE), (0.0, cv2.MOTION_AFFINE)):
        Wi = W.astype(np.float32).copy()
        if mode == cv2.MOTION_EUCLIDEAN:
            Uu, Ss, Vt = np.linalg.svd(Wi[:, :2]); Wi[:, :2] = Uu @ Vt
        try:
            a_ = cv2.GaussianBlur(tplC, (0, 0), blur) if blur else tplC; b_ = cv2.GaussianBlur(img, (0, 0), blur) if blur else img
            c_, Wi = cv2.findTransformECC(a_, b_, Wi, mode, (3, 300, 1e-7), mC, 5 if not blur else 1)
        except cv2.error as e:
            if n < 2: print("ECC error", n, mode, str(e)[-200:])
            continue
        if valid(Wi):
            W = Wi; stage = ("E" if mode == cv2.MOTION_EUCLIDEAN else "A") + str(blur)
            if not blur: cc = c_
    if not valid(W): W = W0
    W = full_from_crop(W.astype(np.float64), ix0, iy0)
    Wc = T @ np.vstack([W, [0, 0, 1]]) @ Ti
    q = Wc @ np.r_[c0, 1]; Mx = Wc[:2, :2]
    Uu, Ss, Vt = np.linalg.svd(Mx); R = Uu @ Vt; rot = np.arctan2(R[1, 0], R[0, 0])
    outA.append(dict(W=W.copy(), cc=float(cc), stage=stage, c=q[:2], th=th0 - rot, scale=float(np.sqrt(abs(np.linalg.det(Mx)))), aff=Wc[:2].copy()))
# --- C: 'canny_ransac' (geometric, independent implementation): Canny edges (sigma 1.2 blur, 25/60 thresholds),
# edge pixels with gradient direction within 25 deg of the predicted across-axis direction, outermost per 2-px
# along-axis bin on each side, RANSAC line (1.2 px) + least squares on inliers; top/bottom = median outermost
# edge with along-axis gradient inside |x|<0.6 hw within +-14 px of the predicted ends. Own constant-velocity
# prediction (no input from B). This is the PRIMARY second method; ECC (A) is kept as a photometric diagnostic.
rng = np.random.default_rng(0)
def ransac_line(sv, xv, thr=1.2, it=300):
    best = None
    for _ in range(it):
        i, j = rng.choice(len(sv), 2, replace=False)
        if abs(sv[i] - sv[j]) < 20: continue
        b_ = (xv[j] - xv[i]) / (sv[j] - sv[i]); a_ = xv[i] - b_ * sv[i]
        inl = np.abs(xv - (a_ + b_ * sv)) < thr
        if best is None or inl.sum() > best.sum(): best = inl
    A_ = np.c_[np.ones(best.sum()), sv[best]]; coef = np.linalg.lstsq(A_, xv[best], rcond=None)[0]
    r_ = xv[best] - A_ @ coef
    return coef, r_, best
def method_C(L8, gx, gy, c, th, hl, hw):
    a = np.array([np.sin(th), np.cos(th)]); p = np.array([np.cos(th), -np.sin(th)])
    x0_, y0_ = int(c[0] - 200), int(c[1] - 200)
    E = cv2.Canny(L8[y0_:y0_ + 400, x0_:x0_ + 400], 25, 60, L2gradient=True)
    yy, xx = np.nonzero(E); U = xx + x0_ + 0.5; V = yy + y0_ + 0.5
    G = np.c_[gx[yy + y0_, xx + x0_], gy[yy + y0_, xx + x0_]]; Gn = G / (np.linalg.norm(G, axis=1, keepdims=True) + 1e-9)
    S = (U - c[0]) * a[0] + (V - c[1]) * a[1]; X = (U - c[0]) * p[0] + (V - c[1]) * p[1]
    across = np.abs(Gn @ p) > np.cos(np.radians(25)); along = np.abs(Gn @ a) > np.cos(np.radians(35))
    res = {}
    for side, sg in (("L", -1), ("R", 1)):
        m = across & (np.abs(S) < 0.6 * hl) & (np.abs(X - sg * hw) < 20)
        sb = np.round(S[m] / 2).astype(int); xs_ = X[m]; ss_ = S[m]; keep_s, keep_x = [], []
        for b in np.unique(sb):
            k = sb == b; i = np.argmax(sg * xs_[k]); keep_s.append(ss_[k][i]); keep_x.append(xs_[k][i])
        res[side] = ransac_line(np.array(keep_s), np.array(keep_x))
    (aL, bL), (aR, bR) = res["L"][0], res["R"][0]
    off = (aL + aR) / 2; slope = (bL + bR) / 2; width = (aR - aL) / np.sqrt(1 + slope ** 2)
    c2 = c + off * p; th2 = th + np.arctan(slope)
    a2 = np.array([np.sin(th2), np.cos(th2)]); p2 = np.array([np.cos(th2), -np.sin(th2)])
    S = (U - c2[0]) * a2[0] + (V - c2[1]) * a2[1]; X = (U - c2[0]) * p2[0] + (V - c2[1]) * p2[1]
    tops = S[along & (np.abs(X) < 0.6 * width / 2) & (np.abs(S + hl) < 14)]; bots = S[along & (np.abs(X) < 0.6 * width / 2) & (np.abs(S - hl) < 14)]
    tt = np.percentile(tops, 10) if len(tops) else -hl; tb = np.percentile(bots, 90) if len(bots) else hl
    c3 = c2 + 0.5 * (tt + tb) * a2
    rr = np.r_[res["L"][1], res["R"][1]]; sig = np.sqrt(np.mean(rr ** 2) / max(len(rr) / 2 - 2, 1))
    return dict(c=c3, th=th2, width=width, length=tb - tt, sig_c_perp=sig, ninl=int(res["L"][2].sum() + res["R"][2].sum()))
outC = []; c = c0.copy(); th = th0; hl, hw = hl0, hw0
for n in range(N):
    L8 = np.clip(cv2.GaussianBlur(luma(F[n]), (0, 0), 1.2), 0, 255).astype(np.uint8)
    gx = cv2.Sobel(L8, cv2.CV_32F, 1, 0, ksize=3); gy = cv2.Sobel(L8, cv2.CV_32F, 0, 1, ksize=3)
    if n >= 2: c = 2 * outC[-1]["c"] - outC[-2]["c"]; th = 2 * outC[-1]["th"] - outC[-2]["th"]
    for it in range(3):
        r = method_C(L8, gx, gy, c, th, hl, hw); c, th = r["c"], r["th"]
    hl, hw = 0.8 * hl + 0.2 * r["length"] / 2, 0.8 * hw + 0.2 * r["width"] / 2
    outC.append(r)
cC = np.array([o["c"] for o in outC]); thC = np.array([o["th"] for o in outC])
cA = np.array([o["c"] for o in outA]); thA = np.array([o["th"] for o in outA])
np.savez(os.path.join(CACHE, "weight_raw.npz"), cA=cA, thA=thA, ccA=[o["cc"] for o in outA], scaleA=[o["scale"] for o in outA], affA=[o["aff"] for o in outA],
         cB=cB, thB=thB, cC=cC, thC=thC, widthC=[o["width"] for o in outC], lengthC=[o["length"] for o in outC], sigC=[o["sig_c_perp"] for o in outC], widthB=[o["width"] for o in outB], lengthB=[o["length"] for o in outB], sigperpB=[o["sig_c_perp"] for o in outB],
         sigparB=[o["sig_c_par"] for o in outB], sigthB=[o["sig_th"] for o in outB], quad0=quad)
import pickle; pickle.dump([(o["edgeL"], o["edgeR"], o["wL"], o["wR"], o["top"], o["bot"]) for o in outB], open(os.path.join(CACHE, "weight_edges.pkl"), "wb"))
d = cA - cB; off = np.median(d, 0)
print("A-B offset", off, "rms", np.sqrt(((d - off) ** 2).mean(0)), "max", np.abs(d - off).max(0))
dth = np.degrees(thA - thB); print("tilt A-B median %.3f rms %.3f max %.3f deg" % (np.median(dth), np.sqrt(np.mean((dth - np.median(dth)) ** 2)), np.abs(dth - np.median(dth)).max()))
d = cC - cB; offC = np.median(d, 0); print("C-B offset", offC, "rms", np.sqrt(((d - offC) ** 2).mean(0)), "max", np.abs(d - offC).max(0))
dth = np.degrees(thC - thB); print("tilt C-B median %.3f rms %.3f max %.3f deg" % (np.median(dth), np.sqrt(np.mean((dth - np.median(dth)) ** 2)), np.abs(dth - np.median(dth)).max()))
print("ECC cc min %.4f median %.4f" % (np.nanmin([o["cc"] for o in outA]), np.nanmedian([o["cc"] for o in outA])))
print("width B min/max", min(o["width"] for o in outB), max(o["width"] for o in outB), "length", min(o["length"] for o in outB), max(o["length"] for o in outB))
for n in range(0, N, 12): print(n, np.round(cB[n], 2), "%.2f" % np.degrees(thB[n]), np.round(cA[n] - off, 2), "%.2f" % np.degrees(thA[n]), "%.4f" % outA[n]["scale"], "%.3f" % outA[n]["cc"], "%.1f %.1f" % (outB[n]["width"], outB[n]["length"]))
