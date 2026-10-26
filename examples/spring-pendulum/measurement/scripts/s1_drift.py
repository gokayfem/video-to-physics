"""Stage 1: camera/framing drift. Per-frame homography H_n mapping frame-0 pixel coords -> frame-n pixel
coords, estimated on the static scene (stand, arm, base, pin, background) with the swing region masked.
Init: chained KLT (forward-backward checked) homographies; refinement: direct ECC alignment of frame n
to frame 0 (no accumulation). Also fits a 4-dof similarity for interpretation."""
import cv2, numpy as np, json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from common import *
F = load_frames()
N = len(F)
M = static_mask()
g = [luma(F[n]) for n in range(N)]
# OpenCV uses pixel-centre-at-integer coords; convert: contract = opencv + 0.5. Homography conjugation:
T = np.array([[1, 0, 0.5], [0, 1, 0.5], [0, 0, 1.0]]); Ti = np.linalg.inv(T)
def to_contract(Hcv): return T @ Hcv @ Ti
# --- chained KLT init
H_chain = [np.eye(3)]
ninl = [0]
lk = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 0.01))
for n in range(1, N):
    a = g[n-1].astype(np.uint8); b = g[n].astype(np.uint8)
    p = cv2.goodFeaturesToTrack(a, 1500, 0.005, 8, mask=M)
    q, st, _ = cv2.calcOpticalFlowPyrLK(a, b, p, None, **lk)
    pb, st2, _ = cv2.calcOpticalFlowPyrLK(b, a, q, None, **lk)
    fb = np.linalg.norm((p - pb).reshape(-1, 2), axis=1)
    ok = (st.ravel() == 1) & (st2.ravel() == 1) & (fb < 0.3)
    H, inl = cv2.findHomography(p[ok], q[ok], cv2.RANSAC, 0.7)
    H_chain.append(H @ H_chain[-1]); ninl.append(int(inl.sum()))
# --- ECC refinement against frame 0 (opencv coordinates internally)
g0 = g[0] / 255.0
Hs = [np.eye(3)]; cc = [1.0]
crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-7)
for n in range(1, N):
    W = np.linalg.inv(H_chain[n]).astype(np.float32)  # ECC warp maps template(frame0)?? see below
    # findTransformECC(template, input, warp): finds warp s.t. input(warp(x)) ~ template(x), i.e. warp: frame0->frame n
    W = H_chain[n].astype(np.float32)
    try:
        c, W = cv2.findTransformECC(g0.astype(np.float32), (g[n] / 255.0).astype(np.float32), W,
                                    cv2.MOTION_HOMOGRAPHY, crit, M, 5)
    except cv2.error as e:
        print("ECC fail", n, e); c = np.nan; W = H_chain[n].astype(np.float32)
    Hs.append(W.astype(np.float64)); cc.append(float(c))
Hs = [to_contract(H / H[2, 2]) for H in Hs]
Hc = [to_contract(H / H[2, 2]) for H in H_chain]
np.save(os.path.join(CACHE, "H_ecc.npy"), np.array(Hs)); np.save(os.path.join(CACHE, "H_chain.npy"), np.array(Hc))
json.dump({"ecc_cc": cc, "klt_inliers": ninl}, open(os.path.join(CACHE, "drift_diag.json"), "w"))
P = np.array([[960, 544], [955, 60], [550, 950], [1380, 1000], [560, 70]], float)
for n in range(0, N, 20):
    a = cv2.perspectiveTransform(P.reshape(-1, 1, 2), Hs[n]).reshape(-1, 2) - P
    b = cv2.perspectiveTransform(P.reshape(-1, 1, 2), Hc[n]).reshape(-1, 2) - P
    print(n, "cc=%.4f" % cc[n], "ecc", np.round(a, 2).tolist(), "maxdiff_vs_chain %.2f" % np.abs(a - b).max())
