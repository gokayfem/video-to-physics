"""Stage 1b: drift model discrimination (zoom vs dolly). KLT features from frame 0 tracked directly into frame n
(initialised by the ECC homography), grouped by scene depth layer (frame front: posts+crossbar+base plate; table top;
backdrop wall). A pure zoom/pan (camera centre fixed) maps every layer by the same homography; a dolly gives
layer-dependent scale. Fits a similarity per layer and the global H residual per layer."""
import cv2, numpy as np, json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from common import *
F = load_frames(); H = np.load(os.path.join(CACHE, "H_ecc.npy"))
M = static_mask(); g0 = luma(F[0]).astype(np.uint8)
p = cv2.goodFeaturesToTrack(g0, 4000, 0.003, 7, mask=M).reshape(-1, 2) + 0.5  # contract coords
def layer(u, v):
    if (490 <= u <= 625 and 50 <= v <= 1000) or (1295 <= u <= 1425 and 50 <= v <= 1000): return "posts"
    if 620 <= u <= 1300 and 60 <= v <= 120: return "crossbar"
    if 475 <= u <= 1440 and v >= 985: return "base_plate"
    if v >= 845: return "table"
    if v <= 820 and (u < 470 or u > 1450 or (640 < u < 1280 and 130 < v < 820)): return "backdrop"
    return None
L = np.array([layer(u, v) for u, v in p], object); keep = L != None; p = p[keep]; L = L[keep]
lk = dict(winSize=(25, 25), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.005))
out = {"n_features": {k: int((L == k).sum()) for k in set(L)}, "frames": {}}
def sim_fit(a, b):
    A = []; y = []
    for (x0, y0), (x1, y1) in zip(a, b):
        A += [[x0, -y0, 1, 0], [y0, x0, 0, 1]]; y += [x1, y1]
    s, *_ = np.linalg.lstsq(np.array(A), np.array(y), rcond=None); return s
c = np.array([960., 544.])
for n in [40, 80, 120, 160, 200, 240]:
    gn = luma(F[n]).astype(np.uint8); q0 = Happly(H[n], p)
    q, st, _ = cv2.calcOpticalFlowPyrLK(g0, gn, (p - 0.5).astype(np.float32).reshape(-1, 1, 2), (q0 - 0.5).astype(np.float32).reshape(-1, 1, 2), flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **lk)
    pb, st2, _ = cv2.calcOpticalFlowPyrLK(gn, g0, q, None, **lk)
    q = q.reshape(-1, 2) + 0.5; fb = np.linalg.norm(pb.reshape(-1, 2) + 0.5 - p, axis=1)
    ok = (st.ravel() == 1) & (st2.ravel() == 1) & (fb < 0.3)
    rec = {}
    for k in sorted(set(L)):
        m = ok & (L == k)
        if m.sum() < 6: continue
        r = np.linalg.norm(q[m] - Happly(H[n], p[m]), axis=1)
        s = sim_fit(p[m] - c, q[m] - c); sc = float(np.hypot(s[0], s[1]))
        rec[k] = {"n": int(m.sum()), "H_resid_median_px": round(float(np.median(r)), 3), "H_resid_rms_px": round(float(np.sqrt(np.mean(np.minimum(r, 5) ** 2))), 3),
                  "similarity_scale": round(sc, 5), "tx": round(float(s[2]), 2), "ty": round(float(s[3]), 2)}
    out["frames"][n] = rec; print(n, {k: (v["n"], v["similarity_scale"], v["H_resid_median_px"]) for k, v in rec.items()})
json.dump(out, open(os.path.join(MEAS, "drift_parallax.json"), "w"), indent=1)
