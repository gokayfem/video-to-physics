"""Stage 2: static landmarks over all frames.
For each landmark (frame-0 position p0, contract pixel coords) and each frame n:
  - warp frame n into frame-0 geometry with the ECC drift homography H_n (local window),
  - ZNCC-match the frame-0 patch in a +-10 px window, parabolic subpixel peak -> residual d (frame-0 coords),
  - raw position in frame n = H_n(p0 + d).
Also an H-independent raw track (template from frame 0 matched directly in raw frame n, scale ignored)
as a cross-check of the raw motion."""
import cv2, numpy as np, os, sys, csv, json
sys.path.insert(0, os.path.dirname(__file__))
from common import *
F = load_frames(); N = len(F)
H = np.load(os.path.join(CACHE, "H_ecc.npy"))
LM = [  # name, u0, v0, half-size, kind  (frame-0 contract px)
    ("postL_top_outer", 490.0, 56.0, 14, "left post top, outer (left) rounded corner"),
    ("postL_top_inner", 608.0, 56.0, 14, "left post top, inner (right) rounded corner"),
    ("bar_L_top", 622.0, 63.0, 12, "crossbar top edge meets left post"),
    ("bar_L_bottom", 626.0, 118.0, 12, "crossbar bottom edge meets left post"),
    ("bar_R_top", 1296.0, 63.0, 12, "crossbar top edge meets right post"),
    ("bar_R_bottom", 1294.0, 118.0, 12, "crossbar bottom edge meets right post"),
    ("postR_top_inner", 1302.0, 57.0, 14, "right post top, inner (left) rounded corner"),
    ("postR_top_outer", 1421.0, 57.0, 14, "right post top, outer (right) rounded corner"),
    ("postL_wood_mid", 550.0, 520.0, 16, "left post front face wood grain (texture patch)"),
    ("postR_wood_mid", 1360.0, 520.0, 16, "right post front face wood grain (texture patch)"),
    ("postL_bottom_inner", 627.0, 1002.0, 14, "left post inner edge meets base plate"),
    ("postR_bottom_inner", 1288.0, 1002.0, 14, "right post inner edge meets base plate"),
    ("screw_L", 516.0, 1042.0, 12, "left base screw head"),
    ("screw_R", 1402.0, 1042.0, 12, "right base screw head"),
    ("plate_L_corner", 482.0, 1050.0, 14, "base plate left front corner"),
    ("plate_R_corner", 1437.0, 1050.0, 14, "base plate right front corner"),
    ("table_L", 300.0, 960.0, 16, "table wood grain patch left"),
    ("table_R", 1640.0, 960.0, 16, "table wood grain patch right"),
    ("hook_top", 957.0, 52.0, 10, "spring top hook apex over the crossbar (NOT assumed static; tested)"),
]
g = [luma(F[n]) for n in range(N)]
def warp_local(img, Hn, u0, v0, R):
    """returns patch W(x) = img(Hn x) for frame-0 coords x in [u0-R,u0+R]x[v0-R,v0+R] at integer+0.5 centres"""
    xs = np.arange(-R, R + 1) + np.floor(u0) + 0.5; ys = np.arange(-R, R + 1) + np.floor(v0) + 0.5
    X, Y = np.meshgrid(xs, ys)
    P = np.stack([X.ravel(), Y.ravel(), np.ones(X.size)])
    Q = Hn @ P; Q = Q[:2] / Q[2]
    out = cv2.remap(img, (Q[0] - 0.5).reshape(X.shape).astype(np.float32), (Q[1] - 0.5).reshape(X.shape).astype(np.float32), cv2.INTER_CUBIC)
    return out, xs[0], ys[0]
def match(tpl, srch):
    r = cv2.matchTemplate(srch, tpl, cv2.TM_CCOEFF_NORMED)
    _, mx, _, (x, y) = cv2.minMaxLoc(r)
    dx = dy = 0.0
    if 0 < x < r.shape[1] - 1:
        a, b, c = r[y, x - 1], r[y, x], r[y, x + 1]; den = a - 2 * b + c; dx = 0.5 * (a - c) / den if den < 0 else 0
    if 0 < y < r.shape[0] - 1:
        a, b, c = r[y - 1, x], r[y, x], r[y + 1, x]; den = a - 2 * b + c; dy = 0.5 * (a - c) / den if den < 0 else 0
    return x + dx, y + dy, mx
SR = 10
rows = []
for name, u0, v0, h, desc in LM:
    tpl, tu, tv = warp_local(g[0], np.eye(3), u0, v0, h)
    for n in range(N):
        srch, su, sv = warp_local(g[n], H[n], u0, v0, h + SR)
        x, y, cc = match(tpl, srch)
        d = np.array([su + x - tu, sv + y - tv])  # residual in frame-0 coords
        p = np.array([u0 + d[0], v0 + d[1], 1.0]); q = H[n] @ p; q = q[:2] / q[2]
        # H-independent raw match (frame-0 template directly in raw frame n)
        srch2, su2, sv2 = warp_local(g[n], np.eye(3), u0, v0, h + 35)
        x2, y2, cc2 = match(tpl, srch2)
        rows.append([name, n, n / FPS, q[0], q[1], d[0], d[1], cc, su2 + x2 - tu + u0, sv2 + y2 - tv + v0, cc2])
os.makedirs(MEAS, exist_ok=True)
with open(os.path.join(MEAS, "static_landmarks.csv"), "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["landmark", "frame", "t_s", "u_px", "v_px", "resid_u_after_drift_px", "resid_v_after_drift_px", "zncc_drift_guided",
                "u_raw_direct_px", "v_raw_direct_px", "zncc_raw_direct", "method"])
    for r in rows:
        w.writerow([r[0], r[1], "%.6f" % r[2]] + ["%.3f" % x for x in r[3:7]] + ["%.4f" % r[7], "%.3f" % r[8], "%.3f" % r[9], "%.4f" % r[10],
                   "zncc_template_frame0_in_drift_warped_frame+parabolic_subpixel"])
json.dump([dict(name=a, u0=b, v0=c, half=d, desc=e) for a, b, c, d, e in LM], open(os.path.join(MEAS, "static_landmarks_def.json"), "w"), indent=1)
import collections
D = collections.defaultdict(list)
for r in rows: D[r[0]].append(r)
print("%-24s %8s %8s %8s %8s %8s %8s %6s" % ("landmark", "du_end", "dv_end", "maxdisp", "res_rms", "res_max", "dir_vs_g", "ccmin"))
for name, *_ in LM:
    A = np.array([r[1:] for r in D[name]], float)
    disp = np.hypot(A[:, 2] - A[0, 2], A[:, 3] - A[0, 3]); res = np.hypot(A[:, 4], A[:, 5])
    dd = np.hypot(A[:, 7] - A[:, 2], A[:, 8] - A[:, 3])
    print("%-24s %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f %6.3f" % (name, A[-1, 2] - A[0, 2], A[-1, 3] - A[0, 3], disp.max(), np.sqrt((res**2).mean()), res.max(), dd.max(), A[:, 6].min()))
