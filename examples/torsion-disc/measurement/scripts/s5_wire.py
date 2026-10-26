"""Stage 5: torsion wire line per frame (raw frame-n px). Rows from 8 px below the clamp bottom to 6 px above the disc
silhouette, every 2 px: dark-line centre = minimum of (row - running median 17 px) within +-8 px of the prediction,
parabolic sub-pixel, weight = contrast; robust (Huber-like) weighted line u = a + b (v - 400).
Top point W_top = line at the clamp-bottom height (frame-0 v = 192.5 mapped with the chained KLT homography);
bottom point W_bot = line meets the disc silhouette upper arc. Writes measurement/wire_track.csv."""
import numpy as np, cv2, os, sys, csv
from scipy.ndimage import median_filter
sys.path.insert(0, os.path.dirname(__file__))
from common import *
F = load_frames(); N = len(F); H = np.load(os.path.join(CACHE, "H_chain.npy")); E = np.load(os.path.join(CACHE, "disc_edges.npy"))
rows = []; a, b = 970.5, -0.0255
for n in range(N):
    g = luma(F[n]); top = Happly(H[n], [[976.5, 192.5]])[0]
    up = E[n][np.isfinite(E[n, :, 0])]; near = up[np.abs(up[:, 0] - (a + b * (up[:, 1] - 400))) < 15]; vdisc = near[:, 1].min()
    vs = np.arange(top[1] + 8, vdisc - 6, 2.0); us = []; ws = []
    for v in vs:
        u0 = a + b * (v - 400); i0 = int(u0) - 20; row = g[int(v), i0:i0 + 41]
        hp = row - median_filter(row, 17); c = np.arange(41) + i0 + 0.5
        m = np.abs(c - u0) <= 8; j = np.where(m)[0][np.argmin(hp[m])]
        if j in (0, 40): us.append(np.nan); ws.append(0); continue
        p, q, r = hp[j - 1], hp[j], hp[j + 1]; den = p - 2 * q + r; off = 0.5 * (p - r) / den if den > 0 else 0
        us.append(c[j] + np.clip(off, -1, 1)); ws.append(max(0.0, -q))
    us = np.array(us); ws = np.array(ws); ok = np.isfinite(us) & (ws > 4)
    w = ws.copy()
    for it in range(5):
        A = np.c_[np.ones(ok.sum()), vs[ok] - 400]; sw = np.sqrt(w[ok]); p_ = np.linalg.lstsq(A * sw[:, None], us[ok] * sw, rcond=None)[0]
        r = us - (p_[0] + p_[1] * (vs - 400)); s = max(0.2, 1.4826 * np.median(np.abs(r[ok]))); w = ws / np.maximum(1, np.abs(r) / (2 * s))
        ok = ok & (np.abs(r) < 4 * s + 0.5)
    a, b = p_; rms = float(np.sqrt(np.mean(r[ok] ** 2)))
    # bottom: intersection with the silhouette upper arc (silhouette points within 3 px of the line, smallest v)
    d = up[:, 0] - (a + b * (up[:, 1] - 400)); cand = up[np.abs(d) < 3]; vb = cand[:, 1].min(); ub = a + b * (vb - 400)
    sa = rms / np.sqrt(ok.sum()); sb = rms / np.sqrt(np.sum((vs[ok] - vs[ok].mean()) ** 2))
    rows.append([n, n / FPS, a + b * (top[1] - 400), top[1], ub, vb, np.degrees(np.arctan(b)), rms, int(ok.sum()), sa, np.degrees(sb), a, b])
    if n % 40 == 0: print(n, np.round(rows[-1][:9], 3))
with open(os.path.join(MEAS, "wire_track.csv"), "w", newline="") as fh:
    w_ = csv.writer(fh); w_.writerow(["frame", "t_s", "u_top_px", "v_top_px", "u_bot_px", "v_bot_px", "image_tilt_deg_(+=bottom_toward_+u_is_negative)", "line_rms_px", "n_rows", "sd_u400_px", "sd_tilt_deg", "u_at_v400_px", "du_dv"])
    for r in rows: w_.writerow([int(r[0])] + ["%.5f" % x for x in r[1:]])
