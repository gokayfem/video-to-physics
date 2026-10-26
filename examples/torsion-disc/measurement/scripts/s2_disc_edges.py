"""Stage 2: disc silhouette per frame (image space, raw frame-n px, contract convention).
Feature: normalised brass chroma Y = 100((R+G)/2 - B)/(mean(RGB)+10) (brass 30-40, backdrop/table/shadow ~ 0), Gaussian sigma 1 px.
Init: threshold 40 in an ROI around the disc, largest blob, cv2.fitEllipse.
Sub-pixel: 720 rays from the blob-ellipse centre (elliptically scaled), half-level crossing between the
inside level (median at rho 0.80-0.90) and the outside level (median at rho 1.12-1.25), linear interpolation.
Output: measurement/cache/disc_edges.npy (N, 720, 2) [nan where rejected], measurement/disc_ellipse_image.csv
(fitEllipse on the sub-pixel silhouette: centre, axes, angle; silhouette = top face + rim band)."""
import numpy as np, cv2, csv, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from common import *
F = load_frames(); N = len(F)
U0, V0, U1, V1 = 560, 470, 1380, 900
NA = 720; phi = np.arange(NA) * 2 * np.pi / NA
rho = np.linspace(0.75, 1.3, 221)
edges = np.full((N, NA, 2), np.nan); rows = []
for n in range(N):
    f = np.asarray(F[n], np.float32)[V0:V1, U0:U1]
    Y = 100 * (0.5 * (f[..., 2] + f[..., 1]) - f[..., 0]) / (f.mean(-1) + 10); Y = cv2.GaussianBlur(Y, (0, 0), 1.0)
    m = (Y > 15).astype(np.uint8); m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    k, lab, st, _ = cv2.connectedComponentsWithStats(m); i = 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])
    cnt, _ = cv2.findContours((lab == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    (cx, cy), (w, h), ang = cv2.fitEllipse(max(cnt, key=len))
    a, b = w / 2, h / 2; t = np.radians(ang); e1 = np.array([np.cos(t), np.sin(t)]); e2 = np.array([-np.sin(t), np.cos(t)])
    c = np.array([cx + 0.5, cy + 0.5])  # opencv -> contract (ROI coords)
    D = (a * np.cos(phi))[:, None] * e1 + (b * np.sin(phi))[:, None] * e2   # (NA,2)
    P = c[None, None] + rho[None, :, None] * D[:, None, :]                  # (NA, nr, 2)
    prof = bilinear(Y, P[..., 0], P[..., 1])
    inside = np.median(prof[:, (rho > 0.8) & (rho < 0.9)], 1); outside = np.median(prof[:, (rho > 1.12) & (rho < 1.25)], 1)
    half = 0.5 * (inside + outside)
    for j in range(NA):
        if inside[j] - outside[j] < 12: continue
        p = prof[j]; idx = np.where((p[:-1] >= half[j]) & (p[1:] < half[j]))[0]
        if len(idx) == 0: continue
        q = idx[np.argmin(np.abs(rho[idx] - 1.0))]
        r = rho[q] + (p[q] - half[j]) / (p[q] - p[q + 1]) * (rho[1] - rho[0])
        edges[n, j] = c + r * D[j] + [U0, V0]
    E = edges[n][np.isfinite(edges[n, :, 0])]
    (ex, ey), (ew, eh), ea = cv2.fitEllipse((E - 0.5).astype(np.float32))
    rows.append([n, n / FPS, ex + 0.5, ey + 0.5, max(ew, eh) / 2, min(ew, eh) / 2, ea if ew >= eh else ea - 90, len(E)])
    if n % 40 == 0: print(n, np.round(rows[-1], 2))
np.save(os.path.join(CACHE, "disc_edges.npy"), edges)
with open(os.path.join(MEAS, "disc_ellipse_image.csv"), "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["frame", "t_s", "u_c_px", "v_c_px", "semi_major_px", "semi_minor_px", "angle_deg_opencv", "n_edge_pts"])
    for r in rows: w.writerow([r[0]] + ["%.4f" % x for x in r[1:]])
