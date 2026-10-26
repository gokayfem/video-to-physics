"""Per-frame image-space camera drift (clip 02). Input: measurement/cache/H_ecc.npy (8-dof ECC homography of each
frame vs frame 0 on the static-scene mask, frame-0 px -> frame-n px, contract convention; measurement/scripts/s1_drift.py).
Per frame: a 4-dof similarity x_n = s R(rho) x_0 + t fitted to a 20-px grid of static-mask points mapped by H_n
(least squares), and its max/rms deviation from H_n over the static region (projective/parallax part).
Independent check: the same similarity fitted to the tracked static landmarks (measurement/static_landmarks.csv,
direct ZNCC tracks; texture patches with unreliable tracks excluded) -> landmark rms.
Smooth model: cubic polynomials in t for s, rho, tx, ty; adopted if smooth-vs-raw deviation < 0.3 px at the image corners.
Writes camera/out/drift_similarity_raw.csv and camera/out/drift_fit.json."""
import numpy as np, pathlib, csv, json, sys
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "measurement/scripts"))
from common import static_mask, Happly
H = np.load(ROOT / "measurement/cache/H_ecc.npy"); N = len(H); t = np.arange(N) / 24
M = static_mask(); ys, xs = np.mgrid[10:1078:20, 10:1910:20]; sel = M[ys, xs] > 0; G = np.c_[xs[sel], ys[sel]] + 0.5
LMK_OK = ["postL_top_outer", "postL_top_inner", "bar_L_top", "bar_L_bottom", "bar_R_top", "bar_R_bottom", "postR_top_inner", "postR_top_outer",
          "postL_bottom_inner", "postR_bottom_inner", "plate_L_corner", "plate_R_corner", "screw_L"]
L = {}
for r in csv.DictReader(open(ROOT / "measurement/static_landmarks.csv")):
    if r["landmark"] in LMK_OK: L.setdefault(r["landmark"], []).append([float(r["u_raw_direct_px"]), float(r["v_raw_direct_px"])])
LM = np.array([L[k] for k in LMK_OK]).transpose(1, 0, 2)  # (N, K, 2)
def sim_fit(a, b):
    A = np.zeros((2 * len(a), 4)); A[0::2] = np.c_[a[:, 0], -a[:, 1], np.ones(len(a)), np.zeros(len(a))]; A[1::2] = np.c_[a[:, 1], a[:, 0], np.zeros(len(a)), np.ones(len(a))]
    p = np.linalg.lstsq(A, b.ravel(), rcond=None)[0]; res = (A @ p - b.ravel()).reshape(-1, 2)
    return np.hypot(p[0], p[1]), np.arctan2(p[1], p[0]), p[2], p[3], res
def apply(P, s, r, tx, ty):
    c, sn = np.cos(r), np.sin(r); return s * (P @ np.array([[c, sn], [-sn, c]])) + [tx, ty]
rows = []
for n in range(N):
    q = Happly(H[n], G); s, r, tx, ty, res = sim_fit(G, q); d = np.hypot(*res.T)
    s2, r2, tx2, ty2, res2 = sim_fit(LM[0], LM[n]); d_l = np.hypot(*(apply(LM[0], s, r, tx, ty) - LM[n]).T)
    rows.append([n, t[n], s, r, tx, ty, float(np.sqrt(np.mean(d ** 2))), float(d.max()), float(np.sqrt(np.mean(d_l ** 2))), float(d_l.max()), s2, float(np.sqrt(np.mean(np.sum(res2 ** 2, 1))))])
R = np.array(rows)
polys = {k: np.polyfit(t, R[:, i], 3) for k, i in (("scale_poly", 2), ("roll_poly", 3), ("tx_poly", 4), ("ty_poly", 5))}
corners = np.array([[0, 0], [1920, 0], [0, 1088], [1920, 1088], [960, 544]], float)
dev = [float(np.abs(apply(corners, *R[n, 2:6]) - apply(corners, *[np.polyval(polys[k], t[n]) for k in ("scale_poly", "roll_poly", "tx_poly", "ty_poly")])).max()) for n in range(N)]
with open(ROOT / "camera/out/drift_similarity_raw.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["frame", "t_s", "scale", "roll_rad", "tx_px", "ty_px", "H_vs_sim_rms_px", "H_vs_sim_max_px", "landmarks_vs_sim_rms_px", "landmarks_vs_sim_max_px",
                                    "landmark_only_similarity_scale", "landmark_only_similarity_rms_px", "smooth_vs_raw_maxdev_px"])
    for r_, dv in zip(R, dev): w.writerow([int(r_[0])] + ["%.8g" % x for x in r_[1:]] + ["%.4f" % dv])
fx = R[:, 4] / (1 - R[:, 2] + 1e-12); fy = R[:, 5] / (1 - R[:, 2] + 1e-12); ss = np.abs(1 - R[:, 2]) > 0.01
out = dict(scale_frame240=float(R[-1, 2]), scale_pct_per_s=float(np.polyfit(t, R[:, 2], 1)[0] * 100), roll_max_abs_rad=float(np.abs(R[:, 3]).max()),
           zoom_fixed_point_median_px=[float(np.median(fx[ss])), float(np.median(fy[ss]))], H_vs_sim_rms_median_px=float(np.median(R[:, 6])),
           H_vs_sim_max_px=float(R[:, 7].max()), landmarks_vs_sim_rms_median_px=float(np.median(R[:, 8])), landmarks_vs_sim_rms_max_px=float(R[:, 8].max()),
           landmark_only_sim_rms_median_px=float(np.median(R[:, 11])), smooth_vs_raw_maxdev_px=float(max(dev)), polys={k: v.tolist() for k, v in polys.items()},
           tx_range_px=[float(R[:, 4].min()), float(R[:, 4].max())], ty_range_px=[float(R[:, 5].min()), float(R[:, 5].max())], landmarks_used=LMK_OK)
json.dump(out, open(ROOT / "camera/out/drift_fit.json", "w"), indent=1); print(json.dumps({k: v for k, v in out.items() if k != "polys"}, indent=0))
