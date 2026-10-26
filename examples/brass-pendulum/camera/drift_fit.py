"""Per-frame image-space camera drift (camera role).
Static stand features tracked from reference frame 120 (camera/track_static.py) are fitted per frame
with a 4-dof similarity  x_n = s_n R(rho_n) x_120 + t_n  (robust: LMedS then least squares on
inliers < 1.5 px).  Also fits a full affine and a homography for model comparison, and checks for
depth-dependent parallax (post/arm vs base front residual means) that a dolly would produce.
Outputs camera/out/drift_similarity_raw.csv; camera/drift.csv (per-frame camera: smoothed s, t,
roll -> focal_px, principal point / Blender shift) is written by camera/calibrate.py which knows f."""
import cv2, numpy as np, pathlib, csv
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = pathlib.Path(__file__).resolve().parents[1]
D = np.load(ROOT / "camera/out/static_tracks.npz"); T = D["pts"]; REF = int(D["ref_frame"])
good = np.isfinite(T[..., 0]).mean(0) > 0.9; T = T[:, good]; P0 = T[REF]
rows = []
for n in range(T.shape[0]):
    ok = np.isfinite(T[n, :, 0]); a = P0[ok].astype(np.float64); b = T[n, ok].astype(np.float64)
    M = cv2.estimateAffinePartial2D(a, b, method=cv2.LMEDS)[0]
    r = np.linalg.norm(a @ M[:, :2].T + M[:, 2] - b, axis=1); inl = r < 1.5
    # LS similarity on inliers: b = [[p,-q],[q,p]] a + t
    A = np.zeros((2*inl.sum(), 4)); y = b[inl].ravel()
    A[0::2] = np.c_[a[inl, 0], -a[inl, 1], np.ones(inl.sum()), np.zeros(inl.sum())]
    A[1::2] = np.c_[a[inl, 1], a[inl, 0], np.zeros(inl.sum()), np.ones(inl.sum())]
    p, q, tx, ty = np.linalg.lstsq(A, y, rcond=None)[0]
    res = (A @ [p, q, tx, ty] - y).reshape(-1, 2); rms = float(np.sqrt((res**2).sum(1).mean()))
    s = float(np.hypot(p, q)); rho = float(np.arctan2(q, p))
    Ma = cv2.estimateAffine2D(a[inl], b[inl], method=cv2.LMEDS)[0]
    ra = np.linalg.norm(a[inl] @ Ma[:, :2].T + Ma[:, 2] - b[inl], axis=1)
    rows.append([n, n/24, s, rho, tx, ty, rms, float(np.sqrt((ra**2).mean())), int(inl.sum())])
R = np.array(rows)
with open(ROOT / "camera/out/drift_similarity_raw.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["frame", "t_s", "scale", "roll_rad", "tx_px", "ty_px", "sim_rms_px", "affine_rms_px", "n_inliers"])
    for r in R: w.writerow([int(r[0])] + [f"{x:.6g}" for x in r[1:]])
t = R[:, 1]
print("scale: frame0 %.5f frame240 %.5f  (%.2f%% over clip)" % (R[0, 2], R[-1, 2], 100*(R[-1, 2]/R[0, 2]-1)))
print("roll range [%.2e, %.2e] rad" % (R[:, 3].min(), R[:, 3].max()))
print("sim rms median %.3f max %.3f ; affine rms median %.3f" % (np.median(R[:, 6]), R[:, 6].max(), np.median(R[:, 7])))
fx = R[:, 4]/(1-R[:, 2] + 1e-12); fy = R[:, 5]/(1-R[:, 2] + 1e-12)
sel = np.abs(1-R[:, 2]) > 0.004
print("fixed point of similarity (zoom centre) median u %.1f v %.1f" % (np.median(fx[sel]), np.median(fy[sel])))
for deg in (1, 2, 3):
    c = np.polyfit(t, R[:, 2], deg); print("scale poly deg", deg, "resid sd %.2e" % np.std(R[:, 2]-np.polyval(c, t)))
fig, ax = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
ax[0].plot(t, (R[:, 2]-1)*100, ".", ms=3); ax[0].set_ylabel("scale-1 vs f120 [%]")
ax[1].plot(t, R[:, 4], ".", ms=3, label="tx"); ax[1].plot(t, R[:, 5], ".", ms=3, label="ty"); ax[1].legend(); ax[1].set_ylabel("translation [px]")
ax[2].plot(t, R[:, 6], ".", ms=3, label="similarity rms"); ax[2].plot(t, R[:, 7], ".", ms=3, label="affine rms"); ax[2].legend(); ax[2].set_ylabel("residual [px]"); ax[2].set_xlabel("t [s]")
fig.suptitle("Camera drift from %d tracked static stand features (ref frame %d)" % (good.sum(), REF))
fig.tight_layout(); fig.savefig(ROOT / "camera/out/drift_similarity.png", dpi=110)
