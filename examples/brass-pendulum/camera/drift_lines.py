"""Camera drift check from static silhouette EDGE LINES (camera role, independent of measurement/).
For each frame, the robust sub-pixel silhouette edges of camera/edges.py are re-measured and each
line's signed offset (px, along its normal) relative to the temporal-median plate line is recorded.
A global similarity motion (du, dv, scale about image centre, roll) is then fitted per frame from
all edge points.  Writes camera/out/drift_lines.csv and camera/out/drift_lines.png."""
import cv2, numpy as np, pathlib, csv, json
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import sys; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from edges import EDGES, feature_images, sample_edge, fit_line
ROOT = pathlib.Path(__file__).resolve().parents[1]
REF = ROOT.parents[1] / "reference-batch/clips/01-brass-pendulum.mp4"
USE = ["p_left", "p_right", "p_top_front", "p_top_right", "a_top", "a_end_right", "b_front_left_v",
       "b_back_right_v", "b_left_top", "b_front_bottom", "b_back_top2"]
ref = json.load(open(ROOT / "camera/out/edges_median.json"))
cap = cv2.VideoCapture(str(REF)); n = 0; rows = []
while True:
    ok, img = cap.read()
    if not ok: break
    Fi = feature_images(img); off = []; A = []; b = []
    for name in USE:
        u0, v0, u1, v1, s, ch = EDGES[name]
        pts = sample_edge(Fi, u0, v0, u1, v1, s, ch)
        if len(pts) < 5:
            print("frame", n, name, "too few points", len(pts)); off.append(np.nan); continue
        nn, c, keep, rms = fit_line(pts)
        nr = np.array(ref[name]["line_n"]); cr = ref[name]["line_c"]
        P = pts[keep]; d = P @ nr + cr          # signed distance of this frame's points to plate line
        off.append(np.median(d))
        # linearised similarity: point p moves by (du - th*(y) + s*x, dv + th*x + s*y), x,y about centre
        x = P[:, 0] - 960; y = P[:, 1] - 544
        A.append(np.stack([np.full_like(x, nr[0]), np.full_like(x, nr[1]), nr[0]*x + nr[1]*y, -nr[0]*y + nr[1]*x], 1))
        b.append(d)
    A = np.concatenate(A); b = np.concatenate(b)
    sol = np.linalg.lstsq(A, b, rcond=None)[0]
    res = b - A @ sol
    rows.append([n, n/24, *sol, float(np.sqrt(np.mean(res**2))), *off]); n += 1
R = np.array(rows)
hdr = ["frame", "t_s", "du_px", "dv_px", "scale_minus_1", "roll_rad", "resid_rms_px"] + [f"off_{k}_px" for k in USE]
with open(ROOT / "camera/out/drift_lines.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(hdr)
    for r in R: w.writerow([int(r[0])] + [f"{x:.5g}" for x in r[1:]])
t = R[:, 1]
print("global: du sd %.3f range[%.3f,%.3f] slope %.4f px/s" % (R[:, 2].std(), R[:, 2].min(), R[:, 2].max(), np.polyfit(t, R[:, 2], 1)[0]))
print("global: dv sd %.3f range[%.3f,%.3f] slope %.4f px/s" % (R[:, 3].std(), R[:, 3].min(), R[:, 3].max(), np.polyfit(t, R[:, 3], 1)[0]))
print("global: scale-1 sd %.2e range[%.2e,%.2e] slope %.2e /s" % (R[:, 4].std(), R[:, 4].min(), R[:, 4].max(), np.polyfit(t, R[:, 4], 1)[0]))
print("global: roll sd %.2e rad range[%.2e,%.2e]" % (R[:, 5].std(), R[:, 5].min(), R[:, 5].max()))
print("resid rms median %.3f" % np.median(R[:, 6]))
for i, k in enumerate(USE):
    o = R[:, 7+i]; o = np.where(np.isfinite(o), o, np.nanmedian(o)); print(f"{k:15s} offset sd {o.std():.3f} range [{o.min():+.3f},{o.max():+.3f}] slope {np.polyfit(t, o, 1)[0]:+.4f} px/s")
fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
for i, k in enumerate(USE): ax[0].plot(t, R[:, 7+i], lw=0.8, label=k)
ax[0].set_ylabel("edge offset vs plate [px]"); ax[0].legend(fontsize=6, ncol=3)
ax[1].plot(t, R[:, 2], label="du"); ax[1].plot(t, R[:, 3], label="dv"); ax[1].plot(t, R[:, 4]*960, label="scale-1 (px @ r=960)")
ax[1].set_ylabel("global similarity [px]"); ax[1].set_xlabel("t [s]"); ax[1].legend(fontsize=7)
fig.tight_layout(); fig.savefig(ROOT / "camera/out/drift_lines.png", dpi=110)
