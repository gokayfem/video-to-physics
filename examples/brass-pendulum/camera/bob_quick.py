"""Camera-role AUXILIARY bob track (used only to constrain the swing plane / scale gauge in the
camera calibration when measurement/bob_track.csv is not yet available).
Method: clean background plate (per-pixel median over frames where the bob is >90 px away, from a
first Hough pass), |frame - plate| mask in the swing ROI, sub-pixel edge points on the bob outline
(radial gradient search), algebraic+geometric circle fit excluding the wire-attachment sector.
Output camera/out/bob_quick.csv: frame,t_s,u_px,v_px,r_px,fit_rms_px."""
import cv2, numpy as np, pathlib, csv
from scipy.optimize import least_squares
ROOT = pathlib.Path(__file__).resolve().parents[1]
REF = ROOT.parents[1] / "reference-batch/clips/01-brass-pendulum.mp4"
U0, V0, U1, V1 = 850, 640, 1420, 960
cap = cv2.VideoCapture(str(REF)); F = []
while True:
    ok, f = cap.read()
    if not ok: break
    F.append(f[V0:V1, U0:U1].astype(np.float32))
F = np.stack(F); N = len(F)
G = F.mean(-1)
# pass 1: Hough
c1 = []
for n in range(N):
    g = cv2.GaussianBlur(G[n].astype(np.uint8), (0, 0), 2)
    cs = cv2.HoughCircles(g, cv2.HOUGH_GRADIENT, 1.5, 200, param1=60, param2=25, minRadius=45, maxRadius=65)
    c1.append(cs[0, 0] if cs is not None else [np.nan]*3)
c1 = np.array(c1)
yy, xx = np.mgrid[0:V1-V0, 0:U1-U0]
plate = np.zeros_like(F[0])
far = np.ones((N, V1-V0, U1-U0), bool)
for n in range(N):
    if np.isfinite(c1[n, 0]):
        far[n] = (xx - c1[n, 0])**2 + (yy - c1[n, 1])**2 > 90**2
for ch in range(3):
    A = np.where(far, F[..., ch], np.nan)
    plate[..., ch] = np.nanmedian(A, axis=0)
plate = np.where(np.isnan(plate), np.median(F, 0), plate)
rows = []
def circ_res(p, P): return np.hypot(P[:, 0]-p[0], P[:, 1]-p[1]) - p[2]
for n in range(N):
    D = np.abs(F[n] - plate).sum(-1)
    D = cv2.GaussianBlur(D, (0, 0), 1.0)
    cu, cv_, r0 = c1[n] if np.isfinite(c1[n, 0]) else rows[-1][2:5] if rows else (300, 180, 55)
    pts = []
    for a in np.deg2rad(np.arange(0, 360, 3)):
        if abs(np.rad2deg(a) - 270) < 25: continue      # wire sector (top; image v down => 270 deg)
        rr = np.arange(r0 - 14, r0 + 14, 0.25)
        x = cu + rr*np.cos(a); y = cv_ + rr*np.sin(a)
        prof = cv2.remap(D, x.astype(np.float32).reshape(1, -1), y.astype(np.float32).reshape(1, -1), cv2.INTER_LINEAR).ravel()
        gp = -np.gradient(prof)      # inside bright diff -> outside low: falling edge
        i = int(np.argmax(gp))
        if 0 < i < len(rr)-1:
            d = gp[i-1] - 2*gp[i] + gp[i+1]; di = 0.5*(gp[i-1]-gp[i+1])/d if d != 0 else 0
            pts.append((cu + (rr[i]+0.25*di)*np.cos(a), cv_ + (rr[i]+0.25*di)*np.sin(a)))
    P = np.array(pts)
    s = least_squares(circ_res, [cu, cv_, r0], args=(P,), loss="soft_l1", f_scale=0.7)
    res = circ_res(s.x, P); k = np.abs(res) < 2.0
    s = least_squares(circ_res, s.x, args=(P[k],))
    rms = float(np.sqrt(np.mean(circ_res(s.x, P[k])**2)))
    # contract pixel convention: array index i <-> pixel centre i+0.5
    rows.append([n, n/24, s.x[0] + U0 + 0.5, s.x[1] + V0 + 0.5, s.x[2], rms])
R = np.array(rows)
with open(ROOT / "camera/out/bob_quick.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["frame", "t_s", "u_px", "v_px", "r_px", "fit_rms_px"])
    for r in R: w.writerow([int(r[0])] + [f"{x:.4f}" for x in r[1:]])
print("u range", R[:, 2].min(), R[:, 2].max(), "v range", R[:, 3].min(), R[:, 3].max(),
      "r mean", R[:, 4].mean(), "r sd", R[:, 4].std(), "fit rms median", np.median(R[:, 5]), "max", R[:, 5].max())
