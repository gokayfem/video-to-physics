"""Independent camera-drift check (camera role).  Normalised cross-correlation template matching of
static textured patches of the temporal-median plate in every frame (search +-10 px, parabolic
sub-pixel peak).  Writes camera/out/drift_patches.csv and prints statistics."""
import cv2, numpy as np, pathlib, csv
ROOT = pathlib.Path(__file__).resolve().parents[1]
S = np.load(ROOT / "camera/frames/grey_stack.npy")
med = cv2.cvtColor(cv2.imread(str(ROOT / "camera/frames/median.png")), cv2.COLOR_BGR2GRAY)
# (name, u0, v0, u1, v1): textured, bob/wire/shadow-free patches, full-res px
P = [("post_upper", 700, 200, 830, 400), ("arm_front", 820, 75, 1015, 165),
     ("pin_head", 1035, 75, 1130, 120), ("post_left_edge", 670, 450, 760, 700),
     ("base_left", 520, 930, 900, 1060), ("base_right_end", 1260, 990, 1470, 1080)]
M = 10
def sub(c, i):
    a, b, d = c[i-1], c[i], c[i+1]; den = a - 2*b + d
    return 0.0 if den == 0 else 0.5*(a - d)/den
rows = []
for n in range(S.shape[0]):
    r = [n, n/24]
    for name, u0, v0, u1, v1 in P:
        T = med[v0:v1, u0:u1]
        I = S[n, v0-M:v1+M, u0-M:u1+M]
        c = cv2.matchTemplate(I, T, cv2.TM_CCOEFF_NORMED)
        _, mx, _, (x, y) = cv2.minMaxLoc(c)
        dx = x - M + (sub(c[y, :], x) if 0 < x < c.shape[1]-1 else 0)
        dy = y - M + (sub(c[:, x], y) if 0 < y < c.shape[0]-1 else 0)
        r += [dx, dy, mx]
    rows.append(r)
R = np.array(rows)
out = ROOT / "camera/out"; out.mkdir(exist_ok=True)
with open(out / "drift_patches.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["frame", "t_s"] + [f"{p[0]}_{a}" for p in P for a in ("du_px", "dv_px", "ncc")])
    for r in R: w.writerow([int(r[0])] + [f"{x:.4f}" for x in r[1:]])
t = R[:, 1]
for i, p in enumerate(P):
    du, dv, cc = R[:, 2+3*i], R[:, 3+3*i], R[:, 4+3*i]
    print(f"{p[0]:15s} du mean {du.mean():+.3f} sd {du.std():.3f} [{du.min():+.2f},{du.max():+.2f}] slope {np.polyfit(t,du,1)[0]:+.4f}px/s | "
          f"dv mean {dv.mean():+.3f} sd {dv.std():.3f} [{dv.min():+.2f},{dv.max():+.2f}] slope {np.polyfit(t,dv,1)[0]:+.4f}px/s | ncc min {cc.min():.3f}")
du = R[:, 2::3].mean(1); dv = R[:, 3::3].mean(1)
print("all-patch mean: du sd %.3f max|.| %.3f slope %+.4f px/s ; dv sd %.3f max|.| %.3f slope %+.4f px/s" % (
    du.std(), abs(du).max(), np.polyfit(t, du, 1)[0], dv.std(), abs(dv).max(), np.polyfit(t, dv, 1)[0]))
# first-vs-last second
print("mean shift first 1 s vs last 1 s: du %.3f dv %.3f" % (du[-24:].mean()-du[:24].mean(), dv[-24:].mean()-dv[:24].mean()))
