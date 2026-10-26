"""Pivot image point from the wire (camera role).  In every frame the dark wire ridge is located
(sub-pixel minimum of smoothed grey along rows v=190..(bob top-15)), a line is fitted, the line is
mapped into reference-frame-120 image coordinates with the inverse drift similarity, and the
least-squares intersection of all wire lines gives the pivot image point P_120 (the point the wire
rotates about, as seen by the camera).  Also maps the auxiliary bob track to frame-120 coords.
Outputs camera/out/wire_lines.csv, camera/out/pivot_f120.json, camera/out/bob_quick_f120.csv"""
import cv2, numpy as np, pathlib, csv, json
ROOT = pathlib.Path(__file__).resolve().parents[1]
S = np.load(ROOT / "camera/frames/grey_stack.npy", mmap_mode="r")
B = np.loadtxt(ROOT / "camera/out/bob_quick.csv", delimiter=",", skiprows=1)
Dr = np.loadtxt(ROOT / "camera/out/drift_similarity_raw.csv", delimiter=",", skiprows=1)
def to120(n, u, v):
    s, rho, tx, ty = (Dr[n, k] for k in (2, 3, 4, 5)); c, sn = np.cos(rho), np.sin(rho)
    x, y = (np.asarray(u) - tx)/s, (np.asarray(v) - ty)/s
    return c*x + sn*y, -sn*x + c*y
P_guess = np.array([1087.0, 125.0])
lines = []
for n in range(S.shape[0]):
    g = cv2.GaussianBlur(np.ascontiguousarray(S[n]).astype(np.float32), (0, 0), 1.0)
    bu, bv, br = B[n, 2:5]
    vs = np.arange(190, bv - br - 15, 6.0); pts = []
    for v in vs:
        uc = P_guess[0] + (bu - P_guess[0])*(v - P_guess[1])/(bv - P_guess[1])
        us = np.arange(uc - 7, uc + 7, 0.25)
        prof = cv2.remap(g, (us - 0.5).astype(np.float32)[None], np.full((1, len(us)), v - 0.5, np.float32), cv2.INTER_LINEAR).ravel()
        i = int(np.argmin(prof))
        if 0 < i < len(us)-1 and prof.max() - prof[i] > 8:
            d = prof[i-1] - 2*prof[i] + prof[i+1]; di = 0.5*(prof[i-1]-prof[i+1])/d if d else 0
            pts.append((us[i] + 0.25*di, v))
    P = np.array(pts)
    for _ in range(3):
        c = np.polyfit(P[:, 1], P[:, 0], 1); r = P[:, 0] - np.polyval(c, P[:, 1])
        P = P[np.abs(r) < max(1.0, 3*np.std(r))]
    c = np.polyfit(P[:, 1], P[:, 0], 1); r = P[:, 0] - np.polyval(c, P[:, 1])
    # bob centre distance from the wire line (consistency check)
    db = (bu - np.polyval(c, bv)) / np.hypot(1, c[0])
    # two points on the line mapped to f120 coords
    va, vb = 200.0, 700.0
    ua, ub = np.polyval(c, va), np.polyval(c, vb)
    x1, y1 = to120(n, ua, va); x2, y2 = to120(n, ub, vb)
    lines.append([n, n/24, c[0], c[1], float(np.sqrt(np.mean(r**2))), len(P), db, x1, y1, x2, y2])
L = np.array(lines)
with open(ROOT / "camera/out/wire_lines.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["frame", "t_s", "du_dv", "u_at_v0", "fit_rms_px", "n_pts", "bob_centre_offset_px", "u1_f120", "v1_f120", "u2_f120", "v2_f120"])
    for r in L: w.writerow([int(r[0])] + [f"{x:.6g}" for x in r[1:]])
# LS intersection of lines (in f120 coords)
A = []; b = []
for r in L:
    d = np.array([r[9]-r[7], r[10]-r[8]]); d /= np.linalg.norm(d); nrm = np.array([-d[1], d[0]])
    A.append(nrm); b.append(nrm @ [r[7], r[8]])
A = np.array(A); b = np.array(b)
Pp = np.linalg.lstsq(A, b, rcond=None)[0]; res = A @ Pp - b
# bootstrap uncertainty
rng = np.random.default_rng(1); bs = []
for _ in range(500):
    i = rng.integers(0, len(A), len(A)); bs.append(np.linalg.lstsq(A[i], b[i], rcond=None)[0])
bs = np.array(bs)
print("pivot f120 (u,v) = %.2f %.2f  line-resid rms %.3f px  bootstrap sd %.3f %.3f" % (*Pp, np.sqrt(np.mean(res**2)), *bs.std(0)))
print("wire fit rms median %.3f; bob-centre offset from wire line: mean %.2f sd %.2f px" % (np.median(L[:, 4]), L[:, 6].mean(), L[:, 6].std()))
json.dump({"frame_ref": 120, "u_px": float(Pp[0]), "v_px": float(Pp[1]), "sd_u_px": float(bs[:, 0].std()), "sd_v_px": float(bs[:, 1].std()),
           "line_resid_rms_px": float(np.sqrt(np.mean(res**2))), "method": "LS intersection of 241 per-frame wire lines mapped to frame-120 coords"},
          open(ROOT / "camera/out/pivot_f120.json", "w"), indent=1)
bu, bv = to120(np.arange(len(B)), B[:, 2], B[:, 3])
with open(ROOT / "camera/out/bob_quick_f120.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["frame", "t_s", "u_f120", "v_f120", "r_f120", "fit_rms_px"])
    for k in range(len(B)): w.writerow([k, f"{k/24:.6f}", f"{bu[k]:.4f}", f"{bv[k]:.4f}", f"{B[k,4]/Dr[k,2]:.4f}", f"{B[k,5]:.4f}"])
