"""Stage 4: disc twist psi(t) from the engraved diameter line, measured in the disc's own plane.
Rectification [shared by both methods]: per-frame camera (camfr) + per-frame disc pose (s3: P, n) map disc-plane
coordinates (x along the in-plane projection of world +X, y = n x x, so azimuth theta is counter-clockwise seen
from above, measured from world +X) to image pixels; bilinear sampling of grey luma.
  * top-down rectified image: 601x601, 600 px per D (measurement/cache/rect_cart.npy, uint8)
  * polar image: r = 0.06..0.48 D (85 rows), theta = 0..359.5 deg (720 cols) (measurement/cache/rect_polar.npy)
The wire (vertical, above the disc centre) overlays the far half at theta_w = azimuth of (P - C): masked +-3 deg.
Method A (spatial line fit): black-hat (11 px ellipse) of the rectified image, annulus 0.08-0.47 D; each half-line
  separately: pixels in a +-12 deg wedge around the prediction, weight = black-hat above its 90th-percentile level,
  weighted total-least-squares line -> half-line direction theta_A1, theta_A2 and its distance from the centre.
  Frame-0 prediction from cv2.HoughLines; later frames from linear extrapolation of the previous two frames.
Method B (angular profile): polar grey minus its circular median filter along theta (15.5 deg), averaged over
  4 radial bands; per band and half, parabolic sub-sample minimum within +-12 deg of the prediction;
  band scatter -> formal sigma_B.
psi = circular mean of the two half-line directions (theta1, theta2 - 180), unwrapped in time; kink = theta2 - theta1 - 180.
Writes measurement/twist_track.csv."""
import numpy as np, cv2, os, sys, csv
from scipy.ndimage import median_filter
sys.path.insert(0, os.path.dirname(__file__))
from common import *
import camfr
F = load_frames(); POSE = np.load(os.path.join(CACHE, "disc_pose.npy")); N = len(POSE)
S = 600; XY = (np.arange(601) - 300) / S
RR = np.linspace(0.06, 0.48, 85); TT = np.radians(np.arange(720) * 0.5)
BANDS = [(0.08, 0.18), (0.18, 0.28), (0.28, 0.38), (0.38, 0.48)]
def normal(a, b):
    n = np.array([np.sin(b), -np.sin(a) * np.cos(b), np.cos(a) * np.cos(b)]); return n / np.linalg.norm(n)
def basis(n):
    ex = np.array([1.0, 0, 0]) - n[0] * n; ex /= np.linalg.norm(ex); return ex, np.cross(n, ex)
def wrap(a): return (a + 180) % 360 - 180
def angdiff(a, b): return wrap(a - b)
def rectify(n, apex=(0.0, 0.0)):
    k = camfr.cam(n); P = POSE[n, 2:5]; nn = normal(*POSE[n, 5:7]); ex, ey = basis(nn)
    g = luma(F[n])
    X, Y = np.meshgrid(XY, -XY)  # row 0 = +y (far side up)
    W = P + X[..., None] * ex + Y[..., None] * ey; uv = camfr.proj(W, k)
    cart = bilinear(g, uv[..., 0], uv[..., 1])
    Rg, Tg = np.meshgrid(RR, TT, indexing="ij"); W = P + (apex[0] + Rg * np.cos(Tg))[..., None] * ex + (apex[1] + Rg * np.sin(Tg))[..., None] * ey
    uv = camfr.proj(W, k); pol = bilinear(g, uv[..., 0], uv[..., 1])
    d = P; thw = np.degrees(np.arctan2(d @ ey, d @ ex))  # azimuth of (P - C) in the disc frame (camera at origin)
    return cart, pol, thw
def methodB(pol, pred, thw):
    hp = pol - median_filter(pol, size=(1, 31), mode="wrap")
    th = np.arange(720) * 0.5; res = []
    for half in (0, 180):
        c = pred + half; vals = []
        for r0, r1 in BANDS:
            p = hp[(RR >= r0) & (RR < r1)].mean(0)
            m = (np.abs(angdiff(th, c)) <= 12) & (np.abs(angdiff(th, thw)) > 3)
            idx = np.where(m)[0]; i = idx[np.argmin(p[idx])]
            a, b_, cc = p[(i - 1) % 720], p[i], p[(i + 1) % 720]; den = a - 2 * b_ + cc
            off = 0.5 * (a - cc) / den if den > 0 else 0.0
            vals.append(th[i] + 0.5 * np.clip(off, -1, 1))
        vals = np.array(vals); mu = c + np.mean(angdiff(vals, c))
        res.append((mu, np.std(angdiff(vals, mu), ddof=1) / 2.0, vals))
    return res
BH_K = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
def methodA(cart, pred, thw):
    bh = cv2.morphologyEx(cart.astype(np.float32), cv2.MORPH_BLACKHAT, BH_K)
    X, Y = np.meshgrid(XY, -XY); r = np.hypot(X, Y); th = np.degrees(np.arctan2(Y, X))
    ann = (r > 0.08) & (r < 0.47) & (np.abs(angdiff(th, thw)) > 3)
    out = []
    for half in (0, 180):
        c = pred + half; m = ann & (np.abs(angdiff(th, c)) <= 12)
        v = bh[m]; thr = np.percentile(v, 90); sel = v > thr
        x = X[m][sel]; y = Y[m][sel]; w = v[sel] - thr
        mx, my = np.average(x, weights=w), np.average(y, weights=w)
        C = np.cov(np.c_[x - mx, y - my].T, aweights=w); ev, evec = np.linalg.eigh(C); d = evec[:, 1]
        if d @ [np.cos(np.radians(c)), np.sin(np.radians(c))] < 0: d = -d
        ang = np.degrees(np.arctan2(d[1], d[0])); off = mx * d[1] - my * d[0]   # signed distance of the line from the centre (D)
        sig = np.degrees(np.sqrt(ev[0] / max(ev[1], 1e-12) / max(len(x), 1)))  # rough formal
        out.append((ang, off, sig, len(x)))
    return out, bh
if __name__ == "__main__":
    cart0, pol0, thw0 = rectify(0)
    cv2.imwrite(os.path.join(CACHE, "rect_f000.png"), np.clip(cart0, 0, 255).astype(np.uint8))
    bh0 = cv2.morphologyEx(cart0.astype(np.float32), cv2.MORPH_BLACKHAT, BH_K)
    X, Y = np.meshgrid(XY, -XY); r = np.hypot(X, Y)
    b8 = ((bh0 > np.percentile(bh0[(r > 0.08) & (r < 0.47)], 97)) & (r > 0.08) & (r < 0.47)).astype(np.uint8) * 255
    L = cv2.HoughLines(b8, 1, np.pi / 720, 60)
    rho_, t_ = L[0][0]; d = np.degrees(t_) + 90  # line direction in array coords (row down) -> convert to y-up
    pred0 = wrap(-d); print("Hough frame-0 line direction %.2f deg (y-up)" % pred0)
    # ---- pass 1: method A on the silhouette-centred rectification
    carts = np.zeros((N, 601, 601), np.uint8); RA = []; hist = [pred0, pred0]
    for n in range(N):
        cart, _, thw = rectify(n); carts[n] = np.clip(cart, 0, 255)
        predA = hist[-1] + (hist[-1] - hist[-2]); A, _ = methodA(cart, predA, thw)
        a1, a2 = A[0][0], A[1][0]; mA = a1 + 0.5 * angdiff(a2 - 180, a1); psiA = predA + angdiff(mA, predA)
        RA.append([psiA, predA + angdiff(a1, predA), angdiff(a2 - 180, a1), A[0][1], A[1][1], A[0][3], A[1][3], thw]); hist.append(psiA)
    RA = np.array(RA)
    # ---- apex (rotation centre seen by the line) in the disc-plane frame: signed line offset o = ax sin(psi) - ay cos(psi)
    ps = np.radians(RA[:, 0]); o = 0.5 * (RA[:, 3] - RA[:, 4])  # half2 direction is reversed -> sign flip
    Mx = np.c_[np.sin(ps), -np.cos(ps)]; apex, *_ = np.linalg.lstsq(Mx, o, rcond=None); ores = o - Mx @ apex
    print("apex offset from the silhouette centre (disc plane, D): (%.4f, %.4f), line-offset residual rms %.4f D" % (*apex, np.sqrt(np.mean(ores ** 2))))
    # ---- pass 2: method B on the apex-centred polar image
    rows = []; pols = np.zeros((N, 85, 720), np.float32); histB = [pred0, pred0]
    for n in range(N):
        _, pol, thw = rectify(n, apex); pols[n] = pol
        predB = histB[-1] + (histB[-1] - histB[-2]); B = methodB(pol, predB, thw)
        b1, b2 = B[0][0], B[1][0]; mB = b1 + 0.5 * angdiff(b2 - 180, b1); psiB = predB + angdiff(mB, predB); histB.append(psiB)
        kinkB = angdiff(b2 - 180, b1); sB = 0.5 * np.hypot(B[0][1], B[1][1]); psiA = RA[n, 0]
        rows.append([n, n / FPS, psiA, psiB, angdiff(psiA, psiB), sB, RA[n, 1], RA[n, 2], kinkB, RA[n, 3], RA[n, 4], B[0][1], B[1][1], RA[n, 7], o[n] - Mx[n] @ apex])
        if n % 20 == 0: print(n, "psiA %.2f psiB %.2f kinkA %.2f kinkB %.2f sB %.3f wire %.1f" % (psiA, psiB, RA[n, 2], kinkB, sB, RA[n, 7]))
    np.save(os.path.join(CACHE, "rect_cart.npy"), carts); np.save(os.path.join(CACHE, "rect_polar.npy"), pols)
    np.save(os.path.join(CACHE, "apex_offset.npy"), apex)
    hdr = ["frame", "t_s", "psi_A_deg", "psi_B_deg", "A_minus_B_deg", "sigma_B_formal_deg", "theta_A_half1_deg", "kink_A_deg", "kink_B_deg",
           "offset_A_half1_D", "offset_A_half2_D", "sigma_B_half1_deg", "sigma_B_half2_deg", "wire_azimuth_deg", "line_offset_resid_D"]
    with open(os.path.join(MEAS, "twist_raw.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(hdr)
        for r in rows: w.writerow([int(r[0])] + ["%.5f" % v for v in r[1:]])
