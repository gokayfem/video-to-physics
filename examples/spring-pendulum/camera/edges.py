"""Sub-pixel edge sampling along hand-picked approximate segments of the static plate.

Each edge is given by approximate endpoints (picked by eye on zoomed, gridded crops of the
temporal-median plate camera/frames/median.png).  Along the segment, every `step` px we take a
profile along the normal (+-search px) of a smoothed feature image (grey luminance and the wood
chroma R-B, both normalised) and locate the maximum of |d/dn| with a parabolic sub-pixel fit.
A robust (Huber-like iterative) total-least-squares line is fitted; points > 2.5 px from it are
dropped.  Output: dict name -> dict(points Nx2 (u,v) px [contract convention, pixel centres at
+0.5], line (n, c) with n.(u,v)+c=0, rms_px).
"""
import numpy as np, cv2, json, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]

# name: (u0, v0, u1, v1, search_px, channel) ; channel: 'g' grey, 'c' chroma, 'gc' both
EDGES = {   # clip 02, frame 0 (frame-0 contract px), picked on gridded zooms (camera/crop.py)
    # ---- posts (front face wood; inner side faces visible as darker bands) ----
    "pL_front_outer": (489, 90, 499, 980, 7, "gc"),     # x=xL0, y=yp0 (silhouette)
    "pL_front_inner": (609, 130, 616, 975, 3, "g"),     # x=xL1, y=yp0 (wood -> inner face, internal)
    "pL_back_inner":  (620, 130, 628, 975, 6, "g"),     # x=xL1, y=yp1 (inner face -> backdrop/table, silhouette)
    "pR_front_outer": (1421, 90, 1409, 980, 7, "gc"),   # x=xR1, y=yp0 (silhouette)
    "pR_front_inner": (1301, 130, 1292, 975, 3, "g"),   # x=xR0, y=yp0 (internal)
    "pR_back_inner":  (1290, 130, 1279, 975, 6, "g"),   # x=xR0, y=yp1 (silhouette)
    "pL_top_front":   (500, 55, 598, 55, 6, "g"),       # y=yp0, z=zpt
    "pR_top_front":   (1312, 56, 1410, 56, 6, "g"),
    "pL_bottom_front":(512, 1027, 588, 1027, 6, "gc"),  # y=yp0, z=bz1 (post/plate junction, internal)
    "pR_bottom_front":(1305, 1028, 1395, 1028, 6, "gc"),
    # ---- crossbar (black steel) ----
    "bar_top_a":      (640, 63, 935, 63, 5, "g"),       # y=yc0, z=0 (top silhouette; camera below the bar)
    "bar_top_b":      (980, 63, 1280, 63, 5, "g"),
    "bar_bot_a":      (640, 118, 935, 118, 5, "g"),     # bottom silhouette (front or back edge: decided by camera height)
    "bar_bot_b":      (980, 118, 1280, 118, 5, "g"),
    # ---- base plate (black steel) ----
    "plate_back_a":   (640, 984, 760, 984, 6, "gc"),    # y=by1, z=bz1 (silhouette against the table)
    "plate_back_b":   (1165, 984, 1268, 984, 6, "gc"),
    "plate_front_top":(530, 1055, 1390, 1055, 5, "g"),  # y=by0, z=bz1 (rounded front-top edge highlight, internal)
    "plate_front_bot":(530, 1073, 1390, 1073, 6, "gc"), # y=by0, z=bz0 (contact with table)
    "plate_left_end": (481, 1040, 496, 1002, 5, "gc"),  # x=bx0, z=bz1, along Y (silhouette)
    "plate_right_end":(1409, 1004, 1430, 1040, 5, "gc"),# x=bx1, z=bz1, along Y (silhouette)
}

def feature_images(img_bgr):
    f = img_bgr.astype(np.float32)
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    c = f[..., 2] - f[..., 0]
    g = cv2.GaussianBlur(g, (0, 0), 1.2); c = cv2.GaussianBlur(c, (0, 0), 1.2)
    return {"g": g / (g.std() + 1e-6), "c": c / (c.std() + 1e-6)}

def bilinear(I, u, v):
    # u,v in contract convention (pixel centres at +0.5) -> array index coords
    return cv2.remap(I, (np.asarray(u, np.float32) - 0.5).reshape(1, -1),
                     (np.asarray(v, np.float32) - 0.5).reshape(1, -1), cv2.INTER_LINEAR).ravel()

def sample_edge(F, u0, v0, u1, v1, search, ch, step=3.0):
    d = np.array([u1 - u0, v1 - v0], float); Lseg = np.linalg.norm(d); d /= Lseg
    n = np.array([-d[1], d[0]])
    ts = np.arange(0, Lseg + 1e-9, step); offs = np.arange(-search, search + 0.01, 0.5)
    pts = []
    for t in ts:
        base = np.array([u0, v0]) + t * d
        P = base[None, :] + offs[:, None] * n[None, :]
        prof = sum(np.gradient(bilinear(F[k], P[:, 0], P[:, 1]), 0.5) * 0 + 0 for k in [])
        gm = np.zeros(len(offs))
        for k in ch:
            gm += np.abs(np.gradient(bilinear(F[k], P[:, 0], P[:, 1]), 0.5))
        i = int(np.argmax(gm))
        if i == 0 or i == len(offs) - 1: continue
        a, b, c = gm[i-1], gm[i], gm[i+1]; den = a - 2*b + c
        di = 0.5*(a - c)/den if den != 0 else 0.0
        pts.append(base + (offs[i] + di*0.5) * n)
    return np.array(pts)

def fit_line(pts, it=6):
    w = np.ones(len(pts)); keep = np.ones(len(pts), bool)
    for _ in range(it):
        m = (pts[keep] * w[keep, None]).sum(0) / w[keep].sum()
        X = (pts[keep] - m) * np.sqrt(w[keep, None])
        _, _, Vt = np.linalg.svd(X); n = Vt[1]; c = -n @ m
        r = pts @ n + c
        s = max(1.4826*np.median(np.abs(r[keep])), 0.3)
        w = 1.0/np.maximum(1.0, np.abs(r)/(1.5*s)); k2 = np.abs(r) < max(2.5, 3*s)
        if k2.sum() >= max(3, len(pts)//3): keep = k2
    r = pts[keep] @ n + c
    return n, c, keep, float(np.sqrt(np.mean(r**2)))

def measure(img_bgr, edges=EDGES):
    F = feature_images(img_bgr); out = {}
    for name, (u0, v0, u1, v1, s, ch) in edges.items():
        pts = sample_edge(F, u0, v0, u1, v1, s, ch)
        if len(pts) < 3:  # edge left the frame (e.g. plate bottom during the push-in)
            out[name] = {"points": np.zeros((0, 2)), "line": (np.array([0.0, 1.0]), 0.0), "rms_px": float("nan"), "n_pts": 0}; continue
        n, c, keep, rms = fit_line(pts)
        out[name] = {"points": pts[keep], "line": (n, c), "rms_px": rms, "n_pts": int(keep.sum())}
    return out

if __name__ == "__main__":
    import sys
    src = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "camera/frames/f_000.png")
    tag = sys.argv[2] if len(sys.argv) > 2 else "f000"
    img = cv2.imread(src)
    res = measure(img)
    ov = img.copy()
    for name, r in res.items():
        print(f"{name:16s} n={r['n_pts']:3d} rms={r['rms_px']:.3f}px  n=({r['line'][0][0]:+.4f},{r['line'][0][1]:+.4f}) c={r['line'][1]:.2f}")
        for p in r["points"]:
            cv2.circle(ov, (int(round((p[0]-0.5)*4)), int(round((p[1]-0.5)*4))), 4, (0, 0, 255), -1, cv2.LINE_AA, 2)
    cv2.imwrite(str(ROOT / f"camera/out/edges_overlay_{tag}.png"), ov)
    json.dump({k: {"points": v["points"].tolist(), "line_n": list(map(float, v["line"][0])), "line_c": float(v["line"][1]),
                   "rms_px": v["rms_px"]} for k, v in res.items()}, open(ROOT / f"camera/out/edges_{tag}.json", "w"), indent=1)
