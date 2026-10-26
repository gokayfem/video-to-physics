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
EDGES = {
    # ---- base (walnut slab) ----
    "b_left_top":     (494, 908, 672, 881, 7, "gc"),   # x=bx0, z=bz1  (Y-line, silhouette)
    "b_back_top":     (860, 880, 1010, 886, 6, "gc"),  # y=by1, z=bz1  (X-line, silhouette)
    "b_back_top2":    (1180, 899, 1235, 903, 6, "gc"),
    "b_front_left_v": (483, 935, 483, 1018, 6, "gc"),  # x=bx0, y=by0  (Z-line, silhouette)
    "b_front_bottom": (500, 1031, 860, 1083, 7, "g"),  # y=by0, z=bz0  (X-line, table contact)
    "b_front_top":    (520, 935, 1220, 1000, 9, "g"),  # y=by0, z=bz1  (X-line, internal)
    "b_front_right_v":(1247, 1015, 1247, 1080, 8, "g"),# x=bx1, y=by0  (Z-line, internal)
    "b_end_top":      (1270, 990, 1430, 945, 9, "g"),  # x=bx1, z=bz1  (Y-line, internal)
    "b_back_right_v": (1487, 965, 1487, 1035, 6, "gc"),# x=bx1, y=by1  (Z-line, silhouette)
    "b_end_bottom":   (1280, 1080, 1460, 1052, 9, "g"),# x=bx1, z=bz0
    # ---- post ----
    "p_left":         (690, 210, 688, 860, 6, "gc"),   # x=px0, y=py0 silhouette
    "p_right":        (838, 220, 838, 860, 6, "gc"),   # x=px1, y=py1 silhouette
    "p_top_front":    (702, 70, 762, 58, 5, "gc"),     # y=py0, z=pz1
    "p_top_right":    (790, 59, 830, 72, 5, "gc"),     # x=px1, z=pz1
    "p_bot_front":    (694, 897, 752, 903, 4, "g"),    # y=py0, z=bz1 (post/base junction)
    "p_bot_right":    (775, 903, 830, 894, 4, "g"),    # x=px1, z=bz1 (post/base junction)
    # ---- arm ----
    "a_top":          (812, 89, 1015, 63, 6, "gc"),    # top silhouette
    "a_bottom":       (852, 178, 1040, 155, 4, "g"),   # bottom silhouette: underside (grey) -> background
    "a_front_bottom2":(852, 166, 1040, 148, 4, "g"),   # front face (dark) -> underside (grey), internal
    "a_end_right":    (1071, 120, 1071, 143, 5, "gc"),
    "a_front_right_v":(1029, 72, 1029, 140, 6, "g"),
    "a_front_bottom": (810, 142, 1020, 133, 7, "g"),
    "a_front_topin":  (812, 100, 1020, 73, 6, "g"),
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
        n, c, keep, rms = fit_line(pts)
        out[name] = {"points": pts[keep], "line": (n, c), "rms_px": rms, "n_pts": int(keep.sum())}
    return out

if __name__ == "__main__":
    import sys
    src = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "camera/frames/f_120.png")
    tag = sys.argv[2] if len(sys.argv) > 2 else "f120"
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
