"""Per-region colour statistics (camera/appearance role).

Regions are polygons defined in REFERENCE FRAME-120 pixel coordinates (contract convention) on
static parts that do not depend on the bob position, plus the bob disc (from the bob track).  For
another reference frame the polygons are mapped with the measured drift similarity; for a render
made with the reference-matched camera of the same frame the same mapping applies, and for a render
at a different resolution the coordinates are scaled.

Stats per region: mean sRGB (0-1, gamma-encoded as stored), mean Oklab L/a/b and chroma (computed
per pixel from linearised sRGB), luma p10/p50/p90 (Rec.709 on encoded values).
CLI: measure reference frames -> camera/out/ref_regions.json and an overlay image of the regions.
"""
import numpy as np, cv2, json, pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[1]
REF = ROOT.parents[1] / "reference-batch/clips/01-brass-pendulum.mp4"

REGIONS = {   # frame-120 px polygons (u, v)
    "background_left":  [(120, 120), (560, 120), (560, 700), (120, 700)],
    "background_right": [(1400, 150), (1820, 150), (1820, 700), (1400, 700)],
    "background_top":   [(250, 10), (1650, 10), (1650, 40), (250, 40)],
    "table":            [(20, 930), (440, 930), (440, 1075), (20, 1075)],
    "table_right":      [(1530, 960), (1900, 960), (1900, 1075), (1530, 1075)],
    "base_top":         [(560, 905), (680, 893), (860, 898), (1000, 905), (1000, 935), (560, 925)],
    "base_front":       [(520, 950), (900, 990), (900, 1050), (520, 1010)],
    "base_end":         [(1290, 985), (1440, 960), (1445, 1020), (1295, 1045)],
    "post_front":       [(700, 260), (760, 260), (760, 820), (700, 820)],
    "post_side":        [(790, 260), (830, 260), (830, 820), (790, 820)],
    "arm_front":        [(840, 100), (1010, 80), (1010, 125), (840, 145)],
}

def _lin(c):
    return np.where(c <= 0.04045, c/12.92, ((c + 0.055)/1.055)**2.4)

def oklab(rgb_srgb):
    l = _lin(rgb_srgb)
    M1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929], [0.2119034982, 0.6806995451, 0.1073969566],
                   [0.0883024619, 0.2817188376, 0.6299787005]])
    M2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468], [1.9779984951, -2.4285922050, 0.4505937099],
                   [0.0259040371, 0.7827717662, -0.8086757660]])
    lms = np.cbrt(l @ M1.T)
    return lms @ M2.T

def masks(shape, sim=(1.0, 0.0, 0.0, 0.0), scale=1.0, bob=None):
    """sim = (s, rho, tx, ty) maps frame-120 coords to the target frame; scale for resolution."""
    s, rho, tx, ty = sim; c, sn = np.cos(rho), np.sin(rho)
    out = {}
    for k, poly in REGIONS.items():
        P = np.array(poly, float)
        Q = s*(P @ np.array([[c, sn], [-sn, c]])) + [tx, ty]
        Q = (Q*scale - 0.5)
        m = np.zeros(shape[:2], np.uint8); cv2.fillPoly(m, [np.round(Q*16).astype(np.int32)], 1, cv2.LINE_8, 4)
        out[k] = m.astype(bool)
    if bob is not None:   # (u, v, r) in target-frame contract px at full res
        m = np.zeros(shape[:2], np.uint8)
        cv2.circle(m, (int(round((bob[0]*scale - 0.5)*16)), int(round((bob[1]*scale - 0.5)*16))), int(0.8*bob[2]*scale*16), 1, -1, cv2.LINE_8, 4)
        out["bob"] = m.astype(bool)
    return out

def stats(img_bgr, M):
    rgb = img_bgr[..., ::-1].astype(np.float64)/255.0
    lab = oklab(rgb.reshape(-1, 3)).reshape(rgb.shape)
    luma = rgb @ [0.2126, 0.7152, 0.0722]
    res = {}
    for k, m in M.items():
        if m.sum() == 0: continue
        L = lab[m]; Y = luma[m]
        res[k] = {"srgb_mean": rgb[m].mean(0).round(4).tolist(), "oklab_L": round(float(L[:, 0].mean()), 4),
                  "oklab_a": round(float(L[:, 1].mean()), 4), "oklab_b": round(float(L[:, 2].mean()), 4),
                  "oklab_C": round(float(np.hypot(L[:, 1], L[:, 2]).mean()), 4),
                  "luma_p10": round(float(np.percentile(Y, 10)), 4), "luma_p50": round(float(np.percentile(Y, 50)), 4),
                  "luma_p90": round(float(np.percentile(Y, 90)), 4), "n_px": int(m.sum())}
    return res

def bob_at(frame):
    import csv
    for r in csv.DictReader(open(ROOT / "measurement/bob_track.csv")):
        if int(r["frame"]) == frame: return float(r["u_px"]), float(r["v_px"]), float(r["radius_px"])

def sim_at(frame):
    D = np.loadtxt(ROOT / "camera/out/drift_similarity_raw.csv", delimiter=",", skiprows=1)
    return tuple(D[frame, 2:6])

if __name__ == "__main__":
    frames = [0, 60, 120, 180, 240]
    cap = cv2.VideoCapture(str(REF)); imgs = {}; n = 0
    while True:
        ok, f = cap.read()
        if not ok: break
        if n in frames: imgs[n] = f
        n += 1
    per = {}
    for n in frames:
        M = masks(imgs[n].shape, sim_at(n), 1.0, bob_at(n))
        per[n] = stats(imgs[n], M)
    agg = {}
    for k in per[120]:
        vals = [per[n][k] for n in frames]
        agg[k] = {q: (np.mean([v[q] for v in vals], 0).round(4).tolist() if q == "srgb_mean" else round(float(np.mean([v[q] for v in vals])), 4))
                  for q in vals[0] if q != "n_px"}
        agg[k]["oklab_L_sd_over_frames"] = round(float(np.std([v["oklab_L"] for v in vals])), 4)
    json.dump({"frames": frames, "per_frame": {str(k): v for k, v in per.items()}, "mean": agg,
               "method": "camera/look_measure.py; polygons in frame-120 px mapped by drift similarity; bob disc r*0.8 from measurement/bob_track.csv"},
              open(ROOT / "camera/out/ref_regions.json", "w"), indent=1)
    ov = imgs[120].copy()
    M = masks(ov.shape, sim_at(120), 1.0, bob_at(120))
    for i, (k, m) in enumerate(M.items()):
        col = tuple(int(x) for x in cv2.applyColorMap(np.array([[i*23 % 255]], np.uint8), cv2.COLORMAP_HSV)[0, 0])
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(ov, cnts, -1, col, 2)
        y, x = np.argwhere(m).mean(0).astype(int); cv2.putText(ov, k, (x - 40, y), 0, 0.5, col, 1, cv2.LINE_AA)
    cv2.imwrite(str(ROOT / "camera/out/ref_regions_overlay.jpg"), cv2.resize(ov, (960, 544)))
    for k, v in agg.items():
        print(f"{k:17s} sRGB {v['srgb_mean']}  L {v['oklab_L']:.3f} C {v['oklab_C']:.3f} b {v['oklab_b']:+.3f}  p10/50/90 {v['luma_p10']:.3f}/{v['luma_p50']:.3f}/{v['luma_p90']:.3f}  L-sd {v['oklab_L_sd_over_frames']:.4f}")
