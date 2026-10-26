"""Region metrics, reference vs render (scene role C, look).

.venv/bin/python look/metrics.py --tag v001 [--frame 0]
Inputs: look/out/<tag>_full.png (1920x1088 or 960x544 render through VSP_Cam_Reference at the given frame, weight
posed at the reference's measured H) and look/out/<tag>_crop_<name>.png (full-res border renders, boxes in CROPS).
Colour stats per region (boxes in frame-0 reference pixels; the render uses the same frame-0 camera, so the same
boxes apply; at other frames the boxes are mapped by the measured drift homography): mean sRGB, Oklab L/a/b/C,
luma p10/p50/p90 at 960x544 (INTER_AREA).
Texture stats on full-res boxes (TEX): hp_rms (luma minus Gaussian sigma 3 px), structure-tensor coherence,
dominant streak angle (deg, 0 = horizontal), mean across-streak frequency (cycles/px).
Writes look/out/metrics_<tag>.json.
"""
import argparse, json, pathlib
import numpy as np, cv2

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
REF = ROOT.parents[1] / "reference-batch/clips/02-vertical-spring.mp4"
# colour regions (full-res frame-0 px: u0, v0, u1, v1)
REGIONS = {"backdrop_left": (100, 100, 400, 700), "backdrop_centre": (650, 200, 800, 500),
           "backdrop_right": (1500, 100, 1850, 700), "table_left": (50, 880, 400, 1080), "table_right": (1500, 880, 1900, 1080),
           "post_left": (510, 200, 600, 950), "post_right": (1320, 200, 1400, 950), "crossbar": (650, 72, 900, 110),
           "plate_top": (650, 992, 780, 1048), "plate_front": (600, 1059, 1300, 1070), "weight_side": (810, 800, 900, 900),
           "spring_coils": (900, 300, 990, 480)}
# full-res crops rendered by look/render_look.py (each <= 960x544)
CROPS = {"post": (480, 150, 680, 650), "top": (860, 20, 1100, 260), "spring": (840, 200, 1060, 700),
         "weight": (740, 690, 1060, 1000), "base": (440, 940, 1040, 1088), "table": (1450, 840, 1920, 1088)}
TEX = {"post_left_face": (515, 250, 595, 600), "weight_side": (815, 820, 900, 900), "table_right": (1500, 900, 1900, 1060),
       "plate_top": (660, 995, 1000, 1045), "crossbar": (650, 75, 900, 108)}


def ref_frame(n):
    cap = cv2.VideoCapture(str(REF))
    cap.set(cv2.CAP_PROP_POS_FRAMES, n)
    ok, f = cap.read()
    assert ok
    return f


def oklab(bgr):
    c = bgr[..., ::-1].astype(np.float64) / 255.0
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    M1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929], [0.2119034982, 0.6806995451, 0.1073969566],
                   [0.0883024619, 0.2817188376, 0.6299787005]])
    M2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468], [1.9779984951, -2.4285922050, 0.4505937099],
                   [0.0259040371, 0.7827717662, -0.8086757660]])
    return np.cbrt(lin @ M1.T) @ M2.T


def colour_stats(img960, boxes):
    out = {}
    for k, (u0, v0, u1, v1) in boxes.items():
        c = img960[v0 // 2:v1 // 2, u0 // 2:u1 // 2]
        L = oklab(c)
        Y = (c[..., ::-1].astype(np.float64) / 255.0) @ [0.2126, 0.7152, 0.0722]
        out[k] = {"srgb_mean": c.reshape(-1, 3).mean(0)[::-1].round(1).tolist(), "oklab_L": round(float(L[..., 0].mean()), 4),
                  "oklab_a": round(float(L[..., 1].mean()), 4), "oklab_b": round(float(L[..., 2].mean()), 4),
                  "oklab_C": round(float(np.hypot(L[..., 1], L[..., 2]).mean()), 4),
                  "luma_p10": round(float(np.percentile(Y, 10)), 4), "luma_p50": round(float(np.percentile(Y, 50)), 4),
                  "luma_p90": round(float(np.percentile(Y, 90)), 4)}
    return out


def tex_stats(img_bgr, box):
    u0, v0, u1, v1 = box
    c = img_bgr[v0:v1, u0:u1].astype(np.float64) / 255.0
    Y = c[..., ::-1] @ [0.2126, 0.7152, 0.0722]
    hp = Y - cv2.GaussianBlur(Y, (0, 0), 3.0)
    gx = cv2.Sobel(cv2.GaussianBlur(Y, (0, 0), 1.0), cv2.CV_64F, 1, 0)
    gy = cv2.Sobel(cv2.GaussianBlur(Y, (0, 0), 1.0), cv2.CV_64F, 0, 1)
    Jxx, Jyy, Jxy = (gx * gx).mean(), (gy * gy).mean(), (gx * gy).mean()
    tr, det = Jxx + Jyy, Jxx * Jyy - Jxy ** 2
    l1 = tr / 2 + np.sqrt(max(tr * tr / 4 - det, 0)); l2 = tr - l1
    coh = (l1 - l2) / (l1 + l2 + 1e-12)
    grad_dir = 0.5 * np.degrees(np.arctan2(2 * Jxy, Jxx - Jyy))
    grain = (grad_dir + 90) % 180
    h, w = hp.shape
    win = np.outer(np.hanning(h), np.hanning(w))
    P = np.abs(np.fft.fftshift(np.fft.fft2(hp * win))) ** 2
    fy = np.fft.fftshift(np.fft.fftfreq(h))[:, None]; fx = np.fft.fftshift(np.fft.fftfreq(w))[None, :]
    fr = np.hypot(fx, fy)
    th = np.radians(grad_dir)
    f_par = np.abs(fx * np.cos(th) + fy * np.sin(th))
    band = (fr > 0.01) & (fr < 0.35)
    f_across = float((P * f_par)[band].sum() / P[band].sum())
    return {"hp_rms": round(float(hp.std()), 4), "coherence": round(float(coh), 3), "grain_deg": round(float(grain), 1),
            "f_across_cyc_px": round(f_across, 4)}


def full_from_crops(info):
    """Full-res canvas: upscaled full render, overwritten by the 100 % crops where available."""
    full = cv2.imread(info["files"]["full"])
    canvas = cv2.resize(full, (1920, 1088), interpolation=cv2.INTER_LINEAR)
    for k, b in info.get("crop_boxes", {}).items():
        crop = cv2.imread(info["files"][k])
        canvas[b[1]:b[1] + crop.shape[0], b[0]:b[0] + crop.shape[1]] = crop
    return canvas


def inside_crop(box, info):
    return any(box[0] >= cb[0] and box[1] >= cb[1] and box[2] <= cb[2] and box[3] <= cb[3]
               for cb in info.get("crop_boxes", {}).values())


def grid(img):
    Y = (img[..., ::-1].astype(np.float64) / 255.0) @ [0.2126, 0.7152, 0.0722]
    return cv2.resize(Y, (16, 9), interpolation=cv2.INTER_AREA)


BG_CELLS = [(r, c) for r in range(0, 6) for c in list(range(0, 4)) + list(range(12, 16))]
TABLE_CELLS = [(8, c) for c in (0, 1, 2, 3, 12, 13, 14, 15)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--frame", type=int, default=0)
    a = ap.parse_args()
    ref = ref_frame(a.frame)
    info = json.loads((HERE / "out" / f"{a.tag}_render.json").read_text())
    cur = full_from_crops(info)
    boxes = REGIONS
    rc = colour_stats(cv2.resize(ref, (960, 544), interpolation=cv2.INTER_AREA), boxes)
    cc = colour_stats(cv2.resize(cv2.imread(info["files"]["full"]), (960, 544), interpolation=cv2.INTER_AREA), boxes)
    rt = {k: tex_stats(ref, b) for k, b in TEX.items()}
    ct = {k: tex_stats(cur, b) for k, b in TEX.items() if inside_crop(b, info)}
    res = {"tag": a.tag, "frame": a.frame, "render": info, "method": __doc__, "regions": boxes, "texture_boxes": TEX,
           "reference": {"colour": rc, "texture": rt}, "current": {"colour": cc, "texture": ct}, "delta": {}}
    for k in rc:
        res["delta"][k] = {q: round(cc[k][q] - rc[k][q], 4) for q in ("oklab_L", "oklab_a", "oklab_b", "oklab_C",
                                                                      "luma_p10", "luma_p50", "luma_p90")}
    gr, gc = grid(ref), grid(cv2.imread(info["files"]["full"]))
    st = {}
    for name, cells in (("backdrop", BG_CELLS), ("table", TABLE_CELLS)):
        r = np.array([gr[i] for i in cells]); c = np.array([gc[i] for i in cells])
        st[name] = {"mae": round(float(np.abs(c - r).mean()), 4), "bias": round(float((c - r).mean()), 4),
                    "corr": round(float(np.corrcoef(r, c)[0, 1]), 3)}
    res["luma_grid_16x9"] = {"reference": gr.round(3).tolist(), "current": gc.round(3).tolist(), "stats": st}
    dl = [abs(v["oklab_L"]) for v in res["delta"].values()]
    res["summary"] = {"mean_abs_dL": round(float(np.mean(dl)), 4), "max_abs_dL": round(float(np.max(dl)), 4),
                      "grid": st}
    (HERE / "out" / f"metrics_{a.tag}.json").write_text(json.dumps(res, indent=1))
    print(f"{'region':16s} {'refL':>6s} {'curL':>6s} {'dL':>7s} {'refC':>6s} {'curC':>6s} {'ref b':>6s} {'cur b':>6s}  ref p10/50/90     cur p10/50/90")
    for k in rc:
        r, c = rc[k], cc[k]
        print(f"{k:16s} {r['oklab_L']:6.3f} {c['oklab_L']:6.3f} {c['oklab_L'] - r['oklab_L']:+7.3f} {r['oklab_C']:6.3f} {c['oklab_C']:6.3f} "
              f"{r['oklab_b']:+6.3f} {c['oklab_b']:+6.3f}  {r['luma_p10']:.2f}/{r['luma_p50']:.2f}/{r['luma_p90']:.2f}  {c['luma_p10']:.2f}/{c['luma_p50']:.2f}/{c['luma_p90']:.2f}")
    print("texture (ref | cur): hp_rms, coherence, grain_deg, f_across")
    for k in rt:
        print(f"  {k:15s} {rt[k]} | {ct.get(k)}")
    print("grid", st, "summary", res["summary"]["mean_abs_dL"], res["summary"]["max_abs_dL"])


if __name__ == "__main__":
    main()
