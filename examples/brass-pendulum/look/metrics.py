"""Region metrics, reference vs render (Look role).

.venv/bin/python look/metrics.py --tag v001 [--frame 0]
Colour stats per region (mean sRGB, Oklab L/a/b/C, luma p10/p50/p90) at 960x544 on the reference (INTER_AREA
downscale of the decoded frame) and on look/out/<tag>_full.png, with the region polygons of camera/look_measure.py
mapped to the frame by the camera role's drift similarity (same polygons for the render: it uses the frame-0
fitted camera, silhouette IoU 0.96). Bob disc: reference from measurement/bob_track.csv, render from the projected
bob in <tag>_render.json (radius x 0.8).
Texture stats on full-res close-up crops (same boxes in both images; boxes in look/render_look.py CROPS):
  hp_rms       RMS of luma minus Gaussian(sigma 3 px) inside the texture box (fine-texture contrast, sRGB luma units)
  coherence    structure-tensor coherence (lambda1-lambda2)/(lambda1+lambda2) (1 = perfectly straight streaks)
  grain_deg    dominant streak orientation in the image (deg, 0 = horizontal, 90 = vertical)
  f_across     power-weighted mean spatial frequency across the streaks (cycles per px, 0.01-0.35 band)
Writes look/out/metrics_<tag>.json.
"""
import argparse, json, pathlib, sys
import numpy as np, cv2

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "camera"))
import look_measure as LM  # noqa: E402  (camera role's region definitions; read-only reuse)

REF = ROOT.parents[1] / "reference-batch/clips/01-brass-pendulum.mp4"
# texture boxes (full-res px) inside the close-up crops: flat parts of the faces, away from edges
TEX = {"post_front": (705, 280, 790, 560), "base_front": (560, 960, 1000, 1030), "base_top": (600, 893, 980, 925),
       "arm_front": (850, 95, 1000, 135), "base_end": (1300, 975, 1430, 1035)}


def ref_frame(n):
    cap = cv2.VideoCapture(str(REF))
    cap.set(cv2.CAP_PROP_POS_FRAMES, n)
    ok, f = cap.read()
    assert ok
    return f


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
    grad_dir = 0.5 * np.degrees(np.arctan2(2 * Jxy, Jxx - Jyy))      # dominant gradient direction
    grain = (grad_dir + 90) % 180                                      # streaks run perpendicular to it
    h, w = hp.shape
    win = np.outer(np.hanning(h), np.hanning(w))
    P = np.abs(np.fft.fftshift(np.fft.fft2(hp * win))) ** 2
    fy = np.fft.fftshift(np.fft.fftfreq(h))[:, None]; fx = np.fft.fftshift(np.fft.fftfreq(w))[None, :]
    fr = np.hypot(fx, fy)
    th = np.radians(grad_dir)
    f_par = np.abs(fx * np.cos(th) + fy * np.sin(th))                  # along the gradient = across streaks
    band = (fr > 0.01) & (fr < 0.35)
    f_across = float((P * f_par)[band].sum() / P[band].sum())
    return {"hp_rms": round(float(hp.std()), 4), "coherence": round(float(coh), 3), "grain_deg": round(float(grain), 1),
            "f_across_cyc_px": round(f_across, 4)}


def colour_stats(img960, bob_full, frame):
    M = LM.masks(img960.shape, LM.sim_at(frame), 0.5, bob_full)
    return LM.stats(img960, M)


BG_CELLS = [(r, c) for r in range(0, 7) for c in list(range(0, 5)) + list(range(11, 16))] + [(0, c) for c in range(5, 11)]
TABLE_CELLS = [(8, c) for c in (0, 1, 2, 3, 13, 14, 15)]


def grid(img):
    Y = (img[..., ::-1].astype(np.float64) / 255.0) @ [0.2126, 0.7152, 0.0722]
    return cv2.resize(Y, (16, 9), interpolation=cv2.INTER_AREA)


def grid_stats(gr, gc):
    d = {}
    for name, cells in (("backdrop", BG_CELLS), ("table", TABLE_CELLS)):
        r = np.array([gr[i] for i in cells]); c = np.array([gc[i] for i in cells])
        d[name] = {"mae": round(float(np.abs(c - r).mean()), 4), "bias": round(float((c - r).mean()), 4),
                   "corr": round(float(np.corrcoef(r, c)[0, 1]), 3), "ref_range": round(float(r.max() - r.min()), 3),
                   "cur_range": round(float(c.max() - c.min()), 3)}
    return d


def measure_ref(frame):
    f = ref_frame(frame)
    small = cv2.resize(f, (960, 544), interpolation=cv2.INTER_AREA)
    col = colour_stats(small, LM.bob_at(frame), frame)
    tex = {k: tex_stats(f, b) for k, b in TEX.items()}
    return col, tex, f


def measure_render(tag, frame):
    info = json.loads((HERE / "out" / f"{tag}_render.json").read_text())
    full = cv2.imread(info["files"]["full"])
    small = cv2.resize(full, (960, 544), interpolation=cv2.INTER_AREA) if full.shape[1] != 960 else full
    col = colour_stats(small, tuple(info["bob_px"]), frame)
    tex = {}
    boxes = info.get("crop_boxes", {})
    for k, b in TEX.items():
        for cname, cb in boxes.items():   # texture box must lie inside a rendered full-res crop
            if b[0] >= cb[0] and b[1] >= cb[1] and b[2] <= cb[2] and b[3] <= cb[3]:
                crop = cv2.imread(info["files"][cname])
                canvas = np.zeros((1088, 1920, 3), np.uint8)
                canvas[cb[1]:cb[1] + crop.shape[0], cb[0]:cb[0] + crop.shape[1]] = crop
                tex[k] = tex_stats(canvas, b)
                break
        else:
            if full.shape[1] == 1920:
                tex[k] = tex_stats(full, b)
    return col, tex, info


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--frame", type=int, default=0)
    a = ap.parse_args()
    rc, rt, _ = measure_ref(a.frame)
    cc, ct, info = measure_render(a.tag, a.frame)
    res = {"tag": a.tag, "frame": a.frame, "t_s": a.frame / 24, "render": info, "method": __doc__,
           "reference": {"colour": rc, "texture": rt}, "current": {"colour": cc, "texture": ct}, "delta": {}}
    for k in rc:
        if k in cc:
            res["delta"][k] = {q: round(cc[k][q] - rc[k][q], 4) for q in ("oklab_L", "oklab_a", "oklab_b", "oklab_C",
                                                                          "luma_p10", "luma_p50", "luma_p90")}
    gr = grid(ref_frame(a.frame)); gc = grid(cv2.imread(info["files"]["full"]))
    res["luma_grid_16x9"] = {"reference": gr.round(3).tolist(), "current": gc.round(3).tolist(), "stats": grid_stats(gr, gc),
                             "note": "luma of 16x9 INTER_AREA cells; backdrop cells = rows 0-6 x cols 0-4,11-15 + row 0; table cells = row 8 edges"}
    print("grid", res["luma_grid_16x9"]["stats"])
    out = HERE / "out" / f"metrics_{a.tag}.json"
    out.write_text(json.dumps(res, indent=1))
    print(f"{'region':17s} {'refL':>6s} {'curL':>6s} {'dL':>6s} {'refC':>6s} {'curC':>6s} {'ref b':>6s} {'cur b':>6s}  ref p10/50/90      cur p10/50/90")
    for k in rc:
        if k in cc:
            r, c = rc[k], cc[k]
            print(f"{k:17s} {r['oklab_L']:6.3f} {c['oklab_L']:6.3f} {c['oklab_L']-r['oklab_L']:+6.3f} {r['oklab_C']:6.3f} {c['oklab_C']:6.3f} "
                  f"{r['oklab_b']:+6.3f} {c['oklab_b']:+6.3f}  {r['luma_p10']:.2f}/{r['luma_p50']:.2f}/{r['luma_p90']:.2f}   {c['luma_p10']:.2f}/{c['luma_p50']:.2f}/{c['luma_p90']:.2f}")
    print("texture (ref | cur): hp_rms, coherence, grain_deg, f_across")
    for k in rt:
        print(f"{k:12s} {rt[k]} | {ct.get(k)}")
