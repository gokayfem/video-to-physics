"""Look metrics, reference vs render (scene role C), clip 03.

.venv/bin/python look/metrics.py --tag v001 [--frame 0]
Inputs: look/out/<tag>_full.png (render through TOR_Cam_Reference at that frame, disc at the measured TEST pose) and
<tag>_render.json. Outputs look/out/metrics_<tag>.json and prints a table.
1. Region colour (Oklab L, a, b, C; 960x544 INTER_AREA) in hand boxes (frame-0 px; valid for frame 0 only).
2. Backdrop/table 2D luma map: Oklab L on a 32x18 grid; cells used = background in BOTH images (>= 90 % of pixels with
   chroma < 0.035 and not dark-brown); MAE, bias, correlation; row profile (mean per grid row) and column profile.
3. Disc highlight profile: the disc top-face ellipse is fitted in each image (brass segmentation, upper silhouette),
   Oklab L sampled on rings r = 0.35 / 0.7 at 180 angles (unwarped-circle angle, 0 = ellipse major axis), median
   filtered over 7 samples (removes the engraved line); MAE, correlation, mean L, and bright-lobe angles.
4. Texture of wood (hp_rms, coherence, low-frequency std) in boxes inside the 100 % wood crop.
"""
import argparse, json, pathlib, sys
import numpy as np, cv2

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
REF = ROOT.parents[1] / "reference-batch/clips/03-torsion-disc.mp4"
sys.path.insert(0, str(ROOT / "scene" / "measure"))
REGIONS = {"backdrop_left": (100, 100, 350, 400), "backdrop_centre": (650, 250, 850, 480), "backdrop_right": (1650, 100, 1900, 400),
           "table_left": (50, 700, 250, 1000), "table_centre": (780, 850, 1150, 1050), "table_right": (1650, 700, 1900, 1000),
           "post_left": (435, 250, 505, 800), "post_right": (1460, 250, 1555, 800), "crossbar": (600, 70, 850, 150),
           "foot_left": (350, 960, 500, 1040), "clamp": (910, 60, 1000, 170)}
TEX = {"post_left": (432, 70, 505, 250), "crossbar": (540, 75, 635, 150)}   # inside the 100 % "wood" crop (380, 0, 640, 260)


def ref_frame(n):
    cap = cv2.VideoCapture(str(REF)); cap.set(cv2.CAP_PROP_POS_FRAMES, n); ok, f = cap.read(); assert ok
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
        L = oklab(img960[v0 // 2:v1 // 2, u0 // 2:u1 // 2])
        out[k] = {"L": round(float(L[..., 0].mean()), 4), "a": round(float(L[..., 1].mean()), 4), "b": round(float(L[..., 2].mean()), 4),
                  "C": round(float(np.hypot(L[..., 1], L[..., 2]).mean()), 4),
                  "hue": round(float(np.degrees(np.arctan2(L[..., 2].mean(), L[..., 1].mean()))), 1)}
    return out


def bg_grid(img, nx=32, ny=18):
    L = oklab(img)
    C = np.hypot(L[..., 1], L[..., 2])
    bg = (C < 0.035) & (L[..., 0] > 0.3)
    h, w = L.shape[:2]
    G, M = np.zeros((ny, nx)), np.zeros((ny, nx), bool)
    for j in range(ny):
        for i in range(nx):
            sl = (slice(j * h // ny, (j + 1) * h // ny), slice(i * w // nx, (i + 1) * w // nx))
            G[j, i] = L[..., 0][sl].mean()
            M[j, i] = bg[sl].mean() >= 0.9
    return G, M


def disc_profile(img, ellipse=None):
    from disc_silhouette import oklab as ok2  # noqa: F401  (same formula)
    L = oklab(img)
    if ellipse is not None:
        ex, ey, a, b, thd = ellipse
        th = np.radians(thd)
        return _rings(L, ex, ey, a, b, th)
    roi = (600, 500, 1400, 880)
    u0, v0, u1, v1 = roi
    c = L[v0:v1, u0:u1]
    hue = np.degrees(np.arctan2(c[..., 2], c[..., 1])); C = np.hypot(c[..., 1], c[..., 2])
    m = ((C > 0.018) & (hue > 50) & (hue < 110)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    k, lab, st, _ = cv2.connectedComponentsWithStats(m)
    if k < 2:
        return None
    i = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    cs, _ = cv2.findContours((lab == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    pts = max(cs, key=cv2.contourArea)[:, 0, :].astype(np.float32) + [u0, v0]
    up = pts[pts[:, 1] < np.median(pts[:, 1]) + 5]
    (ex, ey), (MA, ma), ang = cv2.fitEllipse(np.ascontiguousarray(up, dtype=np.float32))
    a, b = max(MA, ma) / 2, min(MA, ma) / 2
    th = np.radians(ang if MA >= ma else ang + 90)
    return _rings(L, ex, ey, a, b, th)


def _rings(L, ex, ey, a, b, th):
    t = np.radians(np.arange(0, 360, 2.0))
    out = {"ellipse": [float(ex), float(ey), float(a), float(b), float(np.degrees(th))]}
    for r in (0.35, 0.7):
        X, Y = r * a * np.cos(t), r * b * np.sin(t)
        U = ex + X * np.cos(th) - Y * np.sin(th); V = ey + X * np.sin(th) + Y * np.cos(th)
        s = cv2.remap(L[..., 0].astype(np.float32), U.astype(np.float32)[None], V.astype(np.float32)[None], cv2.INTER_LINEAR)[0]
        s = np.array([np.median(np.take(s, range(j - 3, j + 4), mode="wrap")) for j in range(len(s))])
        out[f"r{r}"] = s
    return out


def tex_stats(img, box):
    u0, v0, u1, v1 = box
    Y = (img[v0:v1, u0:u1, ::-1].astype(np.float64) / 255.0) @ [0.2126, 0.7152, 0.0722]
    hp = Y - cv2.GaussianBlur(Y, (0, 0), 3.0)
    gx = cv2.Sobel(cv2.GaussianBlur(Y, (0, 0), 1.0), cv2.CV_64F, 1, 0); gy = cv2.Sobel(cv2.GaussianBlur(Y, (0, 0), 1.0), cv2.CV_64F, 0, 1)
    Jxx, Jyy, Jxy = (gx * gx).mean(), (gy * gy).mean(), (gx * gy).mean()
    tr, det = Jxx + Jyy, Jxx * Jyy - Jxy ** 2
    l1 = tr / 2 + np.sqrt(max(tr * tr / 4 - det, 0)); l2 = tr - l1
    return {"hp_rms": round(float(hp.std()), 4), "coherence": round(float((l1 - l2) / (l1 + l2 + 1e-12)), 3),
            "lowfreq_std": round(float(cv2.GaussianBlur(Y, (0, 0), 6.0).std()), 4)}


def review_stats(img, ell):
    """Visual-review style numbers (VR03-07..09): horizon row (max |dL/dv| of the background profile in side columns
    40-300 and 1620-1880, rows 300-950), wall L left/mid/right and wall chroma/hue above the horizon, table high-pass
    detail (std of L minus Gaussian 3 px) below it, brass chroma on the disc top face, and the engraved mark's dip
    depth / FWHM on vertical profiles (frame-0 geometry: the mark runs roughly along the major axis)."""
    lab = oklab(img); L = lab[..., 0]; C = np.hypot(lab[..., 1], lab[..., 2])
    bg = (C < 0.035) & (L > 0.3)
    cols = np.r_[40:300, 1620:1880]
    Ls = oklab(cv2.GaussianBlur(img, (0, 0), 2))[..., 0]
    prof = np.nanmean(np.where(bg[:, cols], Ls[:, cols], np.nan), 1)
    g = np.gradient(cv2.GaussianBlur(np.nan_to_num(prof, nan=np.nanmean(prof))[:, None].astype(np.float32), (0, 0), 3)[:, 0])
    hz = 300 + int(np.argmax(np.abs(g[300:950])))
    wall = bg.copy(); wall[hz - 15:] = False
    table = bg.copy(); table[:hz + 15] = False
    r = {"horizon_v_px": hz}
    for nm, (u0, u1) in (("wall_left", (40, 400)), ("wall_mid", (700, 1250)), ("wall_right", (1550, 1880))):
        m = wall.copy(); m[:, :u0] = False; m[:, u1:] = False
        r[nm + "_L"] = round(float(L[m].mean()), 4) if m.any() else None
    r["wall_C"] = round(float(np.hypot(lab[..., 1][wall].mean(), lab[..., 2][wall].mean())), 4)
    r["wall_hue_deg"] = round(float(np.degrees(np.arctan2(lab[..., 2][wall].mean(), lab[..., 1][wall].mean()))), 1)
    hp = L - cv2.GaussianBlur(L, (0, 0), 3)
    tm = table.copy(); tm[:, 300:1620] = False
    r["table_detail_hp_std"] = round(float(hp[tm].std()), 5)
    if ell is not None:
        ex, ey, a, b, thd = ell
        yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
        th = np.radians(thd)
        X = (xx - ex) * np.cos(th) + (yy - ey) * np.sin(th); Y = -(xx - ex) * np.sin(th) + (yy - ey) * np.cos(th)
        inside = (X / (0.9 * a)) ** 2 + (Y / (0.9 * b)) ** 2 < 1
        r["brass_C"] = round(float(C[inside].mean()), 4)
        r["brass_C_p05_p95"] = [round(float(np.percentile(C[inside], 5)), 4), round(float(np.percentile(C[inside], 95)), 4)]
        # angular structure on rings r 0.4-0.8: power in harmonics 5..60 over power at harmonic 2 (main 2-lobe pattern)
        ratios = []
        for rr in (0.4, 0.55, 0.7, 0.8):
            t = np.linspace(0, 2 * np.pi, 720, endpoint=False)
            Xr, Yr = rr * a * np.cos(t), rr * b * np.sin(t)
            U = ex + Xr * np.cos(th) - Yr * np.sin(th); V = ey + Xr * np.sin(th) + Yr * np.cos(th)
            sL = cv2.remap(L.astype(np.float32), U.astype(np.float32)[None], V.astype(np.float32)[None], cv2.INTER_LINEAR)[0]
            Fp = np.abs(np.fft.rfft(sL - sL.mean())) ** 2
            ratios.append(Fp[5:61].sum() / max(Fp[2], 1e-12))
        r["fine_over_main_lobe_power"] = round(float(np.median(ratios)), 3)
        r["brass_L"] = round(float(L[inside].mean()), 4)
        Y8 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(float)
        fw, dp = [], []
        for u in list(range(int(ex - 0.8 * a), int(ex - 0.2 * a), 6)) + list(range(int(ex + 0.2 * a), int(ex + 0.8 * a), 6)):
            col = Y8[int(ey) - 30:int(ey) + 31, u]
            base = cv2.GaussianBlur(col[:, None], (1, 0), 6)[:, 0] if False else np.convolve(col, np.ones(15) / 15, "same")
            d = base - col
            i = int(np.argmax(d[8:-8])) + 8
            pk = d[i]
            if pk < 6:
                continue
            h = pk / 2; lo = i; hi = i
            while lo > 0 and d[lo] > h: lo -= 1
            while hi < len(d) - 1 and d[hi] > h: hi += 1
            fw.append(hi - lo - 1); dp.append(pk)
        r["mark_fwhm_px"] = round(float(np.median(fw)), 2) if fw else None
        r["mark_dip"] = round(float(np.median(dp)), 1) if dp else None
        r["mark_samples"] = len(fw)
    return r


def compute(tag, frame):
    ref = ref_frame(frame)
    info = json.loads((HERE / "out" / f"{tag}_render.json").read_text())
    cur_s = cv2.imread(info["files"]["full"])
    cur = cv2.resize(cur_s, (1920, 1088), interpolation=cv2.INTER_LINEAR)
    ref960, cur960 = cv2.resize(ref, (960, 544), interpolation=cv2.INTER_AREA), cv2.resize(cur_s, (960, 544), interpolation=cv2.INTER_AREA)
    res = {"tag": tag, "frame": frame, "render": info}
    if frame == 0:
        rc, cc = colour_stats(ref960, REGIONS), colour_stats(cur960, REGIONS)
        res["regions"] = {k: {"ref": rc[k], "cur": cc[k], "dL": round(cc[k]["L"] - rc[k]["L"], 4), "dC": round(cc[k]["C"] - rc[k]["C"], 4),
                              "dhue": round(cc[k]["hue"] - rc[k]["hue"], 1)} for k in rc}
    Gr, Mr = bg_grid(ref960); Gc, Mc = bg_grid(cur960)
    M = Mr & Mc
    d = (Gc - Gr)[M]
    rowp = [(round(float(Gr[j][M[j]].mean()), 3), round(float(Gc[j][M[j]].mean()), 3)) if M[j].any() else None for j in range(M.shape[0])]
    colp = [(round(float(Gr[:, i][M[:, i]].mean()), 3), round(float(Gc[:, i][M[:, i]].mean()), 3)) if M[:, i].any() else None for i in range(M.shape[1])]
    res["backdrop2d"] = {"cells": int(M.sum()), "mae": round(float(np.abs(d).mean()), 4), "bias": round(float(d.mean()), 4),
                         "corr": round(float(np.corrcoef(Gr[M], Gc[M])[0, 1]), 3), "row_profile_ref_cur": rowp, "col_profile_ref_cur": colp,
                         "grid_ref": Gr.round(3).tolist(), "grid_cur": Gc.round(3).tolist(), "mask": M.astype(int).tolist()}
    pr, pc = disc_profile(ref), disc_profile(cur)
    if pr and pc and abs(pc["ellipse"][3] - pr["ellipse"][3]) > 0.15 * pr["ellipse"][3]:
        # segmentation failure on the render (dark cone gaps fall below the chroma threshold): sample the render on the
        # reference ellipse (valid when rendered through the camera of record; flagged in the output)
        pc = disc_profile(cur, ellipse=pr["ellipse"])
        pc["ellipse_from_reference"] = True
    if pr and pc:
        dp = {"ellipse_ref": pr["ellipse"], "ellipse_cur": pc["ellipse"], "cur_sampled_on_ref_ellipse": bool(pc.get("ellipse_from_reference"))}
        for r in ("r0.35", "r0.7"):
            dp[r] = {"mae": round(float(np.abs(pc[r] - pr[r]).mean()), 4), "corr": round(float(np.corrcoef(pr[r], pc[r])[0, 1]), 3),
                     "mean_ref": round(float(pr[r].mean()), 4), "mean_cur": round(float(pc[r].mean()), 4),
                     "std_ref": round(float(pr[r].std()), 4), "std_cur": round(float(pc[r].std()), 4),
                     "ref": pr[r].round(3).tolist(), "cur": pc[r].round(3).tolist()}
        res["disc_profile"] = dp
    ell = res.get("disc_profile", {}).get("ellipse_ref")
    disc_box = info.get("crop_boxes", {}).get("disc")
    cur_full = cur.copy()
    for k, b in info.get("crop_boxes", {}).items():                 # 100 % crops for the review numbers
        cur_full[b[1]:b[3], b[0]:b[2]] = cv2.imread(info["files"][k])
    res["review"] = {"ref": review_stats(ref, ell), "cur": review_stats(cur_full, ell),
                     "note": "cur uses the 100 % crops where rendered (disc, clamp, wood) and the 50 % full frame elsewhere"}
    if frame == 0 and "wood" in info.get("crop_boxes", {}):
        wb = info["crop_boxes"]["wood"]
        canvas = cur.copy()
        canvas[wb[1]:wb[3], wb[0]:wb[2]] = cv2.imread(info["files"]["wood"])
        res["texture"] = {k: {"ref": tex_stats(ref, b), "cur": tex_stats(canvas, b)} for k, b in TEX.items()}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--frame", type=int, default=0)
    a = ap.parse_args()
    res = compute(a.tag, a.frame)
    (HERE / "out" / f"metrics_{a.tag}.json").write_text(json.dumps(res, indent=1))
    if "regions" in res:
        print(f"{'region':16s} {'refL':>6s} {'curL':>6s} {'dL':>7s} {'refC':>6s} {'curC':>6s} {'refh':>6s} {'curh':>6s}")
        for k, v in res["regions"].items():
            print(f"{k:16s} {v['ref']['L']:6.3f} {v['cur']['L']:6.3f} {v['dL']:+7.3f} {v['ref']['C']:6.3f} {v['cur']['C']:6.3f} {v['ref']['hue']:6.0f} {v['cur']['hue']:6.0f}")
        dl = [abs(v["dL"]) for v in res["regions"].values()]
        print("mean|dL|", round(float(np.mean(dl)), 4), "max", round(float(np.max(dl)), 4))
    b = res["backdrop2d"]
    print("backdrop2d cells", b["cells"], "mae", b["mae"], "bias", b["bias"], "corr", b["corr"])
    print(" rows ref/cur:", [x for x in b["row_profile_ref_cur"]])
    print(" cols ref/cur:", [x for x in b["col_profile_ref_cur"][::3]])
    if "disc_profile" in res:
        for r in ("r0.35", "r0.7"):
            q = res["disc_profile"][r]
            print(f" disc {r}: mae {q['mae']} corr {q['corr']} mean {q['mean_ref']}/{q['mean_cur']} std {q['std_ref']}/{q['std_cur']}")
        print(" ellipse ref", np.round(res["disc_profile"]["ellipse_ref"], 1), "cur", np.round(res["disc_profile"]["ellipse_cur"], 1))
    rv = res["review"]
    print(" review ref:", rv["ref"]); print(" review cur:", rv["cur"])
    if "texture" in res:
        for k, v in res["texture"].items():
            print(" tex", k, v["ref"], "|", v["cur"])


if __name__ == "__main__":
    main()
