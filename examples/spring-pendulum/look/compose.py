"""Look comparison sheet: REFERENCE / PREVIOUS / CURRENT + close-ups (ref | prev | cur) + metrics table.

.venv/bin/python look/compose.py --tag v002 --prev v001 --caption "..." [--status "..."]
--prev is a look tag (look/out/<tag>_full.png + crops) or a png path. Writes look/visuals/<tag>_look_compare_fNNN.jpg.
"""
import argparse, json, pathlib, textwrap
import numpy as np, cv2
import metrics as MX

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
FONT = cv2.FONT_HERSHEY_SIMPLEX
BG = (24, 24, 24)


def label(img, lines, h=40):
    bar = np.full((h, img.shape[1], 3), BG, np.uint8)
    for i, (t, s, col) in enumerate(lines):
        cv2.putText(bar, t, (8, 16 + i * 18), FONT, s, col, 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def fit(img, w):
    return cv2.resize(img, (w, int(round(img.shape[0] * w / img.shape[1]))), interpolation=cv2.INTER_AREA)


def fitbox(img, w, h):
    s = min(w / img.shape[1], h / img.shape[0])
    r = cv2.resize(img, (max(1, int(img.shape[1] * s)), max(1, int(img.shape[0] * s))),
                   interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LANCZOS4)
    out = np.full((h, w, 3), BG, np.uint8)
    out[:r.shape[0], :r.shape[1]] = r
    return out


def crops_of(info, full):
    d = {}
    for k, b in info.get("crop_boxes", {}).items():
        d[k] = cv2.imread(info["files"][k])
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--prev", required=True)
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--caption", default="")
    ap.add_argument("--status", default="integrated look - draft")
    a = ap.parse_args()
    t = a.frame / 24
    ref = MX.ref_frame(a.frame)
    info = json.loads((HERE / "out" / f"{a.tag}_render.json").read_text())
    cur = cv2.imread(info["files"]["full"])
    pinfo = None
    if a.prev.endswith((".png", ".jpg")):
        prev = cv2.imread(a.prev if a.prev.startswith("/") else str(ROOT / a.prev)); ptag = pathlib.Path(a.prev).stem
    else:
        pinfo = json.loads((HERE / "out" / f"{a.prev}_render.json").read_text())
        prev = cv2.imread(pinfo["files"]["full"]); ptag = a.prev
    W = 640
    pose = "weight posed at the reference's measured H (TEST pose)" if info.get("pose") else "baked"
    top = np.hstack([
        label(fit(ref, W), [(f"REFERENCE (AI clip) frame {a.frame} (t={t:.3f}s)", 0.5, (255, 255, 255)), ("observed", 0.4, (180, 180, 180))]),
        label(fit(prev, W), [(f"PREVIOUS {ptag} frame {a.frame}", 0.5, (255, 255, 255)), ("previous look round", 0.4, (180, 180, 180))]),
        label(fit(cur, W), [(f"CURRENT {a.tag} frame {a.frame} (t={t:.3f}s)", 0.5, (120, 255, 120)),
                            (f"{info['engine'].replace('BLENDER_', '')} {info['view_transform']}; {pose}", 0.4, (180, 180, 180))])])
    boxes = info.get("crop_boxes", {})
    names = [k for k in ("spring", "weight", "post", "top", "base", "table") if k in boxes]
    cells_all = []
    for k in names:
        u0, v0, u1, v1 = boxes[k]
        r = ref[v0:v1, u0:u1]
        c = cv2.imread(info["files"][k])
        if pinfo and k in pinfo.get("files", {}):
            p, ptxt = cv2.imread(pinfo["files"][k]), "prev"
        else:
            pp = cv2.resize(prev, (1920, 1088), interpolation=cv2.INTER_LINEAR)
            p, ptxt = pp[v0:v1, u0:u1], "prev (upscaled)"
        cells_all.append((k, r, p, c, ptxt))

    def group(k, r, p, c, ptxt, gw, gh):
        cells = []
        for img, txt, col in ((r, f"ref {k}", (255, 255, 255)), (p, ptxt, (200, 200, 200)), (c, f"{a.tag} {k}", (120, 255, 120))):
            cell = fitbox(img, gw // 3 - 4, gh)
            cell = label(cell, [(txt, 0.4, col)], h=20)
            cells.append(np.pad(cell, ((0, 0), (2, 2), (0, 0)), constant_values=24))
        return np.hstack(cells)
    rows = []
    tall = [x for x in cells_all if x[0] in ("spring", "post", "top")]
    wide = [x for x in cells_all if x[0] in ("weight", "base", "table")]
    for grp, gh in ((tall, 360), (wide, 230)):
        if grp:
            gw = 1920 // len(grp)
            row = np.hstack([group(*x, gw, gh) for x in grp])
            rows.append(cv2.copyMakeBorder(row, 0, 0, 0, 1920 - row.shape[1], cv2.BORDER_CONSTANT, value=BG))
    M = json.loads((HERE / "out" / f"metrics_{a.tag}.json").read_text())
    rc, cc = M["reference"]["colour"], M["current"]["colour"]
    lines = [f"metrics {a.tag} frame {a.frame}: region  Oklab L ref/cur (dL)  chroma C ref/cur  b ref/cur  luma p10/p50/p90 ref | cur"]
    for k in rc:
        r, c = rc[k], cc[k]
        lines.append(f"{k:16s} L {r['oklab_L']:.3f}/{c['oklab_L']:.3f} ({c['oklab_L'] - r['oklab_L']:+.3f})  C {r['oklab_C']:.3f}/{c['oklab_C']:.3f}  "
                     f"b {r['oklab_b']:+.3f}/{c['oklab_b']:+.3f}  p {r['luma_p10']:.2f}/{r['luma_p50']:.2f}/{r['luma_p90']:.2f} | {c['luma_p10']:.2f}/{c['luma_p50']:.2f}/{c['luma_p90']:.2f}")
    rt, ct = M["reference"]["texture"], M["current"]["texture"]
    tl = ["texture (ref | cur): hp_rms, coherence, grain deg, f_across cyc/px"]
    for k in rt:
        c = ct.get(k)
        if c:
            tl.append(f"{k:14s} {rt[k]['hp_rms']:.4f}|{c['hp_rms']:.4f}  {rt[k]['coherence']:.2f}|{c['coherence']:.2f}  "
                      f"{rt[k]['grain_deg']:5.1f}|{c['grain_deg']:5.1f}  {rt[k]['f_across_cyc_px']:.3f}|{c['f_across_cyc_px']:.3f}")
    g = M["luma_grid_16x9"]["stats"]
    tl.append(f"luma grid 16x9: backdrop MAE {g['backdrop']['mae']:.3f} bias {g['backdrop']['bias']:+.3f} corr {g['backdrop']['corr']:.2f}; "
              f"table MAE {g['table']['mae']:.3f} bias {g['table']['bias']:+.3f}")
    tl.append(f"mean |dL| {M['summary']['mean_abs_dL']:.3f}, max |dL| {M['summary']['max_abs_dL']:.3f}")
    n = max(len(lines), len(tl))
    tab = np.full((22 + 17 * n, 1920, 3), BG, np.uint8)
    for i, s in enumerate(lines):
        cv2.putText(tab, s, (8, 18 + 17 * i), FONT, 0.42, (230, 230, 230), 1, cv2.LINE_AA)
    for i, s in enumerate(tl):
        cv2.putText(tab, s, (1150, 18 + 17 * i), FONT, 0.42, (230, 230, 230), 1, cv2.LINE_AA)
    cap = textwrap.wrap(a.caption, 200)
    capbar = np.full((14 + 20 * max(1, len(cap)), 1920, 3), (40, 40, 40), np.uint8)
    for i, s in enumerate(cap):
        cv2.putText(capbar, s, (8, 18 + 20 * i), FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    head = np.full((30, 1920, 3), (60, 40, 20), np.uint8)
    cv2.putText(head, f"LOOK {a.tag} | frame {a.frame} t={t:.3f}s | status: {a.status} | camera: {info['camera']}",
                (8, 21), FONT, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    img = np.vstack([head, top] + rows + [tab, capbar])
    out = ROOT / "scene" / "visuals" / f"{a.tag}_look_compare_f{a.frame:03d}.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(out, img.shape)


if __name__ == "__main__":
    main()
