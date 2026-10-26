"""Compose the Look comparison visual: REFERENCE / PREVIOUS / CURRENT + close-ups + metrics table.

.venv/bin/python look/compose.py --tag v002 --prev v001 --caption "..." [--status "integrated look - draft"]
--prev may be a tag (look/out/<tag>_full.png and crops) or a path to a png (e.g. the geometry blockout).
Writes look/visuals/<tag>_look_compare_f000.jpg (1920 wide).
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
    for i, (t, sc_, col) in enumerate(lines):
        cv2.putText(bar, t, (8, 16 + i * 18), FONT, sc_, col, 1, cv2.LINE_AA)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--prev", required=True)
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--caption", default="")
    ap.add_argument("--status", default="integrated look - draft")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    t = a.frame / 24
    ref = MX.ref_frame(a.frame)
    info = json.loads((HERE / "out" / f"{a.tag}_render.json").read_text())
    cur = cv2.imread(info["files"]["full"])
    pinfo = None
    if a.prev.endswith(".png") or a.prev.endswith(".jpg"):
        prev = cv2.imread(str(ROOT / a.prev) if not a.prev.startswith("/") else a.prev); ptag = pathlib.Path(a.prev).stem
    else:
        pinfo = json.loads((HERE / "out" / f"{a.prev}_render.json").read_text())
        prev = cv2.imread(pinfo["files"]["full"]); ptag = a.prev
    W = 640
    posed = f", posed theta={info['posed_theta_deg']} deg" if info.get("posed_theta_deg") is not None else ""
    top = np.hstack([
        label(fit(ref, W), [(f"REFERENCE (AI clip) frame {a.frame} (t={t:.3f}s)", 0.5, (255, 255, 255)), ("observed", 0.4, (180, 180, 180))]),
        label(fit(prev, W), [(f"PREVIOUS {ptag} frame {a.frame}", 0.5, (255, 255, 255)), ("previous render", 0.4, (180, 180, 180))]),
        label(fit(cur, W), [(f"CURRENT {a.tag} frame {a.frame} (t={t:.3f}s)", 0.5, (120, 255, 120)),
                            (f"{a.status}; {info['engine'].replace('BLENDER_', '')} {info['view_transform']}{posed}", 0.4, (180, 180, 180))])])
    # close-ups: for each crop: ref | prev | cur
    rows = []
    boxes = info.get("crop_boxes", {})
    names = [k for k in ("post", "arm_pin", "bob", "base") if k in boxes]
    tiles = []
    for k in names:
        u0, v0, u1, v1 = boxes[k]
        if k == "bob":   # bob crops follow the bob in each image (reference: measured track; render: projection)
            bu, bv, br = MX.LM.bob_at(a.frame); hh = (u1 - u0) // 2
            r = ref[int(bv) - hh:int(bv) + hh, int(bu) - hh:int(bu) + hh]
        else:
            r = ref[v0:v1, u0:u1]
        c = cv2.imread(info["files"][k])
        p = cv2.imread(pinfo["files"][k]) if pinfo and k in pinfo.get("files", {}) else None
        if p is None:
            pp = prev if prev.shape[1] == 1920 else cv2.resize(prev, (1920, 1088), interpolation=cv2.INTER_LINEAR)
            p = pp[v0:v1, u0:u1]
            ptxt = "prev (upscaled)" if prev.shape[1] != 1920 else "prev"
        else:
            ptxt = "prev"
        tiles.append((k, r, p, c, ptxt))
    # layout: post, arm_pin, bob in one row of 3 groups (each group ref|prev|cur); base on its own row
    def group(k, r, p, c, ptxt, gw, gh):
        cells = []
        for img, txt, col in ((r, f"ref {k}", (255, 255, 255)), (p, ptxt, (200, 200, 200)), (c, f"{a.tag} {k}", (120, 255, 120))):
            cell = fitbox(img, gw // 3 - 4, gh)
            cell = label(cell, [(txt, 0.4, col)], h=20)
            cells.append(np.pad(cell, ((0, 0), (2, 2), (0, 0)), constant_values=24))
        return np.hstack(cells)
    small = [tt for tt in tiles if tt[0] != "base"]
    if small:
        gw = 1920 // len(small)
        row = np.hstack([group(*tt, gw, 300) for tt in small])
        row = cv2.copyMakeBorder(row, 0, 0, 0, 1920 - row.shape[1], cv2.BORDER_CONSTANT, value=BG)
        rows.append(row)
    for tt in tiles:
        if tt[0] == "base":
            rows.append(group(*tt, 1920, 200))
    # metrics table
    mfile = HERE / "out" / f"metrics_{a.tag}.json"
    M = json.loads(mfile.read_text())
    lines = [f"metrics {a.tag} frame {a.frame} (960x544 colour; full-res texture)  region: Oklab L ref/cur, chroma C ref/cur, luma p10/p50/p90 ref | cur"]
    rc, cc = M["reference"]["colour"], M["current"]["colour"]
    for k in rc:
        if k in cc:
            r, c = rc[k], cc[k]
            lines.append(f"{k:16s} L {r['oklab_L']:.3f}/{c['oklab_L']:.3f} ({c['oklab_L']-r['oklab_L']:+.3f})  C {r['oklab_C']:.3f}/{c['oklab_C']:.3f}  "
                         f"b {r['oklab_b']:+.3f}/{c['oklab_b']:+.3f}  p {r['luma_p10']:.2f}/{r['luma_p50']:.2f}/{r['luma_p90']:.2f} | {c['luma_p10']:.2f}/{c['luma_p50']:.2f}/{c['luma_p90']:.2f}")
    rt, ct = M["reference"]["texture"], M["current"]["texture"]
    tl = ["texture (ref | cur): hp_rms, coherence, grain deg, f_across cyc/px"]
    for k in rt:
        c = ct.get(k)
        if c:
            tl.append(f"{k:11s} {rt[k]['hp_rms']:.4f}|{c['hp_rms']:.4f}  {rt[k]['coherence']:.2f}|{c['coherence']:.2f}  "
                      f"{rt[k]['grain_deg']:5.1f}|{c['grain_deg']:5.1f}  {rt[k]['f_across_cyc_px']:.3f}|{c['f_across_cyc_px']:.3f}")
    n = max(len(lines), len(tl))
    tab = np.full((22 + 17 * n, 1920, 3), BG, np.uint8)
    for i, s in enumerate(lines):
        cv2.putText(tab, s, (8, 18 + 17 * i), FONT, 0.42, (230, 230, 230), 1, cv2.LINE_AA)
    for i, s in enumerate(tl):
        cv2.putText(tab, s, (1180, 18 + 17 * i), FONT, 0.42, (230, 230, 230), 1, cv2.LINE_AA)
    cap = textwrap.wrap(a.caption, 210)
    capbar = np.full((14 + 20 * max(1, len(cap)), 1920, 3), (40, 40, 40), np.uint8)
    for i, s in enumerate(cap):
        cv2.putText(capbar, s, (8, 18 + 20 * i), FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    head = np.full((30, 1920, 3), (60, 40, 20), np.uint8)
    cv2.putText(head, f"LOOK {a.tag} | frame {a.frame} t={t:.3f}s | status: {a.status} | camera: {info['camera']} (approx, framing not look's job)",
                (8, 21), FONT, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    img = np.vstack([head, top] + rows + [tab, capbar])
    out = a.out or str(HERE / "visuals" / f"{a.tag}_look_compare_f{a.frame:03d}.jpg")
    pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(out, img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(out, img.shape)


if __name__ == "__main__":
    main()
