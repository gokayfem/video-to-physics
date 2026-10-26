"""Look comparison sheet (scene role C): REFERENCE | PREVIOUS | CURRENT full frames, close-ups (clamp, disc, wood) as
ref | prev | cur, backdrop row/column luma profiles and the disc highlight profile, plus a metrics strip.

.venv/bin/python look/compose.py --tag v002 --prev v001 --caption "..." [--frame 0] [--out scene/visuals/...jpg]
"""
import argparse, io, json, pathlib
import numpy as np, cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import metrics as MX

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
FONT = cv2.FONT_HERSHEY_SIMPLEX
BG = (24, 24, 24)


def bar(w, lines, h=None):
    h = h or 10 + 20 * len(lines)
    b = np.full((h, w, 3), BG, np.uint8)
    for i, (t, s, col) in enumerate(lines):
        cv2.putText(b, t, (8, 18 + i * 20), FONT, s, col, 1, cv2.LINE_AA)
    return b


def fitw(img, w):
    return cv2.resize(img, (w, int(w * 1088 / 1920)), interpolation=cv2.INTER_AREA)


def fitbox(img, w, h):
    s = min(w / img.shape[1], h / img.shape[0])
    r = cv2.resize(img, (max(1, int(img.shape[1] * s)), max(1, int(img.shape[0] * s))), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    out = np.full((h, w, 3), BG, np.uint8)
    out[:r.shape[0], :r.shape[1]] = r
    return out


def load_tag(tag):
    info = json.loads((HERE / "out" / f"{tag}_render.json").read_text())
    full = cv2.resize(cv2.imread(info["files"]["full"]), (1920, 1088), interpolation=cv2.INTER_LINEAR)
    crops = {k: cv2.imread(info["files"][k]) for k in info.get("crop_boxes", {})}
    return info, full, crops


def plot(mref, mprev, mcur, w, h):
    fig, ax = plt.subplots(1, 3, figsize=(w / 100, h / 100), dpi=100)
    for m, lab, st in ((mref, "ref", "k-"), (mprev, "prev", "b:"), (mcur, "cur", "r-")):
        if m is None:
            continue
        rows = m["backdrop2d"]["row_profile_ref_cur"]
        cols = m["backdrop2d"]["col_profile_ref_cur"]
        k = 0 if lab == "ref" else 1
        ax[0].plot([r[k] if r else np.nan for r in rows], np.arange(len(rows)), st, label=lab)
        ax[1].plot([c[k] if c else np.nan for c in cols], st, label=lab)
        if "disc_profile" in m:
            dp = m["disc_profile"]["r0.7"]
            ax[2].plot(np.arange(0, 360, 2), dp["ref"] if lab == "ref" else dp["cur"], st, label=lab)
    ax[0].invert_yaxis(); ax[0].set_title("backdrop/table L by grid row", fontsize=8); ax[0].set_xlabel("Oklab L", fontsize=7)
    ax[1].set_title("backdrop/table L by grid column", fontsize=8)
    ax[2].set_title("disc top L on ring r=0.7 (deg from major axis)", fontsize=8)
    for a in ax:
        a.tick_params(labelsize=6); a.grid(alpha=.3); a.legend(fontsize=6)
    fig.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig)
    return cv2.imdecode(np.frombuffer(buf.getvalue(), np.uint8), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--prev", default=None)
    ap.add_argument("--caption", default="")
    ap.add_argument("--status", default="look round, frame through TOR_Cam_Reference (camera of record), disc at the TEST pose from the reference mark (not physics)")
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    ref = MX.ref_frame(a.frame)
    info, cur, ccrops = load_tag(a.tag)
    mcur = json.loads((HERE / "out" / f"metrics_{a.tag}.json").read_text())
    prev, pcrops, mprev = None, {}, None
    if a.prev:
        _, prev, pcrops = load_tag(a.prev)
        p = HERE / "out" / f"metrics_{a.prev}.json"
        mprev = json.loads(p.read_text()) if p.exists() else None
    W = 1920
    pw = W // 3
    tiles = []
    for img, lab in ((ref, f"REFERENCE frame {a.frame}"), (prev, f"PREVIOUS {a.prev}"), (cur, f"CURRENT {a.tag}")):
        t = fitw(img, pw) if img is not None else np.full((int(pw * 1088 / 1920), pw, 3), BG, np.uint8)
        tiles.append(np.vstack([bar(pw, [(lab, 0.55, (255, 255, 255))], 26), t]))
    row1 = np.hstack(tiles)
    rows = [row1]
    cw = W // 3
    crow = []
    for k in ("clamp", "disc", "wood"):
        if k not in info.get("crop_boxes", {}):
            continue
        u0, v0, u1, v1 = info["crop_boxes"][k]
        parts = [ref[v0:v1, u0:u1], pcrops.get(k), ccrops[k]]
        sub = cw // 3
        hh = int(sub * (v1 - v0) / (u1 - u0)) if k != "disc" else int(sub * 0.6)
        ims = [fitbox(p if p is not None else np.zeros_like(parts[0]), sub, max(hh, 60)) for p in parts]
        crow.append(np.vstack([bar(3 * sub, [(f"{k}: ref | prev | cur (100 %)", 0.45, (220, 220, 220))], 22), np.hstack(ims)]))
    if crow:
        H = max(c.shape[0] for c in crow)
        crow = [np.vstack([c, np.full((H - c.shape[0], c.shape[1], 3), BG, np.uint8)]) for c in crow]
        r2 = np.hstack(crow)
        rows.append(np.hstack([r2, np.full((r2.shape[0], W - r2.shape[1], 3), BG, np.uint8)]))
    mref = json.loads(json.dumps(mcur))
    rows.append(plot(mref, mprev, mcur, W, 300))
    lines = [(f"{a.tag}  {a.caption}", 0.55, (255, 255, 255)), (a.status, 0.45, (200, 200, 120))]
    if "regions" in mcur:
        dl = [abs(v["dL"]) for v in mcur["regions"].values()]
        lines.append(("regions dL: " + "  ".join(f"{k} {v['dL']:+.3f}" for k, v in mcur["regions"].items()) + f"   mean|dL| {np.mean(dl):.3f}", 0.4, (200, 200, 200)))
    b = mcur["backdrop2d"]
    s = f"backdrop 2D (32x18 Oklab L, {b['cells']} cells): MAE {b['mae']:.3f} bias {b['bias']:+.3f} corr {b['corr']:.2f}"
    if "disc_profile" in mcur:
        d = mcur["disc_profile"]
        s += f"   disc ring r0.7: MAE {d['r0.7']['mae']:.3f} corr {d['r0.7']['corr']:.2f} std ref/cur {d['r0.7']['std_ref']:.3f}/{d['r0.7']['std_cur']:.3f}"
        s += f"  r0.35: MAE {d['r0.35']['mae']:.3f} corr {d['r0.35']['corr']:.2f}"
    lines.append((s, 0.45, (200, 200, 200)))
    if mprev and "regions" in mprev:
        lines.append((f"previous {a.prev}: mean|dL| {np.mean([abs(v['dL']) for v in mprev['regions'].values()]):.3f}, backdrop MAE {mprev['backdrop2d']['mae']:.3f}, "
                      f"disc r0.7 MAE {mprev.get('disc_profile', {}).get('r0.7', {}).get('mae', float('nan')):.3f}", 0.4, (160, 160, 160)))
    rows.append(bar(W, lines))
    sheet = np.vstack([np.hstack([r, np.full((r.shape[0], W - r.shape[1], 3), BG, np.uint8)]) if r.shape[1] < W else r for r in rows])
    out = pathlib.Path(a.out) if a.out else HERE / "visuals" / f"{a.tag}_look_compare_f{a.frame:03d}.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print("wrote", out)


if __name__ == "__main__":
    main()
