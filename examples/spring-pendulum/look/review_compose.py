"""Review comparison sheet REFERENCE | ck001 | NEW at one frame + close-ups + region metrics + backdrop profile metric.
.venv/bin/python look/review_compose.py --frame 0 --new new_v014 --version v015 --caption "..."
Regions/points are frame-0 pixel definitions (look/metrics.py REGIONS, look/backdrop_profile.py PTS) mapped to frame n by
the measurement role's drift homography (measurement/cache/H_ecc.npy). Weight/spring regions use the weight track."""
import argparse, json, pathlib, sys, textwrap
import numpy as np, cv2
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import metrics as MX
import backdrop_profile as BP
FONT = cv2.FONT_HERSHEY_SIMPLEX
BG = (24, 24, 24)
Hs = np.load(ROOT / "measurement/cache/H_ecc.npy")
WT = np.genfromtxt(ROOT / "measurement/weight_track.csv", delimiter=",", names=True)


def mp(n, u, v):
    q = Hs[n] @ np.array([u, v, 1.0]); return q[0] / q[2], q[1] / q[2]


def regions(n):
    R = {}
    for k, (u0, v0, u1, v1) in MX.REGIONS.items():
        if k in ("weight_side", "spring_coils"):
            continue
        a, b = mp(n, u0, v0), mp(n, u1, v1)
        R[k] = (int(a[0]) // 2 * 2, int(a[1]) // 2 * 2, int(b[0]) // 2 * 2, int(b[1]) // 2 * 2)
    u, v = WT["u_px"][n], WT["v_px"][n]
    R["weight_side"] = (int(u - 40) // 2 * 2, int(v - 50) // 2 * 2, int(u + 40) // 2 * 2, int(v + 40) // 2 * 2)
    return R


def lab(img, lines, h=40):
    bar = np.full((h, img.shape[1], 3), BG, np.uint8)
    for i, (t, s, c) in enumerate(lines):
        cv2.putText(bar, t, (8, 16 + i * 18), FONT, s, c, 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def fitbox(img, w, h):
    s = min(w / img.shape[1], h / img.shape[0])
    r = cv2.resize(img, (max(1, int(img.shape[1] * s)), max(1, int(img.shape[0] * s))), interpolation=cv2.INTER_AREA)
    o = np.full((h, w, 3), BG, np.uint8); o[:r.shape[0], :r.shape[1]] = r; return o


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--frame", type=int, default=0); ap.add_argument("--new", required=True)
    ap.add_argument("--old", default="ck001"); ap.add_argument("--version", required=True); ap.add_argument("--caption", default="")
    ap.add_argument("--old-label", default="baked dynamics, look v010"); ap.add_argument("--new-label", default="same bake, new look")
    ap.add_argument("--title", default="review fixes VR-04..08")
    a = ap.parse_args(); n = a.frame; t = n / 24
    D = HERE / "out" / "review"
    I = {k: json.loads((D / f"{k}_review.json").read_text())["frames"][str(n)] for k in (a.old, a.new)}
    ref = MX.ref_frame(n)
    full = {k: cv2.resize(cv2.imread(I[k]["files"]["full"]), (1920, 1088)) for k in I}
    W = 640
    top = np.hstack([lab(cv2.resize(ref, (W, 363), interpolation=cv2.INTER_AREA), [(f"REFERENCE (AI clip) frame {n} (t={t:.3f}s)", 0.5, (255, 255, 255)), ("observed", 0.4, (180, 180, 180))]),
                     lab(cv2.resize(full[a.old], (W, 363), interpolation=cv2.INTER_AREA), [(f"{a.old} (as reviewed) frame {n}", 0.5, (255, 255, 255)), (a.old_label, 0.4, (180, 180, 180))]),
                     lab(cv2.resize(full[a.new], (W, 363), interpolation=cv2.INTER_AREA), [(f"NEW {a.version} frame {n}", 0.5, (120, 255, 120)), (a.new_label, 0.4, (180, 180, 180))])])
    rows = []
    for k in ("hook_spring", "post", "weight"):
        u0, v0, u1, v1 = I[a.new]["crops"][k]
        cells = []
        for img, txt, col in ((ref[v0:v1, u0:u1], f"ref {k}", (255, 255, 255)), (cv2.imread(I[a.old]["files"][k]), f"{a.old} {k}", (200, 200, 200)),
                              (cv2.imread(I[a.new]["files"][k]), f"{a.version} {k}", (120, 255, 120))):
            c = lab(fitbox(img, 206, 400), [(txt, 0.4, col)], h=20)
            cells.append(np.pad(c, ((0, 0), (2, 2), (0, 0)), constant_values=24))
        rows.append(np.hstack(cells))
    close = np.hstack(rows)
    close = cv2.copyMakeBorder(close, 0, 0, 0, 1920 - close.shape[1], cv2.BORDER_CONSTANT, value=BG)
    # metrics
    R = regions(n)
    small = lambda im: cv2.resize(im, (960, 544), interpolation=cv2.INTER_AREA)
    rc = MX.colour_stats(small(ref), R)
    oc = MX.colour_stats(small(full[a.old]), R)
    nc = MX.colour_stats(small(full[a.new]), R)
    # backdrop profile on mapped points
    pts = {k: [tuple(int(x) for x in mp(n, u, v)) for (u, v) in P] for k, P in BP.PTS.items()}
    BP.PTS = pts
    bo, bn = BP.profiles(ref, full[a.old]), BP.profiles(ref, full[a.new])
    lines = [f"region (frame {n})   ref L | {a.old} dL | NEW dL      ref C | {a.old} C | NEW C"]
    res = {"frame": n, "regions": R, "reference": rc, a.old: oc, a.new: nc, "backdrop_profile": {a.old: bo, a.new: bn}}
    for k in rc:
        lines.append(f"{k:16s} {rc[k]['oklab_L']:.3f} | {oc[k]['oklab_L'] - rc[k]['oklab_L']:+.3f} | {nc[k]['oklab_L'] - rc[k]['oklab_L']:+.3f}"
                     f"      {rc[k]['oklab_C']:.3f} | {oc[k]['oklab_C']:.3f} | {nc[k]['oklab_C']:.3f}")
    ho = lambda c: np.degrees(np.arctan2(c["oklab_b"], c["oklab_a"]))
    tl = [f"backdrop profile (VR-04), mean |dL| over {sum(len(v) for v in pts.values())} pts: {a.old} {bo['summary']['mae_all']:.3f} -> NEW {bn['summary']['mae_all']:.3f}",
          f"  max |dL|: {a.old} {bo['summary']['max_abs']:.3f} -> NEW {bn['summary']['max_abs']:.3f}"]
    for k in ("col700", "row700", "low_peak"):
        tl.append(f"  {k}: ref {bn[k]['ref']}")
        tl.append(f"      {a.old} {bo[k]['cur']}")
        tl.append(f"      NEW {bn[k]['cur']}")
    tl.append(f"hue deg (ref | {a.old} | NEW): posts {ho(rc['post_left']):.0f} | {ho(oc['post_left']):.0f} | {ho(nc['post_left']):.0f};"
              f" weight {ho(rc['weight_side']):.0f} | {ho(oc['weight_side']):.0f} | {ho(nc['weight_side']):.0f}")
    md = lambda c: np.mean([abs(c[k]['oklab_L'] - rc[k]['oklab_L']) for k in rc])
    tl.append(f"mean |dL| over regions: {a.old} {md(oc):.3f} -> NEW {md(nc):.3f}")
    res["summary"] = {"mean_abs_dL_regions": {a.old: md(oc), a.new: md(nc)}, "backdrop_mae": {a.old: bo["summary"]["mae_all"], a.new: bn["summary"]["mae_all"]}}
    nn = max(len(lines), len(tl))
    tab = np.full((22 + 17 * nn, 1920, 3), BG, np.uint8)
    for i, s in enumerate(lines):
        cv2.putText(tab, s, (8, 18 + 17 * i), FONT, 0.42, (230, 230, 230), 1, cv2.LINE_AA)
    for i, s in enumerate(tl):
        cv2.putText(tab, s[:150], (760, 18 + 17 * i), FONT, 0.4, (230, 230, 230), 1, cv2.LINE_AA)
    cap = textwrap.wrap(a.caption, 205)
    capbar = np.full((14 + 20 * max(1, len(cap)), 1920, 3), (40, 40, 40), np.uint8)
    for i, s in enumerate(cap):
        cv2.putText(capbar, s, (8, 18 + 20 * i), FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    head = np.full((30, 1920, 3), (60, 40, 20), np.uint8)
    cv2.putText(head, f"SCENE {a.version} | {a.title} | frame {n} t={t:.3f}s | status: integrated scene - draft | camera of record",
                (8, 21), FONT, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    img = np.vstack([head, top, close, tab, capbar])
    o = ROOT / "scene" / "visuals" / f"{a.version}_review_fixes_f{n:03d}.jpg"
    cv2.imwrite(str(o), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    (HERE / "out" / f"review_{a.version}_f{n:03d}.json").write_text(json.dumps(res, indent=1, default=float))
    print(o, img.shape, res["summary"])


if __name__ == "__main__":
    main()
