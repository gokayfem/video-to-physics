"""Blockout visual (scene role C): reference vs render through TOR_Cam_Reference at one frame, edge overlay, and
100 % close-ups (clamp, disc + mark, wood), all labelled with version / frame / status.

.venv/bin/python scene/make_visual.py --tag <render tag in look/out or --dir> --frame N --version v00X --status "..." --out scene/visuals/...jpg
Edge overlay: Canny edges of the reference (green) and of the render (magenta) on the dimmed reference; their
agreement is summarised by the chamfer distance (median px from render edges to the nearest reference edge, computed on
edges of the static stand only, i.e. outside the disc box).
"""
import argparse, json, pathlib, sys
import numpy as np, cv2

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "look"))
import metrics as MX  # noqa: E402
FONT = cv2.FONT_HERSHEY_SIMPLEX
BG = (24, 24, 24)


def bar(w, lines):
    b = np.full((10 + 22 * len(lines), w, 3), BG, np.uint8)
    for i, (t, s, c) in enumerate(lines):
        cv2.putText(b, t, (8, 20 + 22 * i), FONT, s, c, 1, cv2.LINE_AA)
    return b


def fitbox(img, w, h):
    s = min(w / img.shape[1], h / img.shape[0])
    r = cv2.resize(img, (max(1, int(img.shape[1] * s)), max(1, int(img.shape[0] * s))), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    out = np.full((h, w, 3), BG, np.uint8)
    out[:r.shape[0], :r.shape[1]] = r
    return out


def edges(img):
    g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (0, 0), 1.5)
    return cv2.Canny(g, 25, 60)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--dir", default=str(ROOT / "look" / "out"))
    ap.add_argument("--frame", type=int, required=True)
    ap.add_argument("--version", required=True)
    ap.add_argument("--status", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = pathlib.Path(a.dir)
    info = json.loads((d / f"{a.tag}_render.json").read_text())
    ref = MX.ref_frame(a.frame)
    cur = cv2.resize(cv2.imread(info["files"]["full"]), (1920, 1088), interpolation=cv2.INTER_LINEAR)
    for k, b in info.get("crop_boxes", {}).items():
        cur[b[1]:b[3], b[0]:b[2]] = cv2.imread(info["files"][k])
    er, ec = edges(ref), edges(cur)
    ov = (cv2.cvtColor(cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR) * 0.45).astype(np.uint8)
    ov[er > 0] = (60, 230, 60)
    ov[ec > 0] = (230, 60, 230)
    ov[(er > 0) & (ec > 0)] = (255, 255, 255)
    mask = np.ones_like(er, bool)
    if "disc" in info.get("crop_boxes", {}):
        u0, v0, u1, v1 = info["crop_boxes"]["disc"]
        mask[v0:v1, u0:u1] = False
    dt = cv2.distanceTransform((er == 0).astype(np.uint8), cv2.DIST_L2, 5)
    sel = (ec > 0) & mask
    cham = float(np.median(dt[sel])) if sel.any() else float("nan")
    p90 = float(np.percentile(dt[sel], 90)) if sel.any() else float("nan")
    W = 1920
    w3 = W // 3
    h3 = int(w3 * 1088 / 1920)
    row1 = np.hstack([fitbox(ref, w3, h3), fitbox(cur, w3, h3), fitbox(ov, W - 2 * w3, h3)])
    lab1 = np.hstack([bar(w3, [(f"REFERENCE frame {a.frame}", 0.55, (255, 255, 255))]), bar(w3, [(f"RENDER {a.version} (TOR_Cam_Reference)", 0.55, (255, 255, 255))]),
                      bar(W - 2 * w3, [("edges: reference green, render magenta, both white", 0.5, (255, 255, 255))])])
    tiles = []
    for k in ("clamp", "disc", "wood"):
        if k not in info.get("crop_boxes", {}):
            continue
        u0, v0, u1, v1 = info["crop_boxes"][k]
        pair = np.hstack([ref[v0:v1, u0:u1], cur[v0:v1, u0:u1]])
        tiles.append(np.vstack([bar(w3, [(f"{k}: reference | render (100 %, box {u0},{v0})", 0.45, (220, 220, 220))]), fitbox(pair, w3, 240)]))
    rows = [lab1, row1]
    if tiles:
        rows.append(np.hstack(tiles + [np.full((tiles[0].shape[0], W - w3 * len(tiles), 3), BG, np.uint8)]))
    pose = info.get("pose") or {}
    rows.append(bar(W, [(f"{a.version} | frame {a.frame} | {a.status}", 0.55, (255, 255, 255)),
                        (f"disc: {pose.get('status', 'baked' if info.get('baked') else '-')}; psi {pose.get('psi_deg')} deg, offset {pose.get('offset_D')} D", 0.45, (200, 200, 120)),
                        (f"static-stand edge agreement (render edges outside the disc box -> nearest reference edge): median {cham:.2f} px, p90 {p90:.2f} px", 0.45, (200, 200, 200))]))
    sheet = np.vstack(rows)
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(a.out, sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(json.dumps({"out": a.out, "edge_median_px": round(cham, 2), "edge_p90_px": round(p90, 2)}))


if __name__ == "__main__":
    main()
