"""Labelled blockout visual: REFERENCE | CURRENT render | OVERLAY (render edges on the reference) | close-ups.

.venv/bin/python scene/make_visual.py --render R.png --frame 0 --version v001 --caption "..." \
    [--previous P.png] [--warp-drift] --status "..." --out scene/visuals/v001_blockout_f000.jpg
--warp-drift: the render came from the static APPROXIMATE frame-0 camera; warp it by the measurement role's 2D drift
homography H_n (measurement/cache/H_ecc.npy, frame0 -> frame n) before comparing (disclosed on the image).
Reference frames are decoded read-only from WS/reference-batch/clips/02-vertical-spring.mp4.
"""
import argparse, json, pathlib
import cv2, numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
REF = ROOT.parents[1] / "reference-batch" / "clips" / "02-vertical-spring.mp4"
FONT = cv2.FONT_HERSHEY_SIMPLEX


def ref_frame(n):
    cap = cv2.VideoCapture(str(REF))
    cap.set(cv2.CAP_PROP_POS_FRAMES, n)
    ok, f = cap.read()
    assert ok
    return f


def lab(im, text, sub="", col=(255, 255, 255)):
    bar = np.full((44, im.shape[1], 3), 22, np.uint8)
    cv2.putText(bar, text, (8, 18), FONT, 0.52, col, 1, cv2.LINE_AA)
    if sub:
        cv2.putText(bar, sub, (8, 37), FONT, 0.42, (200, 220, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, im])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", required=True)
    ap.add_argument("--previous")
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--version", required=True)
    ap.add_argument("--status", default="isolated component - draft")
    ap.add_argument("--caption", default="")
    ap.add_argument("--warp-drift", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    n = a.frame
    ref = ref_frame(n)
    cur = cv2.resize(cv2.imread(a.render), (1920, 1088), interpolation=cv2.INTER_LINEAR)
    prev = cv2.resize(cv2.imread(a.previous), (1920, 1088)) if a.previous else None
    warp_note = ""
    if a.warp_drift and n != 0:
        H = np.load(ROOT / "measurement" / "cache" / "H_ecc.npy")[n]
        T = np.array([[1, 0, -0.5], [0, 1, -0.5], [0, 0, 1.0]])           # contract -> opencv pixel coords
        Hcv = T @ H @ np.linalg.inv(T)
        cur = cv2.warpPerspective(cur, Hcv, (1920, 1088), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        if prev is not None:
            prev = cv2.warpPerspective(prev, Hcv, (1920, 1088), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        warp_note = " | render warped by measured 2D drift H_n (approx static camera)"
    pj = pathlib.Path(a.render + ".json")
    info = json.loads(pj.read_text()) if pj.exists() else {}
    pose = info.get("pose")
    pose_txt = "weight: TEST pose at the reference's measured H, body along A->H" if pose else \
        ("weight: baked dynamics" if info.get("baked") else "weight: rest (unbaked)")
    t = n / 24
    tag = f"frame {n} (t={t:.3f}s)"
    W, Hh = 960, 544
    r2, c2 = cv2.resize(ref, (W, Hh), interpolation=cv2.INTER_AREA), cv2.resize(cur, (W, Hh), interpolation=cv2.INTER_AREA)
    edges = cv2.Canny(cv2.cvtColor(c2, cv2.COLOR_BGR2GRAY), 30, 90)
    ov = cv2.cvtColor(cv2.cvtColor(r2, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    ov = (ov * 0.8).astype(np.uint8)
    ov[cv2.dilate(edges, np.ones((2, 2), np.uint8)) > 0] = (0, 255, 0)
    redge = cv2.Canny(cv2.cvtColor(r2, cv2.COLOR_BGR2GRAY), 40, 110)
    ov[(redge > 0) & (edges == 0)] = (255, 0, 255)
    tiles = [lab(r2, f"REFERENCE (AI clip) {tag}", "observed (AI-generated visual reference)"),
             lab(c2, f"{a.version} CURRENT {tag}", a.status + warp_note, (120, 255, 120))]
    tiles.append(lab(ov, f"{a.version} edge overlay {tag}", "green = render edges, magenta = reference-only edges, on grey reference"))
    # close-ups (full res): top loop / spring top, spring bottom + eye, weight, base corner
    boxes = {"top loop + bar": (860, 20, 1100, 260), "lower loop + eye": (780, 600, 1100, 820),
             "weight": (720, 700, 1120, 1000), "base corner": (420, 900, 720, 1088)}
    close = np.full((Hh, W, 3), 22, np.uint8)
    cw, ch = W // 2, (Hh // 2) - 2
    for k, (name, (u0, v0, u1, v1)) in enumerate(boxes.items()):
        pair = np.hstack([ref[v0:v1, u0:u1], np.full((v1 - v0, 4, 3), 255, np.uint8), cur[v0:v1, u0:u1]])
        s = min(cw / pair.shape[1], ch / pair.shape[0])
        pr = cv2.resize(pair, (int(pair.shape[1] * s), int(pair.shape[0] * s)), interpolation=cv2.INTER_AREA)
        y0, x0 = (k // 2) * (Hh // 2), (k % 2) * cw
        close[y0:y0 + pr.shape[0], x0:x0 + pr.shape[1]] = pr
        cv2.putText(close, f"{name}: ref | render", (x0 + 6, y0 + 14), FONT, 0.42, (0, 255, 255), 1, cv2.LINE_AA)
    tiles.append(lab(close, "close-ups (left = reference, right = render)", pose_txt))
    img = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:4])])
    head = np.full((30, img.shape[1], 3), (60, 40, 20), np.uint8)
    cv2.putText(head, f"SCENE {a.version} | {tag} | status: {a.status} | camera: {info.get('camera', '?')}",
                (8, 21), FONT, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
    lines = [a.caption[k:k + 200] for k in range(0, len(a.caption), 200)][:3]
    cap = np.full((14 + 22 * max(1, len(lines)), img.shape[1], 3), 30, np.uint8)
    for i, s in enumerate(lines):
        cv2.putText(cap, s, (8, 20 + 22 * i), FONT, 0.55, (240, 240, 240), 1, cv2.LINE_AA)
    out = np.vstack([head, img, cap])
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(a.out, out, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(a.out, out.shape)


if __name__ == "__main__":
    main()
