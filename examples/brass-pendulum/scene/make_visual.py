"""Compose a labelled comparison: reference frame | current render | previous render (optional) | overlay.

.venv/bin/python scene/make_visual.py --render R.png --frame 0 --version v001 --caption "..." \
    [--previous P.png] --out scene/visuals/v001_blockout_f000.jpg
Reference frames are decoded read-only from WS/reference-batch/clips/01-brass-pendulum.mp4 (never modified).
"""
import argparse, subprocess, pathlib, cv2, numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[1]
REF = ROOT.parents[1] / "reference-batch" / "clips" / "01-brass-pendulum.mp4"
ap = argparse.ArgumentParser()
ap.add_argument("--render", required=True); ap.add_argument("--previous")
ap.add_argument("--frame", type=int, default=0); ap.add_argument("--version", required=True)
ap.add_argument("--status", default="isolated component - draft; approx geometry-role camera"); ap.add_argument("--caption", default="")
ap.add_argument("--out", required=True)
a = ap.parse_args()
raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(REF), "-vf", f"select=eq(n\\,{a.frame}),scale=960:544",
                      "-vframes", "1", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], capture_output=True, check=True).stdout
ref = np.frombuffer(raw, np.uint8).reshape(544, 960, 3).copy()
cur = cv2.resize(cv2.imread(a.render), (960, 544))
t = a.frame / 24
def lab(im, text, sub=""):
    bar = np.zeros((46, im.shape[1], 3), np.uint8)
    cv2.putText(bar, text, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    if sub: cv2.putText(bar, sub, (8, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 220, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, im])
tag = f"frame {a.frame} (t={t:.3f}s)"
# overlay: reference in grey, render edges in green
edges = cv2.Canny(cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY), 40, 120)
ov = cv2.cvtColor(cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
ov[cv2.dilate(edges, np.ones((2, 2), np.uint8)) > 0] = (0, 255, 0)
tiles = [lab(ref, f"REFERENCE (AI clip) {tag}", "observed"),
         lab(cur, f"{a.version} CURRENT blockout {tag}", a.status)]
if a.previous:
    tiles.append(lab(cv2.resize(cv2.imread(a.previous), (960, 544)), f"PREVIOUS render {tag}", "for comparison"))
tiles.append(lab(ov, f"{a.version} overlay: render edges (green) on reference (grey)", tag))
if len(tiles) == 3:
    # pivot / pin close-up: reference vs render, 2x (full-res region u 960-1200, v 20-200 -> half-res 480-600, 10-100)
    zr = cv2.resize(ref[10:146, 470:710], (480, 272), interpolation=cv2.INTER_NEAREST)
    zc = cv2.resize(cur[10:146, 470:710], (480, 272), interpolation=cv2.INTER_NEAREST)
    zb = cv2.resize(ref[300:436, 420:660], (480, 272), interpolation=cv2.INTER_NEAREST)
    zd = cv2.resize(cur[300:436, 420:660], (480, 272), interpolation=cv2.INTER_NEAREST)
    tiles.append(lab(np.vstack([np.hstack([zr, zc]), np.hstack([zb, zd])]),
                 "2x close-ups: left = reference, right = render", "top: arm end / pin / loop;  bottom: bob over base"))
grid = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:4])])
cap = np.full((60, grid.shape[1], 3), 20, np.uint8)
for i, line in enumerate([a.caption[k:k + 230] for k in range(0, len(a.caption), 230)][:2]):
    cv2.putText(cap, line, (10, 24 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 1, cv2.LINE_AA)
cv2.imwrite(a.out, np.vstack([grid, cap]), [cv2.IMWRITE_JPEG_QUALITY, 88])
print(a.out)
