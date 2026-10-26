"""Static-stand silhouette IoU: reference wood mask (HSV threshold, geometry-role heuristic) vs rendered mask.

.venv/bin/python scene/measure/silhouette_iou.py --mask scene/out/mask_vNNN_f000.png --frame 0 --out-png X.png [--json X.json]
Reference mask = walnut-coloured pixels (hue 5-25 deg, sat > 0.28, value 0.12-0.85) minus the bob disc and the
pin box, then morphological close/open and the largest connected component. Heuristic: wood grain highlights
and the base's contact shadow make the reference mask itself uncertain by a few px along edges.
"""
import argparse, json, subprocess, pathlib, cv2, numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[2]
REF = ROOT.parents[1] / "reference-batch" / "clips" / "01-brass-pendulum.mp4"
ap = argparse.ArgumentParser()
ap.add_argument("--mask", required=True); ap.add_argument("--frame", type=int, default=0)
ap.add_argument("--out-png"); ap.add_argument("--json")
a = ap.parse_args()
raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(REF), "-vf", f"select=eq(n\\,{a.frame}),scale=960:544",
                      "-vframes", "1", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], capture_output=True, check=True).stdout
ref = np.frombuffer(raw, np.uint8).reshape(544, 960, 3)
hsv = cv2.cvtColor(ref, cv2.COLOR_BGR2HSV).astype(float)
h, s, v = hsv[..., 0] * 2, hsv[..., 1] / 255, hsv[..., 2] / 255
m = ((h >= 5) & (h <= 28) & (s > 0.28) & (v > 0.10) & (v < 0.85)).astype(np.uint8)
# remove the brass bob (circle fit, half-res) and pin
bob = {0: (519.5, 410.5, 30.0), 120: (547.2, 411.3, 30.0), 240: (549.4, 411.0, 30.0)}.get(a.frame)
if bob:
    cv2.circle(m, (int(bob[0]), int(bob[1])), int(bob[2]), 0, -1)
m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
n, lab, st, _ = cv2.connectedComponentsWithStats(m)
refm = (lab == 1 + np.argmax(st[1:, 4])).astype(np.uint8)
refm = cv2.morphologyEx(refm, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
# fill holes
ff = refm.copy(); cv2.floodFill(ff, np.zeros((546, 962), np.uint8), (0, 0), 1); refm = refm | (1 - ff)
ren = (cv2.imread(a.mask, cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
ren = cv2.resize(ren, (960, 544), interpolation=cv2.INTER_NEAREST)
inter, union = int((refm & ren).sum()), int((refm | ren).sum())
iou = inter / union
res = {"frame": a.frame, "iou": iou, "ref_px": int(refm.sum()), "render_px": int(ren.sum()),
       "ref_only_px": int((refm & (1 - ren)).sum()), "render_only_px": int((ren & (1 - refm)).sum()),
       "resolution": "960x544", "note": "heuristic reference mask; see script docstring"}
print(json.dumps(res))
if a.json: pathlib.Path(a.json).write_text(json.dumps(res, indent=1))
if a.out_png:
    vis = (cv2.cvtColor(cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR) * 0.5).astype(np.uint8)
    vis[(refm == 1) & (ren == 1)] = (90, 160, 90)
    vis[(refm == 1) & (ren == 0)] = (0, 0, 255)      # reference only: red
    vis[(refm == 0) & (ren == 1)] = (255, 160, 0)    # render only: blue
    cv2.putText(vis, f"stand silhouette IoU {iou:.3f}  (green both, red ref-only, blue render-only)  frame {a.frame}",
                (8, 530), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(a.out_png, vis)
