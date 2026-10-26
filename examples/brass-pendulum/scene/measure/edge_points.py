"""Sample base top-silhouette edge points (full-res px) per column: first walnut-coloured pixel from above.

.venv/bin/python scene/measure/edge_points.py --frame 0 --landmarks scene/measure/landmarks_f000.json
Adds/replaces "lines" in the landmark file: base_back_top (between post and the back-right corner) and
base_left_top (left end top edge, FL -> post). Subpixel: linear interpolation of the walnut score across 0.5.
"""
import argparse, json, subprocess, pathlib, cv2, numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[2]
REF = ROOT.parents[1] / "reference-batch" / "clips" / "01-brass-pendulum.mp4"
ap = argparse.ArgumentParser(); ap.add_argument("--frame", type=int, default=0); ap.add_argument("--landmarks", required=True)
a = ap.parse_args()
raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(REF), "-vf", f"select=eq(n\\,{a.frame})", "-vframes", "1",
                      "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], capture_output=True, check=True).stdout
im = np.frombuffer(raw, np.uint8).reshape(1088, 1920, 3)
hsv = cv2.cvtColor(cv2.GaussianBlur(im, (5, 5), 0), cv2.COLOR_BGR2HSV).astype(float)
sat = hsv[..., 1] / 255
d = json.load(open(a.landmarks)); P = d["points"]
def top_edge(u, v0, v1):
    col = sat[v0:v1, u]
    k = np.argmax(col > 0.30)
    if k == 0 or col[k] <= 0.30: return None
    f = (0.30 - col[k - 1]) / (col[k] - col[k - 1])
    return v0 + k - 1 + f + 0.5
post_r = P["post_bot_BR"]["uv"][0]; br = P["base_top_BR"]["uv"][0]; fl = P["base_top_FL"]["uv"][0]
post_l = P["post_bot_FL"]["uv"][0]
bob_u = P["bob_centre"]["uv"][0]; bob_r = P["bob_radius_px"]["value"]
back, left = [], []
for u in range(int(post_r) + 15, int(br) - 40, 20):
    if abs(u - bob_u) < bob_r + 40: continue           # skip bob + its shadow columns
    v = top_edge(u, 820, 1000)
    if v: back.append([u + 0.5, round(v, 2)])
for u in range(int(fl) + 25, int(post_l) - 10, 15):
    v = top_edge(u, 820, 1000)
    if v: left.append([u + 0.5, round(v, 2)])
d["lines"] = {"base_back_top": {"pts": back, "sigma_px": 2.0, "method": "first pixel with HSV saturation > 0.30 from above, 5x5 blur"},
              "base_left_top": {"pts": left, "sigma_px": 2.0, "method": "same"}}
json.dump(d, open(a.landmarks, "w"), indent=1)
print(a.frame, "back", back, "\nleft", left)
