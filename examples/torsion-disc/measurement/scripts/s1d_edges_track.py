"""Stage 1d: static rectilinear frame edges in every frame (the AI camera ORBITS, so a single homography is not a valid
camera model). Seeds: frame-0 segments (camera/edges.py EDGES); frame n seeds = frame n-1 measured segment endpoints mapped
by the frame-to-frame KLT homography (H_chain[n] H_chain[n-1]^-1, measurement/cache/H_chain.npy) and re-measured sub-pixel
(edges.measure). Endpoints are re-projected onto the measured line each frame (no perpendicular accumulation).
Writes measurement/cache/edges_track.json: per frame, per edge: line (n, c), rms, n_pts, endpoints, points (every 3rd)."""
import numpy as np, json, os, sys
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "camera"))
from common import *
import edges as ED
F = load_frames(); N = len(F); Hc = np.load(os.path.join(CACHE, "H_chain.npy"))
seeds = {k: np.array([[v[0], v[1]], [v[2], v[3]]], float) for k, v in ED.EDGES.items()}
out = []
for n in range(N):
    if n > 0:
        Hf = Hc[n] @ np.linalg.inv(Hc[n - 1]); seeds = {k: Happly(Hf, s) for k, s in seeds.items()}
    meas = ED.measure(np.ascontiguousarray(F[n]), {k: (s[0, 0], s[0, 1], s[1, 0], s[1, 1], ED.EDGES[k][4], ED.EDGES[k][5]) for k, s in seeds.items()})
    rec = {}
    for k, m in meas.items():
        nn, c = m["line"]; nn = np.asarray(nn, float)
        if m["n_pts"] >= 5:
            s = seeds[k]; seeds[k] = s - (s @ nn + c)[:, None] * nn[None, :]   # project endpoints onto the measured line
        rec[k] = dict(n=nn.tolist(), c=float(c), rms=m["rms_px"], n_pts=m["n_pts"], ends=seeds[k].tolist(), pts=np.asarray(m["points"])[::3].tolist())
    out.append(rec)
    if n % 40 == 0: print(n, {k: (round(v["rms"], 2), v["n_pts"]) for k, v in rec.items()})
json.dump(out, open(os.path.join(CACHE, "edges_track.json"), "w"))
