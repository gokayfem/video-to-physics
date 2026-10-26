"""Reference motion-blur evidence (scene role): edge width of the weight's left/right silhouette edges vs its horizontal
image speed. Edge width = count of |d profile| > 30 % of its max in a 25-px window centred on the edge, rows v-25..v+25
averaged (clip-01 / visual-review metric). Edge position: weight_track u -/+ width/2 (silhouette half width), refined to
the strongest gradient within +-6 px. Speed: |du/dn| of the drift-stabilised track (px/frame, central difference).
Frames 1..239. Fit: width = a + b * speed. -> scene/measure/mb_reference.json"""
import json, pathlib
import numpy as np, cv2
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
F = np.load(ROOT / "measurement/cache/frames_bgr.npy", mmap_mode="r")
W = np.genfromtxt(ROOT / "measurement/weight_track.csv", delimiter=",", names=True)
sp = np.abs(np.gradient(W["u_stab_frame0_px"]))
import sys
SGN = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0   # +1 verified: slow-frame edges sharpest (2.4 px vs 9.5 px with -1)
rows = []
for n in range(1, 240):
    g = cv2.cvtColor(np.ascontiguousarray(F[n]), cv2.COLOR_BGR2GRAY).astype(np.float32)
    u, v, hw = W["u_px"][n], W["v_px"][n], W["width_px"][n] / 2
    ws, we = [], []
    for side in (-1, 1):
        # average rows along the (tilted) edge: shift each row by tan(tilt) * dy (tilt from weight_track, sign SGN)
        t = np.tan(np.radians(W["tilt_deg"][n])) * SGN
        band = np.zeros(g.shape[1], np.float32)
        for dy in range(-25, 25):
            row = g[int(v) + dy]
            band += np.interp(np.arange(g.shape[1]) + t * dy, np.arange(g.shape[1]), row)
        band /= 50
        x0 = int(round(u + side * hw))
        loc = np.abs(np.diff(band[x0 - 6:x0 + 7]))
        x0 = x0 - 6 + int(np.argmax(loc))
        prof = band[x0 - 12:x0 + 13]
        d = np.abs(np.diff(prof))
        ws.append(float((d > 0.3 * d.max()).sum()))
        we.append(float(abs(band[x0 + 9] - band[x0 - 9]) / max(d.max(), 1e-6)))   # effective width = step / max slope
    rows.append((n, sp[n], np.mean(ws), np.mean(we)))
    continue
R = np.array(rows)
b, a = np.polyfit(R[:, 1], R[:, 2], 1)
b2, a2 = np.polyfit(R[:, 1], R[:, 3], 1)
q25, q75 = np.percentile(R[:, 1], [25, 75])
res = {"method": __doc__, "fit": {"a_px": float(a), "b_px_per_px_per_frame": float(b), "resid_rms_px": float(np.std(R[:, 2] - (a + b * R[:, 1])))},
       "fit_effective_width": {"a_px": float(a2), "b": float(b2), "resid_rms_px": float(np.std(R[:, 3] - (a2 + b2 * R[:, 1]))),
                               "slow_q25": float(R[R[:, 1] < np.percentile(R[:, 1], 25), 3].mean()), "fast_q75": float(R[R[:, 1] > np.percentile(R[:, 1], 75), 3].mean()),
                               "definition": "|p(x+9) - p(x-9)| / max|dp/dx| at the silhouette edge (px); a box blur of length L adds ~L"},
       "speed_px_per_frame": {"max": float(R[:, 1].max()), "q25": float(q25), "q75": float(q75)},
       "width_slow_q25": float(R[R[:, 1] < q25, 2].mean()), "width_fast_q75": float(R[R[:, 1] > q75, 2].mean()),
       "rows": R.round(3).tolist()}
(HERE / "mb_reference.json").write_text(json.dumps(res, indent=1))
print(json.dumps({k: v for k, v in res.items() if k not in ("rows", "method")}, indent=1))
