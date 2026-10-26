"""Control: straightness of a static straight edge (stand left silhouette edge, rows 200-860) measured with the
same quadratic-sagitta statistic as the wire. If lens distortion/compression bent straight lines, it would show here."""
import numpy as np, os, sys, json
sys.path.insert(0, os.path.dirname(__file__))
from common import *
F = load_frames(); out = {}
for edge, (u_lo, u_hi) in {"stand_left_edge": (675, 715), "stand_right_edge": (820, 860)}.items():
    sags = []; maxdev = []
    for n in range(0, 241, 10):
        L = luma(F[n]); vs = np.arange(200, 860); us = []
        for v in vs:
            row = cv2.GaussianBlur(L[v:v + 1, u_lo:u_hi], (5, 1), 1.0).ravel(); g = np.abs(np.diff(row)); k = int(np.argmax(g[1:-1])) + 1
            a, b, c = g[k - 1], g[k], g[k + 1]; den = a - 2 * b + c; us.append(u_lo + k + 1.0 + (0.5 * (a - c) / den if den < 0 else 0))
        us = np.array(us); vv = vs + 0.5
        q = np.polyfit(vv - vv.mean(), us, 2); r = us - np.polyval(q, vv - vv.mean())
        keep = np.abs(r) < 3 * np.std(r) + 0.3; q = np.polyfit(vv[keep] - vv.mean(), us[keep], 2)
        span = vv.max() - vv.min(); sags.append(q[0] * (span / 2) ** 2)
        k = 15; sm = np.convolve(us[keep], np.ones(k) / k, "valid"); vsm = np.convolve(vv[keep], np.ones(k) / k, "valid")
        ch = sm[0] + (sm[-1] - sm[0]) * (vsm - vsm[0]) / (vsm[-1] - vsm[0]); maxdev.append(np.abs(sm - ch).max())
    out[edge] = {"frames": list(range(0, 241, 10)), "sagitta_px": [round(float(x), 3) for x in sags], "max_dev_px": [round(float(x), 3) for x in maxdev],
                 "sagitta_abs_max_px": round(float(np.abs(sags).max()), 3), "max_dev_max_px": round(float(max(maxdev)), 3), "length_px": 660}
json.dump(out, open(os.path.join(MEAS, "control_straight_edges.json"), "w"), indent=1)
print({k: (v["sagitta_abs_max_px"], v["max_dev_max_px"]) for k, v in out.items()})
