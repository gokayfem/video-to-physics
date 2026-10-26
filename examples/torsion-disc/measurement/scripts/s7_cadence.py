"""(adapted from clip-01 s9_cadence.py) Cadence analysis: estimate apparent per-phase timing offsets delta_j (j = n mod 4) of the reference frames.
Model on stabilized bob u and on the image angle: y_n = c_w + A_w cos(w_w (t_n + delta_{n mod 4}/24) + phi_w), with
independent sinusoid parameters per ~1-cycle window w (34 frames) and global delta_0..3 (constrained sum 0).
Reported in frames. This is a diagnostic of the AI reference; it is NOT a retiming to be applied to the simulation."""
import numpy as np, os, sys, json
sys.path.insert(0, os.path.dirname(__file__))
from common import *
from scipy.optimize import least_squares
A = np.load(os.path.join(CACHE, "analysis.npz")); n = np.arange(241); t = n / 24
import csv
RT = list(csv.DictReader(open(os.path.join(MEAS, "ring_track.csv")))); uR = np.array([float(r["u_stab_frame0_px"]) for r in RT]); suR = np.array([float(r["sigma_u_px"]) for r in RT])
res_out = {}
for name, y, sy in [("ring_u_stab_px", uR, np.maximum(suR, 0.2)), ("swing_angle_deg", A["phid"], np.maximum(A["sphid"], 0.03))]:
    W = 31; wins = [(s, min(s + W, 241)) for s in range(0, 241, W)]
    if wins[-1][1] - wins[-1][0] < 20: wins[-2] = (wins[-2][0], 241); wins = wins[:-1]
    def unpack(p):
        d = np.r_[p[:3], -p[:3].sum()]; return d, p[3:].reshape(-1, 4)
    def model(p):
        d, P = unpack(p); out = np.zeros(241)
        for (a, b), (c, Am, om, ph) in zip(wins, P):
            tt = t[a:b] + d[n[a:b] % 4] / 24; out[a:b] = c + Am * np.cos(om * tt + ph)
        return out
    p0 = [0, 0, 0]
    for a, b in wins:
        best = None
        for ph in np.linspace(-np.pi, np.pi, 16, endpoint=False):
            f = lambda q: (q[0] + q[1] * np.cos(q[2] * t[a:b] + q[3]) - y[a:b])
            r = least_squares(f, [y[a:b].mean(), np.ptp(y[a:b]) / 2, 2 * np.pi / 1.30, ph])
            if best is None or r.cost < best.cost: best = r
        p0 += list(best.x)
    r0 = least_squares(lambda p: (model(np.r_[0, 0, 0, p[3:]]) - y) / sy, np.array(p0, float))
    lb = np.full(len(r0.x), -np.inf); ub = np.full(len(r0.x), np.inf); lb[:3] = -1.5; ub[:3] = 1.5
    x0 = r0.x.copy(); x0[:3] = 0.0
    r = least_squares(lambda p: (model(p) - y) / sy, x0, bounds=(lb, ub), x_scale="jac")
    J = r.jac; chi = (r.fun ** 2).sum() / (241 - len(r.x)); cov = np.linalg.inv(J.T @ J) * max(chi, 1)
    d = np.r_[r.x[:3], -r.x[:3].sum()]
    G = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, -1, -1]]); dcov = G @ cov[:3, :3] @ G.T
    res_noc = (model(np.r_[0, 0, 0, r0.x[3:]]) - y); res_c = model(r.x) - y
    res_out[name] = {"delta_frames_by_n_mod_4": [round(float(x), 4) for x in d], "delta_sd_frames": [round(float(x), 4) for x in np.sqrt(np.diag(dcov))],
                     "rms_resid_without_cadence": round(float(np.sqrt(np.mean(res_noc ** 2))), 4),
                     "rms_resid_with_cadence_offsets": round(float(np.sqrt(np.mean(res_c ** 2))), 4), "windows": wins}
    print(name, res_out[name])
v = np.diff(uR); vl = np.convolve(v, np.ones(5) / 5, "same"); ratio = np.where(np.abs(vl) > 3, v / vl, np.nan)
steps = {str(j): float(np.nanmedian(ratio[(n[1:] % 4) == j])) for j in range(4)}
res_out["step_ratio_by_n_mod_4_of_destination_frame"] = steps; print("step ratio", steps)
json.dump(res_out, open(os.path.join(MEAS, "cadence_analysis.json"), "w"), indent=1)
