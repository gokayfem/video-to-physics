"""Stage 9: anchor-window rule (config/CHANGELOG.md 2026-10-26T03:09:32Z) applied as declared, to every measured mode.
Twist psi: no turning points in 0-10 s -> (a)/(b) inapplicable. Sway: turning points of the disc rotation centre Q_x and Q_y
(world, D) and of the disc tilt about Y; each TP from a local parabola (+-4 frames), sigma of the TP value = parabola
residual rms / sqrt(3) (min 3e-4 D / 0.05 deg); half-cycle amplitude = |TP_k - TP_{k-1}| / 2, sigma = hypot / 2.
(a) first TP after which the amplitude increases by > 3 sigma over the previous half-cycle.
(b) passive-decay reference fitted on the first two full cycles (4 half-cycles): ln A = ln A0 - lambda t (lambda >= 0), half-period
    constant; earliest half-cycle end where the cumulative linear trend (fit from t = 0) of amplitude exceeds the reference by
    > 3 sigma (growth), or the half-period trend departs by > 3 sigma.
(c) visible discontinuities: from analysis.json.
Writes measurement/anchor_window.json."""
import numpy as np, json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from common import *
A = np.load(os.path.join(CACHE, "analysis.npz")); t = A["t"]; N = len(t); AN = json.load(open(os.path.join(MEAS, "analysis.json")))
def tps(y, smin):
    ys = np.convolve(y, np.ones(5) / 5, "same"); d = np.sign(np.diff(ys)); out = []
    for i in range(4, N - 5):
        if d[i - 1] != d[i] and d[i] != 0:
            a, b = i - 4, i + 5; c = np.polyfit(t[a:b] - t[i], y[a:b], 2); tm = -c[1] / (2 * c[0]) if c[0] != 0 else 0.0
            if abs(tm) <= 4 / FPS:
                r = y[a:b] - np.polyval(c, t[a:b] - t[i]); out.append((t[i] + tm, np.polyval(c, tm), max(smin, np.sqrt(np.mean(r ** 2)) / np.sqrt(3))))
    o = np.array(out)
    keep = [0]   # enforce alternation (max/min)
    for k in range(1, len(o)):
        if o[k, 0] - o[keep[-1], 0] > 0.15: keep.append(k)
    return o[keep]
def rule(name, y, smin):
    P = tps(y, smin); amp = np.abs(np.diff(P[:, 1])) / 2; sa = np.hypot(P[1:, 2], P[:-1, 2]) / 2; tend = P[1:, 0]; hp = np.diff(P[:, 0])
    res = {"turning_points": [dict(t=float(a), value=float(b), sigma=float(c)) for a, b, c in P], "halfcycle_amplitude": amp.tolist(), "amp_sigma": sa.tolist(),
           "halfperiod_s": hp.tolist(), "halfcycle_end_t": tend.tolist()}
    # (a)
    za = (amp[1:] - amp[:-1]) / np.hypot(sa[1:], sa[:-1]); ia = np.where(za > 3)[0]
    res["a_jump_z"] = za.tolist(); res["a_trigger_t"] = float(P[ia[0] + 1, 0]) if len(ia) else None   # the TP after which the larger half-cycle starts
    # (b)
    k0 = 4; tt = tend[:k0]; la = np.log(amp[:k0]); w = (amp[:k0] / sa[:k0]) ** 2
    Xb = np.c_[np.ones(k0), tt]; cb = np.linalg.lstsq(Xb * np.sqrt(w)[:, None], la * np.sqrt(w), rcond=None)[0]; lam = max(0.0, -cb[1]); lnA0 = cb[0] if -cb[1] >= 0 else np.average(la, weights=w)
    hp0 = hp[:k0].mean(); hps = np.std(hp[:k0], ddof=1)
    trig_b = None; zs = []
    for m in range(k0 + 2, len(amp) + 1):
        ref = np.exp(lnA0 - lam * tend[:m]); dev = amp[:m] - ref
        X = np.c_[np.ones(m), tend[:m]]; c, *_ = np.linalg.lstsq(X / sa[:m, None], dev / sa[:m], rcond=None); cov = np.linalg.inv((X / sa[:m, None]).T @ (X / sa[:m, None]))
        trend_end = c[0] + c[1] * tend[m - 1]; s_end = np.sqrt(np.array([1, tend[m - 1]]) @ cov @ np.array([1, tend[m - 1]]))
        dh = hp[:m] - hp0; Xh = np.c_[np.ones(m), tend[:m]]; ch, *_ = np.linalg.lstsq(Xh, dh, rcond=None); rh = dh - Xh @ ch; sh = max(np.std(rh, ddof=2), 1 / FPS / 4) if m > 3 else hps
        covh = np.linalg.inv(Xh.T @ Xh) * sh ** 2; th_end = ch[0] + ch[1] * tend[m - 1]; sth = np.sqrt(np.array([1, tend[m - 1]]) @ covh @ np.array([1, tend[m - 1]]))
        za_ = trend_end / s_end; zh = th_end / sth; zs.append(dict(t=float(tend[m - 1]), amp_trend_z=float(za_), halfperiod_trend_z=float(zh)))
        if trig_b is None and (za_ > 3 or abs(zh) > 3): trig_b = float(tend[m - 1])
    res.update(b_reference=dict(A0=float(np.exp(lnA0)), lambda_per_s=float(lam), halfperiod_s=float(hp0)), b_trend_z=zs, b_trigger_t=trig_b)
    return res
modes = {"sway_Qx_D": rule("Qx", A["Q"][:, 0], 3e-4), "sway_Qy_D": rule("Qy", A["Q"][:, 1], 1e-3), "disc_tilt_about_Y_deg": rule("tilt_Y", A["ty"], 0.05)}
trig = []
for k, r in modes.items():
    for key in ("a_trigger_t", "b_trigger_t"):
        if r[key] is not None: trig.append((r[key], k, key))
# supplementary: 2D sway amplitude (isotropic-oscillator invariant) sqrt(x^2 + y^2 + (xdot^2 + ydot^2)/w^2), 11-frame windows
Q = A["Q"]; w0 = 2 * np.pi / AN["sway"]["Qx_D"]["period_s"]
x = Q[:, 0] - np.polyval(np.polyfit(t, Q[:, 0], 2), t); y = Q[:, 1] - np.polyval(np.polyfit(t, Q[:, 1], 2), t)
dv = lambda z: np.gradient(np.convolve(z, np.ones(3) / 3, "same"), 1 / FPS)
E = np.sqrt(x ** 2 + y ** 2 + (dv(x) ** 2 + dv(y) ** 2) / w0 ** 2)
wins = [(k, k + 11) for k in range(2, 230, 11)]; em = np.array([E[a:b].mean() for a, b in wins]); es = np.array([E[a:b].std() / np.sqrt(3) for a, b in wins]); tc = np.array([t[(a + b) // 2] for a, b in wins])
zj = (em[1:] - em[:-1]) / np.hypot(es[1:], es[:-1])
ref = np.polyfit(tc[:4], np.log(em[:4]), 1); lam = max(0.0, -ref[0]); A0 = np.exp(np.polyval(ref, 0)) if -ref[0] >= 0 else em[:4].mean()
zc = (em - A0 * np.exp(-lam * tc)) / es; trig_e = next((float(tc[i]) for i in range(4, len(em)) if zc[i] > 3 and zc[i - 1] > 3), None)
energy = {"window_centres_s": tc.tolist(), "amp_D": em.tolist(), "sd_D": es.tolist(), "jump_z": zj.tolist(), "ref_A0": float(A0), "ref_lambda_per_s": float(lam), "excess_z": zc.tolist(),
          "cumulative_trigger_t (2 consecutive windows > 3 sigma above the passive reference)": trig_e,
          "note": "the swing exchanges energy between x and y (spherical pendulum), so per-axis amplitudes are not modes; this invariant is the physically meaningful sway amplitude"}
cands = sorted(trig); Ta = None; why = None
if cands: Ta, why = max(2.5, cands[0][0]), "earliest trigger %s %s at %.3f s; floor 2.5 s applied" % (cands[0][1], cands[0][2], cands[0][0])
early = [c for c in cands if c[0] < 2.5]
out = {"rule": "CHANGELOG 2026-10-26T03:09:32Z (declared before measurement)", "twist_psi": AN["anchor_window"]["a_halfcycle_amplitude_jump"],
       "modes": modes, "c_discontinuities": AN["anchor_window"]["c_visible_discontinuity"],
       "triggers_sorted": [dict(t=a, mode=b, test=c) for a, b, c in cands], "sway_2D_amplitude_supplementary": energy, "triggers_before_minimum_2.5s": [dict(t=a, mode=b, test=c) for a, b, c in early],
       "T_a_s": Ta if Ta is not None else 10.0, "T_a_reason": why if Ta is not None else "no trigger: default 10 s, last 2.5 s held out",
       "held_out_s": None if Ta is not None else [7.5, 10.0]}
json.dump(out, open(os.path.join(MEAS, "anchor_window.json"), "w"), indent=1)
for k, r in modes.items():
    print(k, "amps", np.round(r["halfcycle_amplitude"], 4).tolist()); print("  sig", np.round(r["amp_sigma"], 4).tolist()); print("  hp", np.round(r["halfperiod_s"], 3).tolist())
    print("  a_z", np.round(r["a_jump_z"], 2).tolist(), "a_trig", r["a_trigger_t"], "b", r["b_reference"], "b_trig", r["b_trigger_t"])
    print("  b z", [(round(z["t"], 2), round(z["amp_trend_z"], 1), round(z["halfperiod_trend_z"], 1)) for z in r["b_trend_z"]])
print("T_a", out["T_a_s"], out["T_a_reason"], "early:", out["triggers_before_minimum_2.5s"])
