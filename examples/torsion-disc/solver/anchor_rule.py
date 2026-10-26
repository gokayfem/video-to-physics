"""Apply the CHANGELOG anchor-window rule (declared 2026-10-26T03:09:32Z, before measurement) to measured mode series.

T_a = the earliest of
  (a) JUMP: the first turning point after which a mode's half-cycle amplitude INCREASES by > 3 sigma over the previous one;
  (b) DRIFT: the earliest turning point at which the mode's amplitude or half-period TREND (weighted linear fit over the
      turning points from t = 0 up to that point) departs from a passive-decay model fitted on the first two full cycles
      (amplitude A0 exp(-gamma t), gamma >= 0; constant half-period) by > 3 sigma in the non-physical direction
      (amplitude growth; any half-period change, since the twist is linear and the sway amplitudes are small);
  (c) a visible discontinuity (passed in by measurement via --discontinuity-s, not detected here);
  minimum 2.5 s; if none, T_a = 10 s and the last 2.5 s are held out (fit on 0-7.5 s).
Turning points: local quadratic least-squares fits (+-k samples) with covariance from the per-frame sigma
(-> sigma of the extremum value and time).  A mode is used only if its half-cycle amplitude exceeds 5 sigma.

  .venv/bin/python solver/anchor_rule.py --obs measurement/tracks_for_fit.csv --modes psi pt:B nrm --out solver/out/anchor_rule.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from fit_projected import load_obs, sinefit, twist_series  # noqa: E402


def extrema(t, y, s, k):
    """local LS quadratic extrema: list of dict(t, A (value), sA, st)."""
    out = []
    for i in range(k, len(y) - k):
        win = slice(i - k, i + k + 1)
        if not (y[i] >= y[win].max() or y[i] <= y[win].min()):
            continue
        if (y[i] - y[i - 1]) * (y[i + 1] - y[i]) >= 0:
            continue
        tt = t[win] - t[i]; w = 1.0 / s[win] ** 2
        A = np.column_stack([np.ones_like(tt), tt, tt * tt])
        Cinv = A.T @ (A * w[:, None]); C = np.linalg.inv(Cinv); c = C @ (A.T @ (w * y[win]))
        if c[2] == 0:
            continue
        te = -c[1] / (2 * c[2]); ye = c[0] - c[1] ** 2 / (4 * c[2])
        if abs(te) > (k + 0.5) * (t[1] - t[0]):
            continue
        J_t = np.array([0.0, -1 / (2 * c[2]), c[1] / (2 * c[2] ** 2)])
        J_y = np.array([1.0, -c[1] / (2 * c[2]), c[1] ** 2 / (4 * c[2] ** 2)])
        out.append(dict(t=float(t[i] + te), y=float(ye), st=float(math.sqrt(J_t @ C @ J_t)), sy=float(math.sqrt(J_y @ C @ J_y))))
    return out


def analyse_mode(name, t, y, s, k, t_end=10.0):
    # centre from the first two cycles (sine fit), half-cycle amplitudes about it
    ext = extrema(t, y, s, k)
    if len(ext) < 6:
        return {"mode": name, "usable": False, "reason": f"only {len(ext)} turning points"}
    t2 = ext[4]["t"] + 1e-9                                   # first two full cycles ~ first 5 extrema
    m = t <= t2
    T_half0 = (ext[4]["t"] - ext[0]["t"]) / 4
    w, c = sinefit(t[m], y[m], np.linspace(0.6, 1.6, 400) * math.pi / T_half0)
    centre = c[2]
    amps = np.array([abs(e["y"] - centre) for e in ext]); sA = np.array([e["sy"] for e in ext]); te = np.array([e["t"] for e in ext])
    st = np.array([e["st"] for e in ext])
    res = {"mode": name, "usable": bool(np.median(amps / sA) > 5), "centre": float(centre), "omega_first_two_cycles": float(w),
           "turning_points": [{"t": float(a), "A": float(b), "sA": float(c_), "st": float(d)} for a, b, c_, d in zip(te, amps, sA, st)]}
    if not res["usable"]:
        res["reason"] = "amplitude < 5 sigma"; return res
    trig = []
    # (a) jump
    for i in range(len(amps) - 1):
        if amps[i + 1] - amps[i] > 3 * math.hypot(sA[i], sA[i + 1]):
            trig.append({"rule": "a_jump", "t_s": float(te[i]), "dA": float(amps[i + 1] - amps[i]), "sigma": float(math.hypot(sA[i], sA[i + 1]))})
            break
    # (b) drift vs a passive model fitted on the first two cycles
    n0 = 5
    tt0, aa0, ss0 = te[:n0], amps[:n0], sA[:n0]
    best = None
    for g in np.linspace(0, 1.0, 2001):                     # gamma >= 0 grid, A0 closed form (weighted)
        f = np.exp(-g * tt0); A0 = np.sum(aa0 * f / ss0 ** 2) / np.sum(f * f / ss0 ** 2)
        chi = np.sum(((aa0 - A0 * f) / ss0) ** 2)
        if best is None or chi < best[0]:
            best = (chi, g, A0)
    _, gam, A0 = best
    sA0 = 1 / math.sqrt(np.sum(np.exp(-2 * gam * tt0) / ss0 ** 2))
    hp = np.diff(te); shp = np.hypot(st[1:], st[:-1])
    hp0 = np.sum(hp[:n0 - 1] / shp[:n0 - 1] ** 2) / np.sum(1 / shp[:n0 - 1] ** 2); shp0 = 1 / math.sqrt(np.sum(1 / shp[:n0 - 1] ** 2))
    res["passive_model"] = {"A0": float(A0), "gamma": float(gam), "sigma_A0": float(sA0), "half_period_s": float(hp0), "sigma_half_period_s": float(shp0)}
    drift_rows = []
    for j in range(n0, len(amps)):
        X = np.column_stack([np.ones(j + 1), te[:j + 1]]); W = 1 / sA[:j + 1] ** 2
        Cc = np.linalg.inv(X.T @ (X * W[:, None])); cc = Cc @ (X.T @ (W * amps[:j + 1]))
        xv = np.array([1.0, te[j]]); trend = xv @ cc; strend = math.sqrt(xv @ Cc @ xv)
        model = A0 * math.exp(-gam * te[j]); smodel = sA0 * math.exp(-gam * te[j])
        zA = (trend - model) / math.hypot(strend, smodel)
        Xh = np.column_stack([np.ones(j), te[1:j + 1]]); Wh = 1 / shp[:j] ** 2
        Ch = np.linalg.inv(Xh.T @ (Xh * Wh[:, None])); ch = Ch @ (Xh.T @ (Wh * hp[:j]))
        xh = np.array([1.0, te[j]]); htr = xh @ ch; shtr = math.sqrt(xh @ Ch @ xh)
        zP = (htr - hp0) / math.hypot(shtr, shp0)
        drift_rows.append({"t": float(te[j]), "z_amp": float(zA), "z_half_period": float(zP)})
        if zA > 3:
            trig.append({"rule": "b_drift_amplitude", "t_s": float(te[j]), "z": float(zA)}); break
        if abs(zP) > 3:
            trig.append({"rule": "b_drift_half_period", "t_s": float(te[j]), "z": float(zP)}); break
    res["drift_test"] = drift_rows
    res["triggers"] = trig
    res["T_trigger_s"] = min([x["t_s"] for x in trig]) if trig else None
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs", required=True); ap.add_argument("--modes", nargs="+", default=["psi", "pt:B", "nrm"])
    ap.add_argument("--k", type=int, default=3, help="half window (samples) of the local quadratic extremum fit")
    ap.add_argument("--discontinuity-s", type=float, default=None)
    ap.add_argument("--deciding", nargs="+", default=["twist_rad"],
                    help="modes whose triggers set T_a (default: the twist, the only single passive linear mode; the sway/tilt "
                         "coordinates exchange energy physically (two coupled modes per axis + gyroscopic precession of the swing "
                         "plane), so their per-coordinate amplitude tests are reported as information; see solver/NOTES.md)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    d = load_obs(a.obs, a.modes)
    t = d["t"]
    modes = []
    for ch in a.modes:
        if ch == "psi":
            y = twist_series(d["y"]["psi"][:, 0], float(np.median(d["y"]["psi"][:48, 0])))
            modes.append(analyse_mode("twist_rad", t, y, d["s"]["psi"][:, 0], a.k))
        elif ch.startswith("pt:"):
            for i, ax in enumerate("uv"):
                modes.append(analyse_mode(f"{ch[3:]}_{ax}_px", t, d["y"][ch][:, i], d["s"][ch][:, i], a.k))
        elif ch == "nrm":
            for i, ax in enumerate("xy"):
                modes.append(analyse_mode(f"normal_{ax}", t, d["y"][ch][:, i], d["s"][ch][:, i], a.k))
    trig = [(m["T_trigger_s"], m["mode"]) for m in modes if m.get("usable") and m.get("T_trigger_s") is not None and m["mode"] in a.deciding]
    info_trig = [(m["T_trigger_s"], m["mode"]) for m in modes if m.get("usable") and m.get("T_trigger_s") is not None and m["mode"] not in a.deciding]
    if a.discontinuity_s is not None:
        trig.append((a.discontinuity_s, "c_discontinuity"))
    if trig:
        Ta, who = min(trig); Ta_used = max(2.5, Ta); hold = None
    else:
        Ta, who, Ta_used, hold = None, None, 7.5, "no trigger: T_a = 10 s, last 2.5 s held out -> fit window 0-7.5 s"
    out = {"rule": "CHANGELOG 2026-10-26T03:09:32Z", "T_a_raw_s": Ta, "triggered_by": who, "T_a_fit_window_s": Ta_used,
           "note": hold, "deciding_modes": a.deciding, "informational_triggers": info_trig, "modes": modes}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"T_a_raw_s": Ta, "triggered_by": who, "informational_triggers": info_trig, "T_a_fit_window_s": Ta_used, "note": hold,
                      "per_mode": {m["mode"]: (m.get("usable"), m.get("T_trigger_s"), [x["rule"] for x in m.get("triggers", [])]) for m in modes}}, indent=1))


if __name__ == "__main__":
    main()
