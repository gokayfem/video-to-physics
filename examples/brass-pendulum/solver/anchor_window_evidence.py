"""Evidence for the anchor-window end T_a, T_a sensitivity, and the damping-identifiability tests.

Inputs: measurement/summary.json (turning points, half-cycle amplitudes, zero crossings), and the
solver/out/fit_anchored/Ta_*/fit_projected.json runs (solver/fit_projected.py --t-anchor T).
Outputs: solver/out/fit_anchored/anchor_evidence.json, solver/visuals/<ver>_anchor_window_evidence.png

Onset criterion (fixed before looking at R2/R3): the non-physical growth starts in the first half-cycle
whose amplitude exceeds the previous half-cycle by more than 3 combined sigma.  That half-cycle begins at
the preceding turning point, so the last physically consistent sample is that turning point; T_a is the
first frame time at or after it.
Usage: .venv/bin/python solver/anchor_window_evidence.py [--version v003]
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
G = 9.80665


def conditional_damping(t_list):
    """DIAGNOSTIC: damping identifiability if the equilibrium direction were offset (th_off free)."""
    import fit_projected as FP
    from fit import load_track
    d = load_track(ROOT / "measurement" / "bob_track.csv"); fr = np.round(d["t"] * 24).astype(int)
    F = FP.Fitter(d, FP.Cams(fr))
    base = json.loads((ROOT / "solver/out/fit_anchored/Ta_4.0/fit_projected.json").read_text())["params"]
    out = {}
    for T in t_list:
        an = d["t"] <= T + 1e-9
        names = ["L_eq", "theta0", "omega0", "k", "th_off"]
        R0 = F.fit(dict(base, th_off=0.0126, gamma=0.0), names, an)
        R1 = F.fit(dict(R0["P"], gamma=0.003), names + ["gamma"], an)
        g = R1["P"]["gamma"]
        uv, _, _ = F.model(R0["P"]); e = uv - d["y"]
        out[str(T)] = {"M0_bic": R0["bic"], "M1_bic": R1["bic"], "gamma": g, "gamma_sd_scaled": R1["sd_scaled"]["gamma"],
                       "Q": math.sqrt(G / R1["P"]["L_eq"]) / (2 * g) if g > 0 else None, "th_off_rad": R0["P"]["th_off"],
                       "th_off_deg": math.degrees(R0["P"]["th_off"]), "anchor_rms_px_M0": float(np.sqrt(np.mean(np.sum(e[an] ** 2, 1)))),
                       "L_eq": R0["P"]["L_eq"]}
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--version", default="v003"); a = ap.parse_args()
    s = json.loads((ROOT / "measurement" / "summary.json").read_text())
    hc = s["amplitude"]["half_cycle_peak_to_peak_half_deg"]
    tps = s["amplitude"]["turning_points"]
    onset = None
    for i in range(1, len(hc)):
        dA = hc[i]["amp_deg"] - hc[i - 1]["amp_deg"]; sd = math.hypot(hc[i]["sd_deg"], hc[i - 1]["sd_deg"])
        if dA > 3 * sd:
            onset = {"half_cycle_index": i, "t_center_s": hc[i]["t_s"], "amp_prev_deg": hc[i - 1]["amp_deg"],
                     "amp_deg": hc[i]["amp_deg"], "increase_sigma": dA / sd}
            break
    # the half-cycle centred at t_center spans the turning points either side of that zero crossing
    tp_before = max(tp["t_s"] for tp in tps if tp["t_s"] < onset["t_center_s"])
    T_a = math.ceil(tp_before * 24 - 1e-9) / 24
    zc = s["period"]["zero_crossings"]
    ev = {"criterion": "first half-cycle amplitude > previous + 3 sigma (measurement/summary.json, image angle)",
          "onset": onset, "last_physical_turning_point_s": tp_before, "T_a_s": T_a, "T_a_frame": int(round(T_a * 24)),
          "turning_points_image_deg": [(tp["t_s"], tp["angle_minus_offset_deg"]) for tp in tps],
          "half_cycle_amp_deg": [(h["t_s"], h["amp_deg"], h["sd_deg"]) for h in hc],
          "full_periods_s": zc["full_periods_s"],
          "period_note": "period wander (1.357-1.434 s) is already present in the opening (1.370, 1.396, 1.417, 1.397 s): it cannot be excluded by any window; it limits the anchor fit"}
    runs = {}
    for T in ("3.5", "3.6", "4.0", "4.5"):
        r = json.loads((ROOT / f"solver/out/fit_anchored/Ta_{T}/fit_projected.json").read_text())
        m = r["metrics"]; e = r["model_selection"]["damping_evidence"]; b = r["bootstrap"]
        runs[T] = {"L_eq": r["params"]["L_eq"], "L_eq_sd_boot": b["L_eq"]["sd"], "theta0": r["params"]["theta0"],
                   "omega0": r["params"]["omega0"], "k": r["params"]["k"], "T_model_s": r["derived"]["T_model_s"],
                   "rms_anchor_px": m["rms_anchor_px"], "R2_rms_px": m["rms_fit_px"], "R3_rms_px": m["rms_holdout_px"],
                   "phase_end_cycles": m["phase"]["phase_error_end_cycles"],
                   "M1_gamma": r["model_selection"]["M1"]["gamma"], "M1_Q": r["model_selection"]["M1"]["Q_implied"],
                   "M1_gamma_boot95": [e["bootstrap_M1_gamma"]["p2.5"], e["bootstrap_M1_gamma"]["p97.5"]],
                   "damping_stat_identified": e["statistically_identified_BIC_3sigma"],
                   "damping_boot_identified": e["bootstrap_identified"], "Q_plausible": e["Q_plausible"],
                   "M1_n_mod4_eq0": e["M1_on_n_mod4_eq0_frames_only"], "adopted": e["adopted"]}
    ev["T_a_sensitivity"] = runs
    ev["conditional_damping_with_equilibrium_offset_DIAGNOSTIC"] = conditional_damping([3.6, 4.0, 4.5])
    ev["damping_decision"] = ("Q = 1000 kept (declared assumption). Under the production geometry (gravity vertical, hinge P_ring) "
                              "linear damping is NOT robust: not identified for T_a = 3.5/3.6 s (bootstrap 95 % includes 0); "
                              "identified for T_a = 4.0/4.5 s but Q ~ 44-52 (< 100, implausible) and driven by left/right swing "
                              "asymmetry. Only if the equilibrium direction is offset by ~+0.7 deg (diagnostic, not adopted) does "
                              "damping become consistent across windows (Q ~ 100-140).")
    out = ROOT / "solver/out/fit_anchored/anchor_evidence.json"
    out.write_text(json.dumps(ev, indent=2, default=float) + "\n")

    # visual
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2), dpi=110)
    t_h = [h["t_s"] for h in hc]; A_h = [h["amp_deg"] for h in hc]; s_h = [h["sd_deg"] for h in hc]
    ax[0].errorbar(t_h, A_h, yerr=np.array(s_h) * 3, fmt="o-", ms=4, label="half-cycle amplitude (image angle, ±3σ)")
    tpp = [(tp["t_s"], abs(tp["angle_minus_offset_deg"]), tp["angle_minus_offset_deg"] > 0) for tp in tps]
    ax[0].plot([x for x, y, r in tpp if r], [y for x, y, r in tpp if r], "^", color="tab:red", label="right turning points")
    ax[0].plot([x for x, y, r in tpp if not r], [y for x, y, r in tpp if not r], "v", color="tab:green", label="left turning points")
    ax[0].axvspan(0, T_a, color="tab:blue", alpha=0.12, label=f"anchor window 0–{T_a:.3f} s")
    ax[0].axvline(onset["t_center_s"], color="k", ls=":", label=f"growth onset half-cycle ({onset['increase_sigma']:.0f}σ jump)")
    ax[0].set_xlabel("t [s]"); ax[0].set_ylabel("amplitude [deg, image]"); ax[0].legend(fontsize=6.5); ax[0].set_title("why T_a: amplitude", fontsize=9)
    fp = zc["full_periods_s"]; tz = zc["times_s"]
    ax[1].plot([(tz[i] + tz[i + 2]) / 2 for i in range(len(fp))], fp, "o-", ms=4)
    ax[1].axvspan(0, T_a, color="tab:blue", alpha=0.12); ax[1].set_xlabel("t [s]"); ax[1].set_ylabel("full period [s]")
    ax[1].set_title("period wander (present inside the window too)", fontsize=9)
    Ts = [float(k) for k in runs]; ax2 = ax[2]
    ax2.plot(Ts, [runs[k]["rms_anchor_px"] for k in runs], "o-", label="anchor RMS px")
    ax2.plot(Ts, [runs[k]["R2_rms_px"] for k in runs], "s-", label="R2 (t<7.5) RMS px")
    ax2.plot(Ts, [runs[k]["R3_rms_px"] for k in runs], "d-", label="R3 (t≥7.5) RMS px")
    ax2.axvline(T_a, color="tab:blue", ls="--"); ax2.set_xlabel("T_a [s]"); ax2.legend(fontsize=7)
    ax3 = ax2.twinx(); ax3.plot(Ts, [runs[k]["L_eq"] * 1000 for k in runs], "k^:", label="L_eq [mm]"); ax3.set_ylabel("L_eq [mm]")
    ax2.set_title("T_a sensitivity", fontsize=9)
    fig.suptitle(f"sim01 dynamics {a.version} | isolated component – draft | anchor-window evidence (measured image amplitudes from measurement/summary.json; "
                 f"T_a = {T_a:.3f} s by a pre-set 3σ onset criterion)", fontsize=7.5)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    vp = ROOT / "solver/visuals" / f"{a.version}_anchor_window_evidence.png"; fig.savefig(vp)
    print(json.dumps({"onset": onset, "T_a": T_a, "tp_before": tp_before}, indent=1)); print(vp)
    print(json.dumps(ev["conditional_damping_with_equilibrium_offset_DIAGNOSTIC"], indent=1))


if __name__ == "__main__":
    main()
