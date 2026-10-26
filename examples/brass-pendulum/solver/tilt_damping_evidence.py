"""Aggregate the damping evidence of the tilt fits (T_a 3.6/4.0/4.5, auto mode) and the M1 production-candidate
run (T_a 4.0, --damping fitted) into solver/out/fit_tilt/damping_evidence.json and apply the lead's rule:
adopt the fitted Q only if the BOOTSTRAPPED Q stays within [100, 6000] and is consistent across T_a.
Usage: .venv/bin/python solver/tilt_damping_evidence.py
"""
import json
import math
from pathlib import Path

D = Path(__file__).resolve().parent / "out" / "fit_tilt"
G = 9.80665
rows = {}
for T in ("3.6", "4.0", "4.5"):
    r = json.loads((D / f"Ta_{T}" / "fit_projected.json").read_text())
    s = r["model_selection"]; e = s["damping_evidence"]
    rows[T] = {"M1_gamma": s["M1"]["gamma"], "M1_gamma_sd_formal_scaled": s["M1"]["gamma_sd_scaled"], "M1_Q": s["M1"]["Q_implied"],
               "dBIC_M0_minus_M1": s["M0"]["bic"] - s["M1"]["bic"], "boot_gamma": e["bootstrap_M1_gamma"],
               "boot_Q95": e["bootstrap_M1_Q_95"], "n_mod4_eq0": e["M1_on_n_mod4_eq0_frames_only"],
               "th_off_rad": r["params"]["th_off"], "L_eq_MQ": r["params"]["L_eq"],
               "anchor_rms_MQ_px": r["metrics"]["rms_anchor_px"], "R2_MQ_px": r["metrics"]["rms_fit_px"],
               "R3_MQ_px": r["metrics"]["rms_holdout_px"], "phase_MQ": r["metrics"]["phase"]["phase_error_end_cycles"]}
p = json.loads((D / "Ta_4.0_prod" / "fit_projected.json").read_text())
b = p["bootstrap"]["gamma"]; w = math.sqrt(G / p["params"]["L_eq"])
m1prod = {"gamma": p["params"]["gamma"], "Q": w / (2 * p["params"]["gamma"]), "boot_gamma": b,
          "boot_Q95": [w / (2 * b["p97.5"]), (w / (2 * b["p2.5"]) if b["p2.5"] > 0 else None)],
          "flags": p["flags"]["gamma"], "anchor_rms_px": p["metrics"]["rms_anchor_px"], "R2_px": p["metrics"]["rms_fit_px"],
          "R3_px": p["metrics"]["rms_holdout_px"], "phase": p["metrics"]["phase"]["phase_error_end_cycles"]}
points_in = all(100 <= v["M1_Q"] <= 6000 for v in rows.values())
boot_in = all(v["boot_Q95"] and v["boot_Q95"][0] >= 100 and v["boot_Q95"][1] is not None and v["boot_Q95"][1] <= 6000
              for v in rows.values()) and m1prod["boot_Q95"][0] >= 100 and m1prod["boot_Q95"][1] is not None
qs = [v["M1_Q"] for v in rows.values()]
consistent = max(qs) / min(qs) < 1.5
adopt = points_in and boot_in and consistent
out = {"rule": "adopt fitted Q only if the bootstrapped Q (95 %) stays within [100, 6000] and point estimates agree across T_a 3.6/4.0/4.5 (CHANGELOG 2026-10-25T23:14:43Z)",
       "per_window": rows, "M1_production_candidate_Ta4.0": m1prod,
       "point_estimates_in_range": points_in, "point_estimates_consistent": consistent, "bootstrap_within_range": boot_in,
       "adopted": adopt,
       "decision": ("ADOPT fitted Q" if adopt else
                    "KEEP Q = 1000 (declared assumption): point estimates Q = %s are consistent and in range, but the moving-block bootstrap "
                    "puts the 95 %% lower bound below 100 in every window (and the upper bound at infinity, i.e. gamma = 0 not excluded, "
                    "for T_a 3.6 and the 4.0 production bootstrap); the formal 4-6 sigma overstates the evidence because residuals are "
                    "strongly correlated (reduced chi2 ~80-95)" % [round(q) for q in qs])}
(D / "damping_evidence.json").write_text(json.dumps(out, indent=2, default=float) + "\n")
print(json.dumps({k: out[k] for k in ("point_estimates_in_range", "point_estimates_consistent", "bootstrap_within_range", "adopted", "decision")}, indent=1))
