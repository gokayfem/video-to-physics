"""Assemble solver/out/fit_projected/params_fit_provenance.json (input to make_params.py --extra-json) from
fit_projected.json and arc_shape.json.  Deterministic: the fit-file hash excludes its created_utc field.
Usage: .venv/bin/python solver/make_fit_provenance.py
"""
import argparse
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "out"
ap = argparse.ArgumentParser(); ap.add_argument("--fit-dir", default=str(OUT / "fit_anchored" / "Ta_4.0"))
ap.add_argument("--evidence", default=str(OUT / "fit_anchored" / "anchor_evidence.json"))
ap.add_argument("--damping-evidence", default=None); ap.add_argument("--tilt-check", default=None)
args = ap.parse_args()
D = Path(args.fit_dir).resolve()
r = json.loads((D / "fit_projected.json").read_text())
a = json.loads((OUT / "fit_projected" / "arc_shape.json").read_text())
ev = json.loads(Path(args.evidence).read_text()) if Path(args.evidence).exists() else None
b = r["bootstrap"]; m = r["metrics"]
r_nots = {k: v for k, v in r.items() if k != "created_utc"}
ex = {"fit_file": str((D / "fit_projected.json").relative_to(OUT.parent.parent)),
      "fit_window": r.get("fit_window"), "t_anchor_s": r.get("t_anchor_s"),
      "anchor_policy": "CHANGELOG 2026-10-25T23:06:55Z (user decision): physics fitted on the opening window only; correct physics runs unchanged to 10 s; later motion = divergence map",
      "fit_file_sha256_excluding_created_utc": hashlib.sha256(json.dumps(r_nots, sort_keys=True).encode()).hexdigest(),
      "inputs": r["inputs"], "production_fit": r["production_fit"],
      "fitted": {k: {"value": r["params"][k], "sd_formal_scaled": r["sd_scaled"][k], "sd_bootstrap": b[k]["sd"],
                     "ci95_bootstrap": [b[k]["p2.5"], b[k]["p97.5"]]} for k in ["L_eq", "theta0", "omega0", "k"]},
      "k_note": "k = |bob - P_ring| in camera calibration units L_eff (image geometry); camera metre scale L_eff_m = L_c/k",
      "camera_metre_scale_L_eff_m": r["derived"]["camera_metre_scale_L_eff_m"],
      "damping_selection": r["model_selection"],
      "Q_sensitivity": {k: {kk: v[kk] for kk in ("rms_fit_px", "rms_holdout_px", "phase_end")} for k, v in r["Q_sensitivity"].items()},
      "reference_fidelity": {"anchor_rms_px": m.get("rms_anchor_px"), "R2_rms_fit_px": m["rms_fit_px"], "R2_pass": m["R2_pass"], "R3_rms_holdout_px": m["rms_holdout_px"],
                             "R3_phase_end_cycles": m["phase"]["phase_error_end_cycles"], "R3_pass": m["R3_pass"],
                             "supplementary_n_mod4_eq0": m["supplementary_n_mod4_eq0"],
                             "attribution": "reference defects (measurement v001): non-monotonic amplitude +62%, period wander, cadence offsets, swing asymmetry; passive model not bent to match"},
      "reduced_chi2": r["reduced_chi2"], "pivot": "fixed at P_ring (lead decision); free-pivot fits are diagnostics only",
      "arc_shape_reference_defect": {"source": "solver/arc_shape_check.py -> solver/out/fit_projected/arc_shape.json",
                                     "hinge_P_ring_rms_px": a["H1_hinge_P_ring (production)"]["rms_px"],
                                     "free_hinge_rms_px": a["H2_free_hinge_dz (AI apparent, diagnostic)"]["rms_px"],
                                     "max_abs_radial_px_P_ring": a["H1_hinge_P_ring (production)"]["max_abs_radial_px"],
                                     "note": "dynamics-free per-frame-theta fit; arc-shape (rotation point) effect ~0.6 px RMS in quadrature, max 3.8 px radial; reference defect"}}
if ev:
    ex["anchor_window_evidence"] = {"file": "solver/out/fit_anchored/anchor_evidence.json", "criterion": ev["criterion"],
                                    "onset": ev["onset"], "T_a_s": ev["T_a_s"],
                                    "T_a_sensitivity": {k: {kk: v[kk] for kk in ("L_eq", "L_eq_sd_boot", "rms_anchor_px", "R2_rms_px", "R3_rms_px", "phase_end_cycles", "M1_Q", "adopted")}
                                                        for k, v in ev["T_a_sensitivity"].items()},
                                    "damping_decision": ev["damping_decision"],
                                    "conditional_damping_with_equilibrium_offset_DIAGNOSTIC": ev["conditional_damping_with_equilibrium_offset_DIAGNOSTIC"]}
    ex["damping_evidence_production_window"] = r["model_selection"].get("damping_evidence")
if args.damping_evidence:
    ex["damping_evidence_tilt_fits"] = json.loads(Path(args.damping_evidence).read_text())
if args.tilt_check:
    tc = json.loads(Path(args.tilt_check).read_text())
    ex["level_tilt_verification"] = {k: tc[k] for k in ("beta_rad", "check1_max_abs_diff", "check2_blender_euler_vs_Ry_max_abs_diff",
                                                         "check3_anchor_turning_point_asymmetry", "anchor_with_tilt",
                                                         "anchor_without_tilt_same_params", "verdict") if k in tc}
    ex["level_tilt_verification"]["file"] = str(Path(args.tilt_check).resolve().relative_to(OUT.parent.parent))
if r.get("level_tilt"):
    ex["level_tilt_fit"] = r["level_tilt"]
    ex["fitted"]["th_off"] = {"value": r["params"]["th_off"], "sd_formal_scaled": r["sd_scaled"]["th_off"],
                              "sd_bootstrap": b["th_off"]["sd"], "ci95_bootstrap": [b["th_off"]["p2.5"], b["th_off"]["p97.5"]]}
(D / "params_fit_provenance.json").write_text(json.dumps(ex, indent=2, default=float) + "\n")
print("wrote", D / "params_fit_provenance.json")
