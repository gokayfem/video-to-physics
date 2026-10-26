"""Merge cadence/control/verdicts/frame-quality into summary.json and write a sha256 manifest."""
import numpy as np, os, sys, json, hashlib, glob
sys.path.insert(0, os.path.dirname(__file__))
from common import *
S = json.load(open(os.path.join(MEAS, "summary.json")))
A = np.load(os.path.join(CACHE, "analysis.npz")); Bz = np.load(os.path.join(CACHE, "bob_raw.npz"))
cad = json.load(open(os.path.join(MEAS, "cadence_analysis.json"))); ctl = json.load(open(os.path.join(MEAS, "control_straight_edges.json")))
fp = np.array(S["period"]["zero_crossings"]["full_periods_s"]); zc = A["zc"]; tm = 0.5 * (zc[:-2, 0] + zc[2:, 0])
pt = np.polyfit(tm, fp, 1, cov=True)
Ln = A["lenp"] / A["simp"][:, 0]; Rn = Bz["radA"] / A["simp"][:, 0]
S["period"]["full_period_linear_trend_s_per_s"] = [round(float(pt[0][0]), 5), round(float(np.sqrt(pt[1][0, 0])), 5)]
S["cadence_defect"] = {"finding": "4-frame generator cadence: the bob step into every frame n = 4k is ~1.58x the local velocity, the other three steps ~0.80-0.84x. Present in every GOP although the H.264 P-frame grid shifts (P at n mod 4 = 0,2,3,1 in different GOPs), so it is generator-native, not encoder.",
                       "apparent_time_offset_frames_by_n_mod_4": cad["angle_deg"]["delta_frames_by_n_mod_4"], "sd_frames": cad["angle_deg"]["delta_sd_frames"],
                       "equivalent_peak_position_error_px": round(float(0.29 * 25.0), 1),
                       "angle_rms_resid_windowed_sine_without_with_offsets_deg": [cad["angle_deg"]["rms_resid_without_cadence"], cad["angle_deg"]["rms_resid_with_cadence_offsets"]],
                       "u_stab_rms_resid_windowed_sine_without_with_offsets_px": [cad["u_stab_px"]["rms_resid_without_cadence"], cad["u_stab_px"]["rms_resid_with_cadence_offsets"]],
                       "note": "diagnostic only; must not be used to retime the simulation (contract). Expect R2/R3 residual floors of a few px at peak speed from this alone."}
S["suspension"]["control_static_edges"] = {k: {"sagitta_abs_max_px": v["sagitta_abs_max_px"], "max_dev_max_px": v["max_dev_max_px"]} for k, v in ctl.items()}
S["suspension"]["verdict"] = ("CURVED in the image whenever the bob is away from centre: quadratic sagitta from -4.9 to +4.7 px over ~650 px of wire "
                              "(formal sd ~0.03 px), correlated with the swing angle (r=0.94, 0.53 px/deg); near zero crossings within ~1-2 px of straight. "
                              "Control: static stand edges of similar length measured the same way are straight to <=0.45 px, so lens distortion does not explain it. "
                              "A straight 3D wire projects to a straight image line (pinhole), so the bowing is an image-content property of the AI clip.")
S["bob"]["normalized_radius_max_framediff_px"] = round(float(np.abs(np.diff(Rn)).max()), 3)
S["bob"]["normalized_pivot_to_bob_max_framediff_px"] = round(float(np.abs(np.diff(Ln)).max()), 3)
S["verdicts"] = {
  "camera": "NOT static. Smooth monotonic zoom-in/dolly-in: similarity scale 1.0000 -> 1.0327 nearly linear in t, rotation < 0.05 deg, pivot moves (+4.1,-0.8) px, base corners ~25-30 px. A per-frame homography explains all static landmarks to 0.77 px RMS (residual grows to ~1-1.5 px late in the clip: small parallax / non-planarity). camera_drift.csv.",
  "pivot": "Fixed relative to the stand: wire departure point residual after drift 0.67 px RMS; swing-correlated motion <=0.5 px peak-to-peak; the loop on the pin does not visibly rock.",
  "wire": S["suspension"]["verdict"],
  "amplitude": "NOT monotonic decay. Half peak-to-peak image angle 5.77 deg (t=0.86 s) -> 5.45 (t=3.64) -> GROWS to 8.84 (t=7.13) -> 6.48 (t=9.26); per-point sd 0.02-0.04 deg. Net growth over the clip; non-physical for a passive pendulum (reference defect). Camera drift (similarity, rot<0.05 deg) cannot create this because the image angle is similarity-invariant; stabilized-u amplitude shows the same pattern.",
  "period": "Image-space period 1.404 +- 0.003 s (zero crossings), 1.4007 +- 0.0012 s (damped-sine fit, formal; model inadequate, reduced chi2 291). Cycle-to-cycle 1.357-1.434 s (sd 0.026 s, >> timing sd 0.002 s): period itself is not constant. Eyeball 1.25 s was wrong.",
  "phase_t0": "Not at a turning point at t=0: bob is at about -3.3 deg (image angle minus offset, ~57% of amplitude) on the image-left side moving right at ~20 deg/s; first turning point (right) at t=0.522 s; the preceding left turning point extrapolates to t~-0.20 s. Not released from rest at t=0.",
  "bob_shape": "Silhouette is near-circular: ellipse minor/major 0.94-0.995 (median ~0.977, not correlated with speed, r=-0.02, so not motion blur; cause not determined). Apparent radius varies smoothly 55.3-60.2 px, ~5%% larger at the image-right extreme (both methods); max frame-to-frame change of drift-normalized radius %.2f px, of drift-normalized pivot-to-bob distance %.2f px (largest at fast frames, cadence)." % (np.abs(np.diff(Rn)).max(), np.abs(np.diff(Ln)).max())}
S["frame_quality"] = {"codec": "H.264 yuv420p, 5.96 Mbit/s, GOPs of ~30 frames with I/P/B (8 I, 64 P, 169 B)",
                      "chroma": "4:2:0 plus compression makes chroma blocky at 8-16 px; all geometry uses full-resolution luma only",
                      "motion_blur": "visible horizontal blur on the bob's left/right rim near mid-swing (e.g. frames 20-21, 36); top/bottom rims stay sharp; a circle-fit centre is unbiased for symmetric blur; fitted ellipse ratio shows no measurable speed dependence",
                      "wire": "twisted two-strand look, ~5 px wide, dark on light wall; centroid of the dark band is used (darkest pixel alternates between strands)",
                      "highlight": "specular highlight and reflections move inside the bob and are excluded (edge-only fit A; annulus-only template B)"}
json.dump(S, open(os.path.join(MEAS, "summary.json"), "w"), indent=1)
files = sorted([p for p in glob.glob(os.path.join(MEAS, "**", "*"), recursive=True) if os.path.isfile(p) and "/cache/" not in p and not p.endswith("MANIFEST.sha256")])
with open(os.path.join(MEAS, "MANIFEST.sha256"), "w") as fh:
    for p in files: fh.write("%s  %s\n" % (hashlib.sha256(open(p, "rb").read()).hexdigest(), os.path.relpath(p, MEAS)))
print(json.dumps(S["verdicts"], indent=1)); print(S["period"]["full_period_linear_trend_s_per_s"], S["bob"]["normalized_radius_max_framediff_px"], S["bob"]["normalized_pivot_to_bob_max_framediff_px"])
