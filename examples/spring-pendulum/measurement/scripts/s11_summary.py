"""Stage 11: measurement/summary.json (key numbers, verdicts, T_a) + MANIFEST.sha256."""
import numpy as np, os, sys, json, csv, hashlib, glob, datetime
sys.path.insert(0, os.path.dirname(__file__))
from common import *
AN = json.load(open(os.path.join(MEAS, "analysis.json"))); AW = json.load(open(os.path.join(MEAS, "anchor_window.json"))); T3 = json.load(open(os.path.join(MEAS, "tracks_3d.json")))
CA = json.load(open(os.path.join(MEAS, "cadence_analysis.json"))); DF = json.load(open(os.path.join(ROOT, "camera/out/drift_fit.json"))); RP = json.load(open(os.path.join(ROOT, "camera/out/reprojection_by_frame.json")))
PV = json.load(open(os.path.join(ROOT, "camera/out/projection_validation.json"))); CJ = json.load(open(os.path.join(ROOT, "params/camera.json")))
PL = json.load(open(os.path.join(MEAS, "drift_parallax.json")))
M = list(csv.DictReader(open(os.path.join(MEAS, "modes_series.csv")))); f = lambda k: np.array([float(r[k]) for r in M])
ph = f("swing_angle_deg"); W = list(csv.DictReader(open(os.path.join(MEAS, "weight_track.csv")))); tb = np.array([float(r["tilt_stab_frame0_deg"]) for r in W]); tc = np.array([float(r["tilt_methodC_deg"]) for r in W])
sw = AN["swing_damped_sine"]; hc = AN["half_cycles_swing"]
S = {
 "version": "v001", "status": "reference measurement - draft", "created_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
 "reference": {"path": "WS/reference-batch/clips/02-vertical-spring.mp4", "sha256_verified": VIDEO_SHA, "frames": 241, "fps": 24, "size": [1920, 1088], "codec": "h264 yuv420p 8.3 Mbit/s"},
 "labels": "[obs] measured from pixels; [fit] model fitted to measurements; [inf] inference; [asm] assumption",
 "camera_drift": {"verdict": "[obs] NOT static: smooth push-in, image scale +%.2f %% over 10 s (%.3f %%/s, near-linear) about (%.0f, %.0f) px, pan tx %.1f px / ty %.1f px, roll < %.3f deg" % (
                      100 * (DF["scale_frame240"] - 1), DF["scale_pct_per_s"], *DF["zoom_fixed_point_median_px"], min(DF["tx_range_px"]), min(DF["ty_range_px"]), np.degrees(DF["roll_max_abs_rad"])),
                  "method": "8-dof ECC homography of every frame vs frame 0 on a static mask (KLT-chain init); per-frame similarity extracted",
                  "evidence": {"static_landmark_track_vs_homography_rms_px": "0.25-0.73 for the 8 top-frame landmarks (static_landmarks.csv); post-bottom/plate landmarks 0.7-2.1",
                               "similarity_vs_homography_rms_median_px": DF["H_vs_sim_rms_median_px"], "landmarks_vs_similarity_rms_median_px": DF["landmarks_vs_sim_rms_median_px"],
                               "landmark_displacement_raw_max_px": "25-33 (frame corners) at frame 240", "hook_top_vs_frame_after_drift_rms_px": AN["hook_top_resid_after_drift_rms_px"],
                               "layer_scales_frame240": {"posts": PL["frames"]["240"]["posts"]["similarity_scale"], "table": PL["frames"]["240"]["table"]["similarity_scale"]}},
                  "zoom_vs_dolly": "frame and side table give the same scale within 0.3 %; backdrop textureless (inconclusive). Modelled as zoom (clip-01 precedent); dolly not excluded"},
 "camera": {"f_px": CJ["reference_camera"]["intrinsics"]["f_px"], "f_px_range_variants": CJ["uncertainty"]["f_px_variants_range"], "focal_mm_36": CJ["reference_camera"]["intrinsics"]["focal_mm_36mm_sensor"],
            "C_world_Win": CJ["reference_camera"]["extrinsics"]["C_world"], "distance_to_A_Win": CJ["reference_camera"]["extrinsics"]["distance_to_A"], "pitch_deg": CJ["reference_camera"]["extrinsics"]["view_pitch_deg"],
            "yaw_deg": CJ["reference_camera"]["extrinsics"]["view_yaw_deg_from_+Y_toward_+X"], "unit": "W_in = crossbar inner span = 1",
            "R4_static_reprojection_rms_px": {str(k): round(v["reference"]["rms_px"], 3) for k, v in RP["frames"].items()}, "R4_locked_camera_rms_px": {str(k): round(v["locked"]["rms_px"], 3) for k, v in RP["frames"].items()},
            "R4_pass": RP["all_pass_reference"], "blender_projection_max_err_px": max(r["max_err_px"] for r in PV["results"]), "blender_pass": PV["all_pass"]},
 "tracks": {"weight": {"methods": "B silhouette profiles + Tukey line fits (primary); C Canny + RANSAC lines (check); A photometric ECC affine kept as diagnostic (fails: highlights do not move with the body)",
                       "B_vs_C_rms_px": AN["weight_B_vs_C_rms_px"], "B_vs_C_max_px": AN["weight_B_vs_C_max_px"], "sigma_median_px": AN["weight_sigma_median_px"], "sigma_max_px": AN["weight_sigma_max_px"],
                       "tilt_B_vs_C_rms_deg": AN["weight_tilt_B_vs_C_rms_deg"], "tilt_B_vs_C_const_offset_deg": AN["weight_tilt_B_vs_C_const_offset_deg"], "tilt_sigma_median_deg": AN["weight_tilt_sigma_median_deg"],
                       "note": "B-C differences are swing-correlated (systematic ~2-4 px, ~2 deg), i.e. silhouette definition depends on motion blur/background; treat weight centre to +-3 px and tilt to +-2 deg (static offset +-1.8 deg)"},
            "ring": {"methods": "A radial edges + robust circle fit (primary); B annulus ZNCC (check)", "A_vs_B_rms_px": AN["ring_A_vs_B_rms_px"], "A_vs_B_max_px": AN["ring_A_vs_B_max_px"], "sigma_median_px": AN["ring_sigma_median_px"],
                     "note": "eye-ring centre = spring lower attachment; most precise moving observable; single-frame outlier at frame 116 (method A radius -3 px)"},
            "spring": {"methods": "row-extent midpoint line fit (primary) vs gradient-centroid line fit (check)", "tilt_methods_rms_deg": AN["spring_tilt_extent_vs_centroid_rms_deg"],
                       "body_length_stab_px": AN["spring_body_len_stab_px"], "coils": AN["coils"], "top_attachment": "S-hook static to %.2f px rms (max %.2f) after drift removal: the hook does NOT move" % (AN["hook_top_resid_after_drift_rms_px"], AN["hook_top_resid_max_px"]),
                       "pivot_P_eff_frame0_px": AN["pivot_Peff_frame0_px"], "pivot_P_eff_sd_px": AN["pivot_Peff_sd_px"]},
            "files": ["weight_track.csv", "ring_track.csv", "spring_track.csv", "static_landmarks.csv", "modes_series.csv", "tracks_3d.csv", "camera_drift (cache/H_ecc.npy; camera/drift.csv)"]},
 "three_d": {"assumption": T3["assumption"], "size_tests": T3["size_tests"],
             "verdict": "[obs/inf] no consistent out-of-plane motion: weight-width depth cue gives swing-plane yaw %.0f +- %.0f deg, ring-radius cue %.0f +- %.0f deg (opposite signs -> systematics dominate); quadrature (elliptical) terms consistent with 0. Plane Y=0 adopted; |yaw| <~ 15 deg not excluded" % (
                 T3["size_tests"][0]["swing_plane_yaw_deg"], T3["size_tests"][0]["swing_plane_yaw_sd_deg"], T3["size_tests"][1]["swing_plane_yaw_deg"], T3["size_tests"][1]["swing_plane_yaw_sd_deg"]),
             "P_eff_world_Win": T3["P_eff_world"], "A_to_P_eff_Win": T3["A_to_Peff_Win"], "P_eff_to_ring_Win": T3["r_ring_Win"], "P_eff_to_weight_centre_Win": T3["r_weight_Win"],
             "ring_to_weight_centre_Win": T3["ring_to_weight_centre_Win"], "weight_diameter_Win": T3["weight_size_Win"]["diameter"], "weight_silhouette_length_Win": T3["weight_size_Win"]["silhouette_length"],
             "spring_coil_body_length_Win": T3["spring_body_len_Win"], "theta_ring_deg_range": [T3["theta_ring_deg"]["min"], T3["theta_ring_deg"]["max"]]},
 "modes": {
   "lateral_swing": {"present": True, "period_s_sine_fit": [sw["period_s"], sw["s_period_s"]], "period_s_zero_crossings": AN["swing_period_from_zero_crossings_s"],
                     "amplitude_deg_image": [hc[0]["amp"], max(h["amp"] for h in hc), hc[-1]["amp"]], "amplitude_per_half_cycle": [(round(h["t_mid"], 3), round(h["amp"], 3), round(h["s_amp"], 3)) for h in hc],
                     "half_period_per_half_cycle": [(round(h["t_mid"], 3), round(h["half_period"], 4), round(h["s_half_period"], 4)) for h in hc],
                     "phase_t0": AW["start_of_clip"]["note"], "decay_fit_per_s": [sw["decay_per_s"], sw["s_decay_per_s"]], "sine_chi2nu": sw["chi2nu"]},
   "vertical_bob": {"present": False, "evidence": AN["vertical_bob_search"], "radial_fit": AN["radial_fit"],
                    "verdict": "[obs] no independent axial (bob) oscillation: pivot-ring distance varies only +-1.2 px rms about 546 px, explained by swing-locked w and 2w terms (spectral peaks 0.767 and 1.533 Hz); residual after regression on phi, phi^2, phidot^2 is %.2f px rms with remaining power at the swing frequency only. The prompt's 1 cm vertical bob is absent." % AN["vertical_bob_search"]["residual_rms_px"]},
   "weight_rocking": {"present": True, "gain_tilt_per_swing_B": float(np.polyfit(ph, tb, 1)[0]), "gain_tilt_per_swing_C": float(np.polyfit(ph, tc, 1)[0]), "static_offset_deg_B": float(np.polyfit(ph, tb, 1)[1]),
                      "amplitude_per_half_cycle_deg": [(round(h["t_start"], 3), round(h["amp"], 3), round(h["s_amp"], 3)) for h in AN["half_cycles_weight_tilt_at_swing_tps"]],
                      "verdict": "[obs] the weight tilts in phase with the swing, ~1.3-1.4x the swing angle (i.e. it rocks ~2.5-3 deg beyond the pivot-ring line): consistent with the in-phase mode of a double pendulum (spring + weight hinged at the ring) [inf]"},
   "spring_axis": {"gain_vs_swing": 0.961, "note": "spring axis tilt ~0.96x the P_eff-ring angle; the spring body pivots about its top coil (P_eff), the S-hook is fixed"},
   "out_of_plane": "see three_d", "spin": "[obs] not measurable: brushed axisymmetric cylinder; photometric ECC shows highlight bands do not follow the body (view-dependent reflections)",
   "frequency_ratio": "undefined: no independent axial frequency observed (only swing harmonics); autoparametric 2:1 not testable from the reference"},
 "cadence": {"offsets_frames_by_n_mod_4": CA["ring_u_stab_px"]["delta_frames_by_n_mod_4"], "sd": CA["ring_u_stab_px"]["delta_sd_frames"], "step_ratio_by_destination_n_mod_4": CA["step_ratio_by_n_mod_4_of_destination_frame"],
             "verdict": "[obs] same generator 4-frame cadence as clip 01: steps into n = 0 mod 4 are ~1.67x; diagnostic only, not a retiming licence"},
 "discontinuities": AW["visible_discontinuities"],
 "anchor_window": {"T_a_s": AW["T_a_s"], "held_out_s": AW["held_out_s"], "reason": AW["reason"], "near_misses": AW["near_misses"], "literal_radius_note": AW["literal_note"], "trends_not_tested_by_rule": AW["trends_not_tested_by_rule"]},
 "ai_defects_vs_prompt": ["lateral swing +-8 deg (image) with weight rocking (prompt: straight vertical bob, no sideways motion)", "no visible vertical bob (prompt: ~1 cm bob, decaying)",
                          "swing amplitude grows +4.5 % over 1-4 s then decays (non-physical energy injection)", "half-period lengthens 7.5 % over the clip (period wander)",
                          "camera push-in +4.8 % and pan (prompt: frame perfectly still / locked camera)", "S-hook static while the spring body pivots at its top coil (hook should swing with the spring if hinged at the bar)",
                          "4-frame generator cadence"],
 "visuals": sorted(os.path.basename(p) for p in glob.glob(os.path.join(MEAS, "visuals", "v001_*")))}
def clean(o):
    if isinstance(o, dict): return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)): return round(float(o), 6)
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, np.bool_): return bool(o)
    return o
json.dump(clean(S), open(os.path.join(MEAS, "summary.json"), "w"), indent=1)
files = sorted(p for p in glob.glob(os.path.join(MEAS, "**", "*"), recursive=True) if os.path.isfile(p) and "/cache/" not in p and not p.endswith("MANIFEST.sha256"))
files += sorted(p for p in glob.glob(os.path.join(ROOT, "camera", "**", "*"), recursive=True) if os.path.isfile(p) and "/frames/" not in p and "__pycache__" not in p) + [os.path.join(ROOT, "params/camera.json")]
with open(os.path.join(MEAS, "MANIFEST.sha256"), "w") as fh:
    for p in files: fh.write("%s  %s\n" % (hashlib.sha256(open(p, "rb").read()).hexdigest(), os.path.relpath(p, ROOT)))
print("summary written; manifest", len(files), "files")
