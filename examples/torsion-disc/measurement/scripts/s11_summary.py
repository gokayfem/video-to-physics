"""Stage 11: measurement/summary.json + MANIFEST.sha256."""
import json, os, sys, hashlib, glob
sys.path.insert(0, os.path.dirname(__file__))
from common import *
AN = json.load(open(os.path.join(MEAS, "analysis.json"))); AW = json.load(open(os.path.join(MEAS, "anchor_window.json"))); CJ = json.load(open(os.path.join(ROOT, "params/camera.json")))
RP = json.load(open(os.path.join(ROOT, "camera/out/reprojection_by_frame.json"))); PV = json.load(open(os.path.join(ROOT, "camera/out/projection_validation.json")))
S = {"schema": "sim03-measurement-summary/v1", "version": VERSION, "status": STATUS, "reference_sha256": VIDEO_SHA, "frames": 241, "fps": 24,
     "unit": "D = disc diameter = 1 (declared gauge); metres not set (prompt intent ~0.10 m)",
     "camera": {"smoothed": True, "jerk_right_post_px": {k: json.load(open(os.path.join(ROOT, "camera/out/camera_smoothing.json")))["chosen"]["metrics"]["right_post"][k] for k in ("cam_jerk_median", "cam_jerk_max", "ref_jerk_median", "ref_jerk_max")}, "verdict": CJ["drift"]["verdict"], "f_px_constant": CJ["reference_camera"]["intrinsics"]["f_px"], "f_systematic_range_px": [2550, 2690],
                "azimuth_deg_frame0_240": CJ["drift"]["azimuth_deg_frame0_240"], "distance_to_A_frame0_240_min_max_D": CJ["drift"]["distance_to_A_frame0_240_min_max"],
                "R4_static_rms_px": {str(k): v["reference"]["rms_px"] for k, v in RP["frames"].items()}, "R4_all_frames": RP["all_frames_rms_px"], "R4_pass": RP["all_pass_reference"],
                "R4_locked_camera_rms_px": {str(k): v["locked"]["rms_px"] for k, v in RP["frames"].items()},
                "blender_projection_max_err_px": max(r["max_err_px"] for r in PV["results"]), "blender_pass": PV["all_pass"]},
     "twist": {"definition": "psi = world azimuth (from +X toward +Y, i.e. counter-clockwise seen from above) of the engraved diameter line in the disc plane; unwrapped",
               "psi0_deg": AN["psi"]["psi0_deg"], "total_rotation_deg": AN["psi"]["total_rotation_deg"], "turns": AN["psi"]["turns"], "turning_points": AN["psi"]["turning_points"],
               "rate_by_second_deg_s": AN["psi"]["rate_by_second_deg_s"], "rate_t0_deg_s_(7th-order)": AN["psi"]["rate_t0_deg_s"], "methods": AN["psi_methods"], "models": AN["psi"]["models"],
               "verdict": "NO torsional oscillation in 0-10 s: the disc spins one way (clockwise from above) by 379 deg, rate -25 -> -47 -> -33 deg/s. Fits as the middle of a very slow large oscillation (T ~ 32 s, amplitude ~226 deg) or cubic; neither is identifiable as torsion physics from this clip"},
     "disc": {"centre_mean_D": AN["disc_centre"]["mean_D"], "wire_length_A_to_top_face_D": AN["disc_centre"]["wire_length_A_to_top_face_mean_D"], "thickness_D": AN["thickness_D"],
              "tilt": AN["tilt"], "silhouette_rms_px_median": AN["silhouette_rms_px_median"], "line_kink_deg": AN["kink"], "line_rotation_centre_offset_from_silhouette_centre_D": AN["apex_offset_D"]},
     "sway": {"Qx": AN["sway"]["Qx_D"], "Qy": AN["sway"]["Qy_D"], "tilt_about_Y": AN["sway"]["tilt_about_Y_deg"], "wire_image_tilt": AN["sway"]["wire_image_tilt_deg"], "spectral_peaks_hz": AN["spectral_peak_hz"],
              "swing_angles_deg": AN["swing_angles_deg"], "two_D_amplitude": {k: AW["sway_2D_amplitude_supplementary"][k] for k in ("window_centres_s", "amp_D", "sd_D")}},
     "disc_P_track_for_fit": {"file": "measurement/disc_P_track.csv", "meta": json.load(open(os.path.join(MEAS, "disc_P_track.json"))),
                              "note": "lead decision VR03-05: the dynamics fit uses the silhouette centre P (not the mark rotation centre Q)"},
     "camera_smoothing": {k: json.load(open(os.path.join(ROOT, "camera/out/camera_smoothing.json")))[k] for k in ("chosen_tau_s", "max_deviation_from_raw")},
     "cadence": AN["cadence"], "highlight_lobes": AN["highlight_lobes"],
     "anchor_window": {"T_a_s": AW["T_a_s"], "reason": AW["T_a_reason"], "triggers": AW["triggers_sorted"], "twist": AW["twist_psi"], "discontinuities": AW["c_discontinuities"]},
     "files": {"twist": "measurement/twist_track.csv", "twist_raw": "measurement/twist_raw.csv", "disc": "measurement/disc_track.csv", "disc_pose_raw": "measurement/disc_pose_camcentred.csv (world, file name historical)",
               "disc_ellipse_image": "measurement/disc_ellipse_image.csv", "wire": "measurement/wire_track.csv", "camera": "params/camera.json, camera/drift.csv, camera/project.py",
               "analysis": "measurement/analysis.json, anchor_window.json"}}
json.dump(S, open(os.path.join(MEAS, "summary.json"), "w"), indent=1)
files = sorted(set(glob.glob(os.path.join(ROOT, "measurement/*.csv")) + glob.glob(os.path.join(ROOT, "measurement/*.json")) + glob.glob(os.path.join(ROOT, "measurement/scripts/*.py")) +
                   glob.glob(os.path.join(ROOT, "measurement/visuals/*")) + glob.glob(os.path.join(ROOT, "camera/*.py")) + glob.glob(os.path.join(ROOT, "camera/test/*.py")) + glob.glob(os.path.join(ROOT, "camera/out/*.json")) +
                   [os.path.join(ROOT, "camera/drift.csv"), os.path.join(ROOT, "params/camera.json")]))
with open(os.path.join(MEAS, "MANIFEST.sha256"), "w") as fh:
    for f in files: fh.write("%s  %s\n" % (hashlib.sha256(open(f, "rb").read()).hexdigest(), os.path.relpath(f, ROOT)))
print("summary + manifest:", len(files), "files")
