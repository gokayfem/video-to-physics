"""Write params/camera.json and camera/drift.csv (clip 02) from camera/out/calib_result.json (frame-0 static calibration),
camera/out/calib_variants.json (uncertainty), camera/out/drift_similarity_raw.csv (per-frame drift) and the measurement
pivot (measurement/analysis.json). Scale: unit W_in (crossbar inner span) = 1; metres_per_unit = None until the dynamics
fit provides it (pass --metres-per-unit X to set it; provenance must be stated with --scale-source)."""
import numpy as np, json, pathlib, sys, csv, hashlib, datetime
HERE = pathlib.Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import project as PJ
ROOT = PJ.ROOT; W, H = 1920, 1088
def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()
def arg(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default
mpu = arg("--metres-per-unit"); mpu = float(mpu) if mpu else None
CR = json.load(open(ROOT / "camera/out/calib_result.json")); CV = json.load(open(ROOT / "camera/out/calib_variants.json"))
DF = json.load(open(ROOT / "camera/out/drift_fit.json")); AN = json.load(open(ROOT / "measurement/analysis.json"))
R0 = np.array(CR["R_world_to_cam"]); C0 = np.array(CR["C"]); f0 = CR["f_px"]; c0 = np.array([CR["cx"], CR["cy"]])
# ---- drift.csv: per-frame camera of record (raw per-frame similarity from the ECC homographies)
D = list(csv.DictReader(open(ROOT / "camera/out/drift_similarity_raw.csv")))
hdr = ["frame", "t_s", "scale", "roll_rad", "tx_px", "ty_px", "H_vs_sim_rms_px", "landmarks_vs_sim_rms_px", "smooth_vs_raw_maxdev_px",
       "f_px", "cx_px", "cy_px", "blender_lens_mm", "blender_shift_x", "blender_shift_y", "camera_roll_deg"]
rows = []
for r in D:
    s, rho, tx, ty = (float(r[k]) for k in ("scale", "roll_rad", "tx_px", "ty_px"))
    c, sn = np.cos(rho), np.sin(rho); cn = s * np.array([[c, -sn], [sn, c]]) @ c0 + [tx, ty]; fn = s * f0
    rows.append([int(r["frame"]), float(r["t_s"]), s, rho, tx, ty, float(r["H_vs_sim_rms_px"]), float(r["landmarks_vs_sim_rms_px"]), float(r["smooth_vs_raw_maxdev_px"]),
                 fn, cn[0], cn[1], fn * 36 / W, -(cn[0] - W / 2) / W, (cn[1] - H / 2) / W, np.degrees(rho)])
with open(ROOT / "camera/drift.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(hdr)
    for r in rows: w.writerow([r[0]] + ["%.10g" % x for x in r[1:]])
PJ._DRIFT = None
# ---- pivots
Peff_px = np.array(AN["pivot_Peff_frame0_px"]); cam0 = {"R": R0, "C": C0, "f_px": f0, "cx": c0[0], "cy": c0[1]}
Peff_w = PJ.intersect_plane(Peff_px[0], Peff_px[1], cam0, [0, 1, 0], 0.0)
A_px = PJ.project(np.zeros(3), cam0)
# ---- diagnostic cameras
g = CR["geometry"]; zmid = 0.5 * (g["zpt"] + g["bz0"])
Rf = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0.0]])          # cam x = X, cam y = -Z, cam z = +Y
def look_at_cv(loc, tgt, up=(0, 0, 1.0)):
    z = np.asarray(tgt, float) - loc; z /= np.linalg.norm(z); x = np.cross(z, up); x /= np.linalg.norm(x); y = np.cross(z, x); return np.array([x, y, z])
ov_loc = np.array([2.6, -3.9, 0.55]); ov_tgt = np.array([0.0, 0.0, zmid]); Rov = look_at_cv(ov_loc, ov_tgt)
front = {"R": Rf, "C": np.array([0.0, -10.0, zmid]), "cx": W / 2, "cy": H / 2, "type": "ORTHO", "ortho_scale": 3.0}
over = {"R": Rov, "C": ov_loc, "cx": W / 2, "cy": H / 2, "type": "PERSP", "f_px": 2400.0}
def bl(cam):
    b = PJ.blender_camera_settings(cam); return b
fwd = R0[2]; ref_bl = bl({**cam0, "type": "PERSP"})
summ = CV["summary"]
out = {
 "schema": "sim02-camera/v1", "owner": "Measure+Camera (A)", "status": "draft v001: frame-0 static calibration + per-frame measured drift (zoom/pan)",
 "created_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
 "conventions": {"world": "config/coordinates.md: origin A = spring top attachment (hook bearing on crossbar top face, crossbar mid-depth); X along the crossbar toward image right; Y horizontal away from camera; Z up (gravity -Z assumed)",
                 "pixels": "CONTRACT: origin top-left corner of top-left pixel, pixel centres at +0.5, 1920x1088",
                 "R_world_to_cv": "OpenCV camera axes (x right, y down, z forward): Xc = R (X - C)",
                 "blender": "camera matrix_world rotation = R^T diag(1,-1,-1); sensor_fit HORIZONTAL, sensor 36 mm; shift_x = -(cx-960)/1920, shift_y = (cy-544)/1920",
                 "utility": "camera/project.py: load_camera(frame=n) (camera of record), load_camera(kind='locked'|'front_ortho'|'overview'), project, ray, intersect_plane, blender_camera_settings"},
 "scale": {"unit": "W_in = crossbar inner span (distance between the inner faces of the two posts) = 1",
           "metres_per_unit": mpu, "metres_per_unit_source": arg("--scale-source", "not set: metric scale is model-conditioned by the dynamics fit (period + g = 9.80665 m/s^2)"),
           "provisional_hint": "simple-pendulum length for the measured swing period 1.3007 s is 0.4199 m; the image pivot (spring coil top P_eff) to weight-centre distance is ~0.98 W_in, so W_in ~ 0.43 m (rough, depends on the compound/elastic model; NOT adopted)"},
 "reference_camera": {
   "description": "camera of record: static calibration on reference frame 0 (t = 0); per-frame camera = measured image-space drift realised as focal length / principal point / roll (camera/drift.csv)",
   "reference_frame": 0,
   "intrinsics": {"resolution": [W, H], "pixel_aspect": 1.0, "f_px": f0, "cx_px": c0[0], "cy_px": c0[1], "focal_mm_36mm_sensor": f0 * 36 / W,
                  "hfov_deg": float(np.degrees(2 * np.arctan(W / 2 / f0))), "vfov_deg": float(np.degrees(2 * np.arctan(H / 2 / f0))),
                  "principal_point_assumption": "image centre (free principal point is degenerate with pitch/f: runs to v=377 px with f=2605 at the same RMS)", "lens_distortion": "none (assumed)"},
   "extrinsics": {"R_world_to_cv": R0.tolist(), "C_world": C0.tolist(), "C_world_sd": CR["C_sd"], "distance_to_A": float(np.linalg.norm(C0)),
                  "view_pitch_deg": CR["view_pitch_deg"], "view_yaw_deg_from_+Y_toward_+X": CR["view_yaw_deg_from_Y"],
                  "roll_deg": float(np.degrees(np.arctan2(R0[0, 2], -R0[1, 2]))),
                  "camera_height_note": "camera centre is between the crossbar top (z=0) and bottom (z=%.3f): neither the top face nor the underside of the crossbar is visible; plate top visible" % g["zc0"]},
   "blender_frame0": {"location": ref_bl["location"], "rotation_matrix": ref_bl["rotation_matrix"], "lens_mm": ref_bl["lens_mm"], "sensor_width_mm": 36.0, "sensor_fit": "HORIZONTAL",
                      "shift_x": ref_bl["shift_x"], "shift_y": ref_bl["shift_y"], "per_frame": "lens_mm, shift_x/y, roll from camera/drift.csv (or project.load_camera(frame=n) + blender_camera_settings)"}},
 "drift": {"verdict": "REAL: smooth near-linear push-in, image scale +%.2f %% over 10 s (%.3f %%/s) about a fixed point near (%.0f, %.0f) px, pan up to %.1f px (tx) / %.1f px (ty), roll < %.3f deg" % (
               100 * (DF["scale_frame240"] - 1), DF["scale_pct_per_s"], *DF["zoom_fixed_point_median_px"], min(DF["tx_range_px"]), min(DF["ty_range_px"]), np.degrees(DF["roll_max_abs_rad"])),
           "model": "per-frame similarity (from the 8-dof static-mask homography) realised as zoom (focal length s*f), principal-point shift s R(rho) c + t and roll rho; camera centre and orientation fixed",
           "zoom_vs_dolly": "frame (posts) and side-table layers give the same scale within 0.3 % (measurement/drift_parallax.json, drift_layers_ecc.json); the textureless backdrop cannot discriminate. A dolly is not excluded; zoom adopted (clip-01 precedent)",
           "similarity_vs_homography_rms_median_px": DF["H_vs_sim_rms_median_px"], "similarity_vs_homography_max_px": DF["H_vs_sim_max_px"],
           "landmarks_vs_similarity_rms_median_px": DF["landmarks_vs_sim_rms_median_px"], "landmarks_vs_similarity_rms_max_px": DF["landmarks_vs_sim_rms_max_px"],
           "cubic_smooth_vs_raw_maxdev_px": DF["smooth_vs_raw_maxdev_px"], "per_frame_table": "camera/drift.csv (raw per-frame values are the camera of record)",
           "smooth_model_polys_vs_frame0": DF["polys"]},
 "locked_cameras": {
   "locked_reference": {"description": "frame-0 camera held for all frames (drift-free diagnostic)", "blender": ref_bl},
   "front_ortho": {"description": "orthographic, looking along +Y at the frame front (X right, Z up); diagnostic of in-plane motion", "cv": {"type": "ORTHO", "R_world_to_cv": Rf.tolist(), "C": front["C"].tolist(), "ortho_scale": 3.0},
                   "blender": bl(front)},
   "overview": {"description": "locked 3/4 overview from front-right, slightly above the crossbar; shows depth (Y) motion", "cv": {"type": "PERSP", "R_world_to_cv": Rov.tolist(), "C": ov_loc.tolist(), "f_px": 2400.0},
                "blender": bl(over)}},
 "key_points": {"A_world": [0, 0, 0], "A_image_frame0_px": [float(A_px[0]), float(A_px[1])], "A_note": "A is hidden behind the crossbar front face (camera below the top face); hook apex visible ~10 px above",
                "P_eff_image_frame0_px": Peff_px.tolist(), "P_eff_world_on_plane_Y0": Peff_w.tolist(),
                "P_eff_note": "least-squares intersection of the 241 spring-axis lines (frame-0 geometry): the spring body swings about its top coil (lower end of the S-hook), not about A; the S-hook stays fixed to <=1.2 px (measurement/analysis.json). Assumes the spring hangs in the plane Y = 0 through A"},
 "frame_geometry_implied_by_calibration": {"units": "W_in", "values": CR["geometry"], "sd": CR["geometry_sd"],
     "boxes": "posts x[xc-0.5-wp, xc-0.5] & [xc+0.5, xc+0.5+wp], y[yp0, yp0+dp], z[bz1, zpt]; crossbar x[xc-0.5, xc+0.5], y[-dc/2, dc/2], z[zc0, 0]; plate x[bx0,bx1], y[by0,by1], z[bz0,bz1]",
     "priors": "posts centred on crossbar in Y (sd 0.02), square post section (sd 15 %), plate centred on frame in X (sd 0.02), crossbar depth 0.08 +- 0.04 (weakly observed)"},
 "fit_quality": {"static_edge_points_rms_px_frame0": CR["static_rms_px"], "n_points": CR["n_points"], "per_edge": CR["per_edge"], "R4_target_px": 2.0, "R4_pass_frame0": CR["static_rms_px"] <= 2.0,
                 "per_frame_reprojection": "camera/out/reprojection_by_frame.json"},
 "uncertainty": {"f_px_sd_formal": CR["f_sd_px"], "f_px_variants_sd": summ["f_sd"], "f_px_variants_range": [summ["f_min"], summ["f_max"]],
                 "distance_to_A_range_Win": [summ["dist_min"], summ["dist_max"]], "pitch_range_deg": summ["pitch_range"],
                 "f_profile": {k: {"rms": v["static_rms"], "cost": v["cost"], "dist": v["dist_to_A"]} for k, v in CV["f_profile"].items()},
                 "note": "f and distance trade off along the line of sight (ratio fixed); realistic f range ~3000-4200 px (shallow cost profile); free principal point is degenerate"},
 "provenance": {"reference_sha256": "fee726c2c30d36e5cf403a7152ff8a02d100682ee4333e1a495aa4746f6865cd",
                "inputs": {p: sha(ROOT / p) for p in ["camera/out/edges_f000.json", "camera/out/calib_result.json", "camera/out/calib_variants.json", "camera/out/drift_similarity_raw.csv", "measurement/analysis.json"]},
                "code": {p: sha(ROOT / p) for p in ["camera/calibrate.py", "camera/edges.py", "camera/calib_variants.py", "camera/drift_fit.py", "camera/project.py", "camera/write_params.py", "measurement/scripts/s1_drift.py"]},
                "classes": {"f, R, C (W_in units)": "measured (image-fitted; assumptions: pinhole, centred principal point, rectilinear axis-aligned frame boxes)",
                            "drift": "measured", "metres": "model-conditioned (not yet set)", "P_eff in world": "measured image point + assumption (plane Y=0)"}}}
def clean(o):
    if isinstance(o, dict): return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, np.ndarray)): return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)): return float(o)
    if isinstance(o, np.bool_): return bool(o)
    return o
json.dump(clean(out), open(ROOT / "params/camera.json", "w"), indent=1)
print("f %.1f  C %s  pitch %.2f  A_px %s  Peff_w %s" % (f0, np.round(C0, 4), CR["view_pitch_deg"], np.round(A_px, 2), np.round(Peff_w, 4)))
