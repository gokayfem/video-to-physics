"""Write params/camera.json and camera/drift.csv (clip 03) from camera/out/calib_result.json (frame-0 VP calibration),
camera/out/calib_variants.json, camera/out/drift_similarity_raw.csv (per-frame drift), camera/out/locate.json (camera
position, gauge D = disc diameter = 1). metres_per_unit stays None (model-conditioned; prompt says ~0.10 m)."""
import numpy as np, json, pathlib, sys, csv, hashlib, datetime
HERE = pathlib.Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import project as PJ
ROOT = PJ.ROOT; W, H = 1920, 1088
def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()
CR = json.load(open(ROOT / "camera/out/calib_result.json")); CV = json.load(open(ROOT / "camera/out/calib_variants.json"))
LO = json.load(open(ROOT / "camera/out/locate.json"))
CP = json.load(open(ROOT / "camera/out/camera_poses.json")); CF = json.load(open(ROOT / "camera/out/calib_per_frame_fixf.json"))
CFREE = json.load(open(ROOT / "camera/out/calib_per_frame.json"))
f0 = CP["f_px"]; c0 = np.array([CP["cx"], CP["cy"]]); R0 = np.array(CP["frames"][0]["R"]); C0 = np.array(CP["frames"][0]["C"])
def rotvec(R):
    c = np.clip((np.trace(R) - 1) / 2, -1, 1); th = np.arccos(c)
    if th < 1e-12: return np.zeros(3)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * np.sin(th)); return v * th
hdr = ["frame", "t_s", "Cx", "Cy", "Cz", "rotvec_x", "rotvec_y", "rotvec_z", "f_px", "cx_px", "cy_px", "azimuth_deg", "elevation_deg", "dist_to_A",
       "view_pitch_deg", "view_yaw_deg", "roll_deg", "vp_edge_rms_px", "bundle_edge_rms_px", "A_resid_u_px", "A_resid_v_px", "f_free_vp_px",
       "blender_loc_x", "blender_loc_y", "blender_loc_z", "blender_lens_mm", "blender_shift_x", "blender_shift_y"]
rows = []
for n, (fr, cf, cfr) in enumerate(zip(CP["frames"], CF, CFREE)):
    R = np.array(fr["R"]); C = np.array(fr["C"]); rv = rotvec(R); az = np.degrees(np.arctan2(C[0], -C[1])); el = np.degrees(np.arctan2(C[2], np.hypot(C[0], C[1])))
    bl_ = PJ.blender_camera_settings({"R": R, "C": C, "f_px": f0, "cx": c0[0], "cy": c0[1], "type": "PERSP"})
    rows.append([n, n / 24, *C, *rv, f0, c0[0], c0[1], az, el, np.linalg.norm(C), cf["pitch"], cf["yaw"], cf["roll"], cf["edge_rms"], fr["edge_endpoint_rms_px"], *fr["A_resid_px"], cfr["f_px"],
                 *bl_["location"], bl_["lens_mm"], bl_["shift_x"], bl_["shift_y"]])
with open(ROOT / "camera/drift.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(hdr)
    for r in rows: w.writerow([r[0]] + ["%.10g" % x for x in r[1:]])
PJ._POSES = None
RW = np.array(rows); AZ = RW[:, 11]; EL = RW[:, 12]; DI = RW[:, 13]
DF = dict(az=[float(AZ[0]), float(AZ[-1])], el=[float(EL.min()), float(EL.max())], dist=[float(DI[0]), float(DI[-1]), float(DI.min()), float(DI.max())],
          roll=[float(RW[:, 16].min()), float(RW[:, 16].max())], pitch=[float(RW[:, 14].min()), float(RW[:, 14].max())])
cam0 = {"R": R0, "C": C0, "f_px": f0, "cx": c0[0], "cy": c0[1], "type": "PERSP"}
AN = json.load(open(ROOT / "measurement/analysis.json")); Dc = np.array(AN["disc_centre"]["mean_D"])   # mean disc rotation centre (world, D)
# ---- locked diagnostic cameras
Rtop = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1.0]])        # looking down -Z, image right = +X, image up = +Y
top = {"R": Rtop, "C": Dc + np.r_[0, 0, 6.0], "cx": W / 2, "cy": H / 2, "type": "ORTHO", "ortho_scale": 1.6}
def look_at_cv(loc, tgt, up=(0, 0, 1.0)):
    z = np.asarray(tgt, float) - loc; z /= np.linalg.norm(z); x = np.cross(z, up); x /= np.linalg.norm(x); y = np.cross(z, x); return np.array([x, y, z])
tgt = 0.5 * Dc; ov_loc = tgt + np.array([-4.0, -5.5, 2.6])                # from front-left, above the crossbar (opposite side to the reference camera)
over = {"R": look_at_cv(ov_loc, tgt), "C": ov_loc, "cx": W / 2, "cy": H / 2, "type": "PERSP", "f_px": 2000.0}
VERDICT = ("REAL and LARGE (not a locked camera): the camera ORBITS about a vertical axis through the stand, azimuth %+.1f -> %+.1f deg (%.1f deg in 10 s), "
           "distance to A %.3f -> %.3f D, elevation %.1f..%.1f deg, roll %.2f..%.2f deg; a single static-mask homography cannot represent it (parallax: H-vs-similarity residual 34 px median)"
           % (DF["az"][0], DF["az"][1], DF["az"][1] - DF["az"][0], DF["dist"][0], DF["dist"][1], *DF["el"], *DF["roll"]))
bl = PJ.blender_camera_settings; ref_bl = bl(cam0); summ = CV["summary"]
out = {
 "schema": "sim03-camera/v1", "owner": "Measure+Camera (A)", "status": "draft v001: frame-0 vanishing-point calibration + disc-gauge placement + per-frame measured drift",
 "created_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
 "conventions": {"world": "config/coordinates.md: origin A = where the torsion wire leaves the black clamp; X along the crossbar toward image right; Y horizontal away from the camera; Z up (gravity -Z assumed)",
                 "pixels": "CONTRACT: origin top-left corner of top-left pixel, pixel centres at +0.5, 1920x1088",
                 "R_world_to_cv": "OpenCV camera axes (x right, y down, z forward): Xc = R (X - C)",
                 "blender": "camera matrix_world rotation = R^T diag(1,-1,-1); sensor_fit HORIZONTAL, sensor 36 mm; shift_x = -(cx-960)/1920, shift_y = (cy-544)/1920",
                 "utility": "camera/project.py: load_camera(frame=n) (camera of record), load_camera(kind='locked'|'top_ortho'|'overview'), project, ray, intersect_plane, blender_camera_settings"},
 "scale": {"unit": "D = disc diameter = 1 (declared gauge; the disc fit uses top-face radius 0.5)", "metres_per_unit": None,
           "metres_per_unit_source": "not set: model-conditioned (prompt intent ~0.10 m disc; the torsion period does not fix length without an assumed wire stiffness)"},
 "reference_camera": {
   "description": "camera of record: per-frame measured pose (orbit) with constant intrinsics; frame-0 values listed here, all frames in camera/drift.csv / project.load_camera(frame=n)",
   "reference_frame": 0,
   "intrinsics": {"resolution": [W, H], "pixel_aspect": 1.0, "f_px": f0, "cx_px": c0[0], "cy_px": c0[1], "focal_mm_36mm_sensor": f0 * 36 / W,
                  "hfov_deg": float(np.degrees(2 * np.arctan(W / 2 / f0))), "principal_point_assumption": "image centre (frame-0 free-principal-point VP variant: |f| %.0f px, pp (%.0f, %.0f): poorly determined)" % (abs(CV["free_pp"]["f_px"]), CV["free_pp"]["cx"], CV["free_pp"]["cy"]), "lens_distortion": "none (assumed)"},
   "extrinsics": {"R_world_to_cv": R0.tolist(), "C_world": C0.tolist(), "distance_to_A": float(np.linalg.norm(C0)), 
                  "view_pitch_deg": CF[0]["pitch"], "view_yaw_deg_from_+Y_toward_+X": CF[0]["yaw"], "roll_deg": CF[0]["roll"]},
   "blender_frame0": {**ref_bl, "per_frame": "lens_mm, shift_x/y, roll from camera/drift.csv (or project.load_camera(frame=n) + blender_camera_settings)"}},
 "drift": {"verdict": VERDICT, "model": "per-frame pose: R_n from the vanishing points of the frame edges (constant f), C_n from a bundle adjustment of 7 identity-stable frame edges + the wire-top pixel (image of A); gauge |C_0 - A| from the disc diameter",
           "azimuth_deg_frame0_240": DF["az"], "elevation_deg_range": DF["el"], "distance_to_A_frame0_240_min_max": DF["dist"], "roll_deg_range": DF["roll"],
           "zoom": "not separately identifiable: f is held constant (%.0f px). Free per-frame VP focal length is stable 2630-2690 px over frames 0-140 and diverges later (yaw -> 0 makes the X vanishing point degenerate); the bundle prefers 2547-2602 px. Systematic f range ~2550-2690 px (+-3 %%)" % f0,
           "homography_diagnostic": "measurement/cache/H_ecc.npy, H_chain.npy; similarity fit camera/out/drift_fit.json (H vs similarity 34 px median: not a zoom/pan)",
           "per_frame_table": "camera/drift.csv (C, rotvec, f, pp, Blender location/lens/shift per frame; rotation matrix via project.load_camera)",
           "bundle": CP["summary"]},
 "locked_cameras": {
   "locked_reference": {"description": "frame-0 camera held for all frames (drift-free diagnostic)", "blender": ref_bl},
   "top_ortho": {"description": "orthographic, straight down onto the mean disc centre (image right = +X, image up = +Y); ortho_scale 1.6 D", "cv": {"type": "ORTHO", "R_world_to_cv": Rtop.tolist(), "C": top["C"].tolist(), "ortho_scale": 1.6}, "blender": bl(top)},
   "overview": {"description": "locked 3/4 overview from front-left above the crossbar (opposite side to the reference camera), f = 2000 px", "cv": {"type": "PERSP", "R_world_to_cv": over["R"].tolist(), "C": ov_loc.tolist(), "f_px": 2000.0}, "blender": bl(over)}},
 "key_points": {"A_world": [0, 0, 0], "A_image_frame0_px": [float(x) for x in PJ.project(np.zeros(3), cam0)], "mean_disc_rotation_centre_world": Dc.tolist(),
                "wire_length_A_to_disc_top_face_mean_D": AN["disc_centre"]["wire_length_A_to_top_face_mean_D"], "A_horizontal_miss_at_frame0_D": LO["A_horizontal_miss"],
                "A_note": "A on the frame-0 ray through the wire-top (clamp exit) pixel, at the depth where it lies vertically above the frame-0 disc rotation centre [asm]; the bundle then keeps A on the wire-top ray in every frame (residual in camera/drift.csv)"},
 "frame_geometry_inferred": {"units": "D", "frame0_planes": LO["frame_geometry_inferred_frame0"], "bundle_lines": json.load(open(ROOT / "camera/out/bundle_lines.json")),
     "assumptions": "frame0_planes: crossbar/post front faces flush, crossbar square section (initial); bundle_lines: triangulated through the orbit (point + direction per edge). The AI frame is not perfectly rigid: crossbar edge depth differs left/right of the clamp by up to 0.05 D"},
 "fit_quality": {"vp_edge_rms_px_frame0": CF[0]["edge_rms"], "bundle_edge_endpoint_rms_px_all": CP["summary"]["all_rms_px"], "R4_target_px": 2.0, "per_frame_reprojection": "camera/out/reprojection_by_frame.json"},
 "uncertainty": {"f_px_sd_formal": CR["f_sd_px"], "f_px_jackknife_sd": summ["f_sd"], "f_px_jackknife_range": [summ["f_min"], summ["f_max"]],
                 "pitch_range_deg": summ["pitch_range"], "yaw_range_deg": summ["yaw_range"], "free_pp_variant": CV["free_pp"]},
 "provenance": {"reference_sha256": "c640e5ce848f7a86c0c6d6a2156aa4d89c022c51ba164b5b119bef16b0a2045c",
                "inputs": {p: sha(ROOT / p) for p in ["camera/out/edges_f000.json", "camera/out/calib_per_frame_fixf.json", "camera/out/camera_poses.json", "camera/out/locate.json", "measurement/cache/edges_track.json", "measurement/wire_track.csv"]},
                "code": {p: sha(ROOT / p) for p in ["camera/calibrate.py", "camera/edges.py", "camera/calib_variants.py", "camera/calib_per_frame.py", "camera/pose_bundle.py", "camera/locate.py", "camera/project.py", "camera/write_params.py", "measurement/scripts/s1d_edges_track.py", "measurement/scripts/s1_drift.py"]},
                "classes": {"R_n": "measured (vanishing points; assumptions: pinhole, centred principal point, orthogonal rectilinear frame, constant f)", "C_n": "measured (bundle) + gauge (disc diameter = 1 at frame 0, A vertically above the frame-0 disc centre)",
                            "orbit": "measured", "metres": "model-conditioned (not set)"}}}
def clean(o):
    if isinstance(o, dict): return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, np.ndarray)): return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)): return float(o)
    if isinstance(o, np.bool_): return bool(o)
    return o
json.dump(clean(out), open(ROOT / "params/camera.json", "w"), indent=1)
print("f %.1f  C %s  A_px %s" % (f0, np.round(C0, 4), np.round(PJ.project(np.zeros(3), cam0), 2)))
