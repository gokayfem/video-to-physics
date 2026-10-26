"""Write params/camera.json and camera/drift.csv from the calibration (camera role).

Usage: .venv/bin/python camera/write_params.py [--free-phi]
  default: adopt swing-plane azimuth phi = 0 (swing plane = vertical plane through the arm axis);
           the free-phi estimate and its spread are reported as uncertainty.
Scale: calibration unit = L_eff (radius of the bob-centre arc about its rotation centre). Metres use
       L_c_m from params/physics.json (model-conditioned; re-run this script when it changes).
"""
import numpy as np, json, pathlib, sys, csv, hashlib, datetime
from scipy.spatial.transform import Rotation as Rot
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
FREE = "--free-phi" in sys.argv
sys.argv = [sys.argv[0]] + ([] if FREE else ["--phi0"])
import calibrate as CAL
import project as PJ
ROOT = CAL.ROOT
W, H = 1920, 1088

def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()

obs = CAL.load_obs()
CAL.VERB = False
sol = CAL.solve(obs, verbose=0)
f, cx, cy, rv, C_L, g, phi, e, Rb, th = CAL.unpack(sol.x)
R_stand = Rot.from_rotvec(rv).as_matrix()
_, det = CAL.residuals(sol.x, obs, detail=True)
static = np.concatenate([det[k] for k in CAL.USE_EDGES])
per_edge = {k: round(float(np.sqrt(np.mean(det[k]**2))), 3) for k in CAL.USE_EDGES}
# world frame: X along swing direction (rotate stand frame by phi about Z)
Rz = Rot.from_euler("z", phi).as_matrix()          # columns: world axes expressed in stand frame
R_w = R_stand @ Rz                                   # world -> cv camera
C_wL = Rz.T @ C_L
phys = json.load(open(ROOT / "params/physics.json"))
if "L_eff_m" in phys:
    Lc = float(phys["L_eff_m"]["value"]); LSRC = "params/physics.json L_eff_m.value (metres per camera.json length unit; gauge decision config/CHANGELOG.md 2026-10-25T22:47:40Z)"
else:   # legacy fallback (before the gauge decision): L_c used as the gauge
    Lc = float(phys["L_c"]["value"]) if "L_c" in phys else float(phys["parameters"]["L_c_m"]["value"]); LSRC = "params/physics.json L_c (LEGACY gauge, pre-L_eff_m)"
C_wm = C_wL * Lc

cam_ref = {"R": R_w, "C": C_wm, "f_px": f, "cx": cx, "cy": cy}
bl = PJ.blender_camera_settings(cam_ref)
fwd = R_w.T @ [0, 0, 1.0]
var_path = ROOT / "camera/out/calib_variants.json"
unc = {}
if var_path.exists():
    V = json.load(open(var_path))
    rows = {k: v for k, v in V.items() if not k.startswith("phi0")}
    for q in ("f", "dist", "Cx", "Cy", "Cz", "pitch_deg", "yaw_deg", "roll_deg", "phi_deg", "e", "Rb", "static_rms"):
        a = np.array([v[q] for v in rows.values()])
        unc[q] = {"sd": round(float(a.std()), 5), "min": round(float(a.min()), 5), "max": round(float(a.max()), 5)}
# ---------------- drift ----------------
D = np.loadtxt(ROOT / "camera/out/drift_similarity_raw.csv", delimiter=",", skiprows=1)
t = D[:, 1]
polys = {k: np.polyfit(t, D[:, i], 3) for k, i in (("scale_poly", 2), ("roll_poly", 3), ("tx_poly", 4), ("ty_poly", 5))}
drift_model = {k: v.tolist() for k, v in polys.items()}
drift_model["note"] = ("image-space similarity x_n = s R(rho) x_120 + t (pixels, contract convention) relative to reference frame 120, "
                       "cubic polynomials in t_s fitted to camera/out/drift_similarity_raw.csv; realised in the camera as per-frame "
                       "focal length s*f, principal point s R(rho) c + t and roll rho about the optical axis (extrinsic position fixed)")
P_tmp = {"reference_camera": {"intrinsics": {"f_px": f, "cx_px": cx, "cy_px": cy},
                              "extrinsics": {"R_world_to_cv": R_w.tolist(), "C_world_m": C_wm.tolist()}},
         "drift": {"model": drift_model}}
tmp = ROOT / "camera/out/_camera_tmp.json"; json.dump(P_tmp, open(tmp, "w"))
rows = []
static_feat_rms = D[:, 6]
for n in range(len(D)):
    cm = PJ.load_camera(frame=n, params_path=tmp)
    b = PJ.blender_camera_settings(cm)
    s_sm = np.polyval(polys["scale_poly"], t[n]); r_sm = np.polyval(polys["roll_poly"], t[n])
    tx_sm = np.polyval(polys["tx_poly"], t[n]); ty_sm = np.polyval(polys["ty_poly"], t[n])
    # smoothing residual: displacement difference raw vs smooth at the image corners/centre (px)
    pts = np.array([[0, 0], [W, 0], [0, H], [W, H], [960, 544], [1090, 127], [1090, 835]], float)
    def apply(s, r, tx, ty): c, sn = np.cos(r), np.sin(r); return s*(pts @ np.array([[c, sn], [-sn, c]])) + [tx, ty]
    dev = np.abs(apply(*D[n, 2:6]) - apply(s_sm, r_sm, tx_sm, ty_sm)).max()
    rows.append([n, t[n], *D[n, 2:6], s_sm, r_sm, tx_sm, ty_sm, dev, static_feat_rms[n], cm["f_px"], cm["cx"], cm["cy"],
                 b["lens_mm"], b["shift_x"], b["shift_y"], float(np.degrees(r_sm))])
hdr = ["frame", "t_s", "scale_raw", "roll_raw_rad", "tx_raw_px", "ty_raw_px", "scale", "roll_rad", "tx_px", "ty_px",
       "smooth_vs_raw_maxdev_px", "static_feature_fit_rms_px", "f_px", "cx_px", "cy_px", "blender_lens_mm",
       "blender_shift_x", "blender_shift_y", "camera_roll_deg"]
with open(ROOT / "camera/drift.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(hdr)
    for r in rows: w.writerow([int(r[0])] + [f"{x:.8g}" for x in r[1:]])
tmp.unlink()
R_ = np.array(rows)
# ---------------- diagnostic cameras (world metres) ----------------
def look_at(loc, target, up=(0, 0, 1.0)):
    loc = np.asarray(loc, float); fz = np.asarray(target, float) - loc; fz /= np.linalg.norm(fz)
    x = np.cross(fz, up); x /= np.linalg.norm(x); y = np.cross(x, fz)
    Rbl = np.c_[x, y, -fz]           # Blender camera: local -Z forward, +Y up
    return Rbl
zc = (g["bz0"] + g["pz1"]) / 2
side_loc = np.array([0.0, -8.0, zc]) * Lc
side_R = look_at(side_loc, np.array([0.0, 0.0, zc]) * Lc)
ov_loc = np.array([3.2, -5.2, 1.1]) * Lc; ov_tgt = np.array([-0.15, 0.0, -0.65]) * Lc
ov_R = look_at(ov_loc, ov_tgt)
def eul(Rbl): return [float(a) for a in Rot.from_matrix(Rbl).as_euler("xyz")]
# geometry implied by the calibration (stand frame == world when phi adopted 0)
geo_L = {k: round(float(v), 5) for k, v in g.items()}
geo_m = {k: round(float(v) * Lc, 5) for k, v in g.items()}
theta_deg = np.degrees(th)
out = {
  "schema": "sim01-camera/v1", "owner": "camera/appearance",
  "status": "calibrated (frame-120 static calibration + per-frame drift); scale model-conditioned via params/physics.json L_c_m",
  "created_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
  "conventions": {"world": "CONTRACT: metres, Z up, origin = pivot P_ring (wire departure at bottom of the pin loop), X horizontal in the adopted swing plane toward image right, Y away from camera",
                  "pixels": "CONTRACT: origin top-left corner, pixel centres at +0.5, 1920x1088",
                  "R_world_to_cv": "OpenCV camera axes (x right, y down, z forward): Xc = R (X - C)",
                  "blender": "camera object matrix_world rotation = R_world_to_cv^T diag(1,-1,-1); sensor_fit HORIZONTAL; shift_x = -(cx-960)/1920, shift_y = (cy-544)/1920",
                  "projection_utility": "camera/project.py (load_camera(frame=n), project, ray, intersect_plane)"},
  "scale": {"unit": "L_eff = radius of the bob-centre arc about its rotation centre P_eff (calibration gauge = 1)",
            "L_eff_m": Lc, "L_eff_m_source": "%s (sha256 %s), provenance model-conditioned" % (LSRC, sha(ROOT / "params/physics.json")),
            "note": "all metre values below = (value in L_eff) x L_eff_m; re-run camera/write_params.py when physics L_eff_m changes. L_c (|P_ring - bob centre|) is a different length (physics.json L_c_in_L_eff_units)"},
  "reference_camera": {
    "description": "reference-matched camera; static calibration on reference frame 120 (t=5.0 s); per-frame drift in 'drift' and camera/drift.csv",
    "reference_frame": 120,
    "intrinsics": {"resolution": [W, H], "pixel_aspect": 1.0, "f_px": round(f, 3), "cx_px": cx, "cy_px": cy,
                   "principal_point_assumption": "image centre (free principal point is degenerate: runs to u=2057 px)",
                   "sensor_width_mm": 36.0, "sensor_fit": "HORIZONTAL", "focal_mm": round(f*36/W, 4),
                   "hfov_deg": round(float(np.degrees(2*np.arctan(W/2/f))), 4), "vfov_deg": round(float(np.degrees(2*np.arctan(H/2/f))), 4),
                   "lens_distortion": "none (assumed)"},
    "extrinsics": {"R_world_to_cv": R_w.round(9).tolist(), "C_world_m": C_wm.round(6).tolist(), "C_world_Leff": C_wL.round(5).tolist(),
                   "t_world_to_cv_m": (-R_w @ C_wm).round(6).tolist(),
                   "distance_to_pivot_Leff": round(float(np.linalg.norm(C_wL)), 4), "distance_to_pivot_m": round(float(np.linalg.norm(C_wm)), 5),
                   "view_pitch_deg": round(float(np.degrees(np.arcsin(fwd[2]))), 4),
                   "view_azimuth_deg_from_+Y_toward_+X": round(float(np.degrees(np.arctan2(fwd[0], fwd[1]))), 4),
                   "roll_deg_world_up_vs_image_up_positive_toward_image_right": round(float(np.degrees(np.arctan2(R_w[0, 2], -R_w[1, 2]))), 4)},
    "blender": {"location_m": bl["location"], "rotation_euler_xyz_rad": eul(np.array(bl["rotation_matrix"])),
                "rotation_matrix": np.array(bl["rotation_matrix"]).round(9).tolist(), "lens_mm": bl["lens_mm"],
                "sensor_width_mm": 36.0, "sensor_fit": "HORIZONTAL", "shift_x": bl["shift_x"], "shift_y": bl["shift_y"],
                "per_frame": "lens_mm, shift_x/y and roll from camera/drift.csv (use project.load_camera(frame=n) + blender_camera_settings)"}},
  "drift": {"verdict": "REAL: smooth monotonic push-in (image scale +3.30 % from frame 0 to 240, ~0.33 %/s) with a small pan; no measurable parallax between post/arm and base (similarity rms 0.3-0.8 px, affine/homography no better) -> represented as a zoom (focal-length change) + principal-point shift, camera centre fixed",
            "evidence": ["camera/out/drift_similarity.png", "camera/out/drift_similarity_raw.csv", "camera/drift.csv", "measurement/summary.json camera_drift (independent, agrees: scale 1.0327 at frame 240 vs frame 0)"],
            "scale_frame0": float(D[0, 2]), "scale_frame240": float(D[-1, 2]), "max_abs_roll_rad": float(np.abs(D[:, 3]).max()),
            "smooth_vs_raw_maxdev_px": float(R_[:, 10].max()), "static_feature_similarity_rms_px_median": float(np.median(D[:, 6])),
            "model": drift_model,
            "alternative_dolly": "a camera dolly of equal image scale would produce depth-dependent parallax of order 1 px between post and base front at the clip ends; not observed above 0.3 px, so the zoom representation is preferred (not proven)"},
  "locked_cameras": {
    "locked_reference": {"description": "drift-free diagnostic: frame-120 reference calibration held for all frames (project.load_camera(kind='locked'))",
                         "blender": {"location_m": bl["location"], "rotation_euler_xyz_rad": eul(np.array(bl["rotation_matrix"])), "lens_mm": bl["lens_mm"], "shift_x": bl["shift_x"], "shift_y": bl["shift_y"], "sensor_width_mm": 36.0, "sensor_fit": "HORIZONTAL"}},
    "side_ortho": {"description": "orthographic view straight along +Y onto the swing plane (true theta visible); for diagnostics",
                   "blender": {"type": "ORTHO", "location_m": side_loc.round(6).tolist(), "rotation_euler_xyz_rad": eul(side_R),
                               "ortho_scale_m": round(3.0*Lc, 5), "shift_x": 0.0, "shift_y": 0.0, "sensor_fit": "HORIZONTAL", "clip_end_m": round(20*Lc, 4)}},
    "overview": {"description": "fixed 3/4 overview, longer lens, slightly above the arm, shows swing plane obliquity and the whole stand",
                 "blender": {"type": "PERSP", "location_m": ov_loc.round(6).tolist(), "rotation_euler_xyz_rad": eul(ov_R), "lens_mm": 60.0,
                             "sensor_width_mm": 36.0, "sensor_fit": "HORIZONTAL", "shift_x": 0.0, "shift_y": 0.0}}},
  "swing_plane": {"adopted_phi_deg": 0.0 if not FREE else float(np.degrees(phi)),
                  "definition": "phi = azimuth of the swing direction from the arm axis (+X_arm) toward +Y; world X = swing direction",
                  "free_fit_phi_deg": unc.get("phi_deg"),
                  "angle_between_swing_plane_and_image_plane_deg": round(float(np.degrees(np.arccos(abs(R_w[2] @ [0, 1, 0])))), 3),
                  "note": "bob moves toward the camera on the image-right side (apparent radius +5 % right vs left turning point)",
                  "theta_range_deg_from_this_fit": [round(float(theta_deg.min()), 3), round(float(theta_deg.max()), 3)]},
  "pivot": {"P_ring_image_f120_px": CAL.P_RING.tolist(), "P_ring_world_m": [0, 0, 0],
            "P_eff_world_Leff": [0, 0, round(-float(e), 5)], "P_eff_world_m": [0, 0, round(-float(e)*Lc, 6)],
            "P_eff_note": "the straight lower wire lines (and the bob arc) rotate about a point ~0.08 L_eff below the pin loop; the wire bends near the loop in the AI clip. The rotation radius about P_eff is L_eff; |B - P_ring| at theta=0 is 1+e L_eff. See camera/NOTES.md (request to lead/dynamics)."},
  "bob_radius_from_image_Leff": round(float(Rb), 5), "bob_radius_from_image_m": round(float(Rb)*Lc, 6),
  "stand_geometry_implied_by_calibration": {"frame": "world (stand frame, phi=0)", "units_Leff": geo_L, "metres": geo_m,
      "boxes": "base x[bx0,bx1] y[by0,by1] z[bz0,bz1]; post x[px0,px1] y[py0,py1] z[bz1,pz1]; arm x[..px1..ax1] y[-aw/2,aw/2] z[az0,az1]; rb/rp/ra = fillet radii (silhouette model)",
      "priors": "post centred on arm in Y (sd 0.01), post and arm cross-sections square (sd 15 %), fillet radius <= 0.04"},
  "fit_quality": {"static_edge_points_rms_px": round(float(np.sqrt(np.mean(static**2))), 4), "n_static_edge_points": int(len(static)),
                  "per_edge_rms_px": per_edge, "P_ring_residual_px": np.round(det["P_ring"], 3).tolist(),
                  "P_eff_residual_px": np.round(det.get("P_eff", [np.nan, np.nan]), 3).tolist(),
                  "bob_centre_rms_px": round(float(np.sqrt(np.mean(det["bob_uv"]**2))), 4), "bob_radius_rms_px": round(float(np.sqrt(np.mean(det["bob_r"]**2))), 4),
                  "R4_target_px": 2.0, "R4_pass": bool(np.sqrt(np.mean(static**2)) <= 2.0)},
  "uncertainty": {"method": "spread over calibration variants (jackknife over edges, re-picked edges +-3 px, pivot weights, bob halves, free phi) camera/out/calib_variants.json; units: f px, C/dist in L_eff, angles deg",
                  "variants": unc,
                  "focal_distance_coupling": "f and distance trade off along the line of sight; their ratio (image scale of the stand) is fixed to ~0.1 %"},
  "provenance": {"reference_sha256": "ced22d105d7bf9e93d7c152fab02b62e077cbe54162056114201c50b61f56bae",
                 "inputs": {p: sha(ROOT / p) for p in ["camera/out/edges_f120.json", "camera/out/bob_meas_f120.csv", "camera/out/pivot_f120.json",
                                                         "camera/out/drift_similarity_raw.csv", "measurement/bob_track.csv"] if (ROOT / p).exists()},
                 "code": {p: sha(ROOT / p) for p in ["camera/calibrate.py", "camera/edges.py", "camera/write_params.py", "camera/project.py",
                                                     "camera/drift_fit.py", "camera/track_static.py", "camera/wire_pivot.py", "camera/bob_to_f120.py"]},
                 "provenance_classes": {"f_px, R, C (in L_eff)": "measured (image-fitted, assumptions: pinhole, centred principal point, rectilinear stand)",
                                        "C in metres": "model-conditioned (L_eff_m from the period fit)", "drift": "measured", "phi=0": "assumed (within uncertainty of the free fit)"}}}
json.dump(out, open(ROOT / "params/camera.json", "w"), indent=1)
print("f %.2f px  focal %.3f mm  vfov %.3f deg  C_L %s  dist %.4f L  pitch %.3f  azim %.3f  phi_adopted %s" % (
    f, f*36/W, out["reference_camera"]["intrinsics"]["vfov_deg"], np.round(C_wL, 4), np.linalg.norm(C_wL),
    out["reference_camera"]["extrinsics"]["view_pitch_deg"], out["reference_camera"]["extrinsics"]["view_azimuth_deg_from_+Y_toward_+X"],
    out["swing_plane"]["adopted_phi_deg"]))
print("static rms %.3f px; bob rms %.3f; swing-plane vs image-plane angle %.2f deg; theta range %s" % (
    np.sqrt(np.mean(static**2)), out["fit_quality"]["bob_centre_rms_px"], out["swing_plane"]["angle_between_swing_plane_and_image_plane_deg"],
    out["swing_plane"]["theta_range_deg_from_this_fit"]))
json.dump({"theta_deg_frame": theta_deg.round(5).tolist(), "e_Leff": float(e), "Rb_Leff": float(Rb), "x": sol.x.tolist()},
          open(ROOT / "camera/out/calib_adopted.json", "w"))
