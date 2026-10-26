"""Write params/geometry.json from scene/measure/fit_frame0.json + landmarks_f000.json (scene role).

.venv/bin/python scene/measure/make_geometry_params.py
All lengths are ratios to S = clear span between the posts' inner faces. Metres = ratio * S_m, where S_m comes from
physics.json / camera of record when available (see scene/NOTES.md), else `S_m_assumed` below (ASSUMED).
"""
import json, pathlib, hashlib
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
F = json.loads((HERE / "fit_fixedcam.json").read_text())        # v002: camera of record FIXED
SL = json.loads((HERE / "spring_layout.json").read_text())
CJ = json.loads((ROOT / "params" / "camera.json").read_text())
LM = json.loads((HERE / "landmarks_f000.json").read_text())
G, GS = F["geometry_S"], F["geometry_sd_S"]
P = F["spring_plane_points"]
SZ = LM["sizes_px"]
k = F["px_per_Win"]["f120_v450"]                # px per W_in at the spring plane, frame 120 (sizes read at f120/f240)
kw = F["px_per_Win"]["f120_v860"]


def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def e(v, sd, method, status="measured", **kw_):
    d = {"value": round(float(v), 6), "sd": round(float(sd), 6), "method": method, "status": status}
    d.update(kw_)
    return d


def mean_px(d):
    vals = [v for kk, v in d.items() if kk.startswith("f")]
    return float(np.mean(vals)), float(d.get("sigma", 2.0))


fit = "box-model fit with the camera of record FIXED (params/camera.json load_camera(frame=0); scene/measure/fit_fixedcam.py), 1-sigma from the fit covariance"
R = {}
R["post_width_x"] = e(G["w_p"], GS["w_p"], fit)
R["post_depth_y"] = e(G["d_p"], max(GS["d_p"], 0.03), fit + "; weakly observed (inner faces 10-14 px wide); prior d_p = w_p +- 0.05", "fitted+prior")
R["post_top_z"] = e(G["z_pt"], GS["z_pt"], fit + "; above A (bar top)")
R["base_top_z"] = e(G["z_bt"], GS["z_bt"], fit + "; base-plate top below A")
R["frame_centre_x"] = e(G["x_c"], GS["x_c"], fit + "; frame centre relative to A (A under the top loop)")
R["bar_height_z"] = e(G["h_b"], GS["h_b"], fit)
R["bar_depth_y"] = e(0.5 * G["h_b"], 0.25 * G["h_b"], "not visible from the front: 0.5 x bar height (flat bar)", "assumed")
R["plate_width_x"] = e(G["W_pl"], GS["W_pl"], fit + "; front-corner silhouettes (2 px residual from the plan-view corner rounding)")
R["plate_depth_y"] = e(G["D_pl"], GS["D_pl"], fit)
R["plate_centre_y"] = e(G["y_pl"], GS["y_pl"], fit + "; prior 0 +- 0.05", "fitted+prior")
R["plate_thickness"] = e(G["t_pl"], GS["t_pl"], fit)
R["table_back_edge_y"] = e(G["y_wall"], GS["y_wall"], fit + "; table/backdrop crease row 827-830 px", "fitted")
R["plate_corner_radius"] = e(22 / kw, 5 / kw, "front-left/right plan-view corner rounding read on 2x zooms (~22 px)", "approximate")
R["plate_top_edge_radius"] = e(4 / kw, 2 / kw, "highlight band on the plate's front top edge (~4 px)", "approximate")
R["wood_edge_radius"] = e(3 / k, 2 / k, "post edge softening (~3 px)", "approximate")
R["screw_head_radius"] = e(13 / kw, 3 / kw, "front screw heads (u 510 / 1399, v 1042-1045), ~26 px wide", "approximate")
R["screw_inset_x"] = e(33 / kw, 4 / kw, "screw centre to plate side edge (~33 px)", "approximate")
R["screw_inset_y"] = e(0.075, 0.02, "screw centre to plate front edge, from v 1042 between the front (1056) and back (984) top edges", "approximate")
R["screw_head_height"] = e(6 / kw, 3 / kw, "~6 px", "approximate")

# spring and hardware (spring plane y = 0)
zA_loop_top_outer = P["top_loop_outer_top"]["f000"]["xz_Win"][1]
zA_loop_bot_outer = P["top_loop_outer_bottom"]["f000"]["xz_Win"][1]
hw_px, hw_sig = mean_px(SZ["hook_wire_diameter"])
sw_px, sw_sig = mean_px(SZ["spring_wire_diameter"])
co_px, co_sig = mean_px(SZ["spring_coil_outer_width"])
R["hook_wire_radius"] = e(14.5 / 2 / 673.0, 1.0 / 673.0, "VR-08 (visual review ck001): S-hook shank 14-15 px wide at f0 rows v 140-150 (render was 9 px with the v001 eye reading of 8 px); 673 px/W_in at the spring plane, frame 0", "measured (review)")
R["spring_wire_radius"] = e(sw_px / 2 / k, sw_sig / 2 / k, "coil wire width on zooms (6.5 px)", "approximate")
R["coil_mean_radius"] = e((co_px - sw_px) / 2 / k, co_sig / 2 / k, "outer coil width f120/f240 (78-80 px) minus one wire, /2")
R["top_loop_inner_height"] = e((zA_loop_top_outer - zA_loop_bot_outer) - 2 * hw_px / k, 3 / k,
                              "outer loop height through the camera (v 47 -> 122 px) minus 2 wire diameters; must enclose the bar (h_b)")
R["top_loop_inner_width_y"] = e(0.5 * G["h_b"] + 0.012, 0.01, "loop seen edge-on: bar depth (assumed) + clearance", "assumed")
R["loop_gap_A_px"] = e(6.0, 3.0, "the reference draws the loop's inner top ~6 px ABOVE the bar top edge (floating); the rig rests it on the bar (A = bar top)", "note")
# spring layout v002 (lead CHANGELOG 2026-10-26T00:52Z: fixed hook, swing hinge at P_eff = top coil)
PE = CJ["key_points"]["P_eff_world_on_plane_Y0"]
R["hinge_P_eff_x"] = e(PE[0], 0.003, "camera.json key_points.P_eff_world_on_plane_Y0 (LSQ intersection of 241 spring-axis lines)", "measured")
R["hinge_P_eff_z"] = e(PE[2], 0.003, "camera.json key_points.P_eff_world_on_plane_Y0 (0.225 W_in below A)", "measured")
R["hook_turns"] = {"value": 2, "sd": 0, "method": "lead decision (CHANGELOG): 2 static turns at the hook above the hinge", "status": "counted"}
R["hook_turn_pitch"] = e(2.1 * R["spring_wire_radius"]["value"], 0.002, "close turns (1.05 wire diameters); coil top is 0.012 W_in above P in the tracks", "approximate")
R["coil_count_open"] = {"value": 11, "sd": 0, "method": "measurement/spring_track.csv n_open_gaps median 11 (open pitch 29.1 px median)", "status": "counted"}
R["coil_count_tight"] = {"value": 10, "sd": 1, "method": SZ["coil_count_tight"]["method"], "status": "counted"}
top_off = -SL["coil_top_along_axis_from_P_Win"]["mean"]
R["coil_body_open_length_ref"] = e(SL["open_length_Win"]["mean"] - top_off, SL["open_length_Win"]["sd"],
                                   "P -> tight-section start along P->H, mean over 241 frames (scene/measure/spring_layout.py)")
R["coil_body_tight_length_ref"] = e(SL["tight_length_Win"]["mean"], SL["tight_length_Win"]["sd"], "tight section length, mean over 241 frames")
R["bottom_hardware_length"] = e(SL["bottom_hardware_Win"]["mean"], SL["bottom_hardware_Win"]["sd"], "last coil -> H along the axis, mean over 241 frames")
R["spring_length_ref_PH"] = e(SL["PH_length_Win"]["mean"], SL["PH_length_Win"]["sd"], "|P_eff - H| mean over 241 frames (reference configuration of the section lengths)")
R["bend_length"] = e(0.3 * (SL["open_length_Win"]["mean"] - top_off) / 11, 0.01, "rotation ramps in over 0.3 open turns below P (bending concentrated at the top coil)", "assumed")
L_ref = float(np.hypot(*P["H_hinge"]["f120"]["xz_Win"]))
R["spring_length_ref_AH"] = e(L_ref, 4 / k, "|A - H| at frame 120 (provenance; gauge fallback)")
ll_px, ll_sig = mean_px(SZ["lower_loop_length_outer"])
R["lower_loop_inner_width_y"] = e(SZ["lower_loop_width_visible"]["f240"] / k + 0.02, 0.01,
                                 "loop seen nearly edge-on (16 px visible); width across the eye wire assumed 0.02 S larger", "assumed")
# weight
eo_px, eo_sig = mean_px(SZ["eye_outer_diameter"])
ew_px, ew_sig = mean_px(SZ["eye_wire_diameter"])
cd_px, cd_sig = mean_px(SZ["cylinder_diameter"])
ch_px, ch_sig = mean_px(SZ["cylinder_height_axis"])
ew_px = 12.0     # VR-08 (visual review ck001): the reference eye-ring wire reads thicker than my v001 reading (10 px)
R["eye_wire_radius"] = e(ew_px / 2 / kw, ew_sig / 2 / kw, "eye ring band width: 12 px (VR-08 review; v001 read 10 px); outer diameter kept at the measured 65-67 px", "approximate (review)")
R["eye_ring_radius"] = e((eo_px - ew_px) / 2 / kw, eo_sig / 2 / kw, "eye outer diameter (65-67 px) minus one wire, /2 (centreline)")
# v002: weight size from the measurement role's 241-frame silhouette track (measurement/weight_track.csv width_px,
# length_px), scaled per frame by the camera-of-record zoom; my v001 eye readings (150-153 px) overestimated the width
# (luma profile at f120 row 880: rim to rim ~146 px horizontal).
import sys as _sys
_sys.path.insert(0, str(ROOT / "camera"))
import project as _PJ
_W = np.genfromtxt(ROOT / "measurement" / "weight_track.csv", delimiter=",", names=True)
_f0 = _PJ.load_camera(frame=0)["f_px"]
_s = np.array([_PJ.load_camera(frame=int(n))["f_px"] for n in _W["frame"]]) / _f0
_k0 = F["px_per_Win"]["f000_v860"]
_D = _W["width_px"] / (_k0 * _s); _Ls = _W["length_px"] / (_k0 * _s)
_c = _PJ.load_camera(frame=120); _C = np.asarray(_c["C"]); _wc = np.array([-0.08, 0.0, -1.23])
_elev = float(np.arcsin((_C[2] - _wc[2]) / np.linalg.norm(_C - _wc)))
R["weight_radius"] = e(np.median(_D) / 2, max(_D.std(), 0.004) / 2, "measurement/weight_track.csv width_px median over 241 frames / per-frame px per W_in (v001 eye reading 0.1144 was 5 % high)")
R["weight_height"] = e(np.median(_Ls) - np.median(_D) * np.sin(_elev), 0.01,
                      f"silhouette length (median {np.median(_Ls):.4f} W_in, includes the top-face ellipse) minus D sin(view elevation {np.degrees(_elev):.1f} deg)")
R["weight_top_edge_radius"] = e(5 / kw, 2 / kw, "rounded top rim (~5 px)", "approximate")
R["weight_bottom_chamfer"] = e(3 / kw, 2 / kw, "small bottom chamfer (~3 px)", "approximate")
R["lower_loop_inner_length"] = e(max(ll_px / k - 2 * hw_px / k, 0.03), ll_sig / k,
                                "outer loop length 80-82 px minus 2 wire diameters (loop top merges into the last coil)", "approximate")
Hz = [SL["H_xz_Win"][f] for f in ("f000", "f120", "f240")]
derived = {
    "H_to_cylinder_centre": round(2 * R["eye_ring_radius"]["value"] + 0.0 + R["weight_height"]["value"] / 2, 6),
    "H_to_cylinder_centre_note": "H = eye inner top; eye centre = H - (R_e - r_e); eye outer bottom touches the cylinder top: top = H - 2 R_e; centre = top - h/2",
    "AH_frames_xz_S": {"f000": Hz[0], "f120": Hz[1], "f240": Hz[2]},
    "AH_length_frames_S": [round(float(np.hypot(*h)), 5) for h in Hz],
    "rest_clearance_weight_to_plate_S": round(-(Hz[1][1]) - 0, 6),
}
derived["rest_clearance_weight_to_plate_S"] = round(abs(G["z_bt"]) - (abs(Hz[1][1]) + 2 * R["eye_ring_radius"]["value"] + R["weight_height"]["value"]), 6)
out = {
    "version": "v002",
    "owner": "scene (role C)",
    "units": "ratios to W_in (= S) = clear span between the posts' inner faces, the camera of record unit. Metres = ratio x S_m (metres per W_in, see S_m_*).",
    "S_m_assumed": {"value": 0.25, "status": "ASSUMED placeholder until physics.json/camera.json give a metre gauge",
                    "note": "build.py reads the gauge in this order: physics.json S_m | scene_unit_S_m, camera.json scale.S_m, then this value"},
    "world": "config/coordinates.md: origin A = top-loop bearing point on the bar top (y = 0 through the bar centre), X along the bar to image right, Y away from the camera, Z up",
    "ratios": R,
    "derived": derived,
    "camera_frame0_approx": json.loads((HERE / "fit_frame0_camera.json").read_text()),
    "fit_quality": {"rms_sigma": F["rms_sigma"], "n_obs": F["n_obs"], "n_par": F["n_par"], "edge_residuals_px": F["edge_residuals_px"]},
    "camera_json_implied_frame_Win": F["camera_json_implied_frame"],
    "sources": {"fit": "scene/measure/fit_fixedcam.json", "fit_sha256": sha(HERE / "fit_fixedcam.json"),
                "spring_layout": "scene/measure/spring_layout.json", "spring_layout_sha256": sha(HERE / "spring_layout.json"),
                "camera_json_sha256": sha(ROOT / "params" / "camera.json"),
                "landmarks": "scene/measure/landmarks_f000.json", "landmarks_sha256": sha(HERE / "landmarks_f000.json"),
                "homography": "measurement/cache/H_ecc.npy (measurement role, frame0->frame n)"},
}
(ROOT / "params" / "geometry.json").write_text(json.dumps(out, indent=1))
for kk, v in R.items():
    print(f"{kk:28s} {v['value']:.5f} +- {v['sd']:.5f}  {v['status']}")
print(json.dumps(derived, indent=1))
