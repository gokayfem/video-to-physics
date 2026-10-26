"""params/geometry.json from the scene-role measurements (role C). Unit D = disc diameter (the camera role's unit).
.venv/bin/python scene/measure/make_geometry_params.py
Inputs: scene/measure/fit_geometry.json (joint landmark + silhouette fit, approximate cameras 0/120/240),
scene/measure/mb_reference.json (mark width), hand-read hardware sizes (below, from 3.5x zooms; px -> D with the local
scale 575 px/D at the clamp, 523 px/D at the disc, frame 120).
"""
import json, pathlib, hashlib, datetime

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def main():
    fit = json.loads((HERE / "fit_geometry.json").read_text())          # approximate cameras (fallback) + sd
    g, sd = fit["geometry_D"], fit["geometry_sd_D"]
    F = "fitted (scene/measure/fit_geometry.py, frames 0/120/240, approximate cameras)"
    camref = HERE / "fit_geometry_camref.json"
    if camref.exists():                                                  # refit through the camera of record wins
        cr = json.loads(camref.read_text())
        g = cr["geometry_D"]
        F = ("fitted through the CAMERA OF RECORD (scene/measure/fit_geometry.py --camera-of-record, params/camera.json "
             f"sha256 {cr['camera_json_sha256'][:12]}..., frames 0/120/240)")
    mb = json.loads((HERE / "mb_reference.json").read_text())

    def m(v, prov, sigma=None, note=None):
        d = {"value": round(float(v), 5), "provenance": prov}
        if sigma is not None:
            d["sigma"] = round(float(sigma), 5)
        if note:
            d["note"] = note
        return d
    Z = "measured px on 3.5x zooms, frame 120 (scene/measure/zoom.py); px/D 575 at the clamp"
    R = {
        "disc_diameter": m(1.0, "unit"),
        "disc_thickness": m(0.031, "camera role (lead message 2026-10-26: rim thickness 0.031 D through the camera of record); scene-role "
                            "cross-check: rim band 21 px at ~620 px/D, ~19 deg elevation, frame 240 -> ~0.035 D; silhouette fit "
                            f"{g['t']:.3f} D (confounded with disc tilt)"),
        "disc_edge_radius": m(0.004, "assumed (bright rim highlight ~2 px)"),
        "wire_length_A_to_disc_top": m(g["L"], F, sd["L"], "A = wire exit at the clamp collet bottom"),
        "wire_radius": m(0.0014, "measured: wire FWHM ~1.5-2 px at ~560 px/D (frames 0/120/240, rows v 260-540)",
                         note="the AI draws the wire very faint; physical wire d from dynamics is reported separately"),
        "mark_width": m(0.010, "VR03-08: visual review measured FWHM 0.010 D at 100 %; scene-role luma-dip FWHM 5.9-6.4 px at "
                        "508-573 px/D (camera of record, frames 70-74/174-178) = 0.011 D incl. ~1-2 px AI/JPEG blur; shader "
                        "groove (flat core, walls 30 %) renders FWHM ~0.9 x width"),
        "mark_depth": m(0.0015, "assumed (shader bump only)"),
        "mark_kind": "full diameter, straight (scene/measure/mark_analysis.json: diameter in every sampled frame; kink "
                     "mostly < 4 deg, up to ~10-20 deg in a few frames = AI inconsistency, not modelled)",
        "span_inner": m(g["S"], F, sd["S"], "clear span between the posts' inner faces"),
        "frame_centre_x": m(g["xc"], F, sd["xc"], "frame centre X relative to A"),
        "post_width_x": m(g["pw"], F, sd["pw"]),
        "post_depth_y": m(g["pd"], F, sd["pd"]),
        "post_top_z": m(g["zt"], F, sd["zt"], "post tops flush with the crossbar top"),
        "bar_height_z": m(g["bh"], F, sd["bh"]),
        "bar_depth_y": m(g["pd"], "assumed = post depth (top faces of bar and posts read as one plane)"),
        "foot_top_z": m(g["z_ft"], F, sd["z_ft"]),
        "foot_height": m(0.15, "measured ~0.146 D (frame-0 left-foot back end, 73 px at ~500 px/D); same stock as the posts"),
        "foot_width_x": m(g["fw"], F, sd["fw"]),
        "foot_front_y": m(g["y_ff"], F, sd["y_ff"]),
        "foot_back_y": m(g["y_fb"], F, sd["y_fb"]),
        "wood_edge_radius": m(0.006, "assumed (soft arrises, 2-3 px)"),
        "clamp_width_x": m(g["cw"], F, sd["cw"]),
        "clamp_depth_y": m(2 * abs(g["cy"]), F + " (front face y; symmetric about the bar)", 2 * sd["cy"]),
        "clamp_top_z": m(g["cz_top"], F, sd["cz_top"]),
        "clamp_bottom_z": m(g["cz_bot"], F, sd["cz_bot"], "collet height = clamp_bottom_z (A at the collet bottom)"),
        "clamp_sheet": m(0.014, "assumed (bent-sheet edge ~8 px wide on the side flange, frame 0)"),
        "clamp_bend_radius": m(0.02, "measured roughly: rounded top-front bend ~10 px"),
        "clamp_top_screw_radius": m(0.036, Z + ": head diameter 41 px"),
        "clamp_top_screw_height": m(0.03, Z + ": head height 17 px"),
        "clamp_top_screw_y": m(0.07, Z + ": two heads, one behind the other, |y| ~ 0.07 D"),
        "clamp_front_screw_radius": m(0.022, "measured: button head ~28 px (head 0.048 D) at 578 px/D, frame 240 (v014 review: 0.029 read too large incl. washer)"),
        "clamp_front_screw_z_above_bottom": m(0.042, Z + ": centre 24 px above the clamp bottom"),
        "clamp_front_screw_height": m(0.006, "VR03-06 v002: flatter button head (assumed)"),
        "collet_radius": m(0.037, Z + ": 43 px wide"),
        "clamp_front_washer_radius": m(0.031, "VR03-06: chrome washer under the front button screw, ~36 px wide at 578 px/D (frame 240 zoom)"),
        "clamp_front_washer_height": m(0.004, "assumed"),
        "clamp_front_socket_radius": m(0.009, "VR03-06: hex socket ~10 px across flats at 578 px/D (frame 240 zoom)"),
    }
    R["table_z"] = m(g["z_ft"] - 0.15, "derived: foot_top_z - foot_height")
    cams = {}
    for n, c in fit["cameras"].items():
        cams[n] = {"R_world_to_cv": c["R_world_to_cv"], "C_D": c["C_D"], "f_px": c["f_px"], "principal_point_px": c["principal_point_px"],
                   "disc_offset_D": c["disc_offset_D"]}
    if camref.exists():
        R["_camref_fit_residuals_px"] = "scene/measure/fit_geometry_camref.json residuals_px (upper frame/clamp <= 3 px, feet <= 7 px, disc silhouette rms 3-6 px)"
    out = {
        "version": "v002" if camref.exists() else "v001", "owner": "scene role C", "generated_utc": datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "unit": "D = disc diameter (camera role's unit). Metres = value x D_m.",
        "D_m_assumed": {"value": 0.10, "provenance": "assumed: prompt 'about ten centimetres across'; build.py prefers physics.json "
                        "geometry_m.disc_radius x 2 (same assumption) when present"},
        "world": "origin A (wire exit at the clamp collet bottom), X along the crossbar (image right), Y away from the camera, Z up",
        "ratios": R,
        "camera_approx": {"status": "APPROXIMATE scene-role cameras (joint fit, robust loss); used only until params/camera.json + "
                          "camera/project.py (camera of record) exist", "frames": cams,
                          "between_keys": "rotation slerp, centre and f linear between frames 0/120/240", "fit_rms_normalised": fit["rms_normalised"]},
        "residual_note": "the AI frame is not perfectly rigid over the camera orbit: feet / back-corner landmarks disagree by up to "
                         "~30 px at frame 240 (fit_geometry.json residuals_px); upper frame and clamp <= ~6 px",
        "inputs_sha256": {p: sha(HERE / p) for p in ("fit_geometry.json", "fit_geometry_camref.json", "landmarks.json", "mb_reference.json") if (HERE / p).exists()},
    }
    (ROOT / "params" / "geometry.json").write_text(json.dumps(out, indent=1))
    print("wrote params/geometry.json", {k: v["value"] for k, v in R.items() if isinstance(v, dict)})


if __name__ == "__main__":
    main()
