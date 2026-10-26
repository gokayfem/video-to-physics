"""Write params/geometry.json from the landmark fit + documented direct readings.

Default (v2, lead decision 2026-10-25T22:42Z): stand proportions from scene/measure/fit_fixedcam.json, i.e. the box
model fitted with the camera FIXED to params/camera.json (camera of record). Pixel-read sizes (pin, loop, wire,
bolt, edge radii) are converted with the camera-of-record image scale at the pivot (K = 700.6 px / px-per-L).
The bob radius ratio keeps the joint-fit value (r over |P_ring - bob|, L_c/r = 12.30) as provenance.
Legacy: --fit scene/measure/fit_joint.json reproduces v1.

Run from ROOT:  .venv/bin/python scene/measure/make_geometry_params.py [--Lc-placeholder 0.388]
All lengths are stored as RATIOS to L_c (pivot-to-bob-centre). The builder multiplies by L_c read from
params/physics.json at build time; "derived_absolute_m" below is informational only, for the L_c given here.
"""
import json, argparse, datetime, hashlib, pathlib

ap = argparse.ArgumentParser()
ap.add_argument("--fit", default="scene/measure/fit_fixedcam.json")
ap.add_argument("--out", default="params/geometry.json")
ap.add_argument("--Lc-placeholder", type=float, default=0.388)
ap.add_argument("--physics", default="params/physics.json")
a = ap.parse_args()
fit = json.load(open(a.fit))
G = fit["geometry"]
FRAMES = "frames 0 (t=0 s) and 120 (t=5 s)"
FIXED = "fixedcam" in a.fit
K = 1.0
if FIXED:
    import sys, numpy as np
    sys.path.insert(0, "camera"); import project as PJ
    LE = json.load(open("params/camera.json"))["scale"]["L_eff_m"]
    c0 = PJ.load_camera(frame=0)
    px_per_L = c0["f_px"] * LE / float(PJ.depth(np.zeros(3), c0))
    K = 700.6 / px_per_L
FIT = (f"least-squares fit of a parametric box model with the camera FIXED to params/camera.json "
       f"(camera/project.py load_camera(frame=n)) to 20/19 hand-read landmarks plus 30/31 base top-edge points "
       f"(edge_points.py, line constraints) in {FRAMES} (scene/measure/fit_fixedcam.py); rms "
       f"{fit['rms_normalised']:.2f} sigma, point residual rms {fit.get('pixel_residual_rms_px_points', 0):.1f} px") if FIXED else \
      (f"joint least-squares fit of a parametric box model + one pinhole camera per frame to 21/20 hand-read "
       f"landmarks plus 30/31 base top-edge points (edge_points.py, line constraints) in {FRAMES} "
       f"(scene/measure/fit_proportions.py, landmarks_f000/f120.json); rms reprojection "
       f"{fit['rms_normalised']:.2f} sigma")

def fv(k, nd=4):
    return round(G[k]["value"], nd), round(max(G[k]["sd"], 1e-4), 4)

R = {}
def put(name, value, sd, method, persp, prov="measured-ratio", note=None):
    R[name] = {"value": round(value, 5), "sd": round(sd, 5), "provenance": prov,
               "perspective_sensitive": persp, "method": method}
    if note:
        R[name]["note"] = note

# --- swing parts (in or next to the swing plane: ratios are robust) -------------------------------------
v, s = fv("r_bob")
put("bob_radius", v, max(s, 0.003), f"{FIT}. Cross-check: direct pixel ratio frame 0, bob radius 56.5 px / "
    "|P - bob centre| 700.6 px = 0.0806 (circle fit to silhouette; P at wire exit (1087,124))", False,
    note="If params/physics.json declares a bob radius the builder uses that value and reports the difference.")
put("wire_diameter", 0.0035 * K, 0.0015 * K, "apparent dark-line width 2.5-4 px over |P-bob| ~700 px, frames 0/120, "
    "per-row profile (scratch wire.py); motion blur/defocus widen it so this is an upper-leaning estimate", False,
    prov="approximate")
put("pin_shaft_radius", 0.022 * K, 0.003 * K, "pin shaft apparent diameter 31 px (frame 0, 4x zoom, v 88.8-120) / ~700 px "
    "per L_c at the pivot depth", False)
put("pin_head_radius", 0.0284, 0.0015, "VR-11 re-measure (frame 120, 8x zoom, camera of record dv/dz = 659.7 px per L_eff "
    "at P): head apparent diameter 37.5 px (frame 0: 38.75 px). ck002 rendered ~35 px. The reference pin is NOT 1.5-2x "
    "thicker; it looks heavier because its fat head starts right at the loop (short exposed shaft)", False)
v, s = fv("x_tip")
put("pin_tip_x", v, s, FIT + " (pin_tip landmark)", True)
put("pin_head_start_x", -0.0212, 0.004, "VR-11 (frame 120, 8x zoom): head shoulder at u~1075 vs P at u 1086.8, du/dx = "
    "556.7 px per L_eff at P -> -0.0212; the wire loop wraps the head's left end. Previous value (v1-v3) 0.004:", True, prov="measured-ratio")
put("pin_head_start_x_v1", 0.004, 0.006, "head shoulder sits just +X of the wire loop (frame 0/120 4x zoom: loop at "
    "u 1074-1085, head from ~1086); placed so the loop (centred at x=0) bears on the shoulder", True, prov="approximate")
put("wire_loop_outer_radius_measured", 0.0341, 0.003, "VR-11 (frame 120, 8x zoom): loop ring outer height 45 px / "
    "659.7 px per L_eff, concentric with the head, no gap; the wire twists into a short V below the ring (ring bottom "
    "~4.5 px above P). Builder uses a TIGHT loop: centreline radius = pin_head_radius + wire_radius + 0.0005 (check: "
    "measured outer 0.0341 vs model 0.0313)", False)
put("wire_loop_radius_v1", 0.028 * K, 0.004 * K, "loop ring around the pin: apparent outer height 44 px (frame 0, 4x zoom, "
    "v 79-123) => centreline radius ~0.028 L_c", False)
v, s = fv("z_pin")
put("pin_axis_z_fit", v, s, FIT + " (free parameter). SINCE v4 (VR-11) THE BUILDER USES THIS (arm centre = pin axis). Earlier note: "
    "the builder used the contact-derived "
    "pin_axis_z = 2*loop_radius - wire_radius - pin_shaft_radius (loop rests on top of the shaft, wire leaves the "
    "loop bottom at P) = %.4f; agreement is a consistency check" % (K * (2 * 0.028 - 0.00175 - 0.022)), True,
    prov="measured-ratio")
# --- stand ---------------------------------------------------------------------------------------------------
v, s = fv("arm_h");  put("arm_height", v, s, FIT, False)
v, s = fv("arm_d");  put("arm_depth", v, s, FIT + "; set mainly by the narrow arm end face width (28-32 px), "
                         "so it depends on the fitted camera yaw (~37 deg)", True)
v, s = fv("x_ae");   put("arm_end_x", v, s, FIT + " (arm end face plane, relative to P)", True)
v, s = fv("post_px"); put("post_size_x", v, s, FIT, True)
v, s = fv("post_py"); put("post_size_y", v, s, FIT, True)
v, s = fv("x_p");    put("post_centre_x", v, s, FIT, True)
v, s = fv("y_p")
put("post_centre_y", 0.0, max(abs(v), s), FIT + f" gives {v:+.4f} +/- {s:.4f}; set to 0 (arm and pin on the "
    "swing plane, post centred on the arm) by design", True, prov="assumed (consistent with fit)")
v, s = fv("z_pt");   put("post_top_z", v, s, FIT, True)
ALT = "" if not FIXED else (" Alternative (camera role's calibration-implied stand, params/camera.json "
       "stand_geometry_implied_by_calibration): %s; kept ours per lead decision, the difference is included in sd.")
CJS = json.load(open("params/camera.json")).get("stand_geometry_implied_by_calibration", {}).get("units_Leff", {}) if FIXED else {}
v, s = fv("W")
altW = CJS.get("bx1", 0) - CJS.get("bx0", 0) if CJS else None
put("base_width", v, max(s, abs(v - altW)) if altW else s, FIT + "; corners are rounded so silhouette corners sit slightly inside." + (ALT % f"{altW:.4f}" if altW else ""), True)
v, s = fv("D")
altD = CJS.get("by1", 0) - CJS.get("by0", 0) if CJS else None
put("base_depth", v, max(s, abs(v - altD)) if altD else s, FIT + "; back-left corner hidden behind the post (weak)." + (ALT % f"{altD:.4f}" if altD else ""), True)
v, s = fv("T")
altT = CJS.get("bz1", 0) - CJS.get("bz0", 0) if CJS else None
put("base_thickness", v, max(s, abs(v - altT)) if altT else s, FIT + "; bottom edges are in contact shadow." + (ALT % f"{altT:.4f}" if altT else ""), True)
v, s = fv("h");      put("h_pivot", v, s, FIT + " (base top plane is z = -h_pivot)", True)
v, s = fv("x_b");    put("base_centre_x", v, s, FIT, True)
v, s = fv("y_b");    put("base_centre_y", v, s, FIT, True)
put("base_corner_radius", 0.03 * K, 0.015 * K, "rounding of the vertical corners seen at the back-right and front-left "
    "silhouettes (~15-25 px), frame 0/120", True, prov="approximate")
put("base_top_edge_radius", 0.022 * K, 0.01 * K, "width of the rounded top-edge highlight band along the front edge "
    "(~12-20 px), frame 0/120", True, prov="approximate")
put("base_bottom_edge_radius", 0.008, 0.006, "bottom edge slightly softened, in shadow", True, prov="assumed")
put("wood_edge_radius", 0.006 * K, 0.003 * K, "post and arm edges softened (~3-5 px highlight), frame 0 2x zoom", True,
    prov="approximate")
put("eyelet_radius", 0.0034, 0.001, "VR-12: brass stem/eyelet where the wire enters the bob, apparent width 4-5 px "
    "(frames 0 and 120, 12x zoom) / 659.7 px per L_eff", False, prov="approximate")
put("eyelet_height", 0.006, 0.003, "VR-12: the bright stem shows 8-12 px long but INSIDE the bob silhouette, ending at its "
    "top edge (frames 0/120, 12x zoom): seen ~6 deg from above, the pole projects ~6 px inside the outline, so the stem "
    "is short. A first try of 0.012 L_eff stuck out above the outline (v012 draft); 0.006 keeps it within", False,
    prov="approximate")
put("bolt_radius", 0.021 * K, 0.005 * K, "steel cap on the post -X face at arm height, apparent height 30 px (frame 0)", True,
    prov="approximate")
put("bolt_protrusion", 0.085, 0.02, "the dark cap shows at u~680-695, v~140-170 in frame 0, i.e. 0-15 px left of "
    "the post's front-left silhouette (u~694). Projecting candidate positions through the frame-0 fitted camera, a "
    "cap on the -X face at y=0 must reach ~0.08-0.1 L_c out of the face to be visible there (0.018 in v004 and 0.03 in "
    "v005 stayed hidden). Likely the end of a through-bolt; the AI reference may be inconsistent here", True, prov="approximate")

d = {k: R[k]["value"] for k in R}
pin_axis_z = d["pin_axis_z_fit"]           # v4 (VR-11): fitted arm/pin axis height; the loop is concentric and tight
derived = {
    "pin_axis_z": pin_axis_z,
    "arm_visible_length": d["arm_end_x"] - (d["post_centre_x"] + d["post_size_x"] / 2),
    "post_height_above_base": d["post_top_z"] + d["h_pivot"],
    "pin_length_from_arm_face": d["pin_tip_x"] - d["arm_end_x"],
}
phys = pathlib.Path(a.physics)
Leff_m = None
try:
    _pj = json.loads(phys.read_text())
    def _find(o, key):
        if isinstance(o, dict):
            for k, x in o.items():
                if k == key: return x
                y = _find(x, key)
                if y is not None: return y
    _v = _pj.get("L_eff_m") or _find(_pj, "camera_metre_scale_L_eff_m")
    Leff_m = float(_v["value"] if isinstance(_v, dict) else _v) if _v is not None else None
except Exception:
    pass
Lc, Lsrc = a.Lc_placeholder, "placeholder (params/physics.json absent at generation time)"
if phys.exists():
    try:
        pj = json.loads(phys.read_text()); 
        cand = pj.get("parameters", pj).get("L_c_m", pj.get("L_c"))
        Lc = float(cand["value"] if isinstance(cand, dict) else cand); Lsrc = "params/physics.json"
    except Exception:
        pass
out = {
    "schema": "pnd-geometry/1",
    "owner": "geometry & rig (scene/)",
    "generated_by": "scene/measure/make_geometry_params.py",
    "generated_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "units": ("every entry in 'ratios' is a length in camera.json units (L_eff; metres = ratio x physics.json L_eff_m, "
              "CHANGELOG 2026-10-25T22:47Z) EXCEPT bob_radius, which is r / |P_ring - bob| = r / L_c (provenance for "
              "dynamics' L_c/r; the builder takes r from physics.json)") if FIXED else
             "every entry in 'ratios' is a length divided by L_c (pivot-to-bob-centre distance); dimensionless",
    "frame": "world frame of CONTRACT.md: pivot P = origin, X image-right in the swing plane, Y away from camera, Z up",
    "scale_note": "L_c is model-conditioned (from the period, owned by dynamics). The builder reads it from "
                  "params/physics.json at build time; never hard-coded.",
    "camera_note": (("Camera FIXED to params/camera.json (sha256 %s) via camera/project.py, per-frame drift included; "
                    "ratios are lengths / L_eff (camera gauge, = physics L_c when camera.json was written). Pixel-read sizes "
                    "scaled by K = %.4f (camera-of-record image scale at the pivot vs the 700.6 px/L used in v1). "
                    "Not fitted, reported: P_ring and bob diagnostics in the fit file (bob centre lies |B-P_ring| = 1.05-1.07 L_eff "
                    "from P_ring: the camera gauge L_eff is the bob arc radius about P_eff, 0.081 below P_ring)."
                    % (hashlib.sha256(pathlib.Path("params/camera.json").read_bytes()).hexdigest(), K))
                   if FIXED else "v1 joint-fit cameras (superseded)"),
    "superseded": "v1 proportions from scene/measure/fit_joint.json (own joint-fit camera, f=2749 px at f120) are superseded by the camera of record" if FIXED else None,
    "fit_source": {"file": a.fit, "sha256": hashlib.sha256(pathlib.Path(a.fit).read_bytes()).hexdigest(),
                   "rms_normalised": fit["rms_normalised"], "n_obs": fit["n_obs"], "n_par": fit["n_par"]},
    "ratios": R,
    "derived_ratios": {k: round(v, 5) for k, v in derived.items()},
    "derived_absolute_m": ({"L_eff_m_used": Leff_m, "L_c_m_used": Lc, "L_c_source": Lsrc,
                            "note": "stand = ratio x L_eff_m; bob radius ratio x L_c (provenance only)",
                            **{k + "_m": round(R[k]["value"] * (Lc if k == "bob_radius" else Leff_m), 6) for k in R},
                            **{k + "_m": round(v * Leff_m, 6) for k, v in derived.items()},
                            "bob_clearance_above_base_m": round(d["h_pivot"] * Leff_m - Lc * (1 + d["bob_radius"]), 6),
                            "wire_free_length_m": round(Lc * (1 - d["bob_radius"]), 6)}
                           if (FIXED and Leff_m) else
                           {"L_c_used": Lc, "L_c_source": Lsrc,
                            **{k + "_m": round(R[k]["value"] * Lc, 6) for k in R},
                            **{k + "_m": round(v * Lc, 6) for k, v in derived.items()}}),
}
pathlib.Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps(out["derived_ratios"], indent=1)); print(out["derived_absolute_m"])
