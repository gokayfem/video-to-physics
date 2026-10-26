# scene/ — geometry, rig & look (role C) NOTES

Status: **v001–v008, integrated scene, draft.**
* Cameras: the **camera of record** (params/camera.json via camera/project.py), keyed per frame in location, rotation, lens and shift.
* Geometry: params/geometry.json **v002**, refit through that camera.
* Look: params/look.json **v016**, after 16 look rounds.
* The checkpoint file `scene/out/torsion.blend` is built with `--look` and is **unbaked**. The bake path has been tested with the placeholder trajectory.

## Visuals (`scene/visuals/`)
| file | what it shows | what remains wrong |
|---|---|---|
| v001_blockout_f000/f120/f240.jpg | Reference vs render through TOR_Cam_Reference, with an edge overlay. Close-ups of the clamp, the disc and mark, and the wood. The disc is at a TEST pose measured from the reference mark. Static-stand edges agree to a median of 2.2 / 2.8 / 3.0 px (p90 11–13 px, mostly wood-grain edges) | f240: the table/wall crease is sharper and slightly higher than the reference's soft band |
| v002_twist_test_poses.jpg | ψ = −40 / 0 / +40°, and +40° with B moved 3 mm and tilted 2°. The mark and the texture rotate with the disc; the turned-finish cones stay with the camera | ring streaks are slightly stronger than in the reference |
| v003…v006_look_round02/07/12_camref/16_final_f000.jpg | Look sheets: reference \| previous \| current, close-ups, backdrop row and column L profiles, disc highlight ring profile, metrics | see "Look" below |
| v007_mark_evidence.jpg | The reference disc unwarped every 20 frames. The mark is a FULL DIAMETER in every frame, sometimes kinked by 5–12° | — |
| v008_motion_blur_evidence.png | Reference mark width (FWHM) against its local speed | — |

## Commands (run from ROOT)
```
.venv/bin/python scene/measure/disc_silhouette.py 0 120 240
.venv/bin/python scene/measure/fit_geometry.py [--camera-of-record]
.venv/bin/python scene/measure/make_geometry_params.py                    # -> params/geometry.json
.venv/bin/python scene/measure/mark_analysis.py --step 4
.venv/bin/python scene/measure/measured_poses.py
.venv/bin/python scene/measure/mb_reference.py
blender -b --factory-startup --python scene/build.py -- --out scene/out/torsion.blend --look [--bake solver/out/trajectory.csv] [--pose "psi,dx_mm,dy_mm,dz_mm,tx,ty"] [--approx-camera]
blender -b scene/out/torsion.blend --python scene/check_rig.py [-- --frames --out X.json]
blender -b scene/out/torsion.blend --python scene/check_camera.py
.venv/bin/python scene/compare_builds.py A.build.json B.build.json        # determinism
look/round.sh vNNN vPREV "caption" [frame]                                  # look round: build, render (lock), metrics, sheet
tools/with_render_lock.sh blender -b X.blend --python look/render_look.py -- --tag T --frame N --crops --auto-crops --outdir D
.venv/bin/python scene/make_visual.py --tag T --dir D --frame N --version vNNN --status "..." --out scene/visuals/...
```

## Update 2 — physics v003 baked (params/physics.json faee420f…, status fitted)

**What build.py now does when physics.json is fitted:**
* D_m = 2 × `geometry_m.disc_radius` = **0.2211 m** (model-conditioned). Geometry D-ratios and the camera gauge (camera.json unit D) are scaled by it.
* Disc thickness = physics 0.006855 m (0.031 D). Wire ℓ = physics 0.19059 m; the geometry.json value is 0.1905 m (+0.03 %).
* TOR_Anchor and the wire top sit at `pivot_world_m`.
* TOR_Disc uses QUATERNION rotation; the mark azimuth μ is physics `mark.mu_body_rad` (−6.796 rad).

**Bake** (`scene/out/baked/`, built with `--look --bake solver/out/trajectory.csv`): bake check P6 disc-origin error 7.5e-9 m, rotation error 4.8e-7; P5 anchor 2e-11 m; attachment gap 0.

**check_rig `--frames`** (961 samples) — **PASS**:
* wire ends ≤ 2.3e-10 m; anchor drift 2e-11 m;
* the wire top lies on the collet bottom, inside its footprint, at every sample;
* no intersections; disc–table clearance ≥ 147 mm;
* wire length 0.1905873 m, constant.

**check_camera** — **PASS**: max 0.0005 px at 29 frames.

**Visual:** `scene/visuals/v009_baked_f000_f120_f240.jpg`. Motion blur is off (config/render.json).

## Bake object contract (matches solver/bake_blender.py)
| object | what it is | channels |
|---|---|---|
| `TOR_Anchor` | empty at the **suspension point**: physics.json `pivot_world_m` when physics is fitted. With v003 that is (+3.9, −1.0, 0) mm from the camera's A, inside the collet footprint (r 8.2 mm). Otherwise it is A = world origin. The clamp and collet mesh stay at A | never animated (custom property `tor_pivot_m`) |
| `TOR_Disc` | Brass disc mesh. **Origin = B = centre of the TOP face.** Local +Z is the disc normal. The engraved full diameter runs along local (cos μ, sin μ, 0), with μ = custom property `tor_mark_azimuth_rad` (from physics.json `mark.mu_body_rad`, default 0). CoM at local (0, 0, −t/2). No parent. **rotation_mode = 'QUATERNION'** | `location` = B and `rotation_quaternion` (wxyz), 4 keys per frame |
| `TOR_WireEnd` | Empty at B with a COPY_LOCATION constraint to TOR_Disc. The bake may key it; the constraint result is identical | (bake keys allowed) |
| `TOR_Wire` | GN 'TOR Wire': a straight tube from TOR_Anchor to TOR_WireEnd, radius 0.0014 D. No drawn twist | never keyed |
| `TOR_Cam_Reference` | Scene camera, keyed per frame (LINEAR) from `blender_camera_settings(load_camera(frame=n))`. Gauge: camera.json unit D × D_m | do not touch |
| `TOR_Cam_Locked`, `TOR_Cam_TopOrtho`, `TOR_Cam_Overview` | locked diagnostic cameras, lock flags set | — |

Bake test (`scene/out/baked_test/`, placeholder physics), from bake_blender's own check:
* P6: disc origin error 3.7e-9 m; rotation-matrix error 6e-8.
* Attachment gap 0; anchor fixed.

check_rig `--frames` on 961 samples:
* wire ends ≤ 3e-11 m;
* no intersections;
* disc–table clearance ≥ 61 mm.

## Geometry v002 (params/geometry.json, unit D = disc diameter)
**Gauge.** D_m = physics.json `geometry_m.disc_radius` × 2, else the assumed 0.10 m (prompt).

**Fit.** Cameras fixed to the camera of record at frames 0/120/240 (`fit_geometry.py --camera-of-record`).
* Inputs: hand-read landmarks (`scene/measure/landmarks.json`) and disc silhouettes. Robust loss.
* Upper frame and clamp residuals ≤ 3 px. Feet ≤ 7 px (one outlier corner, −29 px, at f120). Disc silhouette RMS 3–6 px; the disc is modelled level, while the measurement role sees small tilts.

**Key ratios (D).**
| part | value |
|---|---|
| wire A→disc top | 0.862 (camera role: 0.857) |
| disc thickness | 0.031 (camera role; my rim-band check gives 0.035) |
| inner span | 1.653 |
| frame centre x | −0.001 |
| post | 0.198 × 0.218 |
| post/bar top | +0.245 |
| bar height | 0.201 (bar depth = post depth, assumed) |
| foot top | −1.426 |
| foot | 0.215 wide, 0.15 high (measured roughly), y −0.785…+0.692 |
| clamp | 0.220 × 0.260 × (0.019…0.266); sheet 0.014 assumed |
| top cap screws | r 0.036 |
| front button screw | r 0.024 |
| collet | r 0.037 |
| wire | r 0.0014 (FWHM 1.5–2 px) |
| mark width | 0.0114 D (FWHM 5.9 px) |
| table | z −1.576 |

The superseded approximate scene-role cameras are kept in geometry.json `camera_approx`, used only if camera.json is missing.

## Mark (reference)
**Measured.**
* The mark is a full diameter in every sampled frame: `scene/measure/mark_analysis.json`, `mark_unwarped.jpg`.
* It is mostly straight, with an occasional centre kink of about 5–12°. The measurement role reports a kink of ±6.5°.

**Modelled.** One straight diameter: a groove 0.0114 D wide in the shader, in the disc's local frame (dark core plus bump walls).

**AI inconsistencies.** The prompt asked for a radius; the kink varies between frames. The AI line also turns monotonically, about 1.5°/frame.

## Motion blur (proposal for the lead; config/render.json is not written by me)
* The reference mark width is 5.8–6.5 px FWHM at local speeds of 0–14 px/frame. Linear slope +0.011 px per px/frame; the quadrature fit gives shutter 0.0 (`scene/measure/mb_reference.json`, v008).
* Upper bound on the shutter: about 0.2 frame.
* **Proposal:** `{"motion_blur": {"enabled": false, "shutter": 0.0, "upper_bound": 0.2}}`. If the lead prefers a token blur for parity with clips 01–02, use ≤ 0.1, CENTER.
* build.py reads config/render.json; blur is OFF when the file is absent.

## Look v016 (frame 0, camera of record, EEVEE, AgX Base Contrast, exposure −0.5)
**Region metrics (Oklab).** Mean |dL| 0.031, max 0.058:
| region | dL |
|---|---|
| backdrop (left / centre / right) | +0.043 / −0.014 / +0.056 |
| table (left / centre / right) | +0.040 / +0.015 / +0.004 |
| posts | −0.023 / −0.002 |
| crossbar | +0.030 |
| foot | +0.058 |
| clamp | +0.051 |

**Backdrop and table 2D map** (32×18 Oklab L, 382 background cells): MAE 0.036, bias +0.025, correlation 0.94. v001 was MAE 0.155, correlation 0.66. The row profile reproduces the dark crease band, and the column profile reproduces the left-dark gradient.

**Disc highlight ring profile.**
* r 0.7: MAE 0.079, correlation 0.94.
* r 0.35: MAE 0.070, correlation 0.91.
* Rendered cones are slightly brighter and lower-contrast (std 0.078 vs 0.134).

**Wood texture.** Post hp_rms 0.038 vs 0.065; coherence 0.94 vs 0.92. The fine fleck contrast is about 60 % of the reference.

**Techniques.**
* **Turned brass (EEVEE lobes).** EEVEE has no anisotropic GGX. The turned-brass "cones" are emulated in the disc's object space by view lobes: bright where the radial direction lines up with the view azimuth. This is invariant under twist, like a real turned finish, so the cones stay with the camera while the mark and the ring streaks rotate. The Principled BSDF still carries the anisotropic tangent (circumferential), so a Cycles render gets the real effect.
* **Light linking.** The disc is lit only by its own linked softboxes (back, front, low rim). The key, fill and wall wash exclude the disc. The clamp strip and the screw strip are linked to their parts. The EEVEE light probe is disabled.
* **Crease band.** Modelled as albedo ramps: lower wall and far table.

## Open issues
1. **physics.json is still a placeholder.** ell is 0.0909 m vs the measured 0.0862 m (+5.5 %): baked builds hang the disc lower than the reference. Thickness 3.0 mm vs 3.1 mm. build.py prints a WARNING for both.
2. **Wood.** Fine fleck contrast is lower than the reference; the reference grain reads slightly more oak-like.
3. **Clamp.** +0.05 L. The front button screw has no hex socket or washer detail. The top screw heads are less chromy than in the reference.
4. **Backdrop at f240.** The crease line is sharper than the reference's soft band. The right side of the wall is +0.06 L at frame 0.
5. **Disc.** Ring streaks are a bit visible. The cone-gap hue is slightly olive. The mesh is level; the measured tilt and sway come only from the bake.
6. **Unverified look.** The look was tuned at frame 0 only. Frames 120/240 are checked visually in the blockouts, not by metrics.

## Requests
* **Physics (B):** use ell = 0.862 D (camera role 0.857 D; please reconcile, both are well above the placeholder 0.909 D × 0.1 m) and t = 0.031 D. The mark is a full diameter: mark.mu_body_rad defines one end, and the line runs through B. The contract above uses TOR_WireEnd and Euler XYZ as in bake_blender.py.
* **Lead:** set config/render.json motion blur as proposed above.

## Update 3 — VR03-06..10 look round (look v025, geometry v002 + front-screw detail; motion unchanged)

**Visuals:** `scene/visuals/v010_vr03_look_f000/f120/f240.jpg` — reference | ck001 | new, with close-ups and VR03 numbers. The new frames are the physics v003 bake through the camera of record (camera.json 5037746174b3).

**Changes by finding:**
* **VR03-06 (front screw, collet).**
  * New geometry: a chrome washer (r 0.031 D), a domed button head (r 0.022 D) and a dark hex socket (new part 6, object `TOR_ScrewSockets`).
  * Screws and collet are now chrome (roughness 0.06), lit by a small light-linked panel near the camera direction.
* **VR03-07 (brass).**
  * Brass C 0.034 → 0.046 (ref 0.0485). Ring-profile std 0.078 → 0.120 (ref 0.134); ring MAE 0.035 / 0.029 / 0.046 at f0 / f120 / f240.
  * How: disc-only softboxes are warm-tinted and light-linked to the disc, the white dielectric specular is removed from the brass (it set the gap floor), diffuse share is 15 %, cone gaps are deeper (dark 0), and ring streaks are about 3× weaker.
  * The rim strip is wide (10 × 1.5 D), giving a continuous bright front band.
* **VR03-08 (mark).** Width 0.0095 D. The VR measures 0.010 D; my luma-dip FWHM is 0.011 D including AI blur. Flat dark core.
  * The dip at f0 is still weaker than the reference (26 vs 42 luma, 1–2 px FWHM).
* **VR03-09 (backdrop/table).**
  * The reference wall gradient flips with the orbit (bright right at f0, bright left at f240). That is a WORLD-fixed pool behind the stand seen with parallax, so the wall wash is now a narrow pool centred at x = 0 on the wall (spread 60°).
  * Wall at y = 3.26 D: the reference max-gradient horizon back-projected through the camera of record.
  * Horizon rows (new vs ref): f0 615 vs 615 (ck001 629); f120 576 vs 582; f240 577 vs 578 (ck001 551).
  * Wall L (left / mid / right): f0 0.53 / 0.72 / 0.74 vs 0.50 / 0.72 / 0.72; f240 0.71 / 0.75 / 0.62 vs 0.69 / 0.74 / 0.60 (ck001 right +0.10).
  * Wall C 0.0105–0.011 vs 0.012–0.014, hue −12° vs −7…−11°.
  * Backdrop 2D MAE 0.034 / 0.027 / 0.027, correlation 0.94 / 0.93 / 0.94.
  * Crease ramps softer; table bump and mottle / 3.
* **VR03-10 (walnut).** Foot dL +0.025 (was about +0.12 in the VR). Crossbar +0.025. Streaks finer (130 per D).

**Checks on the new baked file** (scene/out/baked/torsion.blend):
* check_rig `--frames`, 961 samples: PASS (wire ends ≤ 2.3e-10 m, no intersections, including the new socket part).
* check_camera: PASS, 0.0003 px.
* Determinism (two clean baked builds): identical, including bake keyframe hashes.

**Still open:**
* Clamp +0.06 L.
* Mark dip.
* Cone gaps slightly brown/dark compared with the reference's golden gaps.
* Wall hue a few degrees bluer.
* Re-render this sheet when role A publishes the smoothed camera. `look/vr_compose.py` and `look/render_look.py --auto-crops` are re-runnable.

## Update 4 — final look round vs ck002 (look v028; motion and proportions unchanged)

**Visual:** `scene/visuals/v011_vr03_final_look_f000_f240.jpg` — reference | ck002 | new at frames 0 and 240, with clamp and disc close-ups (100 %) and the numbers.

**Changes:**
* **Brass (VR03-07).**
  * Soft cone edges (0–55°).
  * Less saturated base and disc lights.
  * New fine radial texture in the disc frame (angular noise, 140 per turn, ±30 %).
* **Front screw (VR03-06).**
  * Head flattened: `clamp_front_screw_height` 0.012 → 0.006 D. This is the only geometry value changed.
  * The front-screw region is now dark grey and rougher (shader keyed on object y < −0.131 D).
  * The collet has its own chrome material and a linked highlight strip.
* **Mark (VR03-08).** Width 0.010 D; the groove bump was lowered so the cavity reads dark.
* **Backdrop/table (VR03-09).** Wall +6 % then −3 % albedo; table +5 %; table fine bump ÷3.

**Numbers (reference → new, f0 / f240):**
| metric | f0 | f240 |
|---|---|---|
| brass C | 0.0485 → 0.0471 | 0.0451 → 0.0456 |
| brass C p05–p95 | 0.040–0.062 → 0.029–0.064 | 0.036–0.058 → 0.028–0.063 |
| fine/main-lobe power (harmonics 5–60 over harmonic 2, rings 0.4–0.8, 100 % crop) | 0.13 → 0.080 | 0.21 → 0.082 |
| brass L | 0.535 → 0.578 | 0.574 → 0.580 |
| mark dip | 41.7 → 35.8 (FWHM 1 px both) | — |
| backdrop 2D | MAE 0.036, corr 0.94 | MAE 0.029, corr 0.94 |

**Checks on the baked file:** check_rig `--frames` PASS (no intersections); check_camera 0.0003 px.

**Open:** the gap sectors are still greyer than the reference (low chroma p05); fine structure is about half the reference; the clamp region is +0.08 L.
