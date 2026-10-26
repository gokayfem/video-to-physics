# scene/ — geometry, rig & look (role C) NOTES

Status: **v001–v008 integrated scene, draft**. Built from params; look pinned at look.json **v007**. The camera is the scene role's
APPROXIMATE frame-0 fit, because params/camera.json does not exist yet. The checkpoint file `scene/out/spring.blend` is built with
`--look` and is **not baked**: physics.json is still a placeholder (see open issues).

## Visuals (`scene/visuals/`, newest last)
| file | what changed | what remains wrong |
|---|---|---|
| v001_blockout_f000/f120/f240.jpg | GN frame, GN spring and weight through the approximate camera. The weight is posed at the reference's measured H (TEST pose). Edge overlay and close-ups. Frames 120 and 240 are warped by the measured 2D drift H_n | placeholder materials; top-hook S-bend shape differs; camera is static (approximate) |
| v002_look_compare_f000.jpg | look round 1: clip-01 walnut, brushed steel, set, AgX | everything +0.2–0.3 Oklab L too bright |
| v003_look_compare_f000.jpg | −1.2 EV, darker walnut, table grain along Y, tighter pool | backdrop centre and posts too bright |
| v004_look_compare_f000.jpg | strip light for the weight's vertical band, cathedral figure | the strip lit the posts and crossbar (regression) |
| v005_look_compare_f000.jpg | strip light-linked to weight/eye/spring only, backdrop wash | the wash also lit the table; wall edges too bright |
| v006_look_compare_f000.jpg | wash linked to the wall only; matte black steel | crossbar reads grey (+0.11 L) |
| v007_look_compare_f000.jpg (**pinned**) | glossy black steel, redder walnut, stronger table streaks | mean \|dL\| 0.038, max 0.072 (crossbar); wood figure and brushing contrast are lower than the reference |
| v008_spring_test_poses.jpg | spring at \|A−H\| 0.95 / 1.006 / 1.04 S and tilts of ±12° (TEST poses), look v007 | — |

## Build / check commands (from ROOT)
```
.venv/bin/python scene/measure/fit_frame0.py && .venv/bin/python scene/measure/make_geometry_params.py   # -> params/geometry.json
blender -b --factory-startup --python scene/build.py -- --out scene/out/spring.blend --look [--bake solver/out/trajectory.csv] [--pose "Hx,Hy,Hz,rx,ry,rz"]
blender -b scene/out/spring.blend --python scene/check_rig.py [-- --frames]     # -> scene/out/check_rig.json
blender -b scene/out/spring.blend --python scene/check_camera.py                # -> scene/out/check_camera.json
.venv/bin/python scene/compare_builds.py A.build.json B.build.json                                # determinism
look/round.sh vNNN <prev> "caption"      # look round: build --look, render (render lock), look/metrics.py, look/compose.py
tools/with_render_lock.sh blender -b X.blend --python scene/preview_render.py -- --frame N --pose-measured --out X.png
.venv/bin/python scene/make_visual.py --render X.png --frame N --version vNNN --warp-drift --out scene/visuals/...
```

## Object-name contract for the bake (matches solver/bake_blender.py, which runs in-process via runpy)
| object | what it is | bake |
|---|---|---|
| `VSP_Anchor` | empty at A = world origin | never animated (bake checks it) |
| `VSP_Weight` | mesh, **origin = hinge H** (eye inner top = lower-loop inner bottom); body axis = local −Z; cylinder centre at local (0,0,−b_c), b_c = geometry.json `derived.H_to_cylinder_centre` × S_m | `location` = H(t), `rotation_euler` XYZ |
| `VSP_WeightEye` | child of VSP_Weight; torus in the local XZ plane, inner top at the origin | — |
| `VSP_HookLower` | empty, unparented. It has a COPY_LOCATION constraint → VSP_Weight, so the spring's lower end has a single source | keys allowed (the bake keys it) but overridden |
| `VSP_Spring` | GN `VSP Spring`; reads VSP_HookLower | — |
| `VSP_Cam_Reference` | scene camera | — |

Bake test (`scene/out/baked_test/`, placeholder trajectory): bake_blender check passes (P6 1e-5; hook/weight error 1.5e-8 m).
check_rig `--frames` also confirms the attachments (≤ 7e-8 m) and the 22 turns on all 241 frames. But **10 frames FAIL on
weight↔plate intersection**: physics' placeholder b = 0.042 m against our measured 0.068 m, and its swing amplitude, put the
rendered body into the plate.

## Geometry (params/geometry.json v001)
* Unit **S** = clear span between the posts' inner faces. It is a scene-role choice; camera/NOTES.md had no unit yet, so the refit
  through the camera of record is pending.
* Joint fit at frame 0: 11 box-model parameters + a 7-parameter pinhole (principal point fixed at the image centre). 61
  observations (18 edge lines from luma-gradient maxima plus the A gauge), rms 0.53 σ.
* Edge residuals are ≤ 1.4 px, except the inner-front post edges (2.6 px, weak edges) and the plate's rounded front corners
  (2.1 px).
* Fitted camera: f = 3225 px (60.5 mm on a 36 mm sensor), pitch 6.4° down, yaw −0.6°, roll 0.18°, 4.8 S in front of A.
* Spring and weight sizes are read on 2–4× zooms. They are converted to S by ray/plane intersection at y = 0; frames 120 and 240
  are mapped to frame 0 first with measurement's ECC homography.
* Key ratios (S): post 0.177 × 0.162 (depth weakly observed), post top +0.011, base top −1.427, bar h 0.082 (bar depth 0.041
  ASSUMED), plate 1.383 × 0.418 × 0.026.
* Spring: coil mean R 0.054, wire r 0.0048, hook wire r 0.006. 12 open + 10 tight turns (counted). Top hardware 0.232, open
  0.486, tight 0.182 and bottom hardware 0.106 S at |A−H| = 1.0058 S (frame 120).
* Weight: R 0.114, h 0.287, eye R 0.042 / r 0.0075. b_c = 0.228 S.
* Measured |A−H| is 1.008 / 1.006 / 1.008 S at frames 0 / 120 / 240: the AI spring barely changes length. H_x is −0.105 / −0.087
  / +0.060 S (lateral swing).
* Gauge: S_m = physics.json ell_eq / 1.00582. It is currently 0.2983 m (physics status *placeholder*).

## Spring rig (GN `VSP Spring`)
* Axis A→H; the coil count is fixed (an integer number of open + tight spiral turns).
* Pitches: every turn takes the same extension δ = (L_body − L_body,ref)/N, so stiffness per turn is uniform. Tight turns clamp at
  solid pitch (2 r_w).
* The rigid top loop (rounded rectangle in the YZ plane) rests on the bar top at A. The rigid lower loop's inner bottom is at H.
  Bezier transitions join the loops to the coils.
* The whole spring is rotated by Align-Rotation-to-Vector (local Z → A−H). Point attributes `vsp_part` and `vsp_cl` (centreline)
  feed check_rig.
* The tight lower section copies the reference's look. Uniform per-turn extension keeps it physically consistent, and it does
  not affect dynamics (k is fitted).

## check_rig (scene/out/spring.blend, 81 TEST poses: |A−H| 0.95/1.006/1.04 S × in-plane tilt ±12° × out-of-plane ±2° × weight rock ±8°) — **PASS**
| quantity | value | tolerance |
|---|---|---|
| top-loop inner top − A | ≤ 2.0e-9 m (lateral 2.1e-9) | 1e-6 |
| lower-loop inner bottom − H | ≤ 2.5e-8 m (lateral 6.7e-9) | 1e-6 |
| eye inner top − H | ≤ 7.5e-9 m | 1e-6 |
| coil turns | 22.0000005 at every pose | ±1e-3 |
| min coil centreline gap / 2 r_w | 1.62 | ≥ 1 |
| loop↔eye overlap | ≤ 0.026 mm (only when the weight rocks against the axis; fixed-hinge model) | 0.05 mm |
| spring/weight/eye vertices inside bar/posts/plate/weight | 0 | 0 |
| min weight–plate / weight–posts | 0.66 mm (1.04 S with ±12° tilt) / 14.4 mm | > 0 |
| top loop dip into the bar | 2.4e-6 m in-plane; 0.21 mm at ±2° out-of-plane (pivot at A, not at the bar edge; reported) | allowance (d_b/2)·sin\|ty\| |

## check_camera — PASS
max \|Blender − reference projection\| = 0.00033 px at frames 0, 120, 240 and 60.5. The reference projection is the APPROXIMATE
frame-0 camera (geometry.json). Once params/camera.json and camera/project.py exist, build.py builds VSP_Cam_Reference from
`project.load_camera(frame=n)` / `blender_camera_settings` (clip-01 API), keyed per frame, and check_camera switches to
project.py automatically.

## Determinism
Two clean builds (`--look`) in separate processes are identical: 20 objects, graph dump, per-object evaluated mesh hashes
(positions + topology), transforms, dims and notes (`scene/out/det_a|det_b/spring.build.json`). The meshes come from explicit
lists, and build.py sets save_version = 0, so no .blend1 is written.

## Look v007 metrics (frame 0, 960×544, Oklab; render EEVEE AgX Base Contrast, exposure −1.2)
| region | L ref | L cur | dL | C ref/cur |
|---|---|---|---|---|
| backdrop left / centre / right | 0.479 / 0.643 / 0.469 | 0.469 / 0.682 / 0.462 | −0.011 / +0.039 / −0.007 | 0.016/0.017 |
| table left / right | 0.708 / 0.621 | 0.677 / 0.660 | −0.031 / +0.040 | 0.034/0.035, 0.038/0.036 |
| post left / right (walnut) | 0.334 / 0.329 | 0.390 / 0.373 | +0.056 / +0.044 | 0.050/0.043 |
| crossbar / plate top / plate front | 0.225 / 0.283 / 0.164 | 0.297 / 0.334 / 0.223 | +0.072 / +0.051 / +0.059 | ~0.01 |
| weight side / spring coils | 0.538 / 0.611 | 0.545 / 0.652 | +0.008 / +0.041 | 0.028/0.021 |
Mean \|dL\| 0.038 (v002: 0.218). 16×9 luma grid: backdrop MAE 0.023 (corr 0.91), table MAE 0.035.
Texture (ref | cur): post hp_rms 0.0147 | 0.0107, coherence 0.80 | 0.84, grain 87° | 89°; table grain 47° | 44°, hp 0.014 | 0.005;
weight hp 0.024 | 0.012.

## Open issues
1. **Camera of record pending.** The approximate camera is static; the reference pushes in ~4 % by frame 240 (warped overlays
   show the drift). Refit the proportions through camera.json when it lands, and reconcile the unit with camera/NOTES.md.
2. **physics.json is a placeholder.** b = 0.042 m vs measured 0.068 m; ratios r/ell 0.085/0.20/0.14 vs measured 0.1138 / 0.2853 /
   0.2268. Baked placeholder frames intersect the plate.
3. Top hook: our loop is physically in the bar's YZ plane, so it reads edge-on. The reference draws a visible ring floating
   ~6 px above the bar, and a different S-bend into the first coil.
4. Look: the walnut figure and the weight's brushing are about half the reference's fine-texture contrast. The crossbar and
   plate are still +0.05–0.07 L. The table texture is softer than the reference.
5. Out-of-plane rocking of the top loop is simplified (pivot at A; ≤ 0.21 mm dip at 2°). Rocking of the eye on the loop wire uses
   a fixed hinge (≤ 0.026 mm overlap).
6. Bar depth, loop widths and screw positions are assumed or approximate (flagged in geometry.json).

## Requests
* **Physics (B):** set `--ratio-r 0.1138 --ratio-h 0.2853 --ratio-bc 0.2268` (to ell = |A−H| at frame 120). Spring: d/ell 0.00963,
  D/ell 0.1074, 22 turns. Optionally write top-level `S_m`. The base-plate top is 1.427 S below A, so rest clearance is
  ~0.054 S = 16 mm.
* **Camera (A):** please publish the unit / metre gauge in camera.json `scale` (S_m, or a ratio to S) and keep the clip-01
  project.py API.

---
## Update 2 (lead: camera of record, hinge at P_eff, scale, motion blur) — rig v002, look v010

### Visuals
| file | what changed | what remains wrong |
|---|---|---|
| v009_look_compare_f000.jpg | look round 7, the first through the camera of record and rig v002. Black steel is darker and less specular; walnut, brushed-steel and table texture contrast is up. Mean \|dL\| 0.028, max 0.069 (table left) | walnut chroma low (0.042 vs 0.050); weight brushing coherence low |
| v010_camref_f000/f120/f240.jpg | rendered through **VSP_Cam_Reference keyed from camera/project.py** (no warping), with edge overlay and close-ups. The weight is a TEST pose from the reference's measured H and body axis | coil appearance (see v011); the reference loop floats ~6 px above the bar |
| v011_spring_closeups_camref.jpg | full-res spring crops at frames 0/120/240: reference \| v010 \| v011 (spring steel brightened, look v010) | the reference draws coils as flat stacked rings; a real helix at 29 px pitch / 75 px diameter shows sloped front strands |
| v012_motion_blur_evidence.jpg | reference fast/slow frames vs renders at 24 px/frame with shutters 0 / 0.1 / 0.2 / 0.5 | proposal only; the lead sets config/render.json |

### (1) Camera of record + refit
* **Cameras.** `build.py` keys VSP_Cam_Reference on every frame (location, rotation, lens, shift; LINEAR) from
  `blender_camera_settings(load_camera(frame=n))`. camera.json lengths are W_in, multiplied by S_m.
  * Diagnostics VSP_Cam_Locked / VSP_Cam_FrontOrtho / VSP_Cam_Overview come from `load_camera(kind=...)` with
    lock_location/rotation/scale = True.
* **check_camera** switches to project.py automatically: max 0.0002 px at frames 0/120/240/60.5. PASS.
* **Refit.** `scene/measure/fit_fixedcam.py`: camera FIXED, 11 geometry parameters, rms 0.54 σ.
  * The unit W_in equals my S (clear span between the posts' inner faces), so no rescaling was needed.
  * Changes vs v001: d_p 0.188, y_plate −0.047, base top −1.4228, table edge y 1.48. Other ratios changed < 1 %.
* **Weight size.** Now taken from measurement's 241-frame silhouette track: r 0.1058 W_in, h 0.260 W_in.
  * h is the silhouette length minus D·sin(view elevation 9.6°).
  * My v001 eye readings were about 5 % too large; a luma profile across the weight confirms ~146 px at f120.
  * b_c = H → cylinder centre = 0.2126 W_in.

### (2) Spring rig v002 (lead CHANGELOG 2026-10-26T00:52Z)
* **STATIC:** top loop on the bar at A, S-bend, and **2 hook turns** ending at **P_eff** = (−0.0029, 0, −0.2251) W_in
  (camera.json key_points).
* **MOVING:** **11 open turns** (0.476 W_in from P), **10 tight turns** (0.199), and bottom hardware (0.109) down to the
  lower loop whose inner bottom is at H. Section lengths are means over 241 frames from measurement/spring_track.csv
  (`scene/measure/spring_layout.py`); the open pitch is 29 px at frame 0.
  * The moving part is built hanging straight below P_eff and rotated about P_eff onto P→H.
  * The rotation ramps in with a smoothstep over the bend length below P (0.013 W_in, ASSUMED). So the spring bends at its
    top coil and the wire stays continuous with the static hook turns.
  * The swing angle is computed with atan2(|a×b|, a·b). An acos version lost ~2e-5 m near vertical in float32; the bake
    test exposed it.
* **Objects.** New empty `VSP_HingePeff`. Other object names are unchanged. Rest H = P_eff + (0, 0, −0.7835) W_in.
* **check_rig v002** (81 poses: |P−H| 0.75 / 0.7835 / 0.82 W_in × tilt ±12° about P × ±2° out of plane × rock ±8°): **PASS**.

| quantity | value | tolerance |
|---|---|---|
| static parts moved | 3.7e-9 m | 1e-6 |
| moving part rigid about P (fully rotated points) | 5.5e-8 m | 1e-6 |
| top loop on A | 4.8e-9 m | 1e-6 |
| lower loop on H | 3.8e-8 m (lateral 3.3e-7) | 1e-6 |
| eye on H | 1.7e-10 m | 1e-6 |
| turns | 23.0000005 at every pose | ±1e-3 |
| min coil gap | 1.05 × 2 r_w (the hook turns) | ≥ 1 |
| top loop vs bar | +0.037 mm | ≥ 0 |
| intersections | 0 | 0 |
| min weight–plate / weight–posts | 6.7 mm / 33 mm | > 0 |

* **Bake test.** With `--bake solver/out/trajectory.csv` (physics placeholder, now using the measured ratios), `check_rig
  --frames` PASSES on all 241 frames (`scene/out/baked_test/`).

### (3) Scale
* `gauge()` order:
  1. physics.json `S_m` / `metres_per_unit`;
  2. physics.json `camera_scale.camera_units_per_m` (inverted);
  3. camera.json `scale.metres_per_unit`;
  4. ell_eq / |A−H|;
  5. assumed.
* It currently reads physics `camera_scale.camera_units_per_m = 3.333` (provenance *assumed*, status *placeholder*), so
  S_m = 0.300 m per W_in. The source is recorded in every build report.

### (4) Motion blur
* `build.py motion_blur()` reads `config/render.json` `{"motion_blur": {"enabled", "shutter", "position", "eevee_steps"}}`.
  While the file is absent, or the shutter is null or disabled, blur is OFF and the build notes say so.
* **Reference measurement** (`scene/measure/mb_reference.py`): effective edge width = step / max slope of the weight's
  left/right silhouette edges, with rows averaged along the tilted edge.
  * Over 239 frames: width = 2.25 + 0.069 × speed px (residual rms 0.68 px); speed up to 25 px/frame.
  * Without tilt compensation the metric was confounded by the weight's tilt (slow frames = tilted).
* **Calibration renders** (`scene/measure/mb_tune_render.py` / `mb_tune_analyze.py`, look v010): render slope 0.078 /
  0.188 / 0.313 / 0.334 at shutter 0.1 / 0.2 / 0.35 / 0.5. The static width (2.0 px) matches the reference (2.25 px).
* **Proposed `config/render.json`:** `{"motion_blur": {"enabled": true, "shutter": 0.09, "position": "CENTER",
  "uncertainty": 0.05, "evidence": "scene/NOTES.md Update 2; scene/measure/mb_tune.json"}}`. That is about a 32° shutter;
  the reference is much sharper than clip 01's 0.5.

### Look v010 (pinned)
v009 values plus brighter spring/eye steel. Frame-0 metrics (v009 render; v010 changes only the spring/eye materials):
* mean |dL| 0.028 (v007 0.038), max 0.069.
* crossbar +0.019 (was +0.072), plate top +0.031, plate front +0.024, posts +0.045/+0.048, weight −0.007, spring coils +0.021.
* backdrop grid MAE 0.024, table MAE 0.036.
* Texture hp_rms (ref | cur): post 0.0147 | 0.0134, weight 0.024 | 0.018, table 0.014 | 0.0065.

### Determinism
Two clean `--look` builds are identical: 24 objects, mesh hashes, transforms, graph dump and camera keys.

### Open issues
1. physics.json is still a placeholder: gauge assumed; b 0.0680 m vs the mesh's 0.0638 m (measured r/h changed in v002, see
   the request below); trajectory anchored by dynamics' model.
2. The reference draws coils as flat rings (AI drawing); ours are true helices. The reference's dense section is ~1.8 wire
   pitch; ours is 2.0. The reference's top loop floats ~6 px above the bar.
3. Look: walnut chroma is still low; table fine texture is half the reference's; the weight brushing is less coherent.
4. The bend length (0.013 W_in) is an assumption. A smaller value concentrates the kink at P.
5. Motion blur is a proposal (0.09 ± 0.05); it is not applied until config/render.json exists.

### Requests
* **Physics (B):** weight ratios to ell = |A−H| at frame 120 (1.0096 W_in), v002: **r/ell 0.1048, h/ell 0.2579,
  b_c/ell 0.2105**. In W_in: r 0.1058, h 0.2603, b_c 0.2126. Spring: 2 static hook turns, then 11 open + 10 tight turns
  below P_eff; wire d 0.0097, coil D 0.108 W_in. Swing hinge P_eff = (−0.0029, 0, −0.2251) W_in. Please publish metres per
  W_in as a top-level `S_m` (or `metres_per_unit`) once fitted.

---
## Update 3 (lead: visual review ck001, VR-04..VR-08) — geometry v003, look v015

### Visuals
| file | what changed | what remains |
|---|---|---|
| v013_look_compare_f000.jpg, v014_look_compare_f000.jpg | look rounds 8 and 9 (test poses, frame 0) | — |
| **v016_review_fixes_f000/f120/f240.jpg** | REFERENCE \| ck001 (as reviewed) \| NEW, all baked from the same ck001 trajectory, so only scene/look differ. Close-ups of hook+spring, walnut post and weight. Region dL, the backdrop profile metric and hue | weight hue (see open issues); crossbar/plate top +0.04–0.05 L |
| v015_review_fixes_f*.jpg | intermediate (look v014) | superseded by v016 |

### Fixes
* **VR-04 backdrop.**
  * The reviewer's low peak (1087, 746) px is the wall point (0.21, 1.48, −1.27) W_in, just above the table. The new
    `backdrop_pool` is a disk 0.9 W_in in front of the wall, aimed there and linked to the wall only. The wall wash went
    21 → 12.5.
  * New metric `look/backdrop_profile.py`: 47 backdrop-only points (rows 300/500/700, columns 700/1200, the low peak),
    mapped per frame by the drift homography.
  * **Mean \|dL\| 0.080 / 0.079 / 0.074 (ck001) → 0.025 / 0.024 / 0.029** at frames 0/120/240. The low peak went from
    0.53 to 0.765 (reference 0.763). Column u = 700 now rises downward (0.53 → 0.72; reference 0.60 → 0.70).
* **VR-05 weight.**
  * Steel: anisotropic 0.9, roughness 0.18, cool-tinted base.
  * Lights linked to the weight/eye: a second vertical strip (right), a top light (rim and top-face ellipse) and a soft
    frontal reflection. The black steel is rougher, so the plate stops mirroring the new pool.
  * weight_side dL: −0.011 → −0.008 (f0), −0.010 → −0.002 (f120).
* **VR-06 walnut.** Colours scaled R×0.9, G×0.69, B×0.63 overall, plus cathedral figure (amount 0.5, bend 0.12).
  * Posts dL +0.049/+0.045 → +0.034/+0.008 at f0 (similar at f120 and f240).
  * Hue 52° → 46° (reference 40°).
* **VR-07 table / plate.** A table-left light linked to the table only; right fill 15 → 9.
  * Table left −0.069 → −0.005 (f0), −0.073 → −0.009 (f120).
  * Plate front +0.026 → +0.023 at f0 (≈0 at f120); plate top is +0.035…+0.05.
* **VR-08 hardware** (geometry v003):
  * hook/loop wire radius 0.0060 → **0.01077 W_in** (14.5 px wide at 673 px/W_in, the reviewer's 14–15 px);
  * eye wire 10 → 12 px band, r_e 0.00883 W_in. The outer eye diameter is kept, so the centreline R_e is now 0.0397 W_in;
  * H_to_cylinder_centre 0.2096 W_in. spring_layout.json and measured_poses.json were regenerated; H moves 0.003 W_in.
  * VR-16 (tight pitch) is not changed. It would need more tight turns, i.e. a counted-geometry change.
* **Mean \|dL\| over 11 regions:** ck001 0.028 / 0.029 / 0.038 → new 0.019 / 0.019 / 0.027 at frames 0/120/240.

### Checks (final scene/out/spring.blend: look v015, geometry v003, S_m 0.4166 m per W_in from physics.json *fitted*; motion blur ON at shutter 0.09 from config/render.json)
* **check_rig: PASS on 81 poses**, and `--frames` on the ck001-trajectory bake: **241/241 PASS**.
  * Attachments ≤ 1e-7 m; 23 turns at every pose and frame; intersections 0; min weight–plate 10.5 mm.
  * The loop/eye tolerance is now the analytic fixed-hinge bound (r_h + r_e)(1 − cos 8°) + 1e-5 m = 0.09 mm. The old fixed
    0.05 mm no longer covered the thicker wires at the fitted scale. Measured overlap: 0.076 mm, only in the 8° rock test
    poses.
* **check_camera: PASS**, 0.00027 px.
* **Determinism:** two clean builds identical (28 objects).

### Open issues
* The weight hue stays 64–67° (reference 50°) even with a cool base. The region picks up the warm table/backdrop
  reflections; a further fix needs reflection control (e.g. a darker/cooler surround in front of the weight).
* Weight region at f240 −0.09 L: the baked weight is at a different phase from the reference there (dynamics), so the box
  catches backdrop.
* Left post +0.034, crossbar +0.045, plate top +0.04–0.05 L.
* Physics: b now differs by 1.4 % (physics 0.08855 m vs mesh 0.08733 m) after the eye-wire change. The new ratio is
  b_c = 0.2096 W_in.

---
## Update 4 (lead: re-review ck002 — final bounded look round) — look v016, rig v002 + lower-loop wire

### Visuals
| file | what |
|---|---|
| **v017_review_fixes_f000.jpg, v017_review_fixes_f120.jpg** | REFERENCE \| ck002 \| NEW, same baked trajectory (ck002), close-ups, region dL table, backdrop metric |
| **v017_weight_closeup.jpg** | weight + lower loop, 100 % crops at frames 0 and 120 (ref \| ck002 \| new) |
| **v017_weight_steel_profile.png** | the reviewer's weight-steel profile metric (re-implemented in look/weight_steel_local.py from visual_review/scripts/weight_steel.py, same sampling), frames 0 and 120 |

### VR-05 weight steel (frames 0 + 120, reviewer metric)
| | mean L | L < 0.35 | chroma | centre L | bands at −/+0.35 | flanks (\|x\| 0.6–0.75) |
|---|---|---|---|---|---|---|
| reference | 0.489 | 16.6 % | 0.029 | 0.47 | 0.79 / 0.67 | 0.38 |
| ck002 | 0.459 | 34.3 % | 0.019 | 0.61 | 0.49 / 0.53 | 0.31 |
| **v016** | 0.507 | 12.5 % | 0.027 | 0.49 | 0.70 / 0.58 | 0.44 |

* **Cause of the beige central lobe:** the EEVEE sphere light probe at the weight, which mirrored the lit table. Probe
  disabled. Object glossy visibility had no effect in EEVEE, tested.
* **Material:** satin anisotropic warm steel (roughness 0.24, anisotropy 0.75, base 0.72/0.60/0.46).
* **Lighting:** key, softbox, fill and top_back now EXCLUDE the weight (`flag_off_weight`). The weight is lit only by
  light-linked lights:
  * two narrow vertical strips at ±41° off the camera azimuth, aimed at the weight, so their reflections land at ±0.35 of
    the width;
  * dim side panels;
  * a broad dim front panel;
  * a top light behind (top-face ellipse/rim).
  * `weight_front` (the v014 central lobe) was removed.
* **Remaining:** the bands are still about 0.1 L weaker than the reference; the flanks are +0.06.

### VR-18 lower loop
* The new GN input `Lower Loop Wire Radius` is set to the spring wire radius (0.00484 W_in, ~6.5 px). It is used by the
  lower loop and the bottom transition.
* The top loop/S-bend keep the 14.5 px hook wire (VR-08). check_rig uses this radius for the H attachment and the link test.

### VR-07 crossbar
* Own material `VSP_CrossbarSteel`: glossy warm-neutral black dielectric (roughness 0.15, specular 0.2, warm specular tint).
  The plate keeps VSP_BlackSteel.
* Crossbar (left half) dL +0.047 → +0.004, hue 67° → 56° (reference 38–50°). The right half is still +0.10 (ck002 was also
  +0.10).
* Region mean \|dL\| (11 regions): ck002 0.019 / 0.019 → v016 0.016 / 0.018 (frames 0 / 120). Backdrop metric unchanged
  (0.025). The optional backdrop items were not done.

### Checks (scene/out/spring.blend, look v016, S_m 0.4166 from physics.json fitted, motion blur 0.09 from config/render.json)
* **check_rig: PASS**, 81 poses. On the ck002-trajectory bake (look/out/review/new_v016.blend), `--frames`: **241/241 PASS**.
  23 turns, 0 intersections, min weight–plate 10.5 mm, lower-loop attachment 5.6e-8 m.
* **check_camera: PASS**, 0.00024 px.
* **Determinism:** two clean builds identical (29 objects).

### Tools added
* `look/weight_steel_local.py`
* `look/render_review.py --no-crops`
* `look/review_compose.py` (`--old-label` / `--new-label` / `--title`)
* `look.py`: `flag_off_weight`, `receivers`, `specular_tint`, `table_visible_glossy`, and the `crossbar_steel` block.
