# camera/ — clip 02 vertical spring (Measure+Camera role A, v001 draft)

Camera of record: `params/camera.json` + `camera/project.py` (`load_camera(frame=n)`; per-frame zoom/pan from `camera/drift.csv`).
Locked diagnostic cameras: `load_camera(kind="locked")` (frame-0 camera for all frames), `kind="front_ortho"` (orthographic along +Y,
ortho_scale 3.0 W_in), `kind="overview"` (3/4 from front-right, slightly above the crossbar, f = 2400 px). All have Blender settings in camera.json.

## Visuals
- `visuals/v001_reprojection_f120_full.jpg`, `..._zooms.jpg`: frame 120, cyan = projected model edges (camera of record), red = measured
  sub-pixel edge points, yellow = residual x20, green + = origin A. Copy in `measurement/visuals/v001_reprojection_overlay.jpg`.
- `measurement/visuals/v001_camera_drift_raw_vs_corrected.jpg`, `v001_camera_drift_plot.png`: drift raw vs corrected, per-frame scale/pan/residuals.

## Units and frame
World per `config/coordinates.md`: origin A = hook bearing point on the crossbar top face at mid-depth, X along the crossbar (image right),
Y away from camera, Z up. **Length unit W_in = crossbar inner span (inner face to inner face of the posts) = 1.** Metres are not set
(`scale.metres_per_unit = null`): they are model-conditioned by the dynamics. Hint, not adopted: the simple-pendulum length for T = 1.3007 s
is 0.420 m and P_eff→weight centre is 0.998 W_in, so W_in ≈ 0.42–0.43 m. Set it with
`camera/write_params.py --metres-per-unit X --scale-source "..."`.

## Results
- f = **3555 px** (66.7 mm on 36 mm; hfov 30.2°). Formal sd 44 px. Over 26 calibration variants (edge jackknife, priors ×10 looser,
  internal-edge σ ×3, no internal edges, no plate ends) f ranges 3421–3849 px (sd 100). The cost profile is shallow: 3000–4200 px fit
  within 0.69 px RMS. f and distance trade off: distance to A is 5.06–5.69 W_in. A free principal point is degenerate: it moves to v = 377
  with f = 2605 at the same RMS. The principal point is fixed at the image centre.
- The camera centre is C = (−0.038, −5.259, −0.056) W_in. It sits between the crossbar top (z = 0) and bottom (z = −0.082), so neither
  face of the crossbar is visible. Pitch is −7.08° (looking down), yaw +0.49°, roll ≈ 0.
- Frame geometry implied by the calibration (W_in, in camera.json):
  - post width 0.174, post depth 0.201, post top z +0.012
  - crossbar bottom z −0.082, crossbar depth 0.08 (prior; weakly observed)
  - plate x −0.696…0.695, y −0.271…0.155, top z −1.420, bottom z −1.445
- Origin A projects to (956.8, 63.6) px at frame 0, just below the crossbar front-top edge (v ≈ 60). The hook apex is visible about 10 px higher.
- P_eff is the least-squares intersection of the 241 spring axes, at (954.4, 217.7) px in frame 0. On the plane Y = 0 that is
  (−0.003, 0, −0.225) W_in. The spring body swings about its top coil, not about A, and the S-hook stays static to within 0.64 px rms.
- **R4 static reprojection with the camera of record: 0.68 / 0.77 / 0.79 / 0.76 / 0.86 px at frames 0/60/120/180/240 (PASS ≤ 2 px).**
  The locked camera gives 0.68 / 5.0 / 10.1 / 14.8 / 19.4 px, so the drift is real. Points within 8 px of the border are excluded
  (the plate bottom leaves the frame during the push-in).
- **Blender 5.2.2 projection validation: max 0.0031 px (PASS < 0.05).** Checked on a synthetic camera, reference frames 0/120/240,
  subframe 137.5, and the locked, front-ortho and overview cameras (`out/projection_validation.json`).
- **Drift:** image scale +4.76 % from frame 0 to 240 (0.48 %/s, near-linear), fixed point near (973, 512) px. Pan up to −46 px (tx) and
  −25 px (ty) in similarity terms; roll < 0.06°. Similarity vs homography: 0.38 px rms median over the static region. Tracked landmarks
  vs similarity: 0.99 px median. Cubic-smooth vs raw: up to 0.66 px, so the **raw per-frame values are the camera of record**.
  Represented as zoom plus principal-point shift with a fixed camera centre. The frame posts and the side table scale equally within
  0.3 %, but the backdrop is too textureless to test, so a dolly is not excluded.

## Method / commands (from ROOT)
```
.venv/bin/python measurement/scripts/s1_drift.py            # H_ecc (frame 0 -> n) on static mask
.venv/bin/python camera/edges.py camera/frames/f_000.png f000   # 20 rectilinear edges, sub-pixel (frame 0)
.venv/bin/python camera/calibrate.py                         # box model of posts/crossbar/plate + pinhole
.venv/bin/python camera/calib_variants.py                    # uncertainty variants + f profile
.venv/bin/python camera/drift_fit.py                         # per-frame similarity from H + landmark check
.venv/bin/python camera/write_params.py                      # params/camera.json + camera/drift.csv
.venv/bin/python camera/reproject_check.py                   # R4 at frames 0/60/120/180/240 + visuals
tools/with_render_lock.sh blender --background --factory-startup --python camera/test/validate_projection_blender.py
```
Full pipeline: `measurement/run_all.sh`. `camera/crop.py` is the zoom-grid helper used to pick the edges.

## Assumptions / open issues
1. Pinhole camera, no distortion, principal point at the centre, axis-aligned sharp-edged boxes. Soft priors: posts centred on the
   crossbar in Y (σ 0.02), square post section (σ 15 %), plate centred in X (σ 0.02), crossbar depth 0.08 ± 0.04.
2. The internal edges (post front-inner, plate front-top highlight, post bottoms) carry 0.6–1.6 px rms and are down-weighted.
   The post-bottom/plate landmarks track 1–2 px worse than the top under the drift homography: parallax or AI drift near the frame edge.
3. Request to the lead/dynamics (pivot): A (hook on the bar) is static, but the spring axis rotates about P_eff, 0.225 W_in below A
   (top coil). Clip-01 lesson 4 says keep the physical hinge and place parts where the reference shows them. Proposal: model the hook
   as rigid and fixed (observed), with the spring/swing hinge at P_eff. The alternative is a hook hinged at A that visibly swings,
   which would contradict the reference.
4. Request to the builder: frame dimensions above are in W_in; use camera.json `frame_geometry_implied_by_calibration` with the metric
   factor once the dynamics sets it. The camera objects are the builder's; use `blender_camera_settings(load_camera(frame=n))` per frame.

## sha256 (key files; full list in measurement/MANIFEST.sha256)
See measurement/MANIFEST.sha256 (params/camera.json, camera/drift.csv, camera/project.py, camera/calibrate.py, camera/edges.py, camera/write_params.py, camera/drift_fit.py).
