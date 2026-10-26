# camera/ — clip 03 torsion disc (Measure+Camera role A, v001 draft)

Camera of record: `params/camera.json` + `camera/project.py` (`load_camera(frame=n)`, n may be fractional) + per-frame table `camera/drift.csv`.
Diagnostic locked cameras: `kind="locked"` (frame-0 camera for all frames), `kind="top_ortho"` (orthographic straight down onto the mean disc
centre, image right = +X, up = +Y, ortho_scale 1.6 D), `kind="overview"` (3/4 from front-left above the crossbar, f = 2000 px). Blender settings in camera.json.

## Units / frame
World per config/coordinates.md: origin A = where the wire leaves the clamp; X along the crossbar (image right), Y away from camera, Z up.
**Length unit D = disc diameter = 1** (silhouette fit with top-face radius 0.5). metres_per_unit = null (model-conditioned; prompt says ~0.10 m).

## Verdict: the AI camera ORBITS (not a zoom)
- Static-mask homography vs frame 0 (s1_drift, as in clips 01/02) is NOT a valid model here: similarity-vs-homography residual 34 px median,
  ECC diverges after frame ~170; the anaglyph shows parallax between crossbar, posts and feet (visual `measurement/visuals/v001_camera_drift_*`).
- Per-frame pose: camera azimuth about A goes **+15.4° → −8.0° (23.4° orbit in 10 s)**, near-linear; distance to A 4.70 → 4.65 D (−1 %),
  elevation 6.9–7.3°, roll −0.16 → −0.30°, pitch ≈ −14.6…−15.6°.
- Method: (1) 14 rectilinear frame edges tracked sub-pixel in every frame (seeds propagated with the frame-to-frame KLT homography);
  (2) per-frame rotation from the three vanishing directions (posts = Z, crossbar = X, feet = Y), principal point at image centre, **constant f = 2602 px**;
  (3) per-frame position by a bundle over 7 edges that keep their physical identity through the orbit (crossbar front-top/-bottom left and right of the clamp,
  left-post front-outer/front-inner/inner-back corners; the right-post silhouettes switch corner when the camera crosses their face planes and are excluded)
  plus the wire-top pixel (image of A); scale gauge |C0−A| = 4.696 D from the frame-0 disc fit with A vertically above the frame-0 disc rotation centre [asm].
- f: free per-frame VP focal is 2630–2690 px over frames 0–140 and diverges later (X vanishing point → infinity as yaw → 0); a free-f bundle drifts to
  2547–2602 px. Adopted 2602 px; systematic range ~2550–2690 px (±3 %). Free principal point at frame 0 is degenerate (|f| 2265, pp (1061,456)).
- The AI frame is not perfectly rigid: crossbar edge depths differ left/right of the clamp by up to 0.05 D; inferred frame dims (bundle_lines.json,
  locate.json): crossbar height 0.20 D (top z +0.245, bottom z +0.040 above A), front face y ≈ −0.10; left post front corners x −1.053 / −0.852, inner-back y +0.12;
  right post (frame-0 planes) x +0.83 / +1.03. Inner span ≈ 1.70 D.

## Checks
- **R4 static reprojection (camera of record): 0.70 / 0.63 / 0.34 / 0.44 / 0.49 px at frames 0/60/120/180/240; all 241 frames median 0.58, max 0.91 px → PASS.**
  Locked frame-0 camera: 0.70 / 9.3 / 16.2 / 19.4 / 21.0 px. A (wire top) residual ≤ 1.4 px (frame 240 v).
- **Blender 5.2.2 projection validation: max 0.0043 px (PASS < 0.05)** — synthetic camera, reference frames 0/120/240, subframe 137.5, locked, top_ortho,
  overview (`camera/out/projection_validation.json`).

## Commands (from ROOT) — see measurement/run_all.sh
```
.venv/bin/python camera/edges.py camera/frames/f_000.png f000 ; .venv/bin/python camera/calibrate.py ; .venv/bin/python camera/calib_variants.py
.venv/bin/python measurement/scripts/s1d_edges_track.py ; .venv/bin/python camera/calib_per_frame.py --fix-f=2602
.venv/bin/python camera/locate.py ; .venv/bin/python camera/pose_bundle.py --f=2602 --fix-f
.venv/bin/python camera/write_params.py ; .venv/bin/python camera/reproject_check.py
tools/with_render_lock.sh blender --background --factory-startup --python camera/test/validate_projection_blender.py
```
Reused (copied, adapted) from clip 02: edges.py, project.py, write_params.py, reproject_check.py, drift_fit.py, test/validate_projection_blender.py;
new: calibrate.py (vanishing-point form), calib_per_frame.py, pose_bundle.py, locate.py.

## Open issues / requests
1. **Builder (C):** the scene camera must be animated per frame (location + rotation, constant lens) from camera/drift.csv or
   `project.blender_camera_settings(project.load_camera(frame=n))`; this is an orbit, not lens animation.
2. Gauge: A's depth assumes the wire is vertical over the frame-0 disc rotation centre; the measured mean disc centre is then (+0.014, −0.014) D from below A.
3. locate.py uses the line rotation-centre offset from a previous s4 run (bootstrap order); the value changed 0.035 → 0.031 D between runs (A shift ~0.004 D, not re-iterated).

## sha256 (first 16 hex; full list measurement/MANIFEST.sha256)
params/camera.json 84f7deb3b4693396 · camera/drift.csv 9ad7ff6d691a9b74 · camera/project.py d2b2c8c8f2313c08 · camera/pose_bundle.py a4ba5351832a1ed2 · camera/calib_per_frame.py e071996c87007225

## v002 — camera path smoothing (lead VR03-04)
- The raw per-frame pose (VP rotation + bundle position) carried frame-to-frame noise: right-post window jerk of the projected camera median 0.46 / max 4.08 px
  vs the reference's own image motion 0.09 / 0.67 px (same windows/metric as visual_review/scripts/camjitter.py; spikes at frames ~38, 58–63).
- `camera/smooth_path.py`: cubic penalised smoothing spline (lam = tau^3) on the 6 path parameters (azimuth, elevation, distance of C about A; rotation vector
  relative to frame 0), f fixed 2602 px. tau chosen by the data: least smoothing whose projected window motion matches the reference motion to within 0.002 px
  of the best (grid 0.05–5 s) with R4 ≤ 2 px → **tau = 1.2 s**. Max deviation from raw: azimuth 0.20°, elevation 0.12°, distance 0.009 D, rotation 0.22°
  — the measured orbit (+15.4° → −8.0°) is preserved.
- **Result:** right-post jerk smoothed camera median 0.002 / max 0.004 px (reference 0.09 / 0.67 px; the reference's residual jitter is AI/cadence
  and phase-correlation noise, not camera motion we should copy). Velocity rms vs reference motion 0.54 → 0.15 px/frame.
  **R4 per frame: median 0.57, max 0.90 px (frame 189) — PASS**; frames 0/60/120/180/240: 0.74/0.62/0.35/0.45/0.53 px. Blender projection max 0.0047 px PASS.
- Files: camera/out/camera_poses_raw.json (raw), camera/out/camera_poses.json (smoothed = camera of record), camera/out/camera_smoothing.json (grid, metrics, R4 per frame),
  params/camera.json + camera/drift.csv rewritten. Visual: measurement/visuals/v002_camera_jitter_smoothing.png.
- Note: columns view_pitch/yaw/roll_deg, vp_edge_rms_px, f_free_vp_px in drift.csv are the raw per-frame VP diagnostics; Cx..rotvec, blender_* are the smoothed camera.
