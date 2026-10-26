# measurement/ — clip 02 vertical spring, reference measurement (v001, draft)

Role A (Measure+Camera). Image-space measurement of the AI reference `WS/reference-batch/clips/02-vertical-spring.mp4`
(sha256 fee726c2…6865cd verified; 241 frames, 1920x1088, 24 fps, H.264 yuv420p 8.3 Mbit/s, all frames decoded with OpenCV).
Pixels: CONTRACT (top-left corner origin, centres +0.5). Time: frame n ↔ t = n/24 s.
Labels: [obs] measured, [fit] fitted, [inf] inference, [asm] assumption.

## Visuals (measurement/visuals/, burned-in version / frame-time / "reference measurement - draft")
| file | shows |
|---|---|
| `v001_annotated_f000_all_tracks.jpg` | frame 0 with all 241 frames of tracks in frame-0 geometry: weight centre, eye-ring centre, coil-body bottom/top, P_eff, A, spring axis + weight silhouette at both extremes, static landmarks |
| `v001_turning_points_zero_crossings.jpg` | t=0, first R/L turning points, two zero crossings, largest TP, last TP, end — each in its own (raw) geometry with weight silhouette, ring, spring axis |
| `v001_camera_drift_raw_vs_corrected.jpg`, `v001_camera_drift_plot.png` | anaglyph f0 vs f240 raw/corrected + zoom; per-frame scale, pan, residuals |
| `v001_reprojection_overlay.jpg` | camera of record at frame 120 (static RMS 0.79 px) with zooms |
| `v001_amplitude_period_per_halfcycle_Ta.png` | swing angle with TPs/ZCs; swing amplitude and half-period per half-cycle; rocking amplitude; r(t) and spring length; T_a = 10 s and held-out 7.5–10 s marked |
| `v001_spring_coil_crops.jpg` | coil body rectified along its axis every 30 frames with gap ticks / close-wound start / body end |
| `v001_tracking_methods_and_cadence.png` | method differences (weight, ring, spring tilt) and the n mod 4 step pattern |

## Key results
- **Camera [obs]: not static.** Push-in +4.76 % over 10 s (near-linear), pan ≤ 46 px, roll < 0.06°. See `camera/NOTES.md`.
  The S-hook stays fixed to the frame within 0.64 px rms after drift removal.
- **Motion [obs]: a lateral pendulum swing, no vertical bob.**
  - Swing angle of the ring about P_eff (image) is ±7.5–8.2°; in 3D, on plane Y = 0, θ = −8.1…+8.4°.
  - Period **1.3007 ± 0.0007 s** (damped-sine fit, χ²ν = 68, model inadequate); 1.3025 ± 0.0029 s from zero crossings.
  - At t = 0 the weight is essentially at its left turning point: φ = −7.27°, TP at t = 0.047 s (−7.48°).
- **Vertical bob [obs]: absent.**
  - The P_eff–ring distance is 546 ± 1.2 px. Its spectrum shows only 0.767 Hz (swing) and 1.533 Hz (2× swing).
  - Regressing it on φ, φ², φ̇² leaves 0.97 px rms (σ ≈ 0.56), and that remainder sits at the swing frequency.
  - Spring coil-body length (stabilized) is 460.2 ± 1.6 px, also swing-locked.
  - The prompt's ~1 cm decaying bob does not occur. The frequency ratio ω_axial/ω_swing is not measurable.
- **Weight rocking [obs]:** the weight tilt follows the swing with gain 1.29 (method B) / 1.41 (method C), plus a static offset of
  +1.3° (±1.8° systematic). So the weight over-rotates by about 2.5–3° beyond the pivot–ring line.
  [inf] This matches the in-phase mode of a double pendulum (spring link plus weight hinged at the eye-ring).
  The spring axis follows the P_eff→ring line with gain 0.96.
- **Out-of-plane [obs/inf]: no consistent evidence.** Depth from weight width gives yaw −13.7 ± 3.3°; depth from ring radius gives
  +9.7 ± 3.5°. The opposite signs point to systematics, and the elliptical (quadrature) terms are ≈ 0.
  [asm] The swing is taken to lie in the plane Y = 0 through A; |yaw| ≲ 15° is not excluded.
- **Spin [obs]:** not measurable. The cylinder is brushed and axisymmetric; its highlight bands do not follow the body, so photometric
  ECC fails.
- **Amplitude/period per half-cycle [obs]:**
  - Swing amplitude grows 7.75 (first half-cycle) → 7.86 → 8.22° by t ≈ 3.5–4.2 s (+4.5 %, 0.13 ± 0.02 °/s). It then drops to
    7.61° by 6.8 s and stays at 7.6°. A passive system does not gain energy, so this is non-physical injection.
  - Half-period lengthens 0.624 → 0.671 s (+7.5 %, trend 0.0038 ± 0.0006 s/s): period wander.
- **Cadence [obs]:** same generator cadence as clip 01. Offsets by n mod 4 are +0.274, +0.098, −0.094, −0.278 frames (sd 0.008), and
  steps into n ≡ 0 mod 4 are 1.67×. Diagnostic only.
- **Coils [obs]:**
  - 2 top turns at the S-hook, 11 open turns (pitch 29 px), then a close-wound section of about 10 turns (right-edge scallops).
  - A 12th open gap appears in 67 frames, only when the spring is swung to image-right (mean +6.3° vs −2.2°), and reversibly
    (16 transitions). It is the boundary turn opening on the outside of the bend, **not a coil-count change**.
- **Discontinuities [obs]:** none visible. The automated acceleration screen flags only:
  - ring-radius outliers (frame 116: method A −3 px, but a 2× zoom shows an unchanged ring)
  - weight width/tilt outliers at 75–77 and 135–144, which coincide with motion blur and cadence frames.
  The weight silhouette length is constant (sd 0.8 px).

## Anchor window (rule CHANGELOG 2026-10-25T23:41:09Z, applied as declared)
- **T_a = 10.0 s, with the last 2.5 s (7.5–10 s) held out.** No mode has a half-cycle amplitude increase > 3σ, and there is no visible
  discontinuity.
- Swing: the largest step is +2.66σ, at the turning point t = 1.93 s (7.886 → 8.056°). Including the start-of-clip half-cycle,
  0.047→0.664 s, gives z = 1.4.
- Weight rocking: max +0.9σ.
- r(t) is not an oscillation mode. If the rule were applied literally to its raw turning points, it would trigger at t = 9.37 s.
- Flag for the lead: the rule tests only half-cycle-to-half-cycle jumps. The gradual +4.5 % growth (1–4 s) and the 7.5 % half-period
  drift fall inside the 10 s window, so the fit sees non-physical behaviour. Only the lead can amend the rule (e.g. cumulative growth);
  this note does not change it.

## Tracks (all 241 frames, two methods, per-frame sigma)
- `weight_track.csv`: weight silhouette centre (u, v) and tilt (image angle of the downward axis from vertical, + = bottom toward +u),
  raw and in frame-0 geometry, with width/length and sigma.
  - B (primary): silhouette profiles + Tukey line fits.
  - C (check): Canny + RANSAC lines.
  - B–C: 1.7/2.5 px rms (max 3.6/5.2), tilt 1.23° rms, all swing-correlated and systematic. Per-frame σ median 1.5 px / 0.87°.
  - An ECC affine diagnostic is also stored.
- `ring_track.csv`: eye-ring centre (spring lower attachment) and radius.
  - A: radial edges + circle fit. B: annulus ZNCC.
  - A–B: 0.72/1.27 px rms. σ median 0.55/0.58 px. **The most precise moving observable.**
- `spring_track.csv`: coil-body top/bottom points on the axis, axis tilt (extent-midpoint vs gradient-centroid line, 0.46° rms),
  body length/width, and coil statistics (open gaps, pitch, close-wound length, scallops).
- `static_landmarks.csv`: 19 landmarks, drift-guided and raw-direct ZNCC. Wood-grain and table patches are unreliable (AI texture boil);
  hook_top is static.
- `modes_series.csv`: swing angle, r, weight relative tilt, spring tilt, spring length (frame-0 px).
- `tracks_3d.csv` / `tracks_3d.json`: world W_in on plane Y = 0 via the per-frame camera.
  - P_eff = (−0.003, 0, −0.225); P_eff→ring 0.818, P_eff→weight centre 0.998 W_in.
  - Weight diameter 0.212 W_in, silhouette length 0.307 W_in; ring outer radius 0.048 W_in.
- `analysis.json`, `anchor_window.json`, `cadence_analysis.json`, `drift_parallax.json`, `drift_layers_ecc.json`, `summary.json`.

## Commands
`measurement/run_all.sh` from ROOT, about 15 min. Stages:
`s1_drift`, `s1b_parallax`, `s1c_layers_ecc`, `s2_landmarks`, `s4_spring`, `s3_weight`, `s5_ring`, `s6_analysis`, `s7_cadence`,
then the camera stages, then `s8_3d`, `s9_anchor`, `s10_visuals`, `s11_summary` (writes `MANIFEST.sha256`).
Frames are cached in `measurement/cache/` (1.5 GB, deletable).
Adapted from clip 01: `common.py`, `s1_drift`, `s2_landmarks`, `s5_ring` (from `s3_bob`), `s7_cadence` (from `s9_cadence`).

## Open issues / requests
1. **Lead:** decide whether the anchor rule should also cover cumulative amplitude growth and period drift; see Anchor window above.
2. **Dynamics (B):**
   - Use the ring track for the swing (σ ≈ 0.6 px) and the weight tilt for rocking (σ ≈ 2° systematic).
   - The hinge is at P_eff (top coil), not at A. There is no axial mode to fit, so k is not identifiable from the reference
     (only a lower bound on ω_z that keeps the bob invisible).
   - The radial regression coefficients are in `analysis.json` vertical_bob_search; they are perspective/geometry-contaminated.
3. **Builder (C):** 11 open turns at 29 px pitch plus 2 hook-end turns plus ~10 close-wound turns. The close-wound section is
   139 px long (frame-0 px, ≈ 0.21 W_in). Hook and eye-ring geometry are in tracks_3d.json.
4. The weight-centre definition (silhouette midpoint between top and bottom extrema) is offset from the true CoM by the cap-ellipse
   geometry (< 0.01 W_in). Constant offsets between methods are removed by the median.
