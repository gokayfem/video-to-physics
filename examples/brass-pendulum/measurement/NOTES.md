# measurement/ — reference measurement, clip 01 brass pendulum (v001, draft)

Role: reference measurement. Contract v2. Everything here is an image-space measurement of the AI
reference `WS/reference-batch/clips/01-brass-pendulum.mp4` (sha256 `ced22d10…1f56bae`, verified).
No world quantities are produced here. Pixel convention: contract (origin at the top-left corner of the
top-left pixel, pixel centres at +0.5). Time: frame n ↔ t = n/24 s; all 241 frames decoded with OpenCV
(ffprobe also counts 241; pts 0…10.000 s).

Label key used below: **[obs]** measured from pixels; **[fit]** from a fitted model; **[inf]** inference.

## Visuals (v001, burned-in labels, status "reference measurement – draft")
| file | shows | still uncertain |
|---|---|---|
| `visuals/v001_annotated_f000_full_track.jpg` | frame 0 with the full 241-frame bob-centre track (mapped into frame-0 geometry), pivot, wire centreline, fitted bob circle, 16 static landmarks | landmark names are semantic labels on template patches, not surveyed corners |
| `visuals/v001_turning_points_zero_crossings.jpg` | frames 4 (zero crossing), 13/29 (first R/L turning points), 163 (largest L), 179 (R), 240 with per-frame pivot, wire, circle fit and the 10 s track drawn in each frame's own geometry | — |
| `visuals/v001_wire_straightness_rectified.jpg` | wire resampled along its chord, across-wire axis ×6: visible bow at turning points, near-straight at zero crossings | cause of the bow (AI image content) is an inference |
| `visuals/v001_camera_drift_f000_vs_f240.jpg` | anaglyph frame 0 vs 240, raw and after the drift homography: static scene misregistered raw, registered after | residual ~1–1.5 px late in the clip (parallax) |
| `visuals/v001_bob_cadence_f000-012.jpg` | bob crops frames 0–12 with frame steps; steps into n = 4k are ~1.6× larger | — |
| `visuals/v001_pivot_closeups.jpg` | pin/loop ×4 at left TP, zero crossing, right TP, tracked pivot | sub-pixel location of the true rotation axis inside the loop |
| `plots/v001_*.png` | u/v(t), angle + fit + residual, amplitude/period per cycle, tracking residuals, cadence, landmark motion + drift, wire/radius/length vs angle, stabilized path | — |
| `crops/` | bob edge-point overlays (green used / red rejected) for 9 frames; raw mid-wire crops and rectified strips at turning points and zero crossings | — |

## Key results
- **Camera [obs]: not static.** Smooth, near-linear zoom-in/dolly-in: similarity scale 1.0000 → 1.0327 over
  10 s, rotation < 0.05°. Pivot moves (+4.1, −0.8) px, bob rest region (+2.9, +22.9) px, base corners
  ~25–30 px. Per-frame 8-dof homography (`camera_drift.csv`) explains 13 static landmarks to 0.77 px RMS
  (growing to ~1–1.5 px late: small parallax). `pin_shaft_mid` is unreliable (specular, aperture) — ignore.
- **Pivot [obs]:** wire departure point at bottom of the pin loop: (1086.69, 125.01) px at f0,
  (1090.64, 124.56) at f240; after drift removal it is fixed to 0.67 px RMS with ≤0.5 px swing-correlated
  motion. The loop does not rock.
- **Bob [obs]:** two independent methods agree to 0.51/0.34 px RMS (u/v), max 3.1/0.9 px. Per-frame sigma
  median 0.29/0.24 px, max 0.91/0.49 px. Apparent radius 55.3–60.2 px: ~5 % larger at the image-right
  extreme (both methods) [inf: swing has a depth component toward the camera on the right — camera
  specialist should use radius(u) and the asymmetric arc].
- **Period [obs/fit]:** zero crossings (15) → 1.404 ± 0.003 s; damped-sine fit 1.4007 ± 0.0012 s (formal;
  model inadequate, χ²ν = 291). Cycle-to-cycle 1.357–1.434 s, sd 0.026 s ≫ timing sd 0.002 s; linear trend
  +0.0064 ± 0.0021 s per s (lengthening). Eyeball 1.25 s is wrong.
- **Amplitude [obs]: not decaying.** Half peak-to-peak image angle: 5.77° (0.86 s), 5.73, 5.63, 5.49,
  5.45 (3.64 s), 5.93, 6.73, 7.41, 8.47, **8.84 (7.13 s)**, 7.91, 6.86, 6.48 (9.26 s); sd 0.02–0.04°.
  Slight decrease for 3.6 s, then +62 % growth, then decrease. Non-physical for a passive pendulum →
  reference defect. Image angle is similarity-invariant, so the zoom drift cannot produce it.
- **Phase at t = 0 [obs/fit]:** not a turning point. Bob at ≈ −3.3° (≈ 57 % of amplitude, image-left)
  moving right at ≈ 20°/s; first turning point (right) at t = 0.522 ± 0.005 s; previous left turning point
  would be at t ≈ −0.20 s. Not released from rest at t = 0.
- **Wire [obs]: curved.** Quadratic sagitta −4.9…+4.7 px over ~650 px (formal sd ~0.03 px), r = 0.94 with
  angle (0.53 px/deg), ~1–2 px near zero crossings. Control: static stand edges (660 px) measured the
  same way: |sagitta| ≤ 0.45 px. A straight 3D line projects straight (pinhole), so [inf] the bow is AI
  image content, not optics. The earlier "curved wire" review was right; "straight wire" was not.
- **Cadence defect [obs]:** step into every frame n = 4k is 1.58× the local velocity, others 0.80–0.84×,
  in every GOP even where the encoder's P-frame grid is shifted → generator-native (4-frame temporal
  latent). Apparent per-phase time offsets (n mod 4 = 0..3): +0.277, +0.113, −0.094, −0.296 frames
  (sd 0.019) → up to ~±7 px position error at peak speed. Diagnostic only, NOT a retiming licence.
- **Apparent pivot→bob length [obs]:** 696–712 px after drift normalization, smooth function of angle
  (asymmetric, [inf] perspective); no jumps beyond what fast-frame steps imply (max frame diff 2.95 px).
- **Frame quality:** H.264 yuv420p 5.96 Mbit/s, GOP ~30 (8 I / 64 P / 169 B); chroma blocky (4:2:0) →
  luma-only geometry; horizontal motion blur on the bob rim near mid-swing; the wire is a ~5 px
  two-strand band (centroid of dark band used).

## Methods
1. `s1_drift.py`: chained KLT homographies (forward-backward < 0.3 px, RANSAC 0.7 px) as init → ECC
   homography of each frame directly against frame 0 on a static mask (swing region excluded).
2. `s2_landmarks.py`: 16 landmarks, frame-0 patch ZNCC in the drift-warped frame (±10 px), parabolic
   subpixel; also H-independent raw matching as cross-check.
3. `s3_bob.py`: **A** 180 rays, outermost dark→bright luma step in [0.72R, 1.30R], wire rays (±9°)
   excluded, Tukey-IRLS geometric circle fit, free radius, 3+2 re-centring passes. **B** annulus
   (0.80–1.15 R, wire wedge masked) ZNCC, frame-0 template, scale searched 0.94–1.12 + refinement,
   own constant-velocity prediction; constant offset to A removed by the median. Sigma per frame =
   sqrt(fit covariance + local 7-frame RMS(A−B)²/2).
4. `s4_wire.py`: per-row centroid of the dark wire band in ±5 px of the current line, longest continuous
   run, Huber line and quadratic fits, 15-row smoothed max deviation from the end-to-end chord.
5. `s5_analysis.py`: CSV assembly, angle = atan2(u_bob − u_piv, v_bob − v_piv) in the same raw frame;
   damped/undamped sine fits; zero crossings (linear interpolation, sd from angle sigma / slope);
   turning points from local cos fits (±3 frames, ω fixed); log-amplitude and linear trend fits.
6. `s9_cadence.py`, `s8_control_edge.py`, `s6_plots.py`, `s7_visuals.py`, `s10_finalize.py`.

Reproduce: `measurement/run_all.sh` (from ROOT; ~10 min). Decoded frames are cached in
`measurement/cache/frames_bgr.npy` (1.4 GB, deleted after the run; regenerated automatically).

## Outputs
`bob_track.csv`, `static_landmarks.csv` (+ `static_landmarks_def.json`), `camera_drift.csv`,
`suspension.csv`, `angle_series.csv`, `summary.json`, `cadence_analysis.json`,
`control_straight_edges.json`, `plots/`, `visuals/`, `crops/`, `scripts/`. sha256 of every file:
`MANIFEST.sha256`.

Column notes: `bob_track.csv` u_px/v_px are raw-frame coordinates (camera drift included);
`u_stab_frame0_px/v_stab_frame0_px` are the same points mapped to frame-0 geometry with H_n⁻¹.
`camera_drift.csv` H maps frame-0 pixel coords → frame-n pixel coords (contract convention).
`suspension.csv` top = tracked wire departure point; bottom = fitted wire line ∩ bob circle (upper);
`wire_length_px` = their distance (raw, includes zoom); `max_dev_px` = smoothed centreline vs chord.

## Open issues
- The image-plane angle is not a world angle; depth component of the swing (radius and arc asymmetry)
  must be resolved by the camera specialist.
- The bow of the wire means "pivot" is ambiguous at the 10–20 px level along the wire: upper-wire lines
  intersect at (1086.9, 144.5) frame-0 px, full-wire lines at (1087.5, 171.2), vs the visible departure
  point (1086.7, 125.0). The angle series uses the visible departure point; an alternative angle using the
  upper-wire intersection is in `angle_series.csv`.
- A physical model cannot reproduce the amplitude growth, the period variation or the 4-frame cadence;
  R2/R3 residual targets (2–3 px) are likely unattainable against this reference for reasons in the
  reference itself. Suggest the lead decide how R2/R3 are reported (e.g. also on n ≡ 0 mod 4 frames, or with
  the cadence offsets disclosed), without retiming the simulation.

## Requests to other roles
- Camera: use `camera_drift.csv` (per-frame camera move) — a single static camera cannot match all frames
  (static landmarks move up to 30 px). Use bob radius vs u and the stabilized path for the swing-plane
  orientation.
- Dynamics/verifier: amplitude grows 5.45° → 8.84° (image) between t = 3.6 and 7.1 s; do not chase it.
