# measurement/ — clip 03 torsion disc, reference measurement (v001, draft)

Role A (Measure+Camera). Reference `WS/reference-batch/clips/03-torsion-disc.mp4` (sha256 c640e5ce…a2045c verified; 241 frames 1920x1088, 24 fps,
frame n at t = n/24). Pixels per CONTRACT. **Unit D = disc diameter = 1.** Labels: [obs] measured, [fit] fitted, [inf] inference, [asm] assumption.

## Visuals (measurement/visuals/, labelled v001 / frame-time / "reference measurement - draft")
| file | shows |
|---|---|
| v001_annotated_f000.jpg | frame 0: fitted top/bottom disc circles + silhouette points, engraved line (psi), wire top/bottom, A, disc-centre track of all 241 frames (x4 inset) |
| v001_rectified_disc_strip.jpg | disc rectified top-down every 10 frames with the measured line; kymograph of line darkness (theta 0–360 vs t) with psi, psi+180 |
| v001_psi_sway_Ta.png | psi(t) + models, rate, A−B / sigma / cadence / kink, sway X/Y with half-cycle amplitudes, 2D sway amplitude, triggers and T_a |
| v001_camera_drift_raw_vs_corrected.jpg, v001_camera_drift_plot.png | anaglyph f0 vs f240 (parallax = orbit), static edges via camera of record vs locked; azimuth/distance/elevation/f/residuals vs t |
| v001_reprojection_overlay.jpg | frame 240 static edges, camera of record (R4) vs locked camera, zooms |

## Key results
- **Camera [obs]: orbits** +15.4° → −8.0° azimuth about A (23° in 10 s), distance −1 %, constant f 2602 px [asm]. R4 0.34–0.70 px (max 0.91 over all frames).
  Blender projection max 0.004 px. Details: camera/NOTES.md.
- **Twist psi [obs]: NO oscillation.** The line (a full diameter, slightly kinked) turns one way, clockwise seen from above, from −168.4° to −546.9°
  (**−379° = −1.05 turns**). Mean rate per second: −25, −34, −38, −40, −43, −47, −44, −40, −36, −33 °/s. No turning point.
  Models (rms): constant spin 6.0°, constant accel 6.3°, cubic 1.21°, slow sinusoid 1.18° (T = 31.8 ± 0.2 s, amplitude 226 ± 1°: the clip would be the middle of
  one half-swing). The prompt's "a few degrees, back and forth, decaying" and the earlier QA "tens of degrees" are both not what the clip does.
  - Method A (black-hat ridge + weighted TLS per half-line, rectified top-down image) vs Method B (angular darkness profile, polar image, 4 radial bands):
    **A−B rms 0.54°, mean +0.08°, max 2.4°; per-frame sigma_psi median 0.26°** (rolling rms(A−B)/2 ⊕ formal B). Shared systematic: rectification (camera f ±3 %, disc pose)
    — anisotropy error ≲ ±1° peak (sin 2psi pattern) not included in sigma.
  - Rotation centre of the line is offset (0.002, 0.031) D from the silhouette-fitted top-face centre (line passes there; residual 0.006 D rms).
  - Kink (half-line angle difference − 180°): A −0.1 ± 3.0° (range −6.3…+6.5°), B +0.1 ± 3.6°; varies smoothly with psi; 3 small jumps (frames 61, 169, 171) flagged >6 MAD, not visible as shape pops.
- **Disc pose [obs/fit]:** thickness 0.031 D (rim band), silhouette rms 0.90 px median. Wire length A → top face 0.857 D. Mean centre (+0.014, −0.014, −0.857) D.
  Tilt mean 2.1° (max 3.4°); **the disc stays perpendicular to the wire**: tilt about Y vs swing-x gain −0.96 (r = −0.96), tilt about X vs swing-y gain 0.64 (r = 0.88)
  — not level as the prompt says; no tilt dependence on psi (non-circular-disc test, cos/sin 2psi coefficients ≤ 0.17°).
- **Sway [obs]:** pendulum swing of disc+wire, **period 0.948 ± 0.001 s (x), 0.955 ± 0.002 s (y)**; wire image tilt ±1.5° at 0.955 s. Amplitude x 0.026 D
  (swing ptp 5.0°), y (depth) 0.040 D (ptp 8.9°); x/y exchange energy (spherical pendulum). Bob (Qz) std 0.006 D (perspective-coupled, at the sway frequency).
  [inf] If the sway were a simple pendulum of length 0.857 D, D ≈ 0.26 m (prompt 0.10 m) — dynamics' call.
- **2D sway amplitude** (invariant sqrt(x²+y²+(ẋ²+ẏ²)/ω²)): 0.040 → 0.060 D in the first 0.75 s (+50 %), decays to 0.046 by 3 s, grows again to 0.068 D at 5.3 s (+45 %),
  then decays to 0.042 D — **non-physical energy injection** twice.
- **Cadence [obs]:** psi timing offsets by n mod 4: +0.30, −0.16, −0.11, +0.06 frames (sd 0.05); step ratio into n ≡ 1 mod 4 is 1.76× (pattern shifted vs clips 01/02,
  same generator defect). Diagnostic only.
- **Highlights [obs]:** brushed radial lobes do not follow the spin (they move with the camera azimuth, slope 0.80): lighting-locked, as physics expects.
- **Discontinuities [obs]:** none visible (no psi outliers > 5 MAD, line stays one diameter, silhouette rms ≤ 1.29 px).

## Anchor window (CHANGELOG rule, applied as declared) → **T_a = 2.5 s**
- Twist: no turning points → (a)/(b) not applicable. (c): no discontinuity.
- Sway mode (only oscillating mode): (a) jump: y half-cycle amplitude 0.035 → 0.049 D at the TP t = 0.654 s, z = +7.4 (2D invariant agrees: +4.8σ);
  (b) cumulative drift: x amplitude trend departs from the passive reference (first two cycles) by > 3σ at t = 3.68 s; y at 4.37 s; 2D invariant at 4.42 s.
  Earliest trigger 0.654 s < 2.5 s minimum → **T_a = 2.5 s** (floor). Evidence: anchor_window.json, v001_psi_sway_Ta.png.
- Flags for the lead: (1) the rule presumes an oscillating primary mode; the twist is a one-way spin, so a torsion stiffness/damping fit on 0–2.5 s
  (≈ 75° of monotonic rotation, rate −25 → −38 °/s) is close to unconstrained. (2) Per-axis sway amplitudes are not modes (x/y energy exchange);
  the 2D invariant is the physically meaningful test (it gives the same early trigger).

## Files
twist_track.csv (psi, sigma, A, B, rate, kink, lobe, camera azimuth), twist_raw.csv, disc_track.csv (world centre, sd, tilt, swing angles, image u,v),
disc_pose_camcentred.csv (world pose; name historical), disc_ellipse_image.csv, wire_track.csv, analysis.json, anchor_window.json, summary.json, MANIFEST.sha256.
Cache (deletable, 1.6 GB): measurement/cache/.

## Commands
`measurement/run_all.sh` (stage order inside). Reused from clip 02: common.py, s1_drift.py, s7_cadence (logic folded into s6); new: s1d_edges_track, s2_disc_edges,
s3_disc_pose, s4_twist, s5_wire, s6_analysis, s9_anchor, s10_visuals, s11_summary, camfr.py.

## Open issues
1. f only ±3 % determined (VP vs bundle disagree mildly; AI frame not rigid); affects rectification aspect (psi systematic ≲ 1°) and D-scaled positions (±3 %).
2. Depth (y) sway is measured from disc size/ellipse → noisier (sigma ~0.001–0.003 D) than x.
3. Bootstrap order: locate.py needs apex_offset from a previous s4 run (apex changed 0.035 → 0.031 D, A not re-iterated; effect ~0.004 D).

## sha256 (first 16 hex; full list MANIFEST.sha256)
twist_track.csv 4926fe391c4895fb · disc_track.csv b2e0bf818ee898e6 · summary.json 0f86b0fec8e06742 · anchor_window.json bdb655e58550a609 · params/camera.json 84f7deb3b4693396

## v002 (lead VR03-04 / VR03-05, after ck001 visual review)
- Camera of record smoothed (camera/NOTES.md v002): jerk 0.46/4.08 px → 0.002/0.004 px (reference 0.09/0.67); R4 per frame max 0.90 px. All downstream stages re-run
  through the smoothed camera (s3, s4, s6, s9, s12, visuals v002_*). Changes are small: A−B rms 0.55°, sigma_psi 0.26°, total twist −378.5°, sway periods 0.948 / 0.954 s,
  amplitudes 0.026 / 0.039 D. Anchor triggers: Qy jump at 0.66 s (unchanged) → T_a = 2.5 s; Qx cumulative now 4.60 s, Qy 4.36 s.
- **Fit observable P (lead decision VR03-05): `measurement/disc_P_track.csv`** (+ `disc_P_track.json`): per frame P = centre of the fitted top-face circle of the rigid-disc
  silhouette model, world coords (origin A, unit D), with sd_eff (formal ⊕ local high-frequency scatter) and sd_formal, disc tilt, image (u,v) of P, silhouette rms.
  Mean P = (+0.012, −0.045, −0.858) D; std (0.020, 0.033, 0.006) D; sd_eff median (0.0013, 0.0031, 0.0009) D; silhouette rms 0.95 px median.
  Systematics not in sigma: f ±3 % (mainly depth/Y and absolute distance), A placement (A is vertically above the frame-0 Q, so P sits 0.045 D toward the camera from below A —
  the pivot should be refit with P).
- Q − P = +0.031 D in world +Y (constant; P − Q = −0.031 D). Test: the line's perpendicular offset vs psi fits a world-fixed point (rms 0.0058 D) and not a disc-attached chord
  (rms 0.0229 D) → the mark's rotation centre is not attached to the disc (AI inconsistency, or a depth bias of the silhouette centre along the view direction).
- New visual: v002_camera_jitter_smoothing.png (raw vs smoothed path, reference vs camera window motion, jerk, R4 per frame). Other visuals re-rendered as v002_* (v001_* kept).
