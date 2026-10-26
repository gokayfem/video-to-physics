# Simulation 01 — Brass pendulum: report (checkpoint ck003)

**Goal (user):** replace the AI clip's wrong physics with correct physics while visually matching the scene.
**Status:** complete; verified; awaiting human review (user authorized proceeding to clip 02 without waiting).

## What was built
- Editable Blender 5.2.2 scene `checkpoints/ck003/pendulum.blend` (26 objects): Geometry-Nodes stand
  with controls (post height, arm length, base W/D/T, edge radii), pin + wire loop, wire, brass bob with
  eyelet; procedural walnut / brass / steel materials (look v011, frozen by the user); backdrop sweep,
  table, studio lights; cameras from the calibration (reference camera with the measured per-frame
  zoom/pan keyed; locked diagnostic, orthographic side, 3/4 overview); motion blur shutter 0.5.
- Physics: planar rigid compound pendulum (solid brass sphere on a rigid massless wire, hinge at the
  wire's departure from the loop), linear viscous pivot damping, gravity 9.80665 m/s² (assumed).
  Solved offline (RK4, h = 1/3072 s) and baked as 241 Bézier keys (Hermite handles from ω) on
  `PND_Swing` — motion is **baked**, reproducible from `params/physics.json` via `build_all.sh`.

## Physics fitted to the opening window (user decision)
Fitted on t ∈ [0, 4.0] s (rule fixed in advance: first >3σ amplitude jump), then correct physics runs to 10 s.
| Parameter | Value | Provenance |
|---|---|---|
| L_eq (equivalent length) | 0.4751 ± 0.0026 m | model-conditioned (period + assumed g) |
| L_c (hinge → bob centre) | 0.4739 m | model-conditioned |
| bob radius r | 38.5 mm (L_c/r = 12.30 measured) | measured ratio + model-conditioned scale |
| period | 1.3842 s | fitted |
| θ(0), ω(0) | −4.84°, +0.379 rad/s (from true vertical) | fitted |
| static scene tilt vs gravity | +0.722° ± 0.057° | fitted (unlevel table hypothesis) |
| Q (damping) | 1000 (range 300–6000) | **assumed** — not identifiable at a plausible value |
| mass | 2.0 kg (brass 8500 kg/m³) | assumed; dynamics independent of mass |

## Verification (independent verifier, predeclared tolerances, run on the saved checkpoint files)
P1 convergence order 4.000 (finest diff 3e-12 rad) · P2 independent Taylor/Gauss–Legendre solver agrees to
2e-13 rad · P3 exact elliptic period 4e-14 rel · P4 energy balance 1.5e-14 E0, no growth · P5 pivot fixed,
|L−L_c| 4e-8 m, planar · P6 Blender-evaluated vs solver 4e-7 rad (4 subframes/frame) · P7 tension 19.8–20.2 N
· P8 initial conditions/time exact · P9 L×4 / 2° / 15° / damping×10 behave per formulas (1e-8) · P10 clean
rebuilds bitwise identical (motion/geometry); renders differ by 1/255 in 2–3 pixels between separately built
files (cause below Python's visibility). Tilt audit PASS. ck003 re-checked for P5/P6/P8/tilt (motion bitwise
identical to ck002). **None of this proves the real world would do this; it proves the scene does exactly what
the declared physics model says, accurately.**

## Reference fidelity (the AI clip is not ground truth)
- Opening window 0–4 s: projected model vs tracked bob **3.24 px RMS** (tracking σ ≈ 0.3 px; residual is the
  clip's 4-frame cadence jitter + arc-shape defect).
- After 4 s the AI swing grows +62 % and its period wanders 1.36–1.43 s — impossible for a passive pendulum.
  The corrected physics diverges by up to ~45 px there; this divergence map is the "wrong physics" made visible.
- Declared R2/R3 targets (2/3 px over 0–7.5 s / 7.5–10 s) **FAIL** (13.9 / 27.3 px) because of those defects.
- Other AI defects documented, not copied: camera push-in 3.3 % (reproduced as camera motion), bowed wire,
  apparent rotation point 8 % below the visible hinge, cadence jitter.

## Visual review (independent)
Temporal stability PASS (no flicker/popping/sliding; render = solver projection to 0.07 px), silhouette IoU
0.94, camera push-in reproduced. Fixed after review: motion blur, pin/loop, brass eyelet, camera locks.
Remaining (accepted/known): render slightly brighter than reference on wood/table/bob (look frozen by user),
wire ~1.5× thinner than the AI draws it, damping not visible in image (push-in masks a 2 % decay).

## Key files
`review/index.html` (playable review page) · `review/visuals/v004_sbs_ck003_final_f000-240.mp4` ·
`review/visuals/v004_reconstruction_ck003_1920x1088.mp4` · `review/visuals/v003_diagnostic_side_ck002.mp4`
· `config/CHANGELOG.md` (every decision) · `verify/out/ck002/summary.json` · `visual_review/REVIEW.md` ·
`delivery/out/provenance_ck002.json` · `CONTRACT.md`, `config/acceptance.json`.
