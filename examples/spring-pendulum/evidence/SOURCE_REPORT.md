# Simulation 02 — Vertical spring: report (checkpoint ck003)

**Goal (user):** replace the AI clip's wrong physics with correct physics while visually matching the scene.
**Status:** complete; physics verified (ck002, motion bitwise identical in ck003); visual review fixes applied.

## What the AI clip actually shows (measured)
The prompt asked for a weight bobbing ~1 cm vertically. The clip instead shows **no vertical bob at all**: the
weight swings sideways ±8° (period 1.30 s) and over-tilts (1.34× the swing angle), the camera zooms +4.8 %,
the swing amplitude wanders (+4.5 % then −7.5 %) and the half-period drifts +7.5 % — none of which a passive
spring–mass system does. Generator cadence defect again (every 4th frame jumps).

## Corrected physics
A sideways-swinging weight on a spring is a real **elastic pendulum**, so the correct model is the general
physical system, not a forced vertical bob:
- 5-DOF model: elastic spring link from the swing hinge P_eff (top coil, below the fixed S-hook — a hook on a
  bar cannot rotate along the bar) to the weight's eye H; rigid steel cylinder hinged at H (rocking);
  Rayleigh spring mass; linear damping (Q = 500 assumed, not identifiable); gravity 9.80665 (assumed).
- Fitted on 0–7.5 s per the predeclared window rule (T_a = 10 s, last 2.5 s held out); diagnostic fit on
  0–4 s reported. ℓ = 0.3280 ± 0.0009 m, scale 0.4166 m per crossbar span (model-conditioned via the swing
  period), φ0 = −8.3°. k = 1336 N/m and mass 5.2 kg are declared assumptions (no bob in the clip to measure k;
  mass never identifiable). Axial mode starts at rest; coupling predicts only ±2 mm of stretch.
- Solver: RK4 with exact constraint projection (unit body axis), guards on spring compression and constraint
  drift; baked at 4 keys per frame (rocking mode 26 rad/s).

## Verification (independent verifier on ck002; ck003 motion bitwise identical)
P1 order 4.00 · P2 independent Gauss–Legendre solver 4e-13 m · P3 exact static/axial/two swing-mode
frequencies 3.5e-11 · P4 energy balance 9e-14, vertical angular momentum 4e-16 · P5 spring attached at both
ends, 21 turns constant, anchor fixed · P6 Blender vs solver 1.5e-8 m / 3e-8 rad · P7 tension ≥ 51 N · P8
exact · **P9** k×4 / m×4 / L0+50 % / damping×10 per formulas (≤4e-6) — failed on ck001 (solver let |n| drift →
NaN), fixed with constraint projection + re-equilibrated variant starts · P10 rebuilds bitwise identical.

## Reference fidelity (AI clip is not ground truth)
Fitted-window RMS 6.72 px vs 3.24 px target — **FAIL**, caused by the clip's period wander and excess weight
tilt (both documented in the divergence map); after 7.5 s 15.5 px. The render reproduces the simulation to
0.1 px (independent visual review).

## Visual review (independent, two rounds) — fixed
Backdrop gradient was upside-down (fixed: error 0.080 → 0.025 L), weight steel read mirror-like (fixed to the
reference's two-band satin profile), walnut hue, table, hook/loop wire thickness, crossbar tint, motion blur
(shutter 0.09). Remaining residuals: weight highlight bands ~0.1 L weaker, crossbar right half +0.10 L, spring
coils drawn by the AI as flat rings (ours are true helices), tight-coil pitch slightly larger.

## Key files
`review/index.html` · `review/visuals/v003_sbs_ck003_final_f000-240.mp4` ·
`review/visuals/v003_reconstruction_ck003_1920x1088.mp4` · `config/CHANGELOG.md` ·
`verify/out/ck002/summary.json` · `visual_review/REVIEW.md` · `physics/FORMULATION.md` · `solver/NOTES.md`.
