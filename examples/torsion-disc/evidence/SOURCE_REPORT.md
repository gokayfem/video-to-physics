# Simulation 03 — Torsion disc: report (checkpoint ck003)

**Goal (user):** replace the AI clip's wrong physics with correct physics while visually matching the scene.
**Status:** physics verified (ck002; ck003 is look-only, motion bitwise identical); visual review fixes applied.

## What the AI clip actually shows (measured)
The prompt asked for a few degrees of back-and-forth twist. The clip instead turns the disc **one way, −379° in
10 s**, the rate rising then falling (−25 → −47 → −33 °/s) — which is physically the **middle of one slow
torsion half-swing** (period ~36 s), so the twist itself is plausible. The non-physical parts: the small sway
(disc+wire pendulum, 0.95 s) gains energy twice (+50 % at 0.65 s, +45 % at 3–5 s); after 7.5 s the twist slows
faster than physics allows; the engraved line is drawn as a full diameter whose apparent rotation centre stays
fixed in the room, not on the disc; the disc tilts only 0.64× the physical amount in one axis; the camera
ORBITS 23° around the stand; the usual every-4th-frame cadence jitter.

## Corrected physics
- Model `disc_on_wire_rigid6_v1`: rigid brass disc (22.1 cm, model-conditioned from the sway period with
  g = 9.80665 assumed) on an inextensible steel wire 0.191 m; linear torsion (κ/I from the 36.4 s period;
  hard-drawn music wire required for elastic twist at 221° — declared), compound-pendulum sway and tilt,
  rigid disc–wire joint (stiff joint spring), damping assumed (Q_t = 1000, sway/tilt Q = 500; not identifiable).
- Fitted per mode (disclosed deviation from the declared single window, justified by decoupling: sway changes
  twist by ≤ 0.14° over 10 s): twist on 0–7.5 s, sway/tilt on 0–2.5 s; observable = disc silhouette centre
  through the smoothed camera of record. Pivot fitted (3.2, −6.4) mm from the assumed clamp point.
- RK4 with exact quaternion/constraint projection and guards (slack wire, constraint drift, non-finite);
  baked at 4 keys per frame.

## Verification (independent verifier on ck002)
P1 order 3.99 · P2 independent GL4 solver 4.7e-12 m / 2.4e-10 rad · P3 exact torsion (undamped/damped) 3e-14,
swing mode 6e-12 · P4 drift 2.1e-12, damped balance 1.7e-12 · P5 pivot, wire ends ≤ 3.4e-10 m, mark rigid ·
P6 Blender vs solver 7.8e-9 m / 3.1e-7 rad · P7 tension 22.0 N · P8 exact · P9 κ×4, m×4, ℓ×1.5, damping×10 per
formulas · P10 rebuilds bitwise identical. All PASS (first time a clip passed P1–P10 on its first audit).

## Reference fidelity (AI clip is not ground truth)
Disc centre in its window 1.34 px vs 2.2 px target — PASS. Twist in its window **1.04° vs 0.78° — FAIL**
(≈1 Hz wobble of the AI mark). After the windows: centre 14.4 px (AI sway energy injection), twist up to
−11° at 10 s (AI slows too fast). Render reproduces the simulation (twist 0.07°, centre ≤ 0.8 px).

## Visual review (independent, two rounds)
Fixed: camera-path jitter (static-scene jerk 3.6 → 0.18 px), disc outline offset (7.2 → 1.7 px), wall light that
flips with the orbit like the reference, horizon, wall tint, walnut tone, screw/collet detail, brass rim band.
Final round addressed over-saturated brass cones, mark cavity and screw finish (see ck003 sheet). Known residual:
reference wire leaves the clamp ~6 px left of ours (collet mesh kept at the visible clamp; pivot fitted inside it).

## Key files
`review/index.html` · `review/visuals/v003_sbs_ck003_final_f000-240.mp4` ·
`review/visuals/v003_reconstruction_ck003_1920x1088.mp4` · `config/CHANGELOG.md` · `verify/out/*_ck002.json` ·
`visual_review/REVIEW.md` · `physics/FORMULATION.md` · `solver/NOTES.md`.
