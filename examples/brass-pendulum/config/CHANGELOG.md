# Contract / acceptance changelog (lead only)

## 2026-10-25T22:31:55Z — clarifications (no tolerance relaxed)
1. P4 "dE/dt <= 1e-12*E0" was dimensionally ambiguous as declared. Clarified reading (strictest
   consistent one): for every pair of consecutive output samples, E(t_{k+1}) - E(t_k) <= 1e-12*E0
   (dimensionless increment, per 1/96 s sample) — applies to the damped production run AND the
   conservative run. Rationale: round-off in the conservative run is ~1e-14 E0 per sample; a per-second
   rate of 1.85e-12 E0/s corresponds to ~2e-14 E0 per sample, which passes this reading. No numeric
   tolerance changed.
2. Single source of truth for bob radius r and L_c: params/physics.json (dynamics). The L_c/r ratio is a
   measured image quantity supplied by geometry/measurement and passed to solver/make_params.py
   (--ratio); scene/build.py must read r and L_c from params/physics.json, never from geometry.json.
   geometry.json keeps the measured ratio with its uncertainty as provenance.
3. P6/P8 judge samples on t in [0, 10] s only (frames 0..240 incl. subframes up to frame 240 exactly);
   subframes after frame 240 are outside the clip and reported, not judged.

## 2026-10-25T22:33:36Z — reference-fidelity reporting (no tolerance changed)
Measurement v001 established reference defects: camera push-in (scale 1.000->1.033), non-monotonic
amplitude with +62% growth (5.45 deg at 3.6 s -> 8.84 deg at 7.1 s), cycle period 1.357–1.434 s, a
4-frame cadence timing defect (equivalent +0.28/+0.11/-0.09/-0.30 frame offsets) and an image-space
wire bow of up to ~5 px. A passive physical pendulum cannot reproduce the growth or period wander.
Decision: R2/R3 are still evaluated exactly as declared and reported (expected FAIL, attributed to the
reference with evidence). Supplementary, clearly labelled metrics are ADDED (not substituted): residuals
after removing the measured camera drift, residuals on n≡0 mod 4 frames only, and the cadence offsets
disclosed. The simulation is never retimed, amplitude-modulated or bent to match these defects. The
fitted model is the simplest physically consistent one (passive, damping >= 0).

## 2026-10-25T22:42:03Z — camera and pivot decisions (lead)
1. Single camera of record: params/camera.json + camera/project.py (static RMS 0.27 px; per-frame zoom
   model from camera/drift.csv). Geometry proportions must be re-derived through it (the geometry
   role's own joint-fit camera, f=2749 px, is superseded; focal is weakly constrained 2350–2750 px and
   this ambiguity is reported, not hidden).
2. Pivot P stays at the wire-departure point (P_ring), per contract. Physical rationale: a wire looped
   over a pin hinges where it leaves the loop. The reference's apparent rotation centre 0.081 L below
   P_ring comes with a visibly bent upper wire (measurement v001: up to ~5 px bow) — an AI-image defect.
   The fit keeps P fixed at P_ring; a free-pivot-offset fit is run ONLY as a labelled diagnostic to
   quantify the mismatch it causes. No geometry is bent to emulate it.
3. Swing plane = vertical plane through the arm axis (azimuth 0 deg, assumption; fitted +4.3 ± 1.1 deg
   reported). True amplitude is ~±11 deg (36 deg view angle), still a small-amplitude, subtle motion.

## 2026-10-25T22:47:40Z — length gauge decision (lead)
L_c is defined as |P_ring - bob centre| measured through the camera of record (≈1.05–1.07 L_eff, to be
fitted), NOT the camera's arc-radius gauge L_eff (radius about the AI's apparent rotation point 0.081 L
below P_ring). Rationale: keeps the hinge physical (P_ring) and places the bob where the reference shows
it (otherwise the bob renders ~0.06 L_c too high, bob-base clearance 0.124 vs ~0.05 L_c). Consequence:
the rigid arc about P_ring has a slightly larger radius than the AI's apparent arc; the resulting
residual is reported as a reference-defect effect. physics.json must record the gauge explicitly:
"L_eff_m" (metres per camera.json length unit) alongside L_c_m; scene/build.py scales camera
translations by L_eff_m, never by L_c_m.

## 2026-10-25T23:06:55Z — USER DECISION: anchor physics on the opening window
User goal (verbatim intent): replace the AI clip's wrong physics with correct physics while visually
matching the scene. User chose: "Anchor the physics on the opening seconds where the clip behaves
physically, then let correct physics run for the full 10 s."
Implementation: physical parameters (L_eq/period, theta0, omega0, damping if identifiable) are fitted ONLY
on the opening window t in [0, 3.6] s (measurement v001: amplitude onset of non-physical growth after
~3.6 s; exact window end to be justified from data by dynamics, reported). Correct physics then runs
unchanged to 10 s. Everything after the window is reported as a divergence map (AI motion vs corrected
motion), not as a fit target. R2/R3 are still computed exactly as originally declared and reported, plus
anchor-window RMS. Damping: free >= 0 within the window; adopt only if identifiable and physically
plausible, else keep the declared assumption Q = 1000. No tolerance changed.

## 2026-10-25T23:10:47Z — USER DECISION: keep current look as is; Look role stopped
User: "Restart only the independent verifier; keep the current look as is." The Look specialist was
stopped by the user's interrupt after round v011. params/look.json had an unrendered v012 edit (two light
energies 46->36 W and 30->22 W); it is preserved as look/out/v012_unrendered_look.json and NOT used.
params/look.json is pinned to the exact rendered v011 snapshot (look/out/v011_look.json); scene/look.py
unchanged (mtime precedes the v011 render). Ownership of look files passes to the lead (frozen).
Known look residuals at v011 (frame 0, Oklab L ref/cur): base_top 0.569/0.638, table 0.580/0.625,
bob 0.438/0.480 — render slightly bright; accepted as-is per user.

## 2026-10-25T23:14:43Z — lead decision: stand/table tilt relative to gravity (physical, fitted)
Anchored fit (T_a = 4.0 s, rule fixed before looking at R2/R3) leaves an 8.4 px anchor misfit that is
almost entirely a left/right asymmetry (+7.6 deg vs -5.6..-6.2 deg turning points). Interpretation adopted:
the static scene (table + stand + camera, whose roll was referenced to the stand edges at 0.27 px) is
tilted relative to TRUE gravity by a small angle (diagnostic fit +0.72 deg). This is a physical
configuration (a not-quite-level table), invents no force, and gravity stays exactly g along -Z_world.
A camera-only roll was rejected: it would break the 0.27 px stand fit. Implementation: physics.json
gains "level_tilt_rad" (fitted, with bootstrap uncertainty); in Blender the whole static scene (stand,
pivot hardware, table/backdrop, cameras) is rotated about the world Y axis through P by that angle; the
pendulum (PND_Swing) hangs under true gravity. Damping: with the tilt, linear damping is consistently
identifiable across windows (Q ~ 100–140, 4–6 sigma); adopt the bootstrapped fitted Q if it stays within
the physical range (FORMULATION: 100–6000), else keep the Q=1000 assumption. All verification reruns.

## 2026-10-25T23:44:57Z — visual review of ck002: lead dispositions
- VR-05 (major) motion blur missing: ENABLE. Rationale: camera exposure is a camera property (the
  reference shows bob edge smear 5.8 -> 8.7 px slow->fast), not part of the frozen material/lighting look.
  Shutter tuned against measured reference edge widths; value recorded in config/render.json.
- VR-11 pin too thin/satin + loose loop, VR-12 missing brass eyelet: FIX in geometry (pivot P unchanged).
  Pin steel roughness is a material change -> NOT changed (look frozen); thickness/loop are geometry.
- VR-09/VR-10 brightness/hue: accepted per user's frozen-look decision (reported).
- VR-06 R5 as declared (render track vs reference 0-10 s) fails because of reference defects; the render
  reproduces the solver projection to 0.066 px (VR-01). Reported.
- VR-08 damping not visible in image (push-in cancels it): physics unchanged; reported.
