# Batch 2 · Simulation 01 — Ruler buckling: report (checkpoint ck002)

**Goal (user):** keep the AI clip's scene, replace its wrong physics with verified physics.
**Status:** physics verified independently; scene and look reviewed; final render ck002. Awaiting human visual approval.
Not claimed: 100 % real-world accuracy. Material values and damping are assumed (declared below), not measured.

## What the AI clip shows (measured, `measurement/summary.json`)
Camera static (≤ 1.5 px). Weights **appear mid-air** with no hand and the count jumps two at a time (1 → 3 → 5 → 7);
the bench stack **never shrinks**. The strip leans toward camera-left from ~5.0 s (onset rule 5.25 s, 7 weights),
plateaus near −77 mm, then **swings back through vertical and over to the right (+171 mm) under constant load** —
and after ~7.7 s the bending is mostly **edgewise** (strong axis), which a 25 × 0.5 mm strip cannot do.
None of these behaviours were imitated.

## Corrected physics
- **Measured / declared from our own first frame:** geometry (strip 300 × 25 × 0.5 mm, 0.268 m flexible above the jaw,
  top 12 mm stiffened by tabs), 20 g slotted brass weights at 4.5 mm pitch, tip body 13.94 g (plate + 40 mm peg).
- **Assumed (literature values, not measured):** spring-temper 301 stainless, E = 193 GPa; g = 9.81 m/s²;
  damping Q ≈ 50 (Rayleigh α 0.063 s⁻¹, β 3.2e-4 s, + quadratic air drag); imperfection L/1000 toward −Y.
- **Loading (scene design, fixed before solving):** 7 weights seated at 1.0 + 0.6 j s, 0.15 s contact ramp.
- **Model:** geometrically exact planar Kirchhoff rod (inextensible discrete elastica, N = 268) with self-weight and a rigid
  eccentric tip body; generalized-α (ρ∞ 0.85), dt 0.52 ms. Nothing fitted to the reference.
- **Result:** critical load 1.3576 N → the strip goes supercritical **while the 7th weight is seated (4.66 s)**; slow
  topple (10 mm at 5.05 s, 50 mm at 5.79 s), overshoot to −193 mm at 6.8 s, damped sway about the n = 7 equilibrium
  (−150.3 mm, 53.9°) at ~0.35 Hz, always on the same side and always about the weak axis. Max tip rotation 75.4° < 90°,
  so the weights stay pressed on the plate. Root stress at n = 7 static: 252 MPa (elastic for spring temper).

## Verification
**Solver (role C):** Euler P_cr error 2.6e-6 (order 2.00); eccentric-load closed form 7.7e-5; elastica 90° tip angle
P/P_cr 1.39324 vs 1.39320; static n = 6/7/10 within 0.03 mm of the benchmarks; dynamic mesh convergence 0.30 mm (N 134 vs
268), time-step 1.2 µm; energy balance residual 1.9e-4 of peak strain energy.
**Independent verifier (role E, own shooting-elastica code, no solver imports):** W_cr 1.3575726 N, n_cr 6.2224 (agrees
to 5e-9); n = 7 static 150.339 vs 150.316 mm; quasi-static holds n = 0–5 within 0.27 mm; length ≤ 3.2e-7 m; Blender
evaluated mesh vs solver 7.5e-9 m at all 1201 samples; negative controls 6/6 caught.
ck002 audit: **7/7 PASS** — weights peak 0.68–0.73 m/s, seated within 0.1 mm of the physics slot offsets
one sample before each scheduled time, never > 7 on the peg, feeder empties one per transfer, no visibility toggles
(ck001 had failed this item: weights teleported at ~14 m/s). Info: arm links/quill reach up to 6 m/s on empty returns,
mostly above the frame. Post-buckled frames never settle (Q ≈ 50), so the ≤ 1 % post-buckling check rests on the static
n = 7 cross-check, not on the dynamic frames.

## Scene and look
Scene matches our first frame (the frame given to the generator), not the generator's softer grade (frame-0 mean
difference: ours vs our frame0 3.3, reference vs our frame0 9.7, 8-bit). **Our additions, not in the reference:** a
40 mm peg on the platform; a ring feeder at the ruler top holding the 7 weights; an overhead pick-and-place arm (column out
of frame) that carries each weight ~6 cm at ≤ 0.73 m/s and leaves the frame by 5.42 s. The 12 bench weights are untouched
spares. The empty gripper's return between pedestals peaks at 3.2 m/s (2–3 frames). Motion blur on (shutter 0.5).
Scene checks (every frame + contact subframes): 0 intersections, strip at jaw 4.8e-7 m, length 6e-11, seats ±0.07 mm,
clearances ≥ 6.9 mm, camera 6e-4 px. Rebuilds bit-identical (reviewer F).

## Sensitive assumption
The sway amplitude after buckling, and therefore the pose at 10 s (−170 mm), depends on the assumed damping: an undamped
run differs by up to 182 mm. Buckling load, onset weight and the static bent shape do not depend on it.

## Key files
`review/index.html` · `review/final_ck002_1920x1088.mp4` · `review/visuals/ck002_sbs.mp4` · `config/CHANGELOG.md` ·
`solver/visuals/` · `verify/out/ck002_audit.json` · `visual_review/ck001_review.json` · `physics/FORMULATION.md` ·
reproduce: `./build_all.sh <name>` (solver commands in `solver/` per CHANGELOG).
