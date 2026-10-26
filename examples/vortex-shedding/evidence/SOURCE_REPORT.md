# Batch 2 · Simulation 02 — Vortex shedding: report (checkpoint ck004)

**Goal (user):** keep the AI clip's scene, replace its wrong flow with a verified CFD solution.
**Status:** physics verified independently (ck004: 14/14 checks PASS); final render done; awaiting human visual approval.
Not claimed: 100 % real-world accuracy. The flow is a 2-D model of a shallow channel; fluid values are declared, not measured.

## What the AI clip shows (measured, `measurement/summary.json`)
The AI clip runs about **3.5× fast-forward**: the dye front advances at 3.9 U (physical maximum ≈ 1.5 U) and leaves the view
at 4.9 s; shedding at 0.89 Hz (physics: 0.28 Hz); vortices move at 3.2 U (physics: ≈ 0.85–0.9 U) and speed up downstream.
Its vortex rows are 4.7 D apart (physics ≈ 1–2 D), so dye from the outer needles is rolled into the street; the start looks
like an impulsive start; the colour mix is not conserved (magenta 46 % → 5 %); the period wanders by 13 %; the camera drifts
up to 5.9 px. Its vortex spacing (~5.5 cm) is roughly physical. None of these behaviours were imitated.

## Corrected physics
- **Declared:** water ν = 1.0e-6 m²/s; U = 0.015 m/s (mid-depth centreline at the cylinder, empty channel) → Re = 150;
  D = 10 mm; channel 0.6 × 0.12 m, depth 20 mm; our first-frame geometry (needles, cylinder, walls).
- **Model:** 2-D incompressible Navier–Stokes in the dye plane (Basilisk, embedded cylinder, adaptive grid 0.29 mm),
  plug inflow calibrated by an empty-channel run (U_in 0.013874 m/s), no-slip walls, zero-gradient outflow,
  linear floor friction −λu with λ = 6.17e-3 1/s (shallow-wake number 0.0056: street not suppressed).
  Limits: no 3-D floor/surface end effects (h/D = 2), may lower St by a few percent (not quantified).
- **State:** the flow has been running a long time (≫ 10 s needed to start shedding at Re 150), so the 10 s window is cut
  from a settled periodic run (wake KE < 1 % change per period in all bands). The only reference-anchored choice is the
  shedding phase: the lower side (ω > 0) rolls up first, at 0.756 s (AI: 0.75 s).
- **Dye:** real dye barely diffuses (Sc ≈ 1000), so streaks are material lines advected through the solver's velocity
  history from the needle sources (RK4, 1/48 s), drawn as mass-conserving filaments that thin and fade as the flow
  stretches them (σ² = σ0²/λ² + 2·D_m·age, D_m = 1e-9 m²/s), released at −3.875 s so the outer streaks reach x_cyl + 0.6 D at t = 0.
- **Result:** St 0.190 (0.284 Hz, 2.8 periods in 10 s), mean Cd 1.365, Cl′ 0.506; the dye reaches ≈ 13 cm downstream by
  10 s carrying 2–3 vortex pairs; the far channel stays clear.

## Verification
**Solver (C):** Schäfer–Turek 2D-2 at 46 cells/D: St 0.3012 (published 0.295–0.305), Cd_max 3.249 (+0.3 % over band),
Cl_max 1.042 (+3 %); production St converged across L10/L11 (0.3 %); Cd/Cl′ converge to the independent reference at L11.
**Independent verifier (E, FEniCSx IPCS Taylor–Hood, no shared code):** Schäfer–Turek St 0.302, Cd_max → 3.240
(extrapolated); production geometry St 0.1908, Cd 1.368, Cl′ 0.499; negative controls 5/5 caught.
ck004 audit (final): **14/14 PASS** — St 0.1896 (−0.6 % vs independent), Cd 1.364 (−0.3 %), Cl′ 0.508 (+1.7 %),
divergence 1.2 %, no-slip, wake KE −0.11 %/period, vortex convection 0.925 U, dye advected by the solver's velocity (scale 0.999),
tracer mass budget 0.8 %, front at t = 0 +0.57…0.67 D, phase 0.756 s lower side first, Blender wiring. Flow bit-identical to ck003.

## Scene and look
Exact frame-0 camera; dye frames mapped onto the dye sheet through Snell refraction at the water surface (≤ 0.14 px);
occlusion by cylinder, needles and walls; per-frame film grain. Rebuilds are reproducible (state file byte-identical).
**Declared frame-0 deviation:** our first frame showed a magenta ring around the cylinder base, drawn by the potential-flow
look model used for frame 0 (it wraps dye round the cylinder instantly). In the solved flow, dye arriving at the stagnation
point needs ~1–2 s to wrap the cylinder, so the ring forms early in the clip instead of being present at t = 0.

## Key files
`review/index.html` · `review/final_ck004_1920x1088.mp4` · `review/visuals/ck004_sbs.mp4` · `config/CHANGELOG.md` ·
`solver/visuals/` · `verify/out/ck004_audit.json` · `visual_review/ck002_review.json` · `physics/FORMULATION.md` ·
reproduce: `solver/src/reproduce.sh`, `./build_all.sh <name>`.
