# Solvers

Practical recipes for the solver role. The order is always: benchmarks on the closed forms, convergence on the
production configuration, then the production run in the data-contract format with a `meta.json`.

## Contents

- [General rules](#general-rules)
- [Output contracts](#output-contracts)
- [Rigid-body ODEs](#rigid-body-odes)
- [Elastica: static path following and dynamics](#elastica-static-path-following-and-dynamics)
- [Generalised-α time integration](#generalised-α-time-integration)
- [Damping as a declared assumption](#damping-as-a-declared-assumption)
- [Continuum FEA](#continuum-fea)
- [CFD: developed-flow windows](#cfd-developed-flow-windows)
- [CFD: inflow calibration](#cfd-inflow-calibration)
- [Lagrangian dye and mass-conserving filaments](#lagrangian-dye-and-mass-conserving-filaments)
- [Fracture](#fracture)
- [Run hygiene](#run-hygiene)

## General rules

- Develop against synthetic data and the closed forms while measurement runs; take the anchors when they arrive.
- Fixed steps for anything baked (reproducible sample times); tight adaptive tolerances for references.
- Sample at least at frames and 4 subframes per frame (t = n/fps + k/(4 fps)); faster rotations (above a few
  rad/s) need dense samples so the bake matches between frames.
- Record in `meta.json`: parameters with labels, code sha256, parameter-file sha256, levels, step sizes,
  tolerances, key results, anchored values, wall time.
- Ship a reproduce script; document manual stop points if a run is continued from a restart.

## Output contracts

| Output | Format |
|---|---|
| rigid motion | csv or npz: t, generalized coordinates and velocities, body positions and orientations (quaternions), energies, dissipated work, constraint/contact forces |
| deformation | npz: `positions` (N_samples, rows, cols, 3) world metres on the scene mesh's grid, `t` seconds; optional per-sample arrays (load, state flags) |
| tracers | 16-bit RGB PNG per frame, channel k = tracer k in 0..1; `meta.json` with plane extent, px size, pixel-centre rule, row/column orientation |
| fields | npz per frame on a regular grid (u, v, ω, p) + probe time series (forces, wake probes) |

Agree the key names with the scene role at kickoff; a loader that accepts a documented alias or two saves a round.

## Rigid-body ODEs

- Reduced coordinates with exact constraints (rolling angle tied to arc length, inextensible links).
- Orientations: quaternions or exact projection after each step, plus guards that stop the run on constraint
  drift or non-finite state.
- Events (impacts, contact changes, turning points) located to < 1e-9 s with a root finder; restart the step there.
- RK4 at a step that converges to 1e-7 over the clip, observed order 4; an independent Gauss–Legendre or DOP853
  reference for the verifier.
- Track energy and the integral of dissipation; a conservative variant must hold energy to < 1e-8.
- Run the control-response variants (stiffness, length, damping changes) early as a self-check, each from its own
  re-equilibrated initial state.

## Elastica: static path following and dynamics

`v2p/elastica.py` implements a discrete Kirchhoff rod with link angles as unknowns (exact inextensibility).

- **Static path following**: continue in load with a small imperfection or with branch switching at the
  bifurcation. Report the critical load from the linearised problem and compare with the actual-BC reference
  (tip body, eccentricity, self-weight).
- **Benchmarks**: Euler P_cr (clamped–free), eccentric closed form, Greenhill, elliptic elastica tip rotation vs
  load. Mesh convergence on N, 2N, 4N with observed order 2.
- **Dynamics**: the same discretisation with mass and rotary inertia per link, a rigid tip body, load steps with
  short contact ramps, generalised-α in time.
- Check post-buckled rotation of any tip platform stays below the angle at which carried loads would slide off.

## Generalised-α time integration

- Second-order accurate, unconditionally stable for linear problems, with controllable high-frequency dissipation
  through ρ∞ (spectral radius at infinite frequency). ρ∞ around 0.8–0.9 damps numerical noise without visibly
  damping the physical modes; ρ∞ = 1 is energy-conserving (no numerical dissipation).
- Report the numerical dissipation separately from the declared physical damping (run a ρ∞ = 1 or
  halved-step variant).
- Time-step convergence on three steps with observed order ≈ 2.

## Damping as a declared assumption

- Damping is rarely identifiable from a short or generated clip. Declare the law (linear viscous, Rayleigh
  α + β, quadratic air drag) and a value as a quality factor Q with a plausible range.
- Fit damping only if it is identifiable inside the anchor window and the fitted value is physically plausible.
  Never use negative damping or a drive to match a generator's energy gain.
- Run the sensitivity case (undamped or the range ends) and report the effect on what the viewer sees.

## Continuum FEA

- FEniCSx: mesh with gmsh, at least three refinement levels; quadratic elements for bending-dominated problems,
  or use a rod/shell model to avoid locking.
- Nonlinear statics: load stepping with Newton; arc-length continuation near limit points.
- Dynamics: generalised-α or Newmark; Rayleigh damping declared.
- Map results onto the scene mesh grid in the deformation npz; the scene does not re-interpolate.

## CFD: developed-flow windows

- Many flows in a clip are developed: a wake that has been shedding for a long time. An impulsive start can take
  far longer than the clip to reach that state (at moderate Re, many tens of D/U).
- Run a precursor until the state is periodic, then cut the clip window from the settled run. Anchor only the
  phase on the reference (which side rolls up first, at what time) by shifting the window by whole or partial
  periods.
- **Settle per band**, not only on a global force: split the wake into bands (near, mid, far, outlet) and
  require the kinetic energy of each band to change < 1 % per period. A restart sends a pulse downstream one
  band per period; wait until it has left the outlet.
- Converge on three levels (for adaptive grids: max level L, L+1, L+2). Check St **and** force amplitudes and
  fields; coarse grids match St early while C_D and C_L′ are still far off.

## CFD: inflow calibration

- Decide the velocity definition in the contract (for example the centreline approach speed at the body station
  in an empty channel).
- Calibrate the inflow per solver with an **empty-channel run**: adjust the inlet value until the defined
  velocity matches. Correlation-based inlet values can be off by a few percent, enough to move St and C_D
  outside tolerance.
- Shallow layers: add a declared linear floor friction −λu (λ from the slowest vertical mode of the layer) and
  check the shallow-wake stability number; run a λ = 0 twin.
- Correct literature comparisons for blockage.

## Lagrangian dye and mass-conserving filaments

Real dye in water barely diffuses (Sc ~ 1000). Eulerian tracers at practical resolution smear it by millimetres
and make fronts threshold-dependent. `v2p/dye.py` advects dye as material lines through the stored velocity
history instead.

1. Store the velocity history of the clip window (and the release lead-in) at the render cadence or finer.
2. Release material points continuously from the source positions, starting early enough that the front sits
   where frame 0 shows it at t = 0 (release time from measured travel of the advected particles).
3. Advect forward with RK4 at ≤ 1/(2 fps); forward material lines are much cheaper than per-pixel back-tracing.
4. Render each segment as a Gaussian filament whose width and peak follow the local stretch λ and age:
   σ² = σ₀²/λ² + 2·D_m·age, c = σ₀/(λσ). Stretched streaks thin and fade; the integral is conserved, so the
   verifier's tracer mass budget applies.
5. Write 16-bit tracer PNGs in the data-contract layout; keep an Eulerian tracer as a cross-check.

## Fracture

- Load-up: beam or FEA stress to locate the origin (maximum tension) and the failure load from a declared
  strength range.
- Path: phase-field (AT1/AT2) with a length scale resolved by ≥ 2–4 elements; verify on the standard notched tests.
- After separation: crack surfaces from the phase field define the fragment meshes; fragments become rigid bodies
  with contact.
- Propagation is faster than the frame interval in brittle solids; say so rather than animating a slow crack.

## Run hygiene

- Long runs as tracked background jobs with a log; never hand-written wait loops without a deadline and a failure
  check (see `agents-and-guardrails.md`).
- Keep each delivered version (v001, v002) with its own output directory and meta; never overwrite a version the
  lead has checkpointed.
- Report wall times per run; they set the next case's budget.
