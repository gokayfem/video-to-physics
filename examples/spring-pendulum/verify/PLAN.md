# Clip 02 (vertical spring / elastic pendulum) — independent physics verification plan

Role D (independent verifier). Written 2026-10-26 **before** any production result existed:
`physics/`, `params/`, `solver/out/` and `checkpoints/` were all empty when this was written.
Tolerances come verbatim from `config/acceptance.json` v1 (declared 2026-10-25T23:40:54Z). Nothing here
relaxes them. Where the acceptance text needs interpretation for this mechanism, the interpretation is fixed
below now. Any later change is logged in `verify/NOTES.md` with its reason and time.

## Independence rules
- Nothing in `solver/` is imported, copied or read for guidance. The production solver is only ever **run as
  a subprocess** through its documented command (P1, P3 benchmark runs, P9), and only its CSV/JSON outputs
  are read, as data.
- Parameters reach my code through my own loader `verify/vparams.py`. It searches by alias and records every
  JSON path it used in the report. If the schema is ambiguous, a `--model-json` override in my canonical
  schema (below) wins over everything.
- My integrators are an implicit **Gauss–Legendre collocation method** (s = 4, order 8, symplectic; it keeps
  |q|² of the attitude quaternion exactly) and, as a cross-check, **Gragg–Bulirsch–Stoer extrapolation**
  (modified midpoint + Aitken–Neville in h², up to order 16). Neither is the classical RK4 or DOP853 used in
  clip 01's production code. Both share one right-hand side, `verify/vmodel.py`, which I derived myself (see
  below). The RHS is validated independently of both integrators: by exact solutions, by conservation laws
  computed from separate energy and momentum formulas, and by an analytic linear-mode eigenproblem.

## Model under test (my own derivation; reconciled with physics/FORMULATION.md once it exists)
Anchor A (origin, per `config/coordinates.md`), gravity −Z, g = 9.80665 m/s².
- Spring: linear, straight, from A to the weight's eye E. Length ℓ = |E − A|, axis û = (E − A)/ℓ.
  Tension T = F_i + k(ℓ − L0) (preload F_i defaults to 0). The force on the eye is −T û.
- Spring mass m_s (optional, Rayleigh): the spring stretches uniformly, so the point at fraction s along it
  moves at s·v_E. This gives KE_spring = ½(m_s/3)|v_E|² exactly, and gravity m_s g z_E/2 (plus a constant).
  It is implemented as a point mass μ = m_s/3 rigidly attached at E, with an extra gravity force
  −(m_s/2 − μ)g ẑ applied at E. With m_s = 0 the spring is massless.
- Weight:
  - **point**: mass M at E.
  - **rigid**: mass M, inertia I_c about its CoM C, eye at body-frame offset e_b from C, orientation quaternion
    q. The composite body (weight plus μ at E) has state (x_G, v_G, q, ω_body) and obeys Newton–Euler
    equations about the composite CoM G.
- Damping (light, linear, every term optional):
  - −c_v v_C on the weight CoM (isotropic air drag);
  - −c_a ℓ̇ û at E (axial spring damping);
  - −c_r ω on the body (rocking/hook friction).
  P_diss = c_v|v_C|² + c_a ℓ̇² + c_r|ω|².
- Energy: E = ½M_t|v_G|² + ½ωᵀI_Gω + ½k(ℓ−L0)² + F_i(ℓ−L0) + g(M z_C + (m_s/2) z_E).
- Angular momentum about A: L = M_t (x_G − A) × v_G + R I_G ω_b.
  - L_z is conserved when c_v = c_r = 0 (axial damping c_a keeps L_z).
  - With g = 0, all three components of L are conserved.

## Analytic benchmarks and their regimes of validity
| # | Benchmark | Formula | Regime |
|---|---|---|---|
| B1 | Static equilibrium | ℓ_eq = L0 + (W − F_i)/k, W = (M + m_s/2) g | exact; at rest, hanging vertically |
| B2 | Axial oscillation (exact) | z_E(t) = −ℓ_eq − a e^{−γt}(cos ω_d t + (γ/ω_d) sin ω_d t), ω_z² = k/M_t, M_t = M + m_s/3, γ = (c_a + c_v)/(2M_t), ω_d = √(ω_z² − γ²) | exact for any amplitude while T ≥ 0, released on the vertical with no lateral motion. For rigid: CoM straight below the eye. Damping uses c_v here only for the point model |
| B3 | Conical (circular) orbit (exact) | given α: ℓ = L0 + (W/cos α − F_i)/k, Ω² = (W/cos α)/(M_t ℓ), ρ = ℓ sin α, z = −ℓ cos α; x + iy = ρ e^{iΩt} | exact for any α < 90°, undamped, point model. A true 3-D test that exercises both lateral axes and L_z = M_t ρ²Ω |
| B4 | Small-angle swing | ω_s² = W/(M_t ℓ_eq), i.e. ω_s = √(g/ℓ_eq) when m_s = 0 | asymptotic as amplitude θ → 0. Relative error O(θ²). Judged at θ ≤ 1e-4 rad and shown to converge like θ² |
| B5 | Rigid rocking: planar linear modes | det(K − ω²M) = 0 with M = [[M_t ℓ², M ℓ d], [M ℓ d, M d² + I_yy]], K = diag(W ℓ, M g d), at ℓ = ℓ_eq, d = \|e_b\| (eye straight above the CoM). Axial ω_z² = k/M_t is exact (B2). Out-of-plane modes use I_xx | asymptotic as amplitude → 0. Judged at 1e-5 rad mode amplitude |
| B6 | Energy | E(t) = E0 (undamped). E(t) + ∫P_diss dt = E0 (damped) | exact |
| B7 | Angular momentum | L_z conserved (undamped, or c_a only). Full L conserved for g = 0 | exact |
| B8 | 2:1 autoparametric resonance | tuned by ω_z = 2ω_s (point, m_s = 0: k = 3Mg/L0). Linear-phase swing growth rate σ = (3/4) ε ω_s with ε = a/ℓ_eq, where a is the axial amplitude. Detuned: σ = √((3εω_s/4)² − (δ/2)²) | asymptotic (multiple scales), relative error O(ε). Quantitative: \|σ_meas/σ − 1\| → 0 as ε → 0. Qualitative: near-complete energy transfer, axial ↔ swing |

I derived B8 myself (multiple scales on m_k(ℓθ̈ + 2ℓ̇θ̇) + W sinθ = 0 with ℓ = ℓ_eq(1 + ε cos 2ω_s t)). It is
checked against my own solver before it is used on production data.

## Checks (acceptance item → independent method)
Common: my ref solution at the production params, on the 1/96 s grid, t = 0 … 240.75/24 s. Frame n with
subframe s maps to row 4(n + s) exactly. Canonical CSV columns: `t_s, x_m,y_m,z_m` (eye E),
`vx_m_s..` (v_E), `xc_m..` (weight CoM), `qw,qx,qy,qz`, `wx,wy,wz` (world ω), `ell_m, tension_N, KE_J,
PE_J, E_J, P_diss_W, W_diss_J, Lz`. Production columns are mapped by alias, or by an explicit `--colmap`.

- **P1 — convergence** (`check_convergence.py`). Production is run through its command interface, via a
  template, at ≥ 3 step sizes (or tolerances). The tolerance is max over xyz and samples of the difference
  between the two finest runs, which must be < 1e-7 m (and < 1e-7 rad for attitude, if rigid). The observed
  order log(e1/e2)/log r must be within 0.5 of the order dynamics declares. Round-off-level differences
  (< 1e-12) give INCONCLUSIVE, never PASS. Interpretation for adaptive solvers: the error-vs-tol exponent,
  disclosed as such.
- **P2 — independent benchmark** (`check_physics.py`). My GL4 solution (cross-checked by GBS to ≲ 1e-11)
  vs production at identical params. The tolerance is max |ΔE_pos| < 1e-6 m and max attitude angle
  < 1e-6 rad (velocities reported). A finite-time Lyapunov estimate from two of my runs is reported. If the
  motion is chaotic, that explains, but does not excuse, a FAIL.
- **P3 — analytic limits** (`check_physics.py --bench`). Production is run (subprocess) on params copies in
  the B2 / B3 / B4 (B5 if rigid) configurations, and compared sample-by-sample with the closed forms, plus
  least-squares frequency fits.
  - B2/B3: max |x − x_exact| / amplitude < 1e-7, and fitted ω relative error < 1e-7.
  - B4/B5: fitted ω relative error < 1e-7 at amplitude ≤ 1e-4 rad.
  B1 is checked on a "release at rest at ℓ_eq" run: max |E − E_eq| / ℓ_eq < 1e-7. If no benchmark run is
  supplied, P3 is **NOT RUN**, never PASS.
- **P4 — energy.**
  - The trajectory's E_J column is compared with my recomputation from the state and params. KE and E must
    agree to 1e-9 E0. PE may differ by a constant zero-level offset.
  - Undamped run: |E(t) − E0| / E0 < 1e-8.
  - Damped run: |E(t) + W_diss(t) − E0| < 1e-7 E0. W_diss uses my P_diss recomputation, integrated by
    quintic-Hermite Gauss quadrature using my accelerations, and the column if present.
  - "No growth": forward difference of E ≤ 1e-12 E0 between samples, on the damped run.
  - Added (not a relaxation): L_z drift < 1e-8 L_ref in the undamped run, L_ref = M_t ℓ_eq v_max.
  E0 = E(0) − E_static_min, so that the zero level cannot hide drift. (Energy measured above static
  equilibrium. The PE zero level is arbitrary, so relative drift is quoted against the oscillation energy.)
- **P7 — spring force admissibility.**
  - Tension column ≥ 0 at all samples, and equal to F_i + k(ℓ − L0) recomputed (to 1e-9 relative).
  - Hooked extension spring: compression is inadmissible, so min T ≥ 0 is required from the recomputation
    too, and ℓ ≥ L0 when F_i > 0 (coils closed).
  - If a force vector column exists, its direction is checked: −û (toward A).
- **P8 — initial conditions and time.**
  - Position, velocity, attitude and ω at t = 0 vs params: |Δ| < 1e-9.
  - |t[n] − n/96| ≤ 1e-12 s and the row count.
  - Frequency of the dominant axial and swing modes in the data vs my ref (a 0.5 % time scale shows as
    5e-3; criterion 1e-6 relative).
  - Geometric consistency of the reported columns: ℓ_m = |E − A| and xc = E − R e_b, to 1e-9.
- **P5 — saved, reopened .blend** (`check_blend.py`, runs inside Blender). Every frame 0…240 × subframes
  0, .25, .5, .75 is sampled from the evaluated depsgraph geometry, never custom properties:
  - Anchor: the anchor object and every static (non-weight, non-spring, non-camera) object keep their
    frame-0 world matrix. Drift < 1e-6 m. Anchor world position = A.
  - Spring endpoints: the end-vertex centroids of the evaluated spring mesh, taken along the spring axis.
    The top end is within 1e-5 m of A (+ the declared offset), and the bottom end within 1e-5 m of the
    weight eye (+ offset). No sliding: the top end is constant in world coordinates and the bottom end is
    constant in the weight's local frame, to 1e-5 m.
  - Coil count: per sample, the number of connected wire crossings through 8 half-planes about the spring
    axis (an integer per half-plane, from mesh connectivity). Must be identical at every sample.
    Vertex, edge and face counts must be constant. Informational: mean coil radius and wire radius.
  - Eye vs spring attachment: |E_blend − spring bottom end| (same as above).
- **P6 — Blender vs solver.** E_blend (eye locator world position, or `--eye-local` on the weight object)
  vs my ref: < 1e-5 m. The weight's rotation increment R_b(t)R_b(0)ᵀ vs R_ref(t)R_ref(0)ᵀ must be within
  1e-5 rad (rigid). For the point model, the declared visual-orientation rule is checked (`fixed` or
  `spring_axis`). The same comparison vs trajectory.csv is secondary. Also audited: F-curve channels, key
  count, interpolation, handle slopes vs velocities, NLA/drivers/modifiers, other animated objects.
- **P8 (blend part).** fps 24, fps_base 1, frames 0–240, no time remap, unit scale 1, and E_blend(frame 0)
  equal to the params IC to < 1e-6 m.
- **P9 — controls** (`check_controls.py`). Edited params copies run through the real solver (and the
  build/bake plus `check_blend.py` when a build command is given). Expected values come from formulas only:
  - k×4 → axial period ÷2 (B2).
  - M×4 → axial period ×√((4M + m_s/3)/(M + m_s/3)), which is ×2 for m_s = 0.
  - L0 → L0 + ΔL → swing period from B4 at the new ℓ_eq.
  - damping ×10 → axial envelope e^{−10γt}.
  Each variant is run in the clean regime (axial-only release, or 1e-4 rad swing), tolerance 1e-4
  relative. Also: production variant at the production ICs vs my ref at the same variant params (P2-type
  check, 1e-6 m).
- **P10 — repeatability** (`compare_dumps.py`). Two clean rebuilds must give bitwise-identical keyframe
  arrays and evaluated eye positions, spring end points and coil counts.

## Negative controls (synthetic; all must be detected)
Physics, from my ref solution with injected faults:
- +1e-4 m position error after 5 s → P2.
- 0.5 % time scaling → P2, P8.
- Energy growth: negative damping c → −c → P4. Hidden drift on the undamped run → P4.
- Wrong IC (1e-8) → P8.
- Compressive spring (ℓ < L0) and a tension column with the wrong sign → P7.
- ℓ column inconsistent with xyz → P8.
- Grid labelled n/96 but spaced n/95 → P8.
- Good controls (my GL4 run, and GBS as a different method) must PASS everything.

Blender, on test .blends I build in `verify/test/blends/`:
- good → PASS.
- 50 µm spring endpoint gap → P5.
- Anchor drifting 2 µm → P5.
- Coil count +1 at frame 120 (shape key, same topology) → P5.
- 1e-4 m keyframe error → P6.
- 0.5 % time scale → P6.
- 25 fps scene → P6, P8b.

P1: a stand-in RK4 passes; with an injected O(h²) term it fails on order.
P9: a stand-in that ignores k fails k×4.
P10: good vs rebuild PASS, good vs faulty keys FAIL.
