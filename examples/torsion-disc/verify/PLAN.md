# Clip 03 (torsion disc) — independent physics verification plan

Role D (independent verifier). Written 2026-10-26 **before any production result existed**: `physics/`, `params/`,
`solver/out/` and `checkpoints/` were empty, and `solver/NOTES.md` / `physics/FORMULATION.md` did not exist yet.
Tolerances are copied from `config/acceptance.json` v1 (declared 2026-10-26T03:09:31Z). Nothing here relaxes them.
Where the acceptance text needs an interpretation for this mechanism, the interpretation is fixed below now.
Any later change goes into `verify/NOTES.md` with its reason and time.

## Independence rules
- Nothing in `solver/` is imported, copied or read for guidance. The production solver is only **run as a
  subprocess** through its documented command (P1, P3 benchmark runs, P9). Only its CSV/JSON outputs are read, as data.
- Parameters reach my code through my own loader, `verify/tparams.py`. It searches by alias and records every JSON path it
  used. A `--model-json` override in my canonical schema (fields of `tmodel.Model`) wins over everything.
- The dynamics come from my own derivation (below), coded once in `verify/tmodel.py` as Kane's equations in generalized speeds.
  Two integrators share that right-hand side:
  - **GL4**: 4-stage Gauss–Legendre collocation (order 8, implicit, symplectic). It keeps |q|² of every quaternion exactly,
    which is how attitude is "handled exactly": there is no renormalisation and no drift.
  - **GBS**: Gragg–Bulirsch–Stoer extrapolation, up to order 16, as a cross-check.
  Potential gradients are **exact**: complex-step derivatives of the twist and bending functions, with no finite differences.
  The RHS is validated independently of both integrators, by:
  - exact solutions;
  - conservation and balance laws computed from separate energy and momentum formulas;
  - steady conical orbits derived by plain Newton–Euler algebra (not from the RHS);
  - linear-mode eigenproblems.

## Model under test (my derivation; reconciled with physics/FORMULATION.md once it exists)
Anchor A = clamp point = origin (`config/coordinates.md`). Gravity is −Z, g = 9.80665.
- **Disc**: rigid, mass M, inertia I_c about its CoM C (body frame; body z = disc normal, "up" at rest). The wire attaches at
  body offset p_b from C (default (0, 0, +h/2), the centre of the top face).
- **Wire**: massless, inextensible, length L from A to the attachment P. Its unit vector u = (A − P)/L points up toward A.
  Two kinematic options, chosen to match the formulation:
  - `rigid_pivot`: the disc is locked perpendicular to the wire (disc normal a = u), so wire + disc form one rigid body about
    the fixed point A. 3 DOF, swing (2) + twist (1). State: q_d, ω_body.
  - `string`: the wire pivots freely at A, and the disc can tilt relative to the wire at P, with optional bending stiffness
    k_b (V_b = ½k_b β², or k_b(1 − u·a)). 5 DOF, wire swing (2) + disc attitude (3). State: wire frame quaternion q_w with
    no spin about u (body ω_w = (ωx, ωy, 0)), q_d, ω_db.
- **Torsion**: V_t = ½κψ². ψ is the wire twist. Its definition is a modelling choice that matters at second order in
  swing/tilt (about 1e-4 rad at 1e-2 rad swing, so above the P2 tolerance). It is therefore **selectable and must match
  FORMULATION.md**:
  - `vertical`: swing–twist decomposition about Z, R_d = Rmin(ẑ→a)·Rz(ψ).
  - `transport`: the twist transported through the kinks at A and P, R_d = Rmin(u→a)·Rmin(ẑ→u)·Rz(ψ). This is what a wire
    bent in one plane near each end actually stores.
  - `euler_zyx`: ψ = atan2(R10, R00).
  All three agree exactly in pure twist, and `vertical` ≡ `transport` when a = u.
- **Damping** (all linear, all optional; Rayleigh dissipation D, P_diss = 2D):
  - c_t ψ̇² (torsion);
  - c_s|u̇|² (wire swing at the clamp);
  - c_v|v_C|² (air, linear);
  - c_r|ω_d|² (disc rotation).
- **Equations** (Kane, generalized speeds v): M(q)v̇ = Q − M J_Cᵀ a_C0 − J_ωᵀ(ω×I_wω), with:
  - M(q) = M J_CᵀJ_C + J_ωᵀ I_w J_ω;
  - Q = J_Cᵀ(−Mgẑ − c_v v_C) − J_wᵀ∇_wV − J_ωᵀ∇_dV − c_t ψ̇ ∇ψ − c_s Bᵀ B v − c_r J_ωᵀ ω.
  ∇ denotes the gradient with respect to virtual rotations of the wire and the disc.
- **Observables**:
  - P, C, q_d, ω_d, u, ψ (all three conventions);
  - swing angles of u, and the tilt β = ∠(u, a);
  - KE, PE, E, P_diss, W_diss;
  - L_z about the vertical through A;
  - the wire force on the disc and the tension T = F·u.
- **Momentum balance**: dL_z/dt = τ_z = Q·v_α, where v_α is the global-rotation-about-Z direction.
  - With κ = 0 and no damping: L_z is conserved.
  - With torsion: L_z(t) − L_z(0) = −∫κψ dt (the clamp torque), plus the damping terms.

## Analytic benchmarks and regimes of validity
| # | Benchmark | Formula | Regime |
|---|---|---|---|
| B1 | Static equilibrium | u = a = ẑ, ψ = 0, C = A − (L + d)ẑ stays exactly | exact |
| B2 | Torsion, undamped | ψ = ψ0 cos ω_t t (+ ψ̇0/ω_t sin), **ω_t = √(κ/I_zz)**, T_t = 2π√(I/κ) | **exact for any amplitude** (linear torsion law), released in pure twist on the vertical. I_zz about the wire axis (C on axis) |
| B3 | Torsion, damped | ψ = e^{−γt}(ψ0 cos ω_d t + (ψ̇0 + γψ0)/ω_d sin ω_d t), γ = (c_t + c_r)/(2I_zz), ω_d = √(ω_t² − γ²) | exact, any amplitude, pure twist |
| B4 | Small-angle swing | rigid_pivot: ω² = Mgℓ/(I_yy + Mℓ²), ℓ = L + d. string: det(K − ω²M) = 0, M = [[ML², MLd],[MLd, Md² + I_yy]], K = [[MgL + k_b, −k_b],[−k_b, Mgd + k_b]] (out-of-plane: I_xx) | asymptotic, error O(θ²); judged at 1e-5 rad, with the θ² trend shown |
| B5 | Conical orbit (steady, κ = 0) | rigid_pivot: Ω² = −Mg[ρ×ẑ]_y / [ẑ×(I_Aẑ)]_y. string (k_b = 0): Newton–Euler on the disc with T = Mg/u_z; solve force-x and torque-y for (Ω, tilt) | exact, any cone angle, undamped, κ = 0 (torsion makes a conical orbit non-steady) |
| B6 | Energy | E = E0 (undamped); E + W_diss = E0 (damped) | exact |
| B7 | Angular momentum about the vertical | L_z = const for κ = 0, undamped. With torsion, L_z(t) + ∫(κψ + damping torques) dt = const (generic τ_z = Q·v_α) | exact |
| B8 | Decoupling limits | pure twist → swing ≡ 0 and tilt ≡ 0; planar swing (ψ = 0) → ψ ≡ 0 and y ≡ 0 (mirror symmetry); twist + 1e-5 rad swing → twist frequency ω_t and swing frequency B4 to 1e-7 | exact invariants / asymptotic |

## Checks (acceptance item → independent method)
Common: my GL4 solution at the production params on the 1/96 s grid, t = 0 … 240.75/24 s. Frame n, subframe s → row
4(n + s). Canonical columns: `t_s`, `x_m,y_m,z_m` (disc CoM C), `vx..`, `px_m,py_m,pz_m` (attachment P),
`qw,qx,qy,qz` (disc), `wx,wy,wz` (world ω), `ux,uy,uz`, `psi_rad`, `tension_N`, `KE_J, PE_J, E_J, P_diss_W, W_diss_J, Lz`.
Production columns are mapped by alias, or by an explicit `--colmap`.

- **P1 — convergence** (`check_convergence.py`): ≥ 3 production resolutions run through the documented command.
  - The two finest runs must differ by < 1e-7 m (C, P) and < 1e-7 rad (attitude, ψ).
  - The observed order must be within 0.5 of the declared order.
  - A difference below 1e-12 gives INCONCLUSIVE.
- **P2 — independent benchmark** (`check_physics.py`): GL4 (cross-checked by GBS) vs production at identical params.
  - max |ΔC|, |ΔP| < 1e-6 m.
  - max attitude angle of R_refᵀR_prod < 1e-6 rad, and |Δψ| < 1e-6 rad.
  - A finite-time Lyapunov estimate is reported.
- **P3 — analytic limits** (`check_physics.py --bench`). Production is run (subprocess) on params copies in the B2/B3 pure-twist
  configuration (any amplitude), the B4 1e-5 rad swing and B5 conical (if production accepts κ = 0). Each run is compared
  sample by sample with the closed forms.
  - B2/B3: max |ψ − ψ_exact|/ψ0 < 1e-7, and fitted ω relative error < 1e-7.
  - B4: fitted ω relative error < 1e-7.
  - If no benchmark run can be made, P3 is **NOT RUN**, never PASS.
- **P4 — energy**:
  - The E_J column is checked against my recomputation from the state.
  - Undamped variant: |E − E0|/E_osc < 1e-8.
  - Damped run: |E + W_diss − E0| < 1e-7 E_osc, where W_diss is my augmented-integral recomputation, plus the column.
  - No forward step may have dE > 1e-12 E_osc.
  - Added (not a relaxation): the L_z balance residual < 1e-8 L_ref.
  E_osc = E(0) − E_static, so that the zero level cannot hide drift.
- **P7 — forces**: tension T ≥ 0 at every sample, recomputed from my M a_C + Mgẑ + c_v v_C.
- **P8 — initial conditions and time**:
  - The production state at t = 0 vs params: |Δ| < 1e-9.
  - |t[n] − n/96| ≤ 1e-12, and the row count.
  - A best-fit time-scale estimate vs ref (0.5 % scaling → FAIL; criterion |s − 1| < 1e-6).
  - Column consistency (|P − A| = L, C = P − R p_b, u, ψ vs quaternion) to 1e-9.
- **P5 — saved, reopened .blend** (`check_blend.py`, inside Blender). Frames 0…240 × subframes 0, .25, .5, .75, evaluated
  depsgraph geometry only.
  - Anchor/clamp and all static objects keep their frame-0 matrix to < 1e-6 m, and the anchor is at A.
  - Wire endpoints (extreme vertices of the evaluated wire mesh along its axis, centroid of each end ring): top end within
    1e-5 m of A, bottom end within 1e-5 m of P_blend = disc matrix · p_attach. The wire length is constant to 1e-5 m (inextensible).
  - Topology is constant.
  - **Mark rigidity**: the mark's evaluated vertices, expressed in the disc's local frame, are constant to 1e-6 m, and so is the
    mark's matrix relative to the disc (child or separate object).
- **P6 — Blender vs solver**: the disc world matrix vs my ref, at frames + 4 subframes.
  - Position (origin = CoM, or a declared local offset) < 1e-5 m.
  - Attitude angle(R_refᵀ R_blend R_bind) < 1e-5 rad, where R_bind is fixed at frame 0, and also absolute (bind = I) when the
    builder declares that its disc local axes are the physics body axes.
  - Twist angle extracted from the blend (convention of record) vs ψ_ref < 1e-5 rad.
  - Keyframe audit: channels, key times (1/96 s keys expected), interpolation, handles vs velocity, NLA, drivers, modifiers.
- **P8 (blend part)**: fps 24, fps_base 1, frames 0–240, no time remap, unit scale 1, and the frame-0 disc pose = params IC to 1e-6.
- **P9 — controls** (`check_controls.py`). Params copies are run through the real solver (and `make_variant` re-equilibration
  when dynamics provides it). Expected values come from formulas only:
  - κ×4 → twist period ÷2;
  - I×4 → ×2 (all inertia components scaled ×4, so the B2 regime is unchanged);
  - L → L + ΔL → B4 swing period at the new L;
  - damping×10 → envelope e^{−10γt} (B3).
  Each variant runs in its clean regime (pure twist, or 1e-5 rad swing), with tolerance 1e-4 relative. In addition, the
  production variant at production ICs is compared with my GL4 at the variant params (P2-type, 1e-6).
- **P10 — repeatability** (`compare_dumps.py`): two clean rebuilds must give bitwise-identical keyframe arrays and evaluated
  disc matrices, wire end points, mark vertices and topology.

## Negative controls (synthetic; every one must be detected)
Physics (`test/run_negative_controls_physics.py`), from my ref solution with injected faults:
| Fault | Must fail |
|---|---|
| +1e-4 m position error after 5 s | P2 |
| 1e-4 rad twist error after 5 s | P2 |
| 0.5 % time scaling | P2, P8 |
| Energy growth (negative torsion damping) | P4 |
| Hidden energy drift on the undamped run | P4 |
| Attitude drift (slow extra rotation about the wire axis, 1e-5 rad/s) | P2 |
| Non-unit quaternions written raw | P8 |
| Wrong IC (1e-8) | P8 |
| Negative tension column | P7 |
| |P − A| ≠ L (wire stretch 1e-6) | P8 |
| Grid spaced n/95 | P8 |

The good controls (GL4, and GBS as a different method) must PASS everything.

Blender (`test/run_negative_controls_blend.py`), on tiny test .blends built by my own fixture (`test/make_test_blend.py`):
| Test .blend | Expected |
|---|---|
| good | PASS |
| independent rebuild of good | PASS; P10 identical |
| 1e-4 m keyframe error | P6 |
| 0.5 % time scale | P6 |
| attitude drift 1e-4 rad (twist offset growing in time) | P6 |
| wire bottom endpoint gap 50 µm | P5 |
| anchor drifting 2 µm | P5 |
| mark not rigid (extra 1e-3 rad of mark rotation relative to the disc after frame 120) | P5 |
| 25 fps scene | P6, P8b |
| whole-frame keys only (no subframe keys) | P6 at subframes |

P1: a stand-in RK4 production passes; with an injected O(h²) error term it fails on order.
P9: a stand-in that ignores κ fails κ×4.
P10: good vs rebuild PASS; good vs faulty keys FAIL.

## Not checked / limits (stated up front)
- Wire mass and torsional inertia are ignored unless FORMULATION includes them (then they are added to my model before P2).
- Air damping law and coefficients are whatever production declares: I verify consistency, not identifiability.
- Reference fidelity (R1–R5) and visual review (V1–V4) are not mine.
