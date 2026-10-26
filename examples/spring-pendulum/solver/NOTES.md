# solver/ NOTES (role B: physics + dynamics), clip 02

Status: **v005 FITTED**: solver v2 (projection + guards) and scene geometry v003 (b_c 0.209616 W_in, eye centre 0.030897 W_in below H).
* params/physics.json sha256 962fc5363a6e77ba…; trajectory.csv d45be5ab934d5bd7…; trajectory_meta.json 86b2726a6315…; solver source 708651c7cd86….
* Regenerate with `solver/regen_fitted_params.sh`, then run `run.py` and `selfcheck.py`.
* **Changes from v003 to v005.** Fit refit with the new b_c and e; the change is immaterial. Values are v003 → v005:

  | quantity | v003 | v005 |
  |---|---|---|
  | fit-window RMS | 6.727 px | 6.724 px |
  | hold-out | 15.46 px | 15.46 px |
  | ell | 0.32678 m | 0.32801 ± 0.00088 m |
  | Lcam | 0.78439 W_in | 0.78733 W_in (absorbs the eye-offset change) |
  | S_m | 0.41660 | 0.41660 m/W_in |
  | b | 88.55 mm | **87.33 mm** (= the mesh) |
  | rocking mode | 26.36 rad/s | 26.16 rad/s |
  | κ | 1.0546 | 1.0551 |

  * The diagnostic fit on 0–4 s gives ell 0.32037 m, window RMS 5.23 px.
  * Other numbers in the v003 summary below still hold to the quoted precision.
* **Solver v2 effect on the nominal trajectory:** 7.7e-15 m. v2 adds the guards, and selfcheck adds the checks `constraint_1e-12`, `compression_guard` and the variant convergence check.

## v003 summary (production)
* **Model.**
  * Elastic link (spring) from the swing pivot P = P_eff to the hinge H. P_eff is the top coil on the fixed S-hook, (−0.002875, 0, −0.225071) W_in (lead decision).
  * Rigid steel cylinder hinged at H (eye inner top), Rayleigh spring mass, Q = 500 assumed.
  * k from the coil formula (assumed, G 79.3 GPa, 21 active turns).
  * Axial and rocking modes start at rest: quasi-static stretch, and ψ on the swing-mode shape.
* **Geometry** (params/geometry.json v002, W_in): r 0.105786, h 0.260333, H→cylinder centre 0.21256, H→eye centre 0.03384,
  wire d 0.00943, coil D 0.10518 (centreline).
  * The lead's message quoted d 0.0097 / D 0.108. With those values k would be about 4 % higher, well inside the k range.
* **Fit inputs and free parameters.**
  * Tracks: ring = eye centre (σ ≈ 0.55 px), weight centre (σ ≈ 1.5 px), through `camera/project.py` per-frame cameras.
  * Free: ell, Lcam, phi0, phid0.
  * With the scene v002 geometry, the weight-marker nuisance dw = 0.0009 ± 0.0014 W_in is consistent with 0, so it was dropped
    (diagnostic run kept in `solver/out/fit/diag_dw_Ta7.5`).
* **Fitted values** (sd = max(bootstrap, scaled formal); 30 moving-block resamples):
  * ell = |P−H| = **0.32678 ± 0.00088 m** (model-conditioned), boot 95 % [0.32512, 0.32810].
  * Lcam = 0.78439 ± 0.00046 W_in, so **S_m = 1/λ = 0.41661 m per W_in**.
  * phi0 = −0.1452 ± 0.0032 rad (−8.32°); phid0 = +0.052 ± 0.019 rad/s.
  * No flags. Corr(ell, phid0) = +0.85. Reduced χ² = 76.7 (correlated AI residuals).
* **Derived quantities:**

  | quantity | value | provenance |
  |---|---|---|
  | b | 88.6 mm | |
  | r | 44.1 mm | |
  | h | 108.5 mm | |
  | m | 5.23 kg | assumed steel |
  | m_s | 0.275 kg | |
  | k | 1336 N/m | assumed |
  | L0 | 0.2874 m | |
  | static stretch | 39.4 mm | |
  | ω_sw | 4.861 rad/s (T = 1.293 s) | |
  | rocking mode | 26.36 rad/s | |
  | ω_ax | 15.85 rad/s (ω_ax/ω_sw = 3.26, off the 2:1 resonance) | |
  | κ (swing-mode ψ/φ) | 1.055 | |
  | tension | 51.1–54.7 N | |
* **Metrics.**
  * Fit-window RMS **6.73 px** (ring 4.92/1.23 px u/v; weight 8.00/0.92 px).
  * R2 target max(2, 3σ) = 3.24 px: **FAIL**. Cause: the AI half-period lengthens by 7.5 %, and the AI weight tilt is 10.5–11.9° against 8.5° for the physical in-phase tilt.
  * Hold-out 7.5–10 s: 15.46 px (ring 13.6, weight 17.1). Last turning points −21/−37/−57/−80 ms.
  * Supplementary n ≡ 0 mod 4 frames: 6.81 / 13.69 px.
* **Diagnostic fit on 0–4.0 s** (`solver/out/fit/diag_Ta4.0`, visuals `v003_diagTa4_*`):
  * ell = 0.31918 ± 0.0011 m: −2.3 % against production. This is the period-wander systematic.
  * Window RMS 5.24 px; after the window 26.96 px; −167 ms at the last turning point.
* **Damping.**
  * M1 drives γ_sw to the 0 bound in both windows; bootstrap 95 % γ_sw is [0, 0.0139] s⁻¹. BIC prefers M1 only through γ_ax, which is a nuisance on stretch noise.
  * Not identified, so Q = 500 is a declared assumption (range 100–5000). AI amplitude growth is never fitted as negative damping.
* **k** (`solver/out/fit/k_sensitivity.json`), range of |PH| over 10 s:

  | k | ω_ax/ω_sw | range of |PH| |
  |---|---|---|
  | × 0.5 | 2.31 | 15.3 mm (~25 px) |
  | × 1 | 3.26 | 2.65 mm (~4 px) |
  | × 2 | 4.61 | 0.9 mm (~1.5 px) |

  * The reference's ±1.2 px rms radial motion disfavours soft springs near the 2:1 band.
  * A diagnostic fit of rax to the swing-locked 2ω stretch ripple gives k = 1491 N/m (1.12× the coil value).
* **Earlier variant diagnostics** (scene v001 geometry, bootstrap 0; not adopted): free rocking IC, static tilt β = 0.08°, body offset α = −0.6°.
  * Each is sub-degree, or would let the fit chase AI systematics. See v002 history below.
* **Bake.** Test rig at 1/96 s keys: P6 max 4.7e-8 m, 3.0e-8 rad, 6e-8 on the rotation matrix; between samples 5.2e-8 m; anchor 0.
* **selfcheck** (fitted params): ALL PASS.
  * P1 finest pair 3.1e-13 m, orders 4.00.
  * P3: axial 8.6e-10, lateral modes 2.0e-13 / 4.5e-11, point swing 2.9e-13.
  * P4 conservative drift 1.8e-12; damped balance 1.9e-12; max per-sample increase < 0.
  * Production (h = 1/3072) vs the fit model (h = 1/768): 8.4e-11 m.

### Notes for role C (scene) — v002 gauge and rig
* Use `physics.json S_m.value` (0.41661 m per W_in) as the gauge. The scene's old rule ("ell_eq/1.00582") no longer applies, because ell_eq is now |P−H|.
* `pivot_world_m` = (−1.20 mm, 0, −93.8 mm) is the spring's top end. The S-hook A→P is static scenery, and the spring mesh runs P → VSP_HookLower.
* The trajectory's H and X are world positions with A at the origin.

## Interface requests (to the measurement/camera role A and the scene role C)

### For A (measurement): what the fit consumes
Please deliver per-frame image tracks (all 241 frames, contract pixel convention, method A plus the sigma from 2 methods) in
`measurement/` for these points.  The CSV name and columns are flexible, but this layout loads without changes:
`measurement/tracks.csv`: `frame,t_s,<pt>_u_px,<pt>_v_px,<pt>_sigma_u_px,<pt>_sigma_v_px` for each `<pt>` in
* `hook`: the lower hook/ring contact point H (where the weight's ring hangs on the spring's lower hook) -- the physical hinge;
* `wctr`: the weight cylinder's geometric centre (silhouette centroid, or the midpoint of the top/bottom face centres);
* optional `wbot` (bottom-face centre) and `spring_top` (top of the coil, a consistency check of A).
Static/geometry values in `measurement/summary.json` (any key names; please document them):
* A (top-hook bearing point on the crossbar) in pixels, H at rest,
* image ratios: |A-H| at rest / cylinder height, cylinder diameter / height, H -> top face, ring outer diameter,
* the anchor window T_a from the CHANGELOG rule (per mode: lateral swing of the hook, axial bob = |A-H|, weight tilt), with the evidence.
Weight tilt is the angle of the cylinder axis in the image; the fit predicts it from wctr - hook, so a separate tilt track is a
cross-check, not required.

### For C (scene/rig): the rig contract that `solver/bake_blender.py` drives
* `VSP_Anchor` (empty) at A = world origin, never animated.
* `VSP_HookLower` (empty): **location keyed** = H(t) (world), no rotation.  The spring object(s) are driven by A and H
  (e.g. a stretch-to / track-to on a spring mesh rooted at A, or geometry nodes reading both empties): the coil count is fixed
  and the pitch follows |A-H|.  The rigid top hook and bottom hook rotate with the spring axis A->H.
* `VSP_Weight` (empty or mesh, unparented): **origin at the hinge H** (ring/hook contact), body axis = local -Z pointing from
  the hinge to the centre of mass; location keyed = H(t), rotation_euler (XYZ) keyed from the solver quaternion
  (spin about the own axis is zero, but the twist from the geometric phase IS included, so the ring plane may turn slightly).
  Weight mesh children: ring/eye centred so that the ring's contact point is at the origin; cylinder centre at local
  (0, 0, -b_c) with b_c from params/physics.json `geometry_m.H_to_cylinder_centre`.
* Keys on every frame 0..240 AND every 1/96 s sample in between (keys-per-frame 4), FREE Bezier handles at +-1/12 frame from
  the solver velocities (cubic Hermite). VSP_HookLower may carry role C's Copy-Location constraint to VSP_Weight (fine: same H).
* Custom properties `vsp_params_sha256`, `vsp_trajectory_sha256` are written by the bake.

## Progress log
* Solver `solver/spring_pendulum.py` (5-DOF elastic pendulum + hinged rigid cylinder, Rayleigh spring mass, RK4 h = 1/3072 s,
  batched), `run.py`, `selfcheck.py` (P1/P3/P4/P7/P8 all PASS on placeholder params: finest-pair 1.3e-12 m, order 4.00,
  P3 axial 1.5e-10 rel, lateral modes 7.7e-13 / 6.3e-11, point swing 8.9e-13, conservative drift 6.6e-12, balance 4.6e-12).
* Parameter mapping single-sourced in `solver/model_build.py` (fit and make_params share it). Geometry ratios to ell from
  role C (scene/NOTES.md): r/ell 0.1138, h/ell 0.2853, spring d/ell 0.00963, D/ell 0.1074, 22 turns; b is FITTED (bcam).
* Bake: `solver/bake_blender.py` keys VSP_HookLower.location and VSP_Weight.location + rotation_euler with FREE Hermite
  handles on every frame AND every 1/96 s sample (keys-per-frame 4; frame-only keys miss P6 by 1.6e-4 rad because of the
  ~4.6 Hz rocking mode). Test rig: P6 max 3e-8, between-sample check 1.4e-7, anchor 0.
* Fit tooling validated on synthetic data (solver/test/): clean and fake-amplitude-growth cases recover the truth within the
  bootstrap sd (|z| < 4 with 6 resamples); visuals `solver/visuals/test/t001_*`.
* (lead request) The hinge->cylinder-centre offset is now the MEASURED ratio b/ell = 0.2268 (role C) in both the fit and
  params (model_build GEOM_DEFAULT b_mode "measured_ratio"; `--free-bcam` is a diagnostic only). Placeholder physics.json
  regenerated: b = 0.06804 m at ell = 0.30 m (was 0.042), r 34.1 mm, h 85.6 mm, spring d 2.89 mm / D 32.2 mm / 22 turns.
  Geometry provenance per ratio is in physics.json geometry_m.provenance.
* Axial bob is tiny in the reference (lead: |A-H| 1.008/1.006/1.008 S): if omega_axial is not identified in the anchor window
  the fit runs with `--rax-fixed <value>` (declared assumption) and reports the sensitivity.

## For the independent verifier (role D) — exact interface (lead request, CHANGELOG)
### (1) Parameter names (params/physics.json, model_id `elastic_pendulum_hinged_body_v1`)
Read `parameters.<name>.value`: `g` [m/s²], `k` [N/m], `L0` [m] (effective free length A->H incl. hooks/preload), `m` [kg] weight,
`m_s` [kg] spring, `b` [m] hinge H -> weight CoM, `I_perp` [kg m²] transverse inertia about the CoM,
`damping.c_s` [N s/m] (axial dashpot on d|P_eff H|/dt), `damping.c_x` [N s/m] (drag on the CoM velocity), `damping.c_r` [N m s] (on dn/dt).
Also informational: `ell_eq`, `omega_axial`, `damping.gamma_*`. Initial state: `initial_state.X0/V0/n0/u0` (`.value`, 3-vectors RELATIVE TO THE PIVOT P = `pivot_world_m.value` (world = P + state),
CoM position/velocity, body axis n (unit, hinge->CoM), dn/dt). Spin about the axis = 0. Integrator: `integrator.h_s` (1/3072 s).
Loader: `solver/spring_pendulum.model_from_physics_json` (do not import; spec above is complete).
### (2) Initial conditions as eye/hinge state
`solver/out/trajectory_meta.json` → `initial_state_hinge`: `H0_m`, `Hdot0_m_s`, `n0`, `ndot0_1_s`, `omega0_world_rad_s` (= n0 × ndot0),
plus `X0_com_m`, `V0_com_m_s`. Relation: H = X − b n, Ḣ = V − b ṅ. World = gravity frame (Z up, A = origin); a fitted static
tilt `level_tilt_rad` (if any) applies to the static scene/camera only, never to the solver state.
### (3) Spring-mass law (same as the verifier's)
Rayleigh, linear velocity profile: kinetic ½ (m_s/3) |Ḣ|², gravity m_s g H_z / 2 (i.e. m_s/2 at the eye). Massless otherwise
(no internal modes), straight A->H (two-force member). Full Lagrangian / dissipation: physics/FORMULATION.md §3:
`T = ½m|Ẋ|² + ½I⊥|ṅ|² + ½(m_s/3)|Ẋ−bṅ|²`, `V = ½k(|H|−L0)² + m g X_z + ½ m_s g H_z`,
`D = ½c_s(ŝ·Ḣ)² + ½c_x|Ẋ|² + ½c_r|ṅ|²`; ω₃ = 0; the orientation quaternion integrates q̇ = ½(0, n×ṅ)⊗q from the minimal
rotation local −Z → n0. Energy columns: PE zero at static equilibrium (|P_eff H| = L0 + (m + m_s/2)g/k, straight down).
### (4) Step / convergence and commands
* Run: `.venv/bin/python solver/run.py [--params params/physics.json] [--out DIR] [--h STEP]` (STEP must divide 1/96 s;
  default 1/3072). Output `DIR/trajectory.csv` + `trajectory_meta.json`, samples t = k/96 exactly (frame n = row 4n).
* Self-check (P1 ladder h_prod×16 … h_prod/2, P3, P4, P7, P8): `.venv/bin/python solver/selfcheck.py` → `solver/out/selfcheck.json`.
### Trajectory columns (solver/out/trajectory.csv, also listed in trajectory_meta.json `columns`)
`t_s, X_x_m X_y_m X_z_m` (CoM), `V_*_m_s`, `n_x n_y n_z`, `ndot_*`, `H_*_m` (eye/hinge = spring lower end), `Hdot_*_m_s`,
`q_w q_x q_y q_z` (weight orientation, local −Z → n), `eul_x/y/z_rad` (Blender XYZ, unwrapped), `euld_*_rad_s`,
`spring_len_m` (|P_eff H|), `spring_rate_m_s`, `spring_tension_N` (k(|P_eff H|−L0), P_eff = pivot_world_m), `phi_x_rad phi_y_rad` (spring angle about P_eff: atan2(H_x−P_x, −(H_z−P_z)), atan2(H_y−P_y, −(H_z−P_z))),
`psi_x_rad psi_y_rad` (body axis, same convention), `KE_J PE_J E_J P_diss_W W_diss_J`.
### Bake (CHANGELOG): keys on every 1/96 s sample (keys-per-frame 4) for VSP_Weight location + rotation_euler and
VSP_HookLower location, FREE Hermite handles ±1/12 frame from the velocities (`solver/bake_blender.py`, default).

### v2 integrator (verifier P9 finding, CHANGELOG 2026-10-26T01:53:54Z) and P9 variants
* **Solver v2** (`rk4-fixed-step-elastic-pendulum-v2-projected`): after every RK4 step, (n, n', q) are projected exactly
  (n ← n/|n|, n' ← n' − (n·n')n, q ← q/|q|; O(h⁵) per step, so 4th order is kept). The multiplier uses |n|² so that n·n'' = −|n'|² holds exactly.
* **Guards** (fail loudly):
  * `SolverGuardError` on a non-finite state, on | |n|−1 | > 1e-12 or |n·n'| > 1e-12 after projection, and on a per-step pre-projection drift > 1e-8 (step too large).
  * `SpringCompressionError` on tension < 0 at any step node: a hooked spring cannot push (P7).
  * `integrate(..., on_compression="record")` exists only for diagnostics; the fit uses it and rejects compressive parameter vectors.
* **Nominal production trajectory** is unchanged to 7.1e-15 m (|Δn| 9.6e-14). New `trajectory.csv` sha256 = da42bd33dd9e76a0…
  * The change is in bytes only. trajectory_meta now also carries `integrator_stats`: max step drift 2.2e-16, min tension 51.11 N.
* **P9 variants with a re-equilibrated initial state.** Use the variant's own static stretch, the SAME swing amplitude/phase (phi0, phid0 from
  physics.json `primary`), and start the axial and rocking modes at rest:
  ```
  .venv/bin/python solver/make_variant.py --set k=x4 --out solver/out/variants/k_x4.json      # NAME=xF multiplies, NAME=V sets
  .venv/bin/python solver/make_variant.py --set L0=x1.5 --out solver/out/variants/L0_x1.5.json
  .venv/bin/python solver/run.py --params solver/out/variants/k_x4.json --out solver/out/variants/k_x4
  ```
  * Python API: `model_build.reequilibrate(M_variant, phi0, phid0)`. It sets ell = L0 + (m + m_s/2)g/k, the quasi-static stretch for the initial swing
    state, and ψ = κφ on the variant's linear swing-mode shape.
  * Results:

    | variant | ell_eq | min tension | h vs h/2 |
    |---|---|---|---|
    | k×4 | 0.2973 m | 51.9 N | 4.1e-13 m |
    | L0×1.5 | 0.4705 m | 51.5 N | 2.1e-13 m |

  * The nominal-IC variants now raise `SpringCompressionError`: k×4 at t = 0.0605 s (tension −1.1 N), L0×1.5 at t = 0 (−140 N).
  * Run with compression recorded, they stay finite and |n| = 1. h vs h/2 is 1.5e-9 m for k×4 and 8.3e-7 m for L0×1.5 (violent compressive motion).
  * The verifier should build variants by the same rule; the JSON carries the full re-equilibrated state (relative to P).
* Visual: `solver/visuals/v004_integrator_fix_P9.png`.
