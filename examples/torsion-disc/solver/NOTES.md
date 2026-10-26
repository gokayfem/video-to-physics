# solver/ NOTES (role B: physics + dynamics), clip 03 torsion disc

Status: see "Current status" at the bottom (updated with every params version).

## Verifier interface (role D; read this first)
* **Model**: `physics/FORMULATION.md` §3, model id `disc_on_wire_rigid6_v1`. Rigid uniform disc (m, R, t), wire attached at
  B = top-face centre, inextensible wire |B − P| = ell from the suspension point P (= `pivot_world_m`; A + fitted horizontal
  offset). Potential `V = m g X_z + ½ κ ψ² + k_A (1 + s_z) + k_B (1 − s·n)`, ψ = 2 atan2(q_z, q_w) (swing–twist about the body
  z axis), s = (B − P)/|B − P|, n = R(q)(0,0,−1). Rayleigh `D = ½ c_t ψ̇² + ½ c_x |V|² + ½ c_r |ω − (ω·n)n|²`, ψ̇ = G·ω with
  `G = [w y + z x, z y − w x, w² + z²]/(w² + z²)`. Inertia: I_z = m R²/2, I_⊥ = m(3R² + t²)/12 about the CoM X = B − (t/2) e3.
* **Parameters** (`params/physics.json`): `parameters.{m, R, t, ell, kappa, k_A, k_B, g}.value`,
  `parameters.damping.{c_t, c_x, c_r}.value`; ICs `initial_state.{X0 (CoM, m), V0 (m/s), q0 (w,x,y,z body→world), w0 (WORLD
  angular velocity, rad/s)}` — **relative to the pivot** `pivot_world_m.value` (world = pivot + state). `mark.mu_body_rad` = mark
  azimuth in the body frame (rendering only).
* **Production integrator**: RK4, h = 1/3072 s (`integrator.h_s`), output every 1/96 s; `solver/run.py --params P --out DIR --h H`
  (H must divide 1/96 s) for convergence runs. Python API: `torsion_disc.integrate(Model, t_out, h)` → (N, 14) state
  [X(3), V(3), q(4), ω_world(3), W]; `torsion_disc.model_from_physics_json(json)`.
* **trajectory.csv columns**: see `solver/run.py` docstring (t_s, X, V, B, Bdot, q, w, eul, euld, psi, psidot, mark_azimuth, s,
  phi_x/y, theta_x/y, tilt, bend_B, wire_tension_N, wire_torque_z_N_m, KE/PE/E, PE_twist, PE_bend, P_diss, W_diss, qdot).
  Positions are WORLD (A origin, pivot added).
* **Analytic limits** (P3, `solver/selfcheck.py`): pure twist is an exact linear damped oscillator at any amplitude; linear
  in-plane swing+tilt modes from `M = [[m l², m l h],[m l h, m h² + I_⊥]]`, `K = [[m g l + k_A + k_B, −k_B],[−k_B, m g h + k_B]]`,
  h = t/2; ideal-string planar pendulum `T = 4 sqrt(l/g) K(sin²(φ0/2))`.
* **P9 variants**: `solver/make_variant.py --set kappa=x4 --out ...json` writes a variant with a re-equilibrated IC (own
  equilibrium, same twist ψ0/ψ̇0 and swing angles, tilt mode on the variant's own swing-mode shape). Predicted response:
  ω_t = sqrt(κ/I_z); lateral modes from det(K − ω²M) = 0 (written to the variant's `derived`).

## Bake contract (agreed with scene/NOTES.md)
* `TOR_Anchor` empty at the suspension point = `pivot_world_m` (not keyed). **Scene: place it there, not at A**, if the fitted pivot
  offset is non-zero (it is ~0.02 D, see status).
* `TOR_Disc` origin = B (top-face centre), local +Z = disc up-normal, local +X = zero-twist reference; mark (full diameter)
  along local azimuth `mark.mu_body_rad` (scene custom property `tor_mark_azimuth_rad`). `rotation_mode = 'QUATERNION'`.
  `solver/bake_blender.py` keys `location` (B) and `rotation_quaternion` (q, slopes q̇) at every 1/96 s (4 keys/frame), FREE
  Bézier handles = cubic Hermite; checks P5/P6 at frames + subframes and midpoints.
  `blender -b X.blend --python solver/bake_blender.py -- --trajectory solver/out/trajectory.csv --save --report solver/out/bake_check.json`
* `TOR_Wire`: scene GN straight wire TOR_Anchor → TOR_Disc origin (visual; the 5–10 mm bending boundary layers are not drawn).

## Interface (measurement → fit)
`solver/prepare_tracks.py` → `solver/out/tracks_for_fit.csv` (channels pt:B, Bw, psi, nrm) from measurement/disc_track.csv,
twist_track.csv, disc_pose_camcentred.csv; px sigma from the camera-of-record Jacobian (floor 0.5 px); tilt sigma floor 0.004 rad
(AI systematics; formal pose sd 4e-4 rad is 10–20× below the tilt sine-fit residual).

## Commands
```
.venv/bin/python solver/prepare_tracks.py
.venv/bin/python solver/fit_projected.py --obs solver/out/tracks_for_fit.csv --channels pt:B psi nrm --geom-json solver/geom_measured.json \
   --scale-mode swing --bend-ends rigid_disc --fit-pivot --om-t-start 0.15 0.2 0.3 --t-anchor 2.5 --t-anchor-channel psi=7.5 \
   --nboot 20 --out solver/out/fit/prod_permode                    # PRODUCTION (lead per-mode windows)
   ... --t-anchor 2.5 --nboot 10 --out solver/out/fit/diag_global2.5  # rule-compliant global window (diagnostic)
solver/regen_fitted_params.sh          # make_params --status fitted (exact argv stored in physics.json generator)
.venv/bin/python solver/run.py && .venv/bin/python solver/selfcheck.py
.venv/bin/python solver/visuals/make_visuals.py --version vNNN ; .venv/bin/python solver/visuals/make_fit_visuals.py --fit solver/out/fit/prod_permode --version vNNN
.venv/bin/python solver/anchor_rule.py --obs solver/out/tracks_for_fit.csv --modes psi pt:B nrm --out solver/out/anchor_rule.json   # cross-check of the rule
```
Synthetic validation: `solver/test/make_synthetic.py` (clean + fake growth), fits in solver/test/out_{clean,growth}.

## Current status: v003 FITTED (2026-10-26)
* **Files.** params/physics.json sha256 faee420fd876…; trajectory.csv 2e5101130d14…; solver source 4ba334f64b11….
* **Regenerate:** solver/regen_fitted_params.sh, then run.py.
* **Fit.** `solver/out/fit/prod_permode` uses the lead's per-mode windows: twist 0–7.5 s, sway/tilt 0–2.5 s. The model is rigid_disc with the scale from the sway period and the pivot offset fitted.
  * 20 moving-block bootstrap resamples; sd = max(bootstrap, scaled formal).
  * Q_t = 1000 is declared: M1 drives γ_t to the 0 bound (ΔBIC +5.8).
* **Fitted values:**

  | quantity | value |
  |---|---|
  | ell (A→top face) | 0.1906 ± 0.0016 m |
  | Lcam | 0.8619 ± 0.0039 D |
  | D | 0.221 m (model-conditioned) |
  | ω_t | 0.1728 ± 0.0108 rad/s → T_t = 36.4 ± 2.3 s |
  | ψ0 | 221.3 ± 11.8° |
  | ψ̇0 | −23.5 ± 1.1 °/s |
  | μ | −389.4 ± 11.4° |
  | swing φx0 / φy0 | −1.97 ± 0.04° / +0.61 ± 0.11° |
  | swing rate φ̇y0 | 15.2 ± 0.8 °/s |
  | pivot Ax | +0.0176 ± 0.0027 D (3.9 mm) |
  | pivot Ay | −0.005 ± 0.012 D (not identified) |

  * ω_t, ψ0 and μ are strongly correlated (0.97–0.999): the 7.5 s window covers 1/5 of a cycle.
* **Derived quantities** (brass ρ = 8500, steel G and E assumed):

  | quantity | value |
  |---|---|
  | R | 110.6 mm |
  | t | 6.9 mm |
  | m | 2.24 kg |
  | κ | 4.08e-4 N m/rad |
  | wire d | 0.32 mm |
  | max shear strain | 3.2e-3 (music wire) |
  | T_swing | 0.914 s |
  | joint mode | 114 rad/s |
  | θ/φ in swing | 1.004 |
  | tension | 21.96–21.99 N |
* **Anchor RMS.**

  | channel | anchor RMS | target / σ | result | after the window |
  |---|---|---|---|---|
  | B | 1.47 px | target 2.0 px | **PASS** | 14.9 px (AI sway injections + period drift) |
  | twist | 1.05° | 3σ = 0.79° | **misses** | 7.3° (−11° at 10 s) |
  | tilt | 0.97° | σ floor 0.23° | — | 2.8° |

  * The twist residual is a ~1 Hz, ±1–2° wiggle at the sway frequency: an AI mark/rectification systematic.
* **Diagnostic, rule-compliant global window 0–2.5 s** (`solver/out/fit/diag_global2.5`, 10 resamples):
  * T_t = 12.9 ± 2.0 s; twist RMS 0.62° in the window and 46° after it;
  * B 1.47 px; ell 0.1945 m.
  * The torsion period is not identified by 75° of rotation. This is the reason for the lead's per-mode deviation.
* **Coupling** (full model vs the other mode's initial conditions removed):
  * sway → twist ≤ 0.14°;
  * twist → B ≤ 3.7 mm over 10 s (gyroscopic precession of the swing ellipse; ≲ 1 mm in 0–2.5 s).
* **selfcheck** (fitted params): **ALL PASS**.
  * P1 finest pair 3.9e-12 m / 2.4e-10 rad, orders 4.00.
  * P3:
    * pure twist 4e-14 at 30° and 60°;
    * linear modes 2.7e-13 and 1.6e-8 (joint mode);
    * elliptic pendulum 2.5e-13;
    * rigid compound-pendulum check, informational.
  * P4 conservative 3.1e-12, damped balance 2.7e-12.
  * Guards raise as designed.
  * Re-equilibrated variants κ×4, ℓ×1.5, m×2, R×1.3 and k×4 converge to h vs h/2 ≤ 3e-11 m / 2e-9 rad.
* **P9 dev variants** (solver/out/variants):

  | variant | T_t | T_swing |
  |---|---|---|
  | κ×4 | 18.185 s (= 36.370/2) | unchanged, 0.9142 s |
  | ℓ×1.5 | 36.370 s (unchanged) | 1.0948 s |
* **Bake (test rig, 4 keys/frame, quaternion):**
  * P6: 7.5e-9 m and 4.8e-7 on the rotation matrix (float32 keys);
  * between samples: 2.3e-8 m;
  * anchor at the pivot: 2e-11;
  * wire length: 1.7e-8 m.

## v004 FITTED (supersedes v003; lead CHANGELOG 2026-10-26T04:37:55Z)
* **What changed.**
  * The camera of record is now SMOOTHED (camera.json 50377461…, drift.csv 13dafee6…).
  * The observable is now the disc silhouette centre P (measurement/disc_P_track.csv, σ = sd_eff through the camera Jacobian, median 0.73 px). It replaces the mark rotation centre Q, which is world-fixed.
  * ψ comes from the re-measured twist_track.csv. The windows are unchanged (twist 0–7.5 s, sway/tilt 0–2.5 s).
  * The v003 fit, params and tracks are kept in solver/out/fit/history_v003/.
* **Files.** params/physics.json 1ca3ae7639e7…; trajectory.csv ea2b7544482a…; solver source unchanged (4ba334f64b11…).
* **Anchor RMS, v003 → v004.**

  | channel | v003 | v004 | target |
  |---|---|---|---|
  | B/P | 1.47 px | 1.34 px | max(2, 3σ) = 2.22 px: PASS |
  | twist | 1.05° | 1.04° | 3σ = 0.78°: misses |
  | tilt | 0.97° | 0.95° | — |

  * After the windows: P 14.4 px, twist 7.4°, tilt 2.7°.
* **Pivot.** Ax = +0.0144 ± 0.0045 D and Ay = −0.029 ± 0.020 D, so pivot_world_m = (+3.18, −6.35, 0) mm.
  * Ay correlates with Lcam at +0.997: the depth along the view is only weakly separable from the wire length.
* **Scale.**
  * ell = 0.1915 ± 0.0018 m.
  * Lcam = 0.8650 ± 0.0064 D, so 4.518 camera units per metre and D = 0.2213 m.
* **Twist.** T_t = 36.39 ± 2.3 s (ω_t 0.1727 ± 0.0111 rad/s), ψ0 = 221.5 ± 12.3°, ψ̇0 = −23.5 ± 1.1 °/s, μ = −389.6 ± 11.9°.
* **Swing.** T_sw = 0.916 s.
* **Wire.** d = 0.317 mm, max shear strain 3.2e-3.
* **Damping.** Q_t = 1000 declared: γ_t sits at the bound, ΔBIC +5.8.
* **Global 0–2.5 s diagnostic.** T_t = 13.1 ± 2.0 s; twist 0.63° in the window and 46.3° after it; P 1.34 px.
* **selfcheck (v004 params):** ALL PASS.
  * P1: 3.6e-12 m / 2.3e-10 rad, orders 3.99.
  * P3: twist 4e-14, modes 2.7e-13 / 1.6e-8, elliptic 2.5e-13.
  * P4: 3.4e-12.
* **Bake, test rig:** P6 7.4e-9 m, rotation 4.8e-7; anchor 2e-10.
