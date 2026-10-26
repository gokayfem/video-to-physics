# Independent physics verification — plan (written before any production result existed)

Author role: independent physics verifier. Written 2026-10-26, before `params/physics.json`,
`physics/FORMULATION.md`, `solver/out/trajectory.csv` or any `.blend` existed (all were absent).
Tolerances are taken verbatim from `config/acceptance.json` v1
(sha256 `8b65d70e…9455f7`). Nothing here relaxes them. Where the acceptance text is ambiguous
the interpretation is written down below *before* results exist; any later change is logged in
`NOTES.md` with a reason.

Independence rules I follow
- No import from, copy of, or reading for guidance of anything in `solver/`. The production solver is
  only ever *executed* as a subprocess (P1, P9) and its CSV/JSON outputs are read as data.
- Physics constants come from `params/physics.json` through my own loader `verify/vparams.py`
  (alias-tolerant; every mapping it made is printed into the reports). A lead-supplied explicit
  `--model-json` override can replace the mapping if the schema is ambiguous.
- My reference integrators are **not** Runge–Kutta/DOP853: (a) a recursive Taylor-series method
  (automatic differentiation of sin/cos, order 24) and (b) implicit Gauss–Legendre collocation
  (s = 4 stages, order 8, Newton-solved). Both are event-aware: integration is split at every
  ω = 0 turning point so that the non-smooth terms `|ω|ω` and `sign(ω)` are smooth inside each
  segment (Coulomb stick condition handled exactly). Both are validated against the Jacobi-elliptic
  closed form and against each other before being used as benchmarks.

Model under test (contract + task brief; will be re-read from `physics/FORMULATION.md` when it exists)

    I_P θ'' = −m g d sin θ − c_v θ' − k_q |θ'| θ' − τ_c sign(θ')
    bob centre = P + L_c (sin θ, 0, −cos θ), P = 0, g = 9.80665

ω_n² = m g d / I_P. Elliptic modulus k = sin(θ_a/2) (θ_a = amplitude); scipy uses the *parameter*
m = k² (`ellipk(m)`, `ellipj(u, m)`) — every formula below states which one is meant.

## Analytic benchmarks and their regimes of validity
| Benchmark | Formula | Valid when |
|---|---|---|
| Exact conservative solution | sin(θ/2) = k sn(ω_n t + u0 \| k²), k² = sin²(θ0/2) + ω0²/(4ω_n²); release from rest: θ = 2 asin(k sn(K − ω_n t)) | c_v = k_q = τ_c = 0, rigid body, fixed frictionless axis, k < 1 (libration) |
| Exact period | T = 4 √(I_P/(m g d)) K(k²) = 2π/(ω_n · AGM(1, cos(θ_a/2))) (two independent evaluations) | same as above |
| Small-angle period | T0 = 2π √(I_P/(m g d)); T/T0 = 1 + θ_a²/16 + 11θ_a⁴/3072 + … | only as θ_a → 0 (used as a limit test, never as the acceptance benchmark at finite amplitude) |
| Linear-damping envelope | A(t) = A0 exp(−c_v t / (2 I_P)), T_d = T/√(1−ζ²) | small amplitude, k_q = τ_c = 0, ζ ≪ 1 |
| Averaged nonlinear decay (first-order averaging, exact elliptic integrals) | E = 2 m g d k²; dE/dt = −c_v⟨ω²⟩ − k_q⟨\|ω\|³⟩ − τ_c⟨\|ω\|⟩ with ⟨ω²⟩ = 4ω_n²[E(k²) − (1−k²)K(k²)]/K(k²), ⟨\|ω\|³⟩ = 8ω_n²(sin θ_a − θ_a cos θ_a)/T(θ_a), ⟨\|ω\|⟩ = 4θ_a/T(θ_a) | light damping (ζ ≪ 1); error O(ζ²) per cycle. Used for P9 decay envelopes at finite amplitude |
| Bob radial force (tension proxy) | T_bob = m_bob (g cos θ + L_c ω²) | radial equation of motion of the bob's centre of mass; exact for any I_P if the bob is attached at the wire end and the wire is massless; if FORMULATION.md includes wire mass/other force split, the column is compared to the formula it declares and the discrepancy is reported |

## Checks

### P1 — convergence of the production integrator (tol: max|θ_{h/2} − θ_{h/4}| < 1e-7 rad; observed order within 0.5 of theoretical)
Method: `check_physics.py convergence --run h=FILE ...` takes ≥3 production trajectories produced
at different step sizes (or tolerance levels) **by the production command interface**, records the
command line, hashes of the files and the solver code hash from `trajectory_meta.json`. If the
dynamics interface supports it, `--cmd-template "… {step} … {out}"` executes the production solver
itself. Errors on the common 1/96 s grid: e1 = ‖θ_h − θ_{h/2}‖∞, e2 = ‖θ_{h/2} − θ_{h/4}‖∞,
p = log2(e1/e2) (for refinement ratio r ≠ 2: log(e1/e2)/log r). Same for ω and E (reported).
For tolerance-controlled adaptive runs, the "order" is the observed exponent of error vs tol, which
is compared with the theoretical value declared by the dynamics author (`--order`), and this
interpretation is disclosed in the report.
Additionally my own two reference solvers are shown to converge at their theoretical orders (GL4: 8).
Why not self-referential: errors are differences between production runs at different resolutions,
and the final error level is independently bounded by P2 against my solver.

### P2 — independent benchmark (tol: max|Δθ| < 1e-6 rad over 0–10 s)
Method: my Taylor-series solver (cross-checked by my GL4 solver to ≲1e-11 rad) at identical params,
evaluated at the trajectory's own time stamps, vs `theta_rad`. Also Δω reported.
Why not self-referential: different integration method, different code, written without seeing the
production code; params read through my own loader.

### P3 — finite-amplitude period (tol: relative error < 1e-7)
Method: periods in the data are measured from successive same-direction zero crossings of θ located
by quintic Hermite interpolation using (θ, ω, α) columns (interpolation error ≪ 1e-10 s at 1/96 s).
Expected: T = 4 √(I_P/(m g d)) K(k²), k = sin(θ_a/2), θ_a from the conservative release
(θ0, ω0 via energy), evaluated by both `scipy.special.ellipk` and the AGM.
P3 needs a *conservative* production run (damping coefficients set to 0 at otherwise identical
params) — passed with `--conservative-trajectory`. If only the damped production trajectory exists,
P3 is reported as NOT RUN (not PASS) and only the informational damped-period comparison is given.
Small-angle limit: for θ0 = 1e-3…1e-1 rad my closed form → T0 with error following θ0²/16 (shown in
the ref_solver validation). Why not self-referential: the benchmark is a closed form.

### P4 — energy (tol: conservative |drift|/E0 < 1e-8; damped |E(t) − E(0) + ∫P_diss dt|/E0 < 1e-7; no growth)
Method, two independent routes that must agree:
1. the trajectory's own columns `KE_J, PE_J, E_J, P_diss_W`;
2. my recomputation from θ, ω and params: KE = ½ I_P ω², PE = m g d (1 − cos θ) (PE compared up to a
   constant offset since the zero level is the author's choice), P_diss = c_v ω² + k_q |ω|³ + τ_c |ω|.
Column-vs-recomputed agreement is required (relative to E0: ≤ 1e-9 for KE/ΔPE/E, ≤ 1e-9·E0 per s
for P_diss) — otherwise the energy bookkeeping does not describe the motion that is shown.
∫P_diss dt by composite Boole/Simpson on the 1/96 s grid (quadrature error estimated by comparing
Simpson and Boole and reported; it must be < 1e-8·E0 for the test to be meaningful).
"No growth" (dE/dt ≤ 1e-12·E0 per second at all samples) is evaluated on the damped trajectory as the
forward difference of the E_J column (and also analytically, ω(I_P α + m g d sin θ)); for a
conservative run it is reported informationally only, because integrator round-off produces
± fluctuations that do not represent growth (interpretation declared now, open question to lead).
Why not self-referential: the recomputation uses only θ, ω and params; energy balance couples
three columns that the solver could write inconsistently.

### P5 — constraints in the saved .blend (pivot drift < 1e-6 m; |L − L_c| < 1e-6 m; |y_bob| < 1e-6 m; wire endpoint gaps < 1e-5 m)
Method (`check_blend.py`, inside Blender, on the *reopened saved* file): for every frame 0…240 and
subframes 0, .25, .5, .75 I set `scene.frame_set(n, subframe=s)`, get the evaluated depsgraph and:
- bob centre = least-squares sphere fit to the evaluated bob mesh vertices in world space (fallback
  and cross-check: world bounding-box centre of the evaluated mesh); never the rig's custom props;
- pivot = world position of the PND_Swing origin and of any pin object; drift relative to (0,0,0);
- L(t) = |bob − P| vs `L_c` from params; y_bob;
- wire ends: evaluated wire mesh vertices projected on the pivot→bob axis, the two end-ring centroids;
  gaps = |top end − P| (or `--wire-top-target`) and |bottom end − bob surface| (sphere fit radius),
  plus the variation of both endpoints in PND_Swing-local coordinates (sliding detection).
Why not self-referential: positions come from Blender's evaluated geometry, not from params or
keyframe values; L_c comes from the physics params.

### P6 — .blend vs solver (tol: max angular error < 1e-5 rad)
θ_blend = atan2(x_bob, −z_bob) from P5's geometric bob centre at the 964 sample times, compared
with **my** reference solution at the params (primary) and with trajectory.csv (secondary).
Also reported: keyframe channel(s), count, interpolation, handle types, whether handles encode ω
(slope of handles vs ω/24), NLA/driver/F-modifier presence, scene fps/fps_base/frame range/time
remap. Motion is disclosed as baked keyframes if F-curve keys exist on PND_Swing.

### P7 — tension > 0
Recomputed T_bob = m_bob(g cos θ + L_c ω²) from θ, ω (columns and my ref solution) — min/max
reported; column `tension_N` also required > 0 and compared with the recomputation.

### P8 — initial conditions and time (solver |Δ| < 1e-9 rad; blend < 1e-6 rad; t = n/96 exact; no time scaling)
θ(0), ω(0) vs params; |t_s[n] − n/96| ≤ 1e-12 s; row count; first/last time; x_m, z_m consistent
with L_c(sin θ, −cos θ); the period measured in the data vs my reference solution's period at the
params' L, g (a 0.5 % time scaling shows as a 5e-3 relative period error). In Blender: fps = 24,
fps_base = 1, frame_start = 0, frame_end = 240, time remap 100/100, no NLA strip scale/repeat,
θ_blend(frame 0) vs θ0.

### P9 — control response (tol: 1e-4 relative vs P3 formula / damping theory)
`check_controls.py` defines variants: L_c×4, θ0 = 2°, θ0 = 15°, damping×10 (all damping
coefficients ×10). It writes edited copies of params to `verify/out/controls/<variant>/` and runs the
real build path via command templates (`--solver-cmd`, `--bake-cmd`, `--blend-check-cmd`) once the
dynamics/rig interfaces exist; expected outcomes are computed only from the formulas above
(period from elliptic K at the measured amplitude of each cycle with damping correction √(1−ζ²);
peak-amplitude decay from the averaged nonlinear decay ODE). Self-test mode uses my own solver as
a stand-in to prove the formula side, and is never reported as P9 PASS.

### P10 — repeatability
`check_blend.py --dump` writes keyframe arrays (co, handles, interpolation) and the evaluated bob
centres to `.npz`; `compare_dumps.py A.npz B.npz` requires bitwise-identical keyframe arrays and
identical evaluated positions for two clean rebuilds.

## Negative controls (must FAIL)
Synthetic trajectories from my ref solver with injected: +1e-4 rad θ error (P2), 0.5 % time scaling
(P2, P8), L_c change in x,z (P8 geometry consistency), energy growth (P4), wrong θ0 (P8),
negative tension (P7). Synthetic .blend files built by `verify/test/make_test_blend.py` with: good rig
(must PASS), bob 1 mm lower (P5 length), 0.5 % time scale (P6), 1e-4 rad keyframe error (P6), bob
y offset (P5), wire top end gap (P5).
