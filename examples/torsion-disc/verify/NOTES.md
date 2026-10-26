# verify/ NOTES (role D, clip 03)

## Log (UTC 2026-10-26)
- **PLAN.md written before any production output existed.** physics/, params/ and solver/out/ were empty when PLAN was
  written.
- **Kit self-test (v001):** synthetic inputs plus a verifier-owned stand-in production (`test/fake_production_solver.py`,
  RK4 on my RHS). Its only purpose is to exercise P1, P3 and P9 through the subprocess command path.
- **Added to P6 (not a relaxation): between-key interpolation probe.** The PLAN negative control "whole-frame keys only"
  does not apply to this clip.
  - At the clip's low frequencies, Hermite keys every whole frame stay within 1e-5 at the subframes: 1.4e-7 m and
    2.5e-7 rad on the test rig.
  - What a 1/96 s sample grid cannot see is stepped (CONSTANT) or LINEAR keys whose key times coincide with the samples.
  - P6 therefore also samples every 3rd frame at subframes .125/.375/.625/.875 against the cubic-Hermite-interpolated
    ref, with the same 1e-5 tolerances.
  - Negative controls: stepped quarter-frame keys and linear whole-frame keys must FAIL. Whole-frame Hermite keys are
    informational.

## Reconciliation with physics/FORMULATION.md (model_id disc_on_wire_rigid6_v1), read 2026-10-26 after PLAN
I read FORMULATION.md (documentation), params/physics.json and the output headers. I did not read any solver/ code. The
`--help` of run.py and make_variant.py was queried as a subprocess, and make_variant was dry-run into verify/out/probe/.
| FORMULATION | my canonical model (tmodel.Model) |
|---|---|
| inextensible wire, Lagrange tension, RK4 + projection | `mode=string`: reduced coordinates (wire frame q_w without spin + disc q_d), GL4 order 8, no projection |
| ψ = 2 atan2(q_z, q_w) (swing–twist about the disc normal) | `twist=vertical` (R = Rmin(ẑ→a) Rz(ψ)); identical by construction |
| V_A = k_A(1 + s_z) | added `k_a`: k_a(1 − u_z), u = −s |
| V_B = k_B(1 − s·n) | `k_b`, `bend_law=cos`: k_b(1 − u·a), n = −a |
| D: c_t ψ̇², c_x\|V\|², c_r\|ω⊥\|² | c_t; `c_v` = c_x; added `c_rp` (ω⊥ = ω − (ω·a)a) = c_r |
| B = X + R(0,0,t/2); I = diag(I_⊥, I_⊥, I_z) | p_b = (0,0,t/2); I_c |
| linear modes K = [[mgℓ + k_A + k_B, −k_B], [−k_B, mgh + k_B]] | `swing_modes` (mine, independent): 9.7993 / 13.5802 rad/s = production `derived` |
- The schema adapter is `tparams._ck_model` / `_ck_edits`. Production trajectory columns (X_*, V_*, B_*, q_*, w_*, psi_rad,
  wire_tension_N, E_J…) are mapped by alias in `tio.COL_ALIASES`.
- P9 uses the dynamics `make_variant` interface (`--set <Model attr>=xK`), with sets in `variant_sets_ck.json`:
  - kappa=x4;
  - m=x4 (realises I×4 at fixed geometry);
  - ell=x1.5;
  - c_t, c_x, c_r all ×10.
  The regime ICs (pure twist, or lowest swing mode at 1e-5 rad) are applied on top by my own edits of the params copy.
- The transport convention (the holonomy of the bent ends) differs from production's ψ by up to 1.7e-4 rad on the
  placeholder run (from the psi-column-vs-conventions diagnostic). FORMULATION §3.2 declares this. It is a **modelling
  choice**, not a verification failure: P2 uses the production's convention.

## Pre-ck001 early look (solver/out placeholder run, NOT a checkpoint)
- P2: |ΔC| 1.1e-13 m, attitude 3.9e-12 rad, ψ 3.3e-13 rad.
- P7: tension 1.9636 to 1.9648 N.
- P8: all at round-off.
- P4 damped balance: 1.2e-12.
- P3: B2 5e-14 and B3 4e-14 relative.
- First-pass FAILs were my harness's fault, and were fixed before re-running:
  - the zero-damping edit missed c_rp;
  - the B4 bench IC excited both swing/tilt modes.

## Commands (ck001; run from ROOT)
```
REF=verify/out/ref_ck001.csv; CK=checkpoints/ck001
.venv/bin/python verify/ref_solver.py --params $CK/params/physics.json --out $REF --lyapunov
.venv/bin/python verify/check_physics.py --params $CK/params/physics.json --trajectory $CK/solver/out/trajectory.csv --ref $REF \
   --solver-cmd "{py} solver/run.py --params {params} --out {outdir}" --bench --lyapunov --out verify/out/check_physics_ck001.json
.venv/bin/python verify/check_convergence.py --order 4 --values 0.0013020833333333333 0.0006510416666666666 0.0003255208333333333 0.00016276041666666666 \
   --cmd-template ".venv/bin/python solver/run.py --params $CK/params/physics.json --h {value} --out {out}_dir && cp {out}_dir/trajectory.csv {out}" \
   --out verify/out/check_convergence_ck001.json
.venv/bin/python verify/check_controls.py --params $CK/params/physics.json --solver-cmd "{py} solver/run.py --params {params} --out {outdir}" \
   --variant-cmd "{py} solver/make_variant.py --params {params} --out {out} {set}" --variant-sets verify/variant_sets_ck.json \
   --out verify/out/check_controls_ck001.json
tools/with_render_lock.sh blender -b $CK/torsion.blend --python verify/check_blend.py -- --params $CK/params/physics.json \
   --ref $REF --trajectory $CK/solver/out/trajectory.csv --disc-name TOR_Disc --wire-name TOR_Wire --anchor-name TOR_Anchor \
   --com-local 0,0,-<t/2> --attach-local 0,0,0 --absolute-axes --out verify/out/check_blend_ck001.json --dump verify/out/check_blend_ck001.npz
SOLVER_CMD='{PY} solver/run.py --params params/physics.json --out solver/out' \
BUILD_CMD='{B} -b --factory-startup --python scene/build.py -- --out scene/out/torsion.blend --bake solver/out/trajectory.csv --look' \
  verify/clean_rebuild_verifier.sh $CK /tmp/<fresh> $REF params/physics.json scene/out/torsion.blend <check_blend args>   # x2, then compare_dumps.py
```
- The scene contract (scene/NOTES.md) puts the TOR_Disc origin at B, the top-face centre, so the CoM is at local
  (0, 0, −t/2), and its local axes are the body axes.
- The mark is the engraved line; I must confirm whether it is a separate object or disc geometry/material when ck001 exists.

## ck001 audit (2026-10-26; checkpoint copies only)
Inputs:
- params/physics.json faee420f…
- solver/out/trajectory.csv 2e510113…
- torsion.blend 0057a004…

All production runs used `checkpoints/ck001/solver/{run.py,make_variant.py}` on params COPIES under verify/out/.

### Kit changes forced by ck001 (logged; no tolerance changed)
1. **Lifted twist.** ψ0 = 221°, so the twist exceeds ±180°.
   - My `vertical` convention is now ψ = 2 atan2(q_z, q_w) of the continuous state quaternion. This is the same swing–twist
     angle on the quaternion double cover, valid for |ψ| < 2π. It matches solver/NOTES.md.
   - Gradients are still exact (complex step through q).
   - P2 compares ψ without 2π wrapping, and FAILs on any quaternion-hemisphere mismatch, so a 2π slip cannot hide.
2. **Pivot.** The suspension point is `pivot_world_m`. ICs are relative to the pivot; trajectory positions are world.
   The adapter maps both.
3. **Bug fix.** `lyapunov_estimate` canonicalised the perturbed quaternion to w ≥ 0, which flipped the lifted twist by 2π.
   It now uses quat_mul. Corrected result: growth ×1.0001 over 10 s, so the motion is not chaotic.
4. **New check `check_mark_shader.py`.** The mark is a shader on TOR_Disc, not an object:
   - Object-space texture coordinates;
   - an OBJECT-type attribute `tor_mark_azimuth_rad`.
   PASS requires:
   - Object-space coordinates only;
   - attribute not animated;
   - azimuth = params μ to 1e-12.
   Negative controls (copies of the ck001 blend under test/blends/): keyed azimuth → FAIL; azimuth +1e-3 rad → FAIL.
5. **New check `check_projected.py`.** Independent recompute of the projected residuals:
   - motion: my GL4 trajectory;
   - observations: measurement/ files directly;
   - camera: the camera of record loaded from the checkpoint's camera/project.py.
6. **`--exclude-static TOR_WireEnd`.** It is a keyed helper with COPY_LOCATION → TOR_Disc; the wire bottom end is judged
   against B anyway.

### Results
| item | result |
|---|---|
| P1 | PASS: finest pair 2.4e-10 (attitude), order 3.996 / 4.001 |
| P2 | PASS: C 4.1e-12 m, B 5.0e-12 m, attitude 2.6e-10 rad, ψ 5.8e-12 rad, 0 hemisphere mismatches |
| P3 | PASS: pure twist 0.5 rad 3.4e-14 / 2.9e-14 (undamped/damped), ω 1.6e-13, γ 2.3e-10; lowest swing mode 6.873195 rad/s to 6.4e-12 |
| P4 | PASS: conservative drift 2.6e-12; damped balance 2.6e-12; E column = my recomputation + constant to 1.8e-12; L_z balance 3.4e-14 |
| P5 | PASS: anchor at pivot 2.2e-11 m; statics 0; wire ends ≤ 2.3e-10 m; wire length variation 1.5e-8 m; disc mesh rigid 9e-17; mark shader PASS |
| P6 | PASS: C 7.5e-9 m, rotation 3.6e-7 rad, ψ 3.6e-7 rad; between-key probe 2.2e-8 m / 2.9e-7 rad |
| P7 | PASS: tension 21.946–21.975 N; column = recomputed to 7e-15 Mg |
| P8 | PASS: IC 7e-18; grid 1.8e-15 s; time scale 1.8e-13; blend part (frame 0) 7.5e-9 m / 5.4e-8 rad |
| P9 | PASS via make_variant (see below) |
| P10 | PASS (see below) |

- **P9 detail.** Ratios: κ×4 → T ×0.500000; m×4 (I×4) → ×2.000000; damping×10 → ×1.000012 (γ law).
  - ℓ×1.5 swing frequency: 5.7391 rad/s (formula) = fit, error 6e-12.
  - The full-IC variants match my GL4 to ≤ 4.3e-12 m and 2.7e-10 rad.
- **P10 detail.**
  - Both clean rebuilds (scratchpad work dirs) reproduce trajectory.csv bitwise (2e510113…).
  - Dumps of rebuild 1 vs rebuild 2 and rebuild 1 vs checkpoint are bitwise identical: keys, disc matrices, wire ends,
    topology, anchor.
  - The .blend file bytes differ between rebuilds (9703fbf0… vs 760a7680…). That is expected, because file headers and
    paths differ; it is not judged.

### Projected residuals (R2, reference fidelity — not P items)
My recompute reproduces production's fit_summary claims.

| channel | mine | production claim | outcome |
|---|---|---|---|
| B, 2-D RMS, 0–2.5 s | 1.46881 px | 1.46881 px | R2 PASS (≤ 2 px) |
| B, after the window | 14.93 px | 14.93 px | — |
| twist, 0–7.5 s | 1.0510° | 1.0510° | misses 3σ = 0.79° (as dynamics disclosed) |
| twist, after the window | 7.30° | — | +10.9° at 10 s |
| tilt angle, 0–2.5 s | 0.9656° | 0.9657° | — |

B residual definition: sqrt(mean(du² + dv²)). The per-component RMS would be 1.04 px.

### Operational note
`clean_rebuild_verifier.sh` extra args must be separate words. Under zsh, an unquoted "$ARGS" string is one word, and the
in-script check_blend failed on it. I re-ran check_blend on both rebuilt blends directly; the rebuilds themselves were valid.

## ck002 audit (2026-10-26; checkpoint copies only)
Inputs:
- params 1ca3ae76…
- trajectory ea2b7544…
- camera.json 50377461…
- torsion.blend 63af1bd3…
- pivot (3.18, −6.35, 0) mm

Synthetic negative-control suites were not re-run: the checker code is unchanged since v001/v002.

### Code change
`check_projected.py --b-track/--b-cols`: the fit observable is now the silhouette top-face centre P, read from
measurement/disc_P_track.csv (u_P_px, v_P_px). This is the model point B.

### Results
| item | result |
|---|---|
| P1 | PASS: order 3.988 / 3.994; finest pair 2.3e-10 |
| P2 | PASS: C 3.8e-12 m, B 4.7e-12 m, attitude 2.4e-10 rad, ψ 5.2e-12 rad, 0 hemisphere mismatches; growth ×1.0001 (not chaotic) |
| P3 | PASS: pure twist 2.3e-14 / 3.0e-14, ω 2e-13, γ 6.5e-12; lowest swing mode 6.4e-12 |
| P4 | PASS: conservative drift 2.1e-12; damped balance 1.7e-12; E column consistent to 1.9e-12; L_z 3.4e-14 |
| P5 | PASS: anchor at pivot 2.2e-10 m; wire ends ≤ 3.4e-10 m; mark shader PASS |
| P6 | PASS: C 7.8e-9 m, rotation 3.1e-7 rad, ψ 3.0e-7 rad, probe 3.5e-7 rad |
| P7 | PASS: tension 22.007–22.032 N |
| P8 | PASS: IC 8.7e-19; time scale 1.8e-13; blend part PASS |
| P9 | PASS: κ×4 period ratio 0.4999999531 (damped law), m×4 1.99999981; ℓ×1.5 swing 5.727 rad/s to 6.3e-12; full-IC variants ≤ 3.9e-12 m |
| P10 | PASS: both rebuilds reproduce trajectory ea2b7544…; dumps bitwise identical rebuild-to-rebuild and rebuild-to-checkpoint |

### Projected residuals, recomputed (claims REPRODUCED)
| channel | mine | production claim | outcome |
|---|---|---|---|
| P, anchor | 1.33692 px | 1.33692 px | R2 PASS |
| P, after the window | 14.38 px | 14.38 px | — |
| twist, anchor | 1.04073° | 1.04073° | MISS vs 3σ = 0.78° (declared FAIL) |
| twist, after the window | 7.40° | 7.40° | — |
| tilt, anchor | 0.9512° | 0.9512° | — |
