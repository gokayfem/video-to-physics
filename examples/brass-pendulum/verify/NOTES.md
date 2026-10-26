# verify/ — independent physics verifier notes (v001, 2026-10-26)

Independence: nothing under `solver/` was imported, copied or read for implementation. The solver was
only run as a subprocess (`solver/run.py --params P --out DIR`, `scene/build.py ... --bake`), and only
its CSV/JSON outputs and the docstring usage lines of `run.py`, `bake_blender.py` and `build.py` were read.
`PLAN.md` was written before any production output existed.

## Visuals (verify/visuals/, v001; status PRELIMINARY because params/physics.json says "status": "placeholder")
| file | what it shows | verdict | what it does NOT establish |
|---|---|---|---|
| v001_P2_benchmark_theta.png | production θ(t) overlaid on the verifier's Taylor reference, plus log \|Δθ\| against the 1e-6 tolerance | PASS (max 2.98e-13 rad) | that the params fit the reference clip. They are placeholders. |
| v001_P1_convergence.png | log-log plot of the production RK4 successive differences vs h (slope 4), with the verifier's own GL4/GL2/Taylor orders beside it | PASS (p_obs 3.9999, finest diff 4.45e-12 rad) | accuracy against the truth. That is P2. |
| v001_P4_energy.png | E(t)/E0 vs 1 − ∫P_diss/E0, the balance residual, and the conservative drift | PASS (balance 5.2e-14, drift 1.9e-14) | that linear viscous damping is the right law. |
| v001_P3_P7_period_tension.png | conservative-run periods vs the exact elliptic period, and tension (column vs recomputed) | PASS / PASS (P3 6.0e-14 rel; T_min 9.1388 N) | anything about the real clip's period. T_meas = 1.25 s is an eye estimate. |
| v001_P5_P6_blend_overlay.png | Blender-evaluated bob θ (reopened .blend, 4 subframes per frame) vs the verifier reference, \|Δθ\| for frames and subframes, and L−L_c and y_bob in µm | P5 PASS, P6 PASS (max 5.39e-7 rad) | the lead's `scene/out/pendulum.blend`. That file is NOT animated (see below). This plot is from the verifier's own run of the documented build+bake command. It says nothing about the render or the look. |
| v001_P9_controls.png | turning-point amplitudes of the production runs vs the averaged elliptic decay law, and max period and envelope errors for L×4, 2°, 15°, damping×10 | PASS (worst 1.1e-8) | P9 through `build_all.sh`, which does not exist yet. It used `scene/build.py --bake`. |
| v001_negative_controls.png | metric/tolerance for each injected fault vs a good control | every fault detected | — (synthetic data only) |
| v001_refsolver_validation.png | the verifier's integrators vs the closed form at 0.5°–170°, and the period's small-angle limit | PASS | — |

## Preliminary results (production inputs as found 2026-10-26 ~01:27 local)
Inputs (sha256): params/physics.json `dd7e90e9…31a4a2` (status placeholder); solver/out/trajectory.csv
`ecb6e274…e44cda`; trajectory_meta.json `fe76c609…82fbeb` (its params hash matched physics.json);
verifier ref `verify/out/ref_trajectory.csv` `f0c16cf7…397f5`; verifier-run builds
`verify/out/prod_runs/build_a/pendulum.blend` `69cbc1e2…e567` and build_b `7b390ca4…e012`, built with
params/geometry.json `953659a5…0702` (geometry.json has since changed to `53ea6202…46d2`);
scene/out/pendulum.blend as found `bedcd857…94fa4` (an earlier `609c803d…` gave the same result).

| item | result | numbers | report |
|---|---|---|---|
| P1 | PASS | RK4 h = 1/384…1/3072 s; diffs 1.14e-9, 7.12e-11, 4.45e-12 rad; p_obs = 3.9999 (theory 4) | out/check_convergence_prelim_v001.json |
| P2 | PASS | max\|Δθ\| = 2.98e-13 rad, max\|Δω\| = 1.5e-12; ref Taylor vs GL4 2.9e-15 | out/check_physics_prelim_v001.json |
| P3 | PASS | the conservative production run (params copy with c = 0): T = 1.250000000000 s, max rel err 6.0e-14 vs 4√(I/mgd)K | same |
| P4 | PASS | damped balance 5.2e-14 E0; pure-column Simpson 3.3e-10; max forward dE/dt = −8.3e-6 E0/s; conservative drift 1.9e-14; columns = recomputation to 3.6e-16 | same |
| P5 | PASS on the verifier-run baked build | pivot drift 0; \|L−L_c\| ≤ 3.3e-8 m; \|y\| ≤ 1.7e-10 m; wire gaps ≤ 1.6e-8 m; no endpoint sliding | out/check_blend_baked_a_prelim_v001.json |
| P6 | PASS on the verifier-run baked build; FAIL on scene/out/pendulum.blend (not animated) | 5.39e-7 rad vs the verifier ref (on whole frames 2.0e-8); same vs trajectory.csv | baked_a / sceneout json |
| P7 | PASS | T ∈ [9.1388, 9.2894] N, column = recomputation exactly | check_physics |
| P8 | PASS (solver); blend part PASS on the baked build | θ0, ω0 exact; t = n/96 exact; period vs ref 6e-14; fps 24/1, frames 0–240, no time remap, θ_blend(0) error 3.8e-9 | check_physics, check_blend |
| P9 | PASS (solver + build/bake + check_blend) | max period rel err 6.5e-9 and max envelope err 1.1e-8 (damping×10); blend P6 for every variant ≤ 1.3e-6 rad | out/check_controls_prelim_v001.json |
| P10 | PASS (2 verifier-run clean builds) | keyframe co/handles/interp/handle types and evaluated bob centres bitwise identical (.blend file hashes differ, which the criterion allows) | out/check_repeat_prelim_v001.json |

Disclosure: motion in the baked builds is baked keyframes. There is one F-curve, PND_Swing
rotation_euler[1] = −θ, with 241 BEZIER keys, FREE handles at ±1/3 frame and slopes = −ω/24 (to 3.3e-7). There are
no NLA strips, drivers or modifiers. Extrapolation is CONSTANT, so samples after t = 10 s are wrong by up to 1.3e-3 rad.
They are outside the clip and not judged.

## Negative controls (synthetic, verify/test/) — all detected (out/negative_controls_*.json)
Trajectory checks: an injected +1e-4 rad after 5 s fails P2 (1e-4 rad). A 0.5 % time scale fails P2, P3 and P8 (period
5e-3 rel). Writing x,z at L+1.6 mm fails P8 (1.6e-3 m). Negative damping fails P4 and P2. A conservative run with
hidden drift fails P4c (2e-6). θ0 off by 1e-8 fails P8 while P2 still passes, which shows P8 is more sensitive. A
120° release fails P7 (T_min −0.49 N). A t-grid labelled n/96 but spaced at n/95 fails P8. Both good controls (my Taylor
solver, and GL4 as a different method) PASS everything.
Blender checks on test .blend files I built (verify/test/make_test_blend.py): a bob 1 mm lower fails P5. y offset of 2 µm fails P5.
A wire top gap of 50 µm fails P5. Keys at 0.5 % time scale fail P6. Keys +1e-4 rad from frame 120 fail P6. A 25 fps scene fails P6 and P8.
The good .blend PASSes. P10: good vs a rebuild PASS, and good vs the faulty keys FAIL.
P1: a stand-in GL2 solver gives order 4.00, PASS. The same solver with an injected O(h²) error gives order 2.00, FAIL.
P9: a "solver" that ignores the params fails L×4 (5e-1) and damping×10.

## Commands (run from ROOT)
```
PY=.venv/bin/python; B=blender
$PY verify/ref_solver.py --selftest --report verify/out/ref_solver_selftest.json
$PY verify/ref_solver.py --params params/physics.json --out verify/out/ref_trajectory.csv
# conservative production run for P3/P4c (params copy with damping c=0)
$PY -c "import json;P=json.load(open('params/physics.json'));P['damping']['c']['value']=0.0;json.dump(P,open('verify/out/prod_runs/physics_conservative.json','w'),indent=2)"
$PY solver/run.py --params verify/out/prod_runs/physics_conservative.json --out verify/out/prod_runs/conservative
$PY verify/check_physics.py --params params/physics.json --trajectory solver/out/trajectory.csv \
   --conservative-trajectory verify/out/prod_runs/conservative/trajectory.csv --meta solver/out/trajectory_meta.json \
   --out verify/out/check_physics_prelim_v001.json                                   # P2 P3 P4 P7 P8
$PY verify/check_convergence.py --kind step --order 4 --values 0.0026041666666666665 0.0013020833333333333 0.0006510416666666666 0.0003255208333333333 \
   --cmd-template "$PY verify/run_production_at_step.py --params params/physics.json --h {value} --out {out}" \
   --workdir verify/out/prod_runs/conv --out verify/out/check_convergence_prelim_v001.json   # P1
for x in a b; do tools/with_render_lock.sh $B -b --factory-startup --python scene/build.py -- --out $PWD/verify/out/prod_runs/build_$x/pendulum.blend \
   --physics params/physics.json --geometry params/geometry.json --bake solver/out/trajectory.csv; done
$B -b <BLEND> --python verify/check_blend.py -- --params params/physics.json --ref verify/out/ref_trajectory.csv \
   --trajectory solver/out/trajectory.csv --out $PWD/verify/out/check_blend_X.json --dump $PWD/verify/out/check_blend_X.npz   # P5 P6 P8b
$PY verify/compare_dumps.py A.npz B.npz --out verify/out/check_repeat.json                                                   # P10
$PY verify/check_controls.py --params params/physics.json --tag prelim_v001 --include-base \
   --bake-cmd "tools/with_render_lock.sh $B -b --factory-startup --python scene/build.py -- --out {blend} --physics {params} --geometry params/geometry.json --bake {traj}" \
   --out verify/out/check_controls_prelim_v001.json                                  # P9
$PY verify/test/make_synthetic.py && $PY verify/test/make_test_keys.py && $PY verify/test/run_negative_controls_physics.py && $PY verify/test/run_negative_controls_blend.py
$PY verify/make_visuals.py --version v001 --status preliminary
```

## Changes after PLAN.md (logged honestly)
- The P9 averaged viscous law was first coded with ⟨ω²⟩ = 2ω_n²[E−k'²K]/K. The verifier's own self-test caught it
  with a 1.8 % envelope error, and it was corrected to 4ω_n²[…], whose small-angle limit is 2ω_n²k². This was done
  before any production run, and PLAN.md was updated.
- P6/P5 samples after the clip end (frame 240 subframes 0.25–0.75, t > 10 s) are reported but not judged.
- Alias additions to vparams for the production schema: theta_init, omega_init, damping/c.

## Open issues / requests
1. **scene/out/pendulum.blend is not animated** (no animation_data). P6 and P8-blend FAIL on it. The PASS above is
   for the verifier's own run of the documented `scene/build.py --bake` command. The lead's checkpoint .blend needs re-checking once baked.
2. **Bob radius mismatch (information, not a P item):** the rig's bob sphere radius is 0.031438 m (sphere fit),
   but params/physics.json bob_radius is 0.029745 m (+5.7 %). I_P uses the physics radius, so the dynamics model a
   slightly different body from the one rendered. The period effect is ~1.4e-4 relative. Geometry and dynamics should agree on r.
3. P4 "dE/dt ≤ 1e-12·E0 at all samples" is judged on the damped run only. On the conservative run, round-off alone
   gives a forward difference of 1.85e-12 E0/s, both for production and for my own near-exact solver (1.9e-12). Please confirm this interpretation.
4. P9 L×4: the verifier recomputes the derived I_P (and L_eq, wire_length) in the params copy from the params'
   model_equation. If dynamics wants derived fields regenerated by solver/make_params.py, the control should
   go through that path. P9 was run via scene/build.py --bake, because build_all.sh does not exist yet.
5. The results are preliminary: the params are placeholders (T_meas and amplitude are eye estimates). Everything must be re-run on the
   fitted params. The checks are scripted for that (commands above).
6. P1 depends on the production solver honouring params `integrator.h_s` (substeps 4/8/16/32 confirmed in
   the meta files).

---
# v003 — full audit of checkpoint ck001 (status: final for ck001)
Inputs (ONLY the checkpoint's copies; worked on byte-identical copy `verify/out/ck001/ckcopy`, MANIFEST 194/194 OK before and after,
nothing in `checkpoints/ck001` newer than READY): physics.json 86ba7d90…, trajectory.csv f18d7894…, trajectory_meta 43eea035…,
pendulum.blend 14492f92…, geometry 71ab5ead…, camera 3eee3276…, bob_track a9038421…, project.py 2d65216b…. Summary: `out/ck001/summary.json`.
Results: P1–P10 all PASS (numbers in summary.json). Fit-claim audit: RMS fit/hold-out reproduced to <5e-4 px.

Verifier code changes for v003 (logged): run_production_at_step.py `--solver-root` (run a checkpoint copy's solver/run.py, cwd = that tree);
make_visuals.py path options (`--inputs-root --outdir --tag --blend-report --blend-label --checkpoint --skip-synthetic`), wrapped footer;
new `audit_fit_residuals.py` (independent projected-residual recomputation) and `clean_rebuild_verifier.sh` (P10).
Negative controls re-run with current code: all as expected (v001 copies in out/negctl_v001_archive/).

## Commands (from ROOT; repeat for ck002 by replacing ck001)
```
export PYTHONDONTWRITEBYTECODE=1; PY=.venv/bin/python; B=blender; O=verify/out/ck001; C=$PWD/$O/ckcopy
cp -Rp checkpoints/ck001 $C && (cd $C && shasum -a 256 -c MANIFEST.sha256)
$PY verify/ref_solver.py --selftest --report $O/ref_solver_selftest.json
$PY verify/ref_solver.py --params $C/params/physics.json --out $O/ref_trajectory.csv
$PY -c "import json;P=json.load(open('$C/params/physics.json'));P['damping']['c']['value']=0.0;json.dump(P,open('$O/prod_runs/physics_conservative.json','w'),indent=2)"
(cd $C && $OLDPWD/$PY solver/run.py --params $OLDPWD/$O/prod_runs/physics_conservative.json --out $OLDPWD/$O/prod_runs/conservative)
(cd $C && $OLDPWD/$PY solver/run.py --params params/physics.json --out $OLDPWD/$O/prod_runs/nominal)      # bitwise == ck trajectory
$PY verify/check_physics.py --params $C/params/physics.json --trajectory $C/solver/out/trajectory.csv \
   --conservative-trajectory $O/prod_runs/conservative/trajectory.csv --meta $C/solver/out/trajectory_meta.json --label ck001 --out $O/check_physics_ck001.json
$PY verify/check_convergence.py --kind step --order 4 --values 0.0026041666666666665 0.0013020833333333333 0.0006510416666666666 0.0003255208333333333 \
   --cmd-template "$PY verify/run_production_at_step.py --params $C/params/physics.json --solver-root $C --h {value} --out {out}" \
   --workdir $O/prod_runs/conv --label ck001 --out $O/check_convergence_ck001.json
tools/with_render_lock.sh $B -b $C/pendulum.blend --python verify/check_blend.py -- --params $C/params/physics.json --ref $PWD/$O/ref_trajectory.csv \
   --trajectory $C/solver/out/trajectory.csv --out $PWD/$O/check_blend_ck001.json --dump $PWD/$O/check_blend_ck001.npz --label "ck001 checkpoint pendulum.blend (reopened)"
$PY verify/check_controls.py --params $C/params/physics.json --tag ck001 --include-base --solver-cmd "{py} $C/solver/run.py --params {params} --out {outdir}" \
   --bake-cmd "tools/with_render_lock.sh $B -b --factory-startup --python $C/scene/build.py -- --out {blend} --physics {params} --geometry $C/params/geometry.json --camera $C/params/camera.json --bake {traj}" \
   --out $O/check_controls_ck001.json
for x in A B; do verify/clean_rebuild_verifier.sh $O/ckcopy $O/p10/v_rebuild_$x $O/ref_trajectory.csv; done
$PY verify/compare_dumps.py $O/check_blend_ck001.npz $O/p10/v_rebuild_A/check_blend.npz --out $O/p10/compare_ck_vs_A.json   # also ck_vs_B, A_vs_B
$PY verify/audit_fit_residuals.py --ck $O/ckcopy --out $O/fit_audit_ck001.json --fig verify/visuals/v003_fit_residual_map.png --label ck001 --caption "..."
$PY verify/make_visuals.py --version v003 --status "final for ck001" --checkpoint ck001 --inputs-root $O/ckcopy --outdir $O --tag ck001 \
   --blend-report check_blend_ck001 --blend-label "ck001 pendulum.blend, reopened" --skip-synthetic
```
(summary.json was assembled by an inline script from these reports; it records every number and hash.)

## ck001 findings / requests
1. fit_projected.json records camera.json sha 53c55ac6…, but ck001 ships camera.json 3eee3276… (rewritten 22:51:33Z with L_eff_m=0.43911).
   Projection is gauge-invariant: my recomputation with the shipped camera reproduces the claimed RMS to 2e-4 px. Dynamics: please re-hash.
2. ck001 fit window is t<7.5 s (fit made 22:51Z), NOT the user's anchor window [0,3.6] s (decision 23:06Z). Anchor-window RMS of ck001 = 8.84 px,
   dominated by a systematic u bias (mean du −5.8 px; model's left turning points ~12 px left of the track). Expected to change in ck002.
3. "max_abs_px" in fit_projected.json is max(|du|,|dv|) per component (48.44), not Euclidean (48.81) — definition should be stated.
4. delivery/clean_rebuild.sh cannot run on ck001: it requires verify/*.py inside the checkpoint (missing_required_inputs). P10 used my own path.
5. Building from a checkpoint tree in place writes __pycache__ into it (Blender's Python ignored PYTHONDONTWRITEBYTECODE: camera/, scene/, vendor/).
   Seen in my copy only; never build in checkpoints/ directly.
6. PND_Cam_Reference is animated (per-frame drift camera); disclosed, not part of P-items.

---
# v004 — full audit of checkpoint ck002 (status: final for ck002)
Inputs (checkpoint copies only; copy verify/out/ck002/ckcopy; MANIFEST 557/557 OK before and after): physics dc20e99c (fitted, anchor t<=4.0 s,
level_tilt_rad +0.012595907, Q=1000 assumed), trajectory 8ff8bb05, pendulum.blend 64c4c5bb (look applied), camera 3eee3276, look 76034a40.
Results: P1–P10 PASS; tilt audit PASS; fit claims (fit_tilt/Ta_4.0) reproduced to <3e-8 px incl. anchor RMS 3.2392 px. Summary: out/ck002/summary.json.
Commands: identical to the v003 block with ck001->ck002, plus:
```
$B -b $C/pendulum.blend --python verify/check_tilt.py -- --params $C/params/physics.json --trajectory $C/solver/out/trajectory.csv --out $PWD/$O/check_tilt_ck002.json
# P9 bake-cmd and P10 rebuilds add --look (as ck002 was built):  verify/clean_rebuild_verifier.sh $O/ckcopy $O/p10/v_rebuild_A $O/ref_trajectory.csv --look
$PY verify/audit_fit_residuals.py --ck $O/ckcopy --fit-json solver/out/fit_tilt/Ta_4.0/fit_projected.json --anchor 4.0 --out $O/fit_audit_ck002.json --fig verify/visuals/v004_fit_residual_map.png ...
$PY verify/plot_tilt.py $O/check_tilt_ck002.json $O/p10/check_tilt_A.json --out verify/visuals/v004_tilt_audit.png --caption ...
delivery/clean_rebuild.sh $PWD/$O/ckcopy $PWD/$O/p10/delivery_rebuild      # NOT_REPRODUCED: scene/vendor (nodekit) not staged
```
New verifier code: check_tilt.py, plot_tilt.py; audit_fit_residuals.py gains --fit-json/--anchor/--no-tilt, applies R_tilt^T and the
camera/physics L_eff gauge (1.001341 for ck002: camera.json L_eff_m 0.43911 vs physics 0.43852). Negative controls: flipped tilt sign -> check_tilt FAIL;
audit without tilt -> claims not reproduced (anchor 8.12 px).
## ck002 findings / requests
1. delivery/clean_rebuild.sh: NOT_REPRODUCED because inputs_manifest.json does not stage scene/vendor/ (build fails: No module named 'nodekit'). Its
   trajectory/meta/params comparisons were IDENTICAL. Delivery: add scene/vendor/** and camera/project.py, camera/drift.csv (build reads them).
2. camera.json (3eee3276, L_eff_m 0.43911) was not re-written for ck002's L_eff_m 0.43852; build and fit both apply the 1.00134 gauge, consistently.
3. Tilt sign: bob rest in stand coordinates = +tilt (physics.json wording); as a world angle, bob minus stand-down = -tilt (lead wording). Both hold.
4. Q stays assumed 1000 although tilt fits give Q~100-140 (bootstrap lower bound <100) — documented in physics.json; physics checks are Q-agnostic.
