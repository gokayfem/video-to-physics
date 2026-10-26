# verify/ — clip 02 independent physics verifier notes (v001, 2026-10-26)

**Independence.** Nothing under `solver/` was imported, copied or read; it was empty while this kit was written.
`PLAN.md` was written before any production output existed (physics/, params/, solver/out/ and checkpoints/ were all empty).
Clip-01 verifier code was reused by copying: `compare_dumps.py`, and the patterns of `check_blend.py` and `check_convergence.py`.

**Status v001.** The kit is built and self-tested on synthetic inputs only. No production result is checked yet.

## Visuals (verify/visuals/)
| file | shows | verdict |
|---|---|---|
| v001_refsolver_validation.png | GL4 against the exact axial solution, the B4 θ² approach to ω_s, the B1–B5 errors, energy / energy balance / L_z, B8 growth vs theory, and B8 energy exchange | self-test PASS |
| v001_negative_controls.png | worst metric / tolerance for 15 physics cases, 10 blend cases, P1/P3/P9/P10 verdicts | every fault detected, every good control passes |
| v001_convergence.png | GL4 order (7.97, 8.01), and P1 on the stand-in RK4: good p = 4.00 PASS; injected O(h²) p = 2.00 FAIL | as expected |

## Reference solver self-test (`out/ref_solver_selftest.json`)
| test | result |
|---|---|
| B1 static equilibrium (point and rigid, m_s and preload) | ≤ 1.8e-16 rel |
| B2 exact axial (point undamped/damped, rigid; GL and GBS) | ≤ 2.1e-12 of amplitude; ω_d and γ fits ≤ 2e-14 |
| B3 exact conical orbit, 5°/30°/60°, with and without m_s and preload | ≤ 5e-14 of ρ; L_z ≤ 9e-15 |
| B4 small-angle swing | rel freq error 2.9e-9 at 1e-4 rad; slope 2.000 (θ²) |
| B5 rigid linear modes (5.18, 20.74 rad/s) | ≤ 2e-10 |
| B6 energy, undamped (point / rigid) | 1.9e-14 / 2.0e-14 |
| B6 damped balance with the integrated W_diss | ≤ 1.2e-14; no growth |
| B7 L_z (undamped, and axial-only damping) | ≤ 6e-16 |
| B7 full L with g = 0 (rigid) | 3.3e-15 |
| GL4 vs GBS on generic 3-D motion | 8e-14 m (point); 1.7e-13 m and 4.8e-12 rad (rigid) |
| GL4 order (stiff case) | 7.97, 8.01 |
| B8 2:1 growth rate vs σ = ¾εω_s | rel err 2.5e-3 / 1.3e-3 / 9.1e-4 at ε = 0.04/0.02/0.01, decreasing with ε |
| B8 energy exchange (ε = 0.05) | swing share reaches 1.000 (lowest-order energy proxy) |

## Negative controls (`out/negative_controls_physics.json`, `out/negative_controls_blend.json`)
All are as expected (ALL_AS_EXPECTED = true in both).

Physics:
- +1e-4 m after 5 s → P2 FAIL.
- 0.5 % time scale → P2 and P8 FAIL (scale estimate 5e-3).
- Negative damping → P4 FAIL.
- Hidden drift on the conservative run (point and rigid) → P4 FAIL, while P2 passes.
- IC off by 1e-8 → P8 FAIL, while P2 passes.
- Tension column with flipped sign → P7 FAIL.
- Compressive spring (ℓ < L0) → P7 FAIL.
- ℓ column +1 µm → P8 FAIL.
- t = n/95 grid → P8 FAIL.

Good controls pass all checks: GL4, GBS, the stand-in RK4 with production-style column names plus a PE zero offset, rigid GL4, and rigid RK4.
- P3 through `run_benchmarks.py` + stand-in RK4: PASS (point B1–B4, rigid B1/B2/B5). A solver that ignores k FAILs.
- P1: RK4 PASS; injected O(h²) FAIL.
- P9: RK4 PASS (period ratio k×4 0.4999985 measured vs 0.4999985 predicted). Ignores-k FAILs.

Blender (test .blends that I built):
- Good rigid (quarter-frame keys) PASS; good point PASS.
- Spring top gap 50 µm → P5 FAIL.
- Anchor drift 2 µm → P5 FAIL.
- Coil count +1 at frame 120 (shape key, same topology) → P5 FAIL.
- Keys +1e-4 m → P6 FAIL.
- Keys at 0.5 % time scale → P6 FAIL.
- 25 fps → P6 and P8b FAIL.
- Frame-rate keys for a rigid body rocking at 20 rad/s → P6 FAIL. This is a genuine detection: 3.8e-4 rad between frames.

P10:
- Good vs an independent build: PASS. This includes `clean_rebuild_verifier.sh` on a fake checkpoint.
- Good vs keys error: FAIL.
- Good vs coil change: FAIL.

## Commands (run from ROOT; PY=.venv/bin/python, B=blender)
```
$PY verify/ref_solver.py --selftest --report verify/out/ref_solver_selftest.json
$PY verify/ref_solver.py --params <CK>/params/physics.json [--model-json M.json] --out verify/out/ckNNN/ref.csv --lyapunov
# P2 P4 P7 P8 (+P3 with --bench-manifest)
$PY verify/run_benchmarks.py --params <CK>/params/physics.json --outdir verify/out/ckNNN/bench --solver-cmd "{py} <CK>/solver/run.py --params {params} --out {outdir}" --cwd <CK>
$PY verify/check_physics.py --params <CK>/params/physics.json --trajectory <CK>/solver/out/trajectory.csv --meta <CK>/solver/out/trajectory_meta.json \
    --ref verify/out/ckNNN/ref.csv --conservative-trajectory <damping-zeroed production run> --bench-manifest verify/out/ckNNN/bench/bench_manifest.json \
    [--colmap colmap.json] --out verify/out/ckNNN/check_physics.json
# P1
$PY verify/check_convergence.py --order <declared> --values h1 h2 h3 h4 --cmd-template "$PY verify/run_production.py --params <CK>/params/physics.json \
    --set <json-path-of-step>={value} --solver-cmd '{{py}} <CK>/solver/run.py --params {{params}} --out {{outdir}}' --cwd <CK> --out {out}" \
    --workdir verify/out/ckNNN/conv --out verify/out/ckNNN/check_convergence.json
# P5 P6 P8b (+P10 dump)
tools/with_render_lock.sh $B -b <CK>/<blend> --python verify/check_blend.py -- --params <CK>/params/physics.json --ref $PWD/verify/out/ckNNN/ref.csv \
    --trajectory <CK>/solver/out/trajectory.csv --out $PWD/verify/out/ckNNN/check_blend.json --dump $PWD/verify/out/ckNNN/check_blend.npz \
    [--weight-name VSP_Weight --eye-name <eye empty> | --eye-local x,y,z] [--spring-name VSP_Spring] [--anchor-name ...] \
    [--weight-origin com|eye|none] [--orientation rigid|fixed|spring_axis] [--top-end-offset m --bottom-end-offset m]
# P9
$PY verify/check_controls.py --params <CK>/params/physics.json --tag ckNNN --solver-cmd "{py} <CK>/solver/run.py --params {params} --out {outdir}" \
    [--bake-cmd "tools/with_render_lock.sh $B -b --factory-startup --python <CK>/scene/build.py -- --out {blend} --bake {traj} ..."] --out verify/out/ckNNN/check_controls.json
# P10
SOLVER_CMD='{PY} solver/run.py ...' BUILD_CMD='{B} -b --factory-startup --python scene/build.py -- ...' \
  verify/clean_rebuild_verifier.sh verify/out/ckNNN/ckcopy verify/out/ckNNN/p10/A verify/out/ckNNN/ref.csv params/physics.json <blend_rel> [check_blend args]
$PY verify/compare_dumps.py verify/out/ckNNN/check_blend.npz verify/out/ckNNN/p10/A/check_blend.npz --out verify/out/ckNNN/check_repeat.json
# self-test of the kit
$PY verify/test/make_synthetic.py && $PY verify/test/make_test_keys.py && $PY verify/test/run_negative_controls_physics.py && \
  tools/with_render_lock.sh $PY verify/test/run_negative_controls_blend.py && $PY verify/make_visuals.py --version v001
```

## Interpretations fixed before results (also in PLAN.md)
- E0 in P4 is the oscillation energy above the static equilibrium, because the PE zero level is arbitrary.
- "No per-sample increase" is judged on the damped run. The undamped run is judged on its drift.
- W_diss is integrated by GL4 propagation from each production state over each 1/96 s interval, not by quadrature
  of the samples. Simpson at 1/96 s would carry an error of about 1e-5 E0, which is above the 1e-7 tolerance.
- The L_z drift check (< 1e-8) is added, not a relaxation.
- P3 is NOT RUN unless production benchmark runs are supplied.

## Open issues / requests (for lead and dynamics)
1. **Rigid rocking and baking.** If the weight rocks as a rigid body (modes around 15–25 rad/s for a 3–5 cm eye offset),
   cubic-Hermite keys only on whole frames give about 4e-4 rad between frames (my test rig). That FAILs P6 (1e-5 rad at
   subframes). Keys every 1/96 s, as in my good test rig, give 3.4e-7 rad.
   The point model at ω_z ≈ 10 rad/s with 1.5 cm amplitude gives 1.7e-6 m on frame-rate keys. That passes, with a margin of 6×.
2. **Schemas are unknown.** `vparams.py` finds parameters by alias, and `vio.py` maps trajectory columns by alias.
   - Parameters: k, rest length, M, m_s, preload, the anchor A, the eye offset e_b, the inertia I_c, damping c_v/c_a/c_r, the
     initial state of the **eye** E0 and v_E0, and q0/ω0.
   - Trajectory columns: eye position and velocity, CoM, quaternion (w, x, y, z), world ω, ℓ, tension, KE/PE/E, P_diss, W_diss.
   Please state the production names, or I will add a `--colmap` / `--model-json`.
   If the ICs are given as angles or a stretch rather than E0/v_E0, `run_benchmarks.py` and `check_controls.py` need
   dynamics to supply variant params, or a documented IC mapping.
3. **Spring-mass model.** My model uses the Rayleigh treatment: kinetic m_s/3 at E, gravity m_s/2 at E. If FORMULATION.md
   uses a different spring-mass law (or m_eff in the axial direction only), P2 and P9 will differ. I will add that law as a
   declared variant; I will not loosen the tolerances.
4. **Blender names.** check_blend looks for VSP_* objects whose names contain weight, spring/coil/helix, anchor, or eye.
   Its spring end-point rule is: the end-ring centroid at the extreme projections on the A→eye axis, plus declared
   offsets. Hooks or loops modelled at the spring ends need `--top-end-offset` / `--bottom-end-offset` from the builder.
5. **Chaos.** The elastic pendulum near 2:1 can be chaotic. `ref_solver.py --lyapunov` reports the separation growth
   (×1.37 over 10 s for the synthetic case). If P2 fails, this explains it but does not excuse it.
6. `clean_rebuild_verifier.sh` needs the checkpoint's documented SOLVER_CMD and BUILD_CMD. It was exercised only on a fake checkpoint.

---
# v002 — full audit of checkpoint ck001 (status: final for ck001)

## Inputs and isolation
- Only the checkpoint's own copies were used, via a byte-identical copy at `verify/out/ck001/ckcopy`.
- MANIFEST check: 156/156 OK, before and after the audit.
- Hashes:
  - physics.json `c71ba937…`
  - trajectory.csv `251e07c1…`
  - spring.blend `18caaad8…`
  - camera.json `df339954…`
- Read for the interface only: solver/NOTES.md "For the independent verifier", scene/NOTES.md, and the build.py docstring. No solver code was imported or read.
- Every production run was a subprocess of `<ckcopy>/solver/run.py`. Rerunning the nominal case reproduces trajectory.csv bitwise.

Summary: `out/ck001/summary.json`.

## Results
| item | verdict | numbers |
|---|---|---|
| P1 | PASS | `run.py --h`, 1/96 … 1/3072 s; orders 3.997, 4.000, 4.000, 4.001; finest pair 5.0e-12 m and 9.2e-12 rad |
| P2 | PASS | max \|ΔH\| 3.7e-13 m, attitude 6.1e-13 rad. Extra 3-D undamped variant (out-of-plane velocity and rocking rate): 6.1e-12 m, 6.7e-11 rad. Lyapunov growth factor 1.00 (regular motion) |
| P3 | PASS | production runs in benchmark configurations: B1 rest 3.4e-16; B2 exact damped axial 8.4e-10 of amplitude (ω 5.9e-12); B5 rigid modes 4.861 and 26.359 rad/s at 1.5e-11 and 3.6e-11 rel |
| P4 | PASS | damped balance 7.6e-14 E_osc (column W 4e-14 from mine); undamped drift 6.0e-14; no per-sample growth (max −1.1e-7 E_osc per sample); KE and E columns match recomputation to 1.3e-12 (PE zero at static equilibrium, offset 25.633 J). L_z: 2.7e-15 on the 3-D variant. The nominal motion is exactly planar, so L_z is trivial there |
| P5 | PASS | see the P5 notes below |
| P6 | PASS | hinge vs my reference 1.5e-8 m and rotation 3.0e-8 rad at frames + 4 subframes; same vs trajectory.csv. 961 keys every 1/96 s, FREE Bézier handles ±1/12 frame. Weight and HookLower are animated; the camera is animated per frame (disclosed) |
| P7 | PASS | tension 51.12–54.66 N, column equal to recomputation |
| P8 | PASS | initial conditions exact (0); t = n/96 to 1.8e-15; time-scale estimate 1 − 8e-14; geometry columns consistent. Blend: fps 24/1, frames 0–240, H(0) 1.2e-8 m |
| P9 | **FAIL** | see the P9 notes below |
| P10 | PASS | two verifier clean rebuilds (`clean_rebuild_verifier.sh`): trajectory bitwise equal; keyframes, eye, weight matrices, spring ends, coil counts and topology bitwise equal (ck vs A, ck vs B, A vs B). The .blend file hashes differ, which the criterion allows |
| Fit audit | reproduced | see the fit-audit notes below |

**P5 details** (frames + 4 subframes):
- Anchor and static objects: 0 drift.
- Static spring parts (top loop, S-bend, hook turns): 0 drift.
- Contact gaps: top loop at A 2.3e-10 m; lower loop at H 1.3e-7 m.
- H out of the lower-loop plane: 9.9e-9 m. H drift in the loop frame: 1.1e-7 m. Contact sliding along the loop: 2.4e-7 m.
- Coil-to-P_eff distance variation: 1.2e-6 m.
- HookLower = weight origin, difference 0.
- Coil winding 21.0000005 turns, variation 8e-10 turns. Half-plane crossing counts constant. Topology 10236/20412/10176 constant.

**P9 details:**
- All formula-regime checks pass:
  - k×4: axial period ratio 0.49999993 measured vs 0.49999993 predicted.
  - M×4: 1.98523 ratio.
  - L0+50 %: swing period ratio 1.157940 vs 1.157938, rel ω 1.8e-6.
  - damping×10: decay rate 2.6e-7 rel.
  - The bake path passes P6 for base, M×4 and damping×10.
- The production-IC sub-check fails for k×4 and L0+50 %. The production solver does not converge in those regimes; details are in finding 1 below.

**Fit audit (independent):**
- Recomputed from my reference trajectory, camera/project.py per-frame cameras and the raw measurement tracks.
- Anchor 0–7.5 s RMS 6.726650 px (claim 6.726650); after 15.461359 px; ring 5.067 (u 4.916 / v 1.229); weight 8.051 (u 7.999 / v 0.916); n≡0 mod 4 6.814 px.
- All claims match to 4.3e-8 px.
- R2 target 3.245 px: **FAIL**, as dynamics also reports.
- The median-σ definition the fit uses is the median of √((σu²+σv²)/2) over all frames. The median of the components would be 1.0686, giving a target of 3.206 px; it fails either way.
- tracks_for_fit equals the measurement tracks exactly.
- The residuals show the 4-frame cadence sawtooth (LESSONS §3) and the AI period wander after about 7.5 s.

## Verifier changes for v002 (logged after the ck001 results existed; no tolerance changed)
- `vparams.py`: adapter for `elastic_pendulum_hinged_body_v1`, from the documented spec. It maps:
  - A = pivot_world_m, e_b = (0, 0, b), I = I_perp isotropic (spin 0);
  - c_x → c_v, c_s → c_a, c_r → c_r;
  - the eye initial state from X0, V0, n0, u0;
  - variant edits back into the schema.
  It reproduces dynamics' ω_ax 15.84928 and lateral modes 4.86122 / 26.35853 exactly.
- `vio.py`: production column aliases (H_*, Hdot_*, X_*, q_*, spring_*), and world ω = n × ṅ.
- `check_blend.py`:
  - `--anchor-world`;
  - `--vsp` mode for the GN spring, which reads vsp_part, vsp_cl and vsp_w;
  - contact gap = |distance to the loop centreline polyline − wire radius|, because straight GN segments carry only end rings;
  - static spring parts given by part id (0, 1, 6), not vsp_w = 0, because unrotated coil vertices translate with the per-turn extension;
  - P_eff is the axis point of the top coil, so the judged quantity is the variation of the centreline-to-P_eff distance, not the distance itself;
  - added during the audit: H in the lower-loop frame and H out of the loop plane. The first fault test showed that a 50 µm out-of-plane hook offset changes the polyline gap only to second order (7.7e-6).
  New production-blend negative controls (copies in out/ck001/negctl):
  - HookLower offset 50 µm → P5 FAIL (5.0e-5 out of plane);
  - anchor drift 2 µm → P5 FAIL;
  - weight rotation +1e-4 rad → P6 FAIL;
  - Open Coils +1 at frame 120 → P5 FAIL (winding 21 → 22, topology change), and P10 FAIL vs ck.
  The synthetic blend controls were re-run with the final code: all as expected.
- `audit_fit_residuals.py` and `make_visuals_ck.py` are new. `check_controls.py` now fits the envelope for damped rigid swing.

## ck001 findings / requests
1. **P9 FAIL: the production integrator is not robust outside the fitted regime.**
   - Same production initial conditions, k×4: the hinge separates from the verifier solution after 1.17 s and becomes NaN at 1.96 s. Production h vs h/2 differ by 0.68 m before NaN, and |n| reaches 2.6, so the body-axis norm is not preserved.
   - Same initial conditions, L0×1.5: production h vs h/2 differ by 3.4 mm, and production vs verifier by 3.7 mm. My solver converges: sub 8 vs sub 32 differ by 3e-14 and 1.4e-11.
   - Both variants are compressive regimes (T_min −101 N / −140 N), which are inadmissible for a hooked spring under P7. The fitted production regime passes P1–P8 with large margin.
   - Request to dynamics: renormalise or constrain n (or integrate a quaternion), and/or step-size control. Re-run P9.
   - The lead decides whether the production-IC sub-check should use admissible (re-equilibrated) initial conditions. I have not relaxed or redefined it.
2. **R2 anchor RMS 6.73 px against a 3.24 px target: FAIL.** The claims are reproduced exactly; this is a fidelity issue, not a physics-validation one.
3. Running build.py from the checkpoint copy writes `__pycache__` into that tree, as in clip 01. It was removed from my copy; nothing was written into checkpoints/.
4. solver/NOTES wording: "k(|AH| − L0)" means |P_eff H| (spring_len is |P−H|; my check of ℓ is against P). Please correct the wording.
5. The P9 `production_ic` bakes for k×4 and L0+50 % failed: bake rc 1 / 2 on the NaN or violent trajectories. Those blends were not checked.

## Commands (ck001; from ROOT, PYTHONDONTWRITEBYTECODE=1, PY=.venv/bin/python, B=blender, O=verify/out/ck001, C=$PWD/$O/ckcopy)
```
cp -Rp checkpoints/ck001 $C && (cd $C && shasum -a 256 -c MANIFEST.sha256)
$PY verify/ref_solver.py --params $C/params/physics.json --out $O/ref_trajectory.csv --lyapunov
$PY verify/run_production.py --params $C/params/physics.json --set /parameters/damping/c_x=0.0 --set /parameters/damping/c_s=0.0 --set /parameters/damping/c_r=0.0 \
    --solver-cmd "{py} solver/run.py --params {params} --out {outdir}" --cwd $C --out $O/prod_runs/conservative.csv
$PY verify/run_benchmarks.py --params $C/params/physics.json --outdir $PWD/$O/bench --solver-cmd "{py} solver/run.py --params {params} --out {outdir}" --cwd $C
$PY verify/check_physics.py --params $C/params/physics.json --trajectory $C/solver/out/trajectory.csv --meta $C/solver/out/trajectory_meta.json --ref $O/ref_trajectory.csv \
    --conservative-trajectory $O/prod_runs/conservative.csv --bench-manifest $O/bench/bench_manifest.json --out $O/check_physics_ck001.json
$PY verify/check_convergence.py --order 4 --values 0.010416666666666666 0.005208333333333333 0.0026041666666666665 0.0013020833333333333 0.0006510416666666666 0.0003255208333333333 \
    --cmd-template "$PY verify/run_production.py --params $C/params/physics.json --solver-cmd '{{py}} solver/run.py --params {{params}} --out {{outdir}} --h {value}' --cwd $C --out {out}" \
    --workdir $O/prod_runs/conv --out $O/check_convergence_ck001_full.json        # judged file check_convergence_ck001.json = finest 4 of the same runs
BARGS="--weight-name VSP_Weight --eye-local 0,0,0 --spring-name VSP_Spring --anchor-name VSP_Anchor --anchor-world 0,0,0 --weight-origin eye --orientation rigid --vsp --exclude-static VSP_HookLower"
tools/with_render_lock.sh $B -b $C/spring.blend --python verify/check_blend.py -- --params $C/params/physics.json --ref $PWD/$O/ref_trajectory.csv \
    --trajectory $C/solver/out/trajectory.csv --out $PWD/$O/check_blend_ck001.json --dump $PWD/$O/check_blend_ck001.npz $BARGS
$PY verify/check_controls.py --params $C/params/physics.json --tag ck001 --solver-cmd "cd $C && {py} solver/run.py --params {params} --out {outdir}" \
    --bake-cmd "cd $C && $PWD/tools/with_render_lock.sh $B -b --factory-startup --python scene/build.py -- --out {blend} --physics {params} --bake {traj} --look" \
    --blend-check-args "$BARGS" --out $O/check_controls_ck001.json
for x in A B; do SOLVER_CMD='{PY} solver/run.py --params params/physics.json --out solver/out' \
  BUILD_CMD='{B} -b --factory-startup --python scene/build.py -- --out scene/out/spring.blend --bake solver/out/trajectory.csv --look' \
  verify/clean_rebuild_verifier.sh $O/ckcopy $O/p10/rebuild_$x $O/ref_trajectory.csv params/physics.json scene/out/spring.blend $BARGS; done
$PY verify/compare_dumps.py $O/check_blend_ck001.npz $O/p10/rebuild_A/check_blend.npz --out $O/p10/compare_ck_vs_A.json   # + ck_vs_B, A_vs_B
$PY verify/audit_fit_residuals.py --ck $O/ckcopy --ref $O/ref_trajectory.csv --anchor 7.5 --out $O/fit_audit_ck001.json --fig verify/visuals/v002_fit_residual_map.png --label "v002 | ck001 | final for ck001"
$B -b $C/spring.blend --python verify/test/inject_prod_blend_fault.py -- --fault <hook_gap_50um|anchor_drift_2um|weight_rot_1e-4|coil_extra_turn> --out $O/negctl/<f>.blend
$PY verify/make_visuals_ck.py --ck ck001 --version v002 --status "final for ck001"
```

---
# v003 — full audit of checkpoint ck002 (status: final for ck002)
Inputs: only the checkpoint's own files, from the copy `verify/out/ck002/ckcopy` (MANIFEST 173/173 OK before and after):
- physics.json `962fc536…`, trajectory.csv `d45be5ab…`, spring.blend `6718ce7f…`, camera.json `df339954…`.
- Solver: `rk4-fixed-step-elastic-pendulum-v2-projected`.
- Re-running the production solver on the nominal params reproduces trajectory.csv bitwise.
- Summary: `out/ck002/summary.json`.

| item | verdict | numbers |
|---|---|---|
| P1 | PASS | orders 4.001 / 3.999; finest pair 5.0e-12 m (9.2e-12 rad) |
| P2 | PASS | 3.7e-13 m, 6.1e-13 rad. The extra 3-D undamped variant gives 5.8e-12 m and 6.5e-11 rad. Lyapunov factor 1.00 |
| P3 | PASS | B1 rest 1.7e-16; B2 exact axial 8.4e-10 (ω 5.9e-12); rigid modes 4.861 and 26.157 rad/s at 1.5e-11 and 3.5e-11 |
| P4 | PASS | damped balance 9.1e-14; undamped drift 7.6e-14; L_z drift on the 3-D variant 4.4e-16 |
| P5 | PASS | static parts 0 drift; top contact 4.8e-10 m; lower contact 9.6e-8 m; H out of the loop plane 9.9e-9 m; sliding 2.6e-7 m; coil to P_eff variation 1.2e-6 m; winding 21.0000005 (variation 8e-10); topology constant |
| P6 | PASS | 1.5e-8 m, 3.0e-8 rad (frames and 4 subframes) |
| P7 | PASS | T_min 51.11 N |
| P8 | PASS | ICs exact; t grid exact; blend H(0) 1.5e-8 m |
| P9 | PASS | details below |
| P10 | PASS | two clean rebuilds: trajectory bitwise; ck/A/B dumps bitwise identical |
| Fit audit | reproduced | anchor RMS 6.724228 px (claimed 6.724228); after the anchor window 15.459 px; ring 5.067 px; weight centre 8.047 px; all claims within 4.1e-8 px. R2 (target 3.245 px) **FAIL**, as declared by dynamics |

P9 details:
- Formula checks pass: k×4, M×4, L0+50 % and damping×10, all within 4e-6 relative.
- Re-equilibrated variants were made with `<ck>/solver/make_variant.py` (subprocess), the documented interface. Each one, at its own initial state, matches my solver to ≤ 5.7e-13 m. All baked builds pass P5, P6 and P8b.
- I checked each variant's initial state independently against the CHANGELOG rule (all consistent):
  - edits applied exactly, other parameters unchanged;
  - φ0 and φ̇0 equal to the base state;
  - ψ/φ equal to my linear-mode κ to 2e-16;
  - axial rate 0;
  - T0 > 0.
- Diagnostic, not judged, starting from the NOMINAL state:
  - k×4 and L0×1.5 now raise `SpringCompressionError` (rc 1). My solver shows minimum tension −101.5 N and −140.8 N within 0.5 s, so the guard is correct.
  - M×4 and damping×10 run normally (no compression).

Production-blend negative controls (copies in `out/ck002/negctl`):
- HookLower offset 50 µm → P5 FAIL.
- Anchor drift 2 µm → P5 FAIL.
- Weight rotation +1e-4 rad → P6 FAIL.
- Open Coils +1 → P5 FAIL, and P10 FAIL vs ck.

The synthetic physics controls were re-run after the `check_controls.py` changes: all as expected.

Verifier code changes for v003:
- `check_controls.py`: `--variant-cmd` (production re-equilibration tool), `--nominal-ic-too` (diagnostic), and the `ic_rule_check` above.
- `make_visuals_ck.py`: labels.
- No tolerance changed.

Commands: as in the v002 block with ck001 → ck002. P9 adds:
`--variant-cmd "cd $C && {py} solver/make_variant.py --params params/physics.json {sets} --out {out}" --nominal-ic-too`.

Open items:
1. R2 fidelity: 6.72 px against 3.24 px. Not a physics-validation item.
2. Building from a checkpoint copy writes `__pycache__` into it. This happened in my copy only, and was removed.
