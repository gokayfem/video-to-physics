# Verification

Verification answers one question: does the checkpoint do exactly what the declared model says, accurately? It is
run by role E on the frozen checkpoint, with code that shares nothing with the solver, against tolerances declared
before any data. It does not answer whether the real world would do this.

## Contents

- [Independence](#independence)
- [The audit list](#the-audit-list)
- [Convergence, observed order and GCI](#convergence-observed-order-and-gci)
- [Energy, mass and equilibrium budgets](#energy-mass-and-equilibrium-budgets)
- [Saved-file checks](#saved-file-checks)
- [Scene logistics](#scene-logistics)
- [Negative controls](#negative-controls)
- [PASS/FAIL reporting](#passfail-reporting)
- [Re-checks of later checkpoints](#re-checks-of-later-checkpoints)

## Independence

- The verifier never imports or reads `solver/` code. It reads production outputs and checkpoints as data and
  re-runs solver studies only through their documented commands.
- It writes its own reference solver with a **different method** (shooting instead of a discrete rod; a
  Taylor–Hood projection scheme instead of an adaptive finite-volume code; Gauss–Legendre instead of RK4) and
  validates it first on the closed forms.
- Its plan (`verify/PLAN.md`) lists checks and tolerances from `acceptance.json` before the first checkpoint.
- In a single-agent run, write the reference solver from `FORMULATION.md` alone, before looking at `solver/`.

## The audit list

Adapt per case; keep the IDs stable across checkpoints.

| ID | Check | Typical pass |
|---|---|---|
| P1 | convergence: ≥ 3 levels in space and time, observed order | order within 0.5 of theory; finest difference below tolerance |
| P2 | independent benchmark: verifier solver vs production at identical parameters | state difference < tolerance; key scalars (P_cr, St, C_D) within declared % |
| P3 | analytic limit in its regime | < 1e-7 relative (ODE); ≤ 0.5–1 % (field solvers) |
| P4 | energy / mass / equilibrium budget | conservative drift < 1e-8; balance residual within declared fraction; no unexplained growth |
| P5 | constraints on the saved .blend (frames + subframes) | anchors fixed, lengths and attachments < 1e-6 m |
| P6 | Blender-evaluated vs solver | < 1e-5 m / rad, or relative 1e-5 for meshes |
| P7 | forces admissible | tension ≥ 0, normal ≥ 0, friction in cone |
| P8 | initial conditions and time | IC equals params; frame n ↔ t = n/fps; no time scaling |
| P9 | control response through the real build path | variants change the dynamics as the equations predict |
| P10 | repeatability | two clean rebuilds give identical keys, caches and evaluated transforms |
| L1 | scene logistics | prop speeds, continuity, counts, visibility (below) |
| R* | reference fidelity (reported separately) | anchor RMS, camera RMS, render-track RMS; divergence map named |

Fluids add: divergence norm, no-slip, St / C_D / C_L′ vs independent reference, vortex convection speed, wake KE
settling per band, dye advected by the solver's velocity (advection scale ≈ 1), tracer mass budget, dye front
position at t = 0, anchored phase. Fracture adds: origin at maximum tension, failure load inside the declared
range, crack-surface energy balance, fragment rigid-body checks.

## Convergence, observed order and GCI

With solutions f₁ (fine), f₂, f₃ (coarse) at refinement ratio r:

- observed order p = ln((f₃ − f₂)/(f₂ − f₁)) / ln r
- Richardson extrapolation f_ext = f₁ + (f₁ − f₂)/(r^p − 1)
- grid convergence index GCI_fine = 1.25 · |(f₁ − f₂)/f₁| / (r^p − 1)

`v2p/checks.py` implements these. Apply them to scalars (P_cr, St, C_D, C_L′, period) and to fields or
trajectories (max norm over the clip). Non-monotone convergence (oscillating p) is reported as such, with the
spread, not forced into a GCI. A frequency can converge while force amplitudes are still 10–20 % off; check both.

## Energy, mass and equilibrium budgets

- Conservative runs: |ΔE|/E₀ below tolerance; no per-sample increase beyond round-off.
- Damped or driven runs: D = W_external − ΔU − ΔV − ΔT must be ≥ −tol and non-decreasing; transfer contact forces
  count as external work.
- Statics: equilibrium residual (curvature vs moment / EI).
- Fluids: divergence, tracer mass budget within the declared %, kinetic energy per wake band per period.
- Inextensible structures: arc length preserved.

## Saved-file checks

Run on the checkpoint's saved .blend, reopened, as a user meets it:

- evaluate transforms or the evaluated mesh at every frame and four subframes; compare with the solver samples;
- fixed anchors and clamps do not move; attachments hold; no interpenetration; clearances above bound;
- camera matches the camera json (reprojection of known points);
- `scripts/blender/check_bake.py` covers the common cases; record its exact arguments in the audit json.

## Scene logistics

Physics-only audits miss props that teleport. Check every carried or placed object:

- peak speed (flag anything above the declared limit, typically ≈ 1 m/s for small hand-placed objects);
- position continuity (no jumps between samples), counts (a stack shrinks as items leave it);
- visibility (no hide/unhide toggles to fake a transfer);
- seat positions within tolerance of the physics load points at the scheduled times.

## Negative controls

A check is credible only if it fails on wrong data. For each main check, plant an error and confirm detection.

| Control | Planted error | Expected |
|---|---|---|
| parameter | stiffness or length +1 % | P2/P3 FAIL |
| constraint | stretch 2e-5 in an inextensible rod | P5 FAIL |
| energy | an injected energy pulse | P4 FAIL |
| time | frame offset or 1 % time scaling | P8 FAIL |
| bake | a perturbed vertex or key | P6 FAIL |
| flow | wrong inlet speed; Eulerian smeared dye | St / front checks FAIL |
| good control | a correct synthetic case | all PASS |

Use the harness in `v2p/checks.py`. Report "controls caught: k/n"; a control that is not caught means the check
is too loose and is fixed before the audit counts.

## PASS/FAIL reporting

`verify/out/<ck>_audit.json`, one entry per check:

```json
{"id": "P2", "quantity": "P_cr_N", "production": 1.35767, "reference": 1.35757, "rel_diff": 7.4e-5,
 "tolerance": 0.005, "result": "PASS", "command": "python verify/checks/check_production.py --ck ck002"}
```

- Lead with a labelled visual (version | check | status), then the table.
- State what was not checked and why (for example "post-buckling ≤ 1 % checked on the static n = 7 state; dynamic
  frames never settle").
- A FAIL names the defect and the owning role; the fix arrives as a new checkpoint.

## Re-checks of later checkpoints

- Motion-identical checkpoints (look or geometry changes only) re-run P5, P6, P8 and a dump comparison; state
  "motion bitwise identical to ckNNN" with hashes.
- Ship `verify/recheck.sh <ck>` that reads the verifier's recorded arguments; re-checks by hand with rounded or
  missing arguments produce false FAILs.
- Check the manifest from inside the checkpoint directory (paths in `MANIFEST.sha256` are relative).
