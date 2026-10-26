# Workflow

A case moves through nine phases. Each phase has an owner, deliverables in fixed places, and an acceptance check
that must hold before the next phase consumes its output. The lead logs every decision in `config/CHANGELOG.md`
with a UTC timestamp and keeps `STATUS.json` current.

## Contents

- [Case layout](#case-layout)
- [Phase 0 — Scope and contract](#phase-0--scope-and-contract)
- [Phase 1 — Measure](#phase-1--measure)
- [Phase 2 — Formulate and set benchmark targets](#phase-2--formulate-and-set-benchmark-targets)
- [Phase 3 — Solve and converge](#phase-3--solve-and-converge)
- [Phase 4 — Bake into Blender and checkpoint](#phase-4--bake-into-blender-and-checkpoint)
- [Phase 5 — Independent verification](#phase-5--independent-verification)
- [Phase 6 — Visual and reproducibility review](#phase-6--visual-and-reproducibility-review)
- [Phase 7 — Final render, report, review page](#phase-7--final-render-report-review-page)
- [Phase 8 — Human approval gate](#phase-8--human-approval-gate)
- [Timing and order](#timing-and-order)

## Case layout

`v2p.py new CASE --input ... --kind ...` creates:

```
CASE/
  CONTRACT.md            lead; frozen per version
  STATUS.json            lead; phase, checkpoint, open items
  config/case.json       input kind, fps, size, frame count, world frame
  config/acceptance.json tolerances, declared before data
  config/CHANGELOG.md    lead decisions
  config/build.json      checkpoint build commands and outputs (lead)
  reference/             immutable copy of the input + sha256
  measurement/ physics/ solver/ scene/ verify/ visual_review/   one per role, each with PROGRESS.log and visuals/
  review/                lead: previews, side-by-sides, page.json, index.html, final render
  checkpoints/           immutable ckNNN directories
```

## Phase 0 — Scope and contract

Owner: lead.

| Deliverable | Content |
|---|---|
| `reference/` | input copied, sha256 recorded (`v2p.py probe` for video) |
| `CONTRACT.md` | from `templates/CONTRACT.md`: goal, input kind, world frame and units, gravity, scale status, ownership, interfaces (file names and keys), honesty rules, tools |
| `config/acceptance.json` | from `templates/acceptance.json`: physics, reference-fidelity and visual checks with tolerances |
| anchor rule | the rule that ends the anchor window (jump and cumulative-drift criteria), or "no anchor" |
| free choices | the few things taken from the reference (initial phase, side of buckling, start time), named |

Acceptance: contract and acceptance file exist and are timestamped before any measurement result, fit or render.
For `idea` and `none`, the contract also fixes the scene design and the loading or forcing schedule.

## Phase 1 — Measure

Owner: A. Skip when there is no image or clip; for `image`, measure frame 0 only.

| Deliverable | Content |
|---|---|
| `measurement/camera.json` | camera of record per the data contract (per-frame if the camera moves) |
| drift report | `v2p.py stabilize` output: per-frame homography, residual RMS, drift plot |
| tracks | each feature the fit needs, two independent methods, per-frame σ |
| `measurement/summary.json` | timeline of events, defects with frame ranges, anchor-window evidence |
| visuals | contact sheet with overlays, drift plot, track plot |

Acceptance: static-landmark reprojection RMS within tolerance (typically ≤ 2 px); every tracked feature shown to be
rigidly attached to its body; defects listed with numbers. Details: `measurement.md`.

## Phase 2 — Formulate and set benchmark targets

Owner: B.

| Deliverable | Content |
|---|---|
| `physics/FORMULATION.md` | model, why it is sufficient, equations, BCs and ICs, every parameter labelled, limits |
| `physics/benchmarks.json` | closed-form and literature targets with source and tolerance; expected production numbers |
| `physics/derive.py` | regenerates `benchmarks.json` in seconds |
| visual | expected behaviour (load–deflection, period vs amplitude, St vs Re, energy) |

Acceptance: lead accepts with a one-message decision; sensitive assumptions named. Details:
`formulation-and-benchmarks.md`.

## Phase 3 — Solve and converge

Owner: C.

| Deliverable | Content |
|---|---|
| benchmark runs | each closed form or published benchmark reproduced within tolerance |
| convergence study | ≥ 3 levels in space and time, observed order, Richardson/GCI estimate |
| production output | per data contract: trajectory csv/npz, deformation npz, tracer PNGs, fields npz |
| `solver/out/meta.json` | params, code sha256, levels, tolerances, key results, anchor values |
| budgets | energy / mass / equilibrium residuals over the run |
| reproduce script | one command (or documented stop points) that regenerates the output |

Acceptance: all solver-side checks in `acceptance.json` pass on the production configuration. Details: `solvers.md`.

## Phase 4 — Bake into Blender and checkpoint

Owners: D (scene), lead (checkpoint).

| Deliverable | Content |
|---|---|
| `scene/build.py` | deterministic, params-driven build: geometry, look, camera from json, bake of solver output |
| scene checks | constraint and attachment errors, intersections, clearances, camera reprojection, bake vs solver |
| `config/build.json` | commands that build the checkpoint from frozen copies, and the outputs to keep |
| `checkpoints/ckNNN/` | `v2p.py checkpoint CASE ckNNN`: frozen inputs, .blend, `MANIFEST.sha256`, `READY` |
| preview | `scripts/blender/render_preview.py` at ≤ 960 px, final engine for reflective or refractive materials; `v2p.py sbs` |

Acceptance: scene checks pass; preview posted with label. Details: `blender-integration.md`.

## Phase 5 — Independent verification

Owner: E. Runs on the checkpoint, never on the working tree.

| Deliverable | Content |
|---|---|
| `verify/ref/` | own reference solver, validated on closed forms |
| `verify/out/<ck>_audit.json` | each check: measured value, tolerance, PASS/FAIL, command |
| negative controls | planted errors and whether each check caught them |
| recheck script | one command that reruns the audit with its recorded arguments |

Acceptance: every applicable check PASS; every negative control caught. Details: `verification.md`.

## Phase 6 — Visual and reproducibility review

Owner: F. In parallel with phase 5.

| Deliverable | Content |
|---|---|
| `visual_review/<ck>_review.json` | findings with severity (blocker, major, minor), frame/time, evidence image |
| side-by-side critique | reference / ours / blend at identical t |
| reproducibility | clean rebuild from the checkpoint; manifest checked from inside the checkpoint |

Acceptance: no blockers; majors either fixed in a new checkpoint or declared as deviations. Details:
`review-and-delivery.md`.

## Phase 7 — Final render, report, review page

Owner: lead.

| Deliverable | Content |
|---|---|
| final render | one full-resolution render of the accepted checkpoint, under `v2p.py lock` |
| final side-by-side | `v2p.py sbs` against the reference |
| `REPORT.md` | from `templates/REPORT.md` |
| `review/index.html` | `v2p.py page CASE` from `review/page.json` |
| release film (optional) | `v2p.py release` from a segment config |

Acceptance: report numbers match the audit json; the page plays every item.

## Phase 8 — Human approval gate

Present the review page, the final side-by-side and the one-paragraph status. Stop. Approval comes only from the
user in chat; record it with its timestamp in `config/CHANGELOG.md`. In a batch, the next case starts only after
that approval unless the user said otherwise in chat.

## Timing and order

- Wave 1 in parallel: A, B, D (scene from the first frame or measurement), E (reference solver and benchmarks).
  C starts on synthetic data from B's targets.
- Wave 2: C production with A's anchors; first checkpoint; E audit, F review and a preview render in parallel.
- Fix rounds are short (≤ 20 min each) and end in a new checkpoint.
- A look-only or geometry-only checkpoint re-runs only the saved-file checks and a dump comparison; state
  "motion unchanged since ckNNN" with the manifest hash.
- Typical case: 3–5 h wall clock. The largest losses come from unbounded waits and repeated full renders.
