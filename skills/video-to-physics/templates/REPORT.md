# {{TITLE}} — report (checkpoint {{CK}})

**Goal:** keep the scene of the reference, replace its wrong physics with verified physics.
**Status:** {{STATUS}}  <!-- e.g. "physics verified independently (k/n PASS); final render done; awaiting human approval" -->
**Not claimed:** real-world accuracy. Declared and assumed values are listed below; the claim is that the render
does exactly what the declared model says, verified.

![final side-by-side]({{SBS_STILL}})

## What the reference shows (measured, `measurement/summary.json`)
<!-- Skip for input kind "none". Events with times; defects with numbers (camera drift px, energy gain %,
time compression factor, teleports, impossible deformation). -->
{{REFERENCE_FINDINGS}}
None of these behaviours were imitated.

## Corrected physics
- **Measured / declared:** {{MEASURED_DECLARED}}
- **Assumed (with sources):** {{ASSUMED}}
- **Fitted / anchored (inside the anchor window, rule declared before fitting):** {{ANCHORED}}
- **Model-conditioned:** {{MODEL_CONDITIONED}}
- **Model:** {{MODEL}}  <!-- equations, solver, discretisation, time integration -->
- **Result:** {{RESULT}}  <!-- the physical story with numbers -->

## Verification
| Check | Value | Tolerance | Result |
|---|---|---|---|
| {{ID}} | {{VALUE}} | {{TOL}} | PASS |

- **Solver (C):** benchmarks, convergence (levels, observed order), budgets.
- **Independent verifier (E, no shared code):** reference agreement, bake vs solver, negative controls k/n.
- **Not checked:** {{NOT_CHECKED}}

## Scene and look
- Frame 0: ours vs reference (and vs our first frame for `idea`).
- **Our additions, not in the reference:** {{ADDITIONS}}
- **Declared deviations:** {{DEVIATIONS}}
- Scene checks: intersections, constraints, camera reprojection, logistics. Reproducibility: clean rebuild result.

## Sensitive assumption
{{SENSITIVE}}  <!-- the assumption that most changes what the viewer sees, and a sensitivity run -->

## Key files
`review/index.html` · `review/final_{{CK}}.mp4` · `review/visuals/{{CK}}_sbs.mp4` · `config/CHANGELOG.md` ·
`physics/FORMULATION.md` · `verify/out/{{CK}}_audit.json` · `visual_review/{{CK}}_review.json`
Rebuild: `python3 <SKILL_DIR>/scripts/v2p.py checkpoint {{CASE}} <new name>` (solver reproduce command: {{REPRODUCE}}).
