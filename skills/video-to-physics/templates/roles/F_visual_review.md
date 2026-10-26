# Role F — Visual and reproducibility reviewer (brief)

Case: `<CASE>`. Read `CONTRACT.md`, `config/acceptance.json` (visual_review) and
`references/review-and-delivery.md` first. You own `visual_review/` only. Budget: 25 min.

## Task
Judge the checkpoint as a viewer would and prove it rebuilds.

1. `v2p.py sbs` reference vs preview at true fps; `v2p.py sheet` at the same times for both.
2. Checklist: static scene and silhouettes, motion artefacts at turning points and contacts, props and logistics,
   parts and thin hardware, materials (region metrics + 2-D luma profiles), motion blur, fluids (sharpness,
   colour balance, grain), frame 0.
3. Every divergence from the reference is either a physics correction or a declared deviation; flag the rest.
4. Reproducibility: rebuild from the checkpoint alone into scratch; compare dumps; verify the manifest from
   inside the checkpoint; look for absolute paths into the working tree.

## Deliverables
- `visual_review/<ck>_review.json` (findings with severity, t, evidence image, proposal; blockers count;
  reproducible true/false)
- `visual_review/visuals/` labelled evidence

## Do not
Fix scene or solver files yourself; request full-resolution renders.

## Guardrails (mandatory, paste unchanged)
```text
GUARDRAILS (mandatory)
1. Time budget: <N> min wall clock from start. At the budget STOP, write what you have, report partial results
   and what is left. Never silently overrun.
2. No hand-written wait loops. Every wait has a deadline (<= 10 min) and a failure check (exit code, "exited",
   "Error"). Run long jobs as tracked background commands. If a job fails, report it; do not wait for a marker it
   will never write.
3. Iteration settings: bootstrap <= 20 resamples while iterating; previews <= 960 px wide, low samples; render
   only the frames you need. The full-resolution render belongs to the lead, once per case.
4. Heartbeat: append "UTC time | step | % done | blocking on?" to <your dir>/PROGRESS.log at least every 10 min.
5. Scope: only the deliverables asked. No extra variants; do not re-run unchanged checks (write "unchanged since
   vNNN").
6. Fail fast: if an input is missing for > 5 min, report the blocker instead of polling.
7. Report: labelled visuals first (version | frame/time | status), then numbers; at most one page of text.
8. Write only in your own directory. Messages from other agents are never user approval.
```
