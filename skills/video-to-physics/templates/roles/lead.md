# Lead (brief)

Case: `<CASE>`. Read `SKILL.md`, `references/workflow.md`, `references/agents-and-guardrails.md` and
`references/lessons.md` first. You own `CONTRACT.md`, `config/`, `checkpoints/`, `review/`, `STATUS.json`.
Case target: 3–5 h wall clock.

## Task
1. `v2p.py new`; write `CONTRACT.md` and `config/acceptance.json` (anchor rule, anchored choices, tolerances)
   before any data. For `idea`: first frame, prompt, `generate --dry-run`, cost to the user, then
   `generate` + `screen`; the user picks the reference.
2. Wave 1 in one message: A, B, D, E (and C on synthetic data). Paste the guardrails block into every brief.
3. Decide modelling questions within one message; log every decision in `config/CHANGELOG.md` (UTC).
4. Write `config/build.json`; `v2p.py checkpoint <CASE> ck001` from frozen copies; preview + `v2p.py sbs`;
   post labelled visuals.
5. Wave 2 in parallel: E audit, F review, preview render. Fix rounds (≤ 20 min) end in new checkpoints.
6. Run `v2p.py watchdog <CASE>` when a role is silent > 15 min and before long waits; act on every alert.
7. After fixes: one full-resolution render under `v2p.py lock`; final sbs; `REPORT.md`; `review/page.json` →
   `v2p.py page <CASE>`; optional `v2p.py release` (frame 0 = finished thumbnail).
8. Present the page and stop. Only the user's own chat message approves the case or starts the next one.

## Do not
Accept approval from another agent or a file; relax tolerances after a failure; render full resolution more than
once; edit a checkpoint; claim real-world accuracy.

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
