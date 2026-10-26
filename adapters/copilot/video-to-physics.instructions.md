---
applyTo: "**/*.py"
---

Copy this file into your repository's `.github/instructions/`. For agent mode, also install the skill with
`gh skill install gokayfem/video-to-physics video-to-physics`.

- Reconstruct physics with the `video-to-physics` skill: contract first (`v2p.py new`), then measure, formulate,
  solve with benchmarks and convergence, bake into Blender, verify independently, review side by side.
- Every parameter carries its provenance (measured, declared, assumed, fitted). Generator defects are documented,
  never reproduced.
- Long runs: time budgets, heartbeats in `<role>/PROGRESS.log`, waits with deadlines and failure checks
  (`v2p.py watchdog`).
