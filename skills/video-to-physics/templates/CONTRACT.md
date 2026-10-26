# {{TITLE}} — contract (lead, v1)

Frozen by the lead on {{DATE}}. Only the lead edits this file; request changes in your role's `NOTES.md`.
Decisions after v1 go to `config/CHANGELOG.md` with a UTC timestamp.

## Goal
Keep the scene of the reference, replace its wrong physics with verified physics. The scene layer (geometry,
camera including its measured motion, materials, light) matches {{SCENE_SOURCE}}. The motion layer is a solved,
benchmarked physical model. Reference defects are measured and reported, never imitated.

## Input
- Kind: `{{INPUT_KIND}}` (ai-video | footage | image | idea | none)
- Reference: `reference/{{REFERENCE_FILE}}`, sha256 `{{SHA256}}` (immutable)
- {{SIZE}}, {{FPS}} fps, {{FRAMES}} frames; frame n at t = n / {{FPS}} s
- Trust: {{TRUST}}  <!-- e.g. "accepted visual reference, not a measurement" for ai-video -->

## World frame and units
- SI units, right-handed, Z up; gravity g = {{G}} m/s² along −Z ({{G_LABEL}}).
- Origin and axes: {{WORLD_FRAME}}
- Blender: 1 unit = 1 m, scene fps = {{FPS}}, frames 0 … {{LAST_FRAME}}.
- Scale: {{SCALE_STATUS}}  <!-- measured from <object> | declared (our first frame) | model-conditioned (period + assumed g) -->
- Camera of record: `{{CAMERA_JSON}}` (pinhole, OpenCV convention, pixel centres at +0.5); never deform geometry to
  explain camera motion.

## Physics
- Phenomenon: {{PHENOMENON}}
- Model (initial choice, B confirms): {{MODEL}}
- Declared / assumed inputs: {{ASSUMED}}
- Initial state: {{INITIAL_STATE}}
- Anchor rule (declared now, before any fit): {{ANCHOR_RULE}}
- Anchored choices (the only things taken from the reference): {{ANCHORED_CHOICES}}
- Loading / forcing schedule and its visible cause: {{SCHEDULE}}
- Standard: benchmarks and tolerances in `config/acceptance.json`; convergence on ≥ 3 levels.

## Ownership (write only in your own paths; read everything)
| Role | Owns |
|---|---|
| Lead | CONTRACT.md, config/, checkpoints/, review/, STATUS.json |
| A Measurement | measurement/ |
| B Formulation | physics/ |
| C Solver | solver/ |
| D Scene | scene/ |
| E Independent verifier | verify/ (never imports or reads solver/ code) |
| F Visual / reproducibility reviewer | visual_review/ |

## Interfaces
- Solver output: `{{SOLVER_OUTPUT}}` + `solver/out/meta.json` (params, code sha256, levels, results).
  <!-- deformation npz: positions (N_samples, rows, cols, 3) world m, t s | tracer PNGs 16-bit RGB + meta.json | trajectory csv/npz -->
- Samples: every frame + 4 subframes (t = n/fps + k/(4 fps)) unless stated.
- Scene: `scene/build.py` deterministic and params-driven; object prefix `{{PREFIX}}_`; camera `{{PREFIX}}_Cam`.
- Checkpoints: `v2p.py checkpoint {{CASE}} ckNNN` from `config/build.json`, built from frozen copies.

## Honesty rules
- Label every value measured / declared / assumed / fitted / model-conditioned / derived.
- No hand-authored motion, per-frame forces, negative damping, hidden drives or time scaling.
- Never claim real-world accuracy; report residuals and what was not checked.
- Every handoff: labelled visuals first (version | frame/time | status), then numbers.

## Tools
Python: `{{PYTHON}}`. Blender 5.2+: `{{BLENDER}}` (renders only via `v2p.py lock -- ...`).
Solver environments: {{SOLVER_ENVS}}. Guardrails: `references/agents-and-guardrails.md` (paste-in block in every brief).
