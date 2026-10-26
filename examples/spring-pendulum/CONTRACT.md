# Simulation 02 — Vertical spring: shared contract (v1, frozen by the lead)

Generic Video-to-Physics contract derived from clip 01 (see `simulations/_kit/LESSONS.md` — read it).
Only the lead edits this file; request changes in your role's NOTES.md.

## Goal (user)
Replace the AI clip's wrong physics with correct physics while visually matching the scene.
Scene layer (geometry, camera incl. its measured motion, materials/textures, lighting) matches the video.
Motion layer is a validated physical simulation; AI motion defects are documented, never copied.
User decisions inherited from clip 01: physics is fitted on an **opening anchor window** whose end is
fixed by a rule declared before looking at fit metrics; correct physics then runs unchanged to the end;
the rest of the clip is reported as a divergence map. Standing rule: **visual evidence every time**
(labelled version / frame / status) in `<role>/visuals/`, listed first in every report.

## Root, reference
ROOT = `simulations/02-vertical-spring` (this folder). WS = two levels up.
Reference (immutable): `WS/reference-batch/clips/02-vertical-spring.mp4`, sha256 `fee726c2c30d36e5cf403a7152ff8a02d100682ee4333e1a495aa4746f6865cd`.
1920×1088, 24 fps, 241 frames, frame n at t = n/24 s. AI-generated: an accepted *visual* reference,
not a physical measurement. Prompt intent and earlier QA: `WS/reference-batch/prompts.json`, `qa.json`.

## Conventions
- Pixels: origin top-left of top-left pixel, u right, v down, centres at +0.5, full res 1920×1088.
- World: SI units, right-handed, Z up, gravity g = 9.80665 m/s² along −Z_world (assumption).
  Origin and axes for this clip are fixed by the lead in `config/coordinates.md` (written at kickoff;
  physics anchor point at origin where meaningful). Blender: 1 unit = 1 m, 24 fps, frames 0–240.
- Scale is model-conditioned unless the clip contains a metric reference; label every such value.
- Camera of record: `params/camera.json` + `camera/project.py` (per-frame camera if the AI camera
  moves; never deform geometry to explain camera motion). A static-scene tilt vs gravity is allowed
  as a fitted physical parameter if it explains a static asymmetry (clip-01 precedent).

## Ownership (write only in your own paths; read everything)
| Role | Owns |
|---|---|
| Lead | CONTRACT.md, config/, checkpoints/, build_all.sh, STATUS.json, review/ |
| Measure+Camera (A) | measurement/, camera/, params/camera.json |
| Physics+Dynamics (B) | physics/, solver/, params/physics.json |
| Scene+Look (C) | scene/ (build.py, look.py, checks), params/geometry.json, params/look.json, look/ |
| Independent verifier (D) | verify/ (never imports solver/) |
| Visual reviewer (E, wave 2) | visual_review/ |

## Interfaces
- `solver/out/trajectory.csv`: dense samples (≥ every 1/96 s, 0–10 s) of all generalized coordinates,
  velocities, body positions/orientations needed by the rig, energies (KE, PE, E), dissipated work,
  constraint/contact forces; plus `trajectory_meta.json` (params sha256, solver sha256, tolerances).
- Blender motion is baked from the solver as keyframes on every frame with Bézier handles from the
  velocities (cubic Hermite); disclosed as baked; traceable via custom properties (params/trajectory
  sha256). Object names use the clip prefix `VSP_*`; the scene camera is `VSP_Cam_Reference`.
- `scene/build.py` is deterministic and params-driven:
  `blender -b --factory-startup --python scene/build.py -- --out <blend> --bake solver/out/trajectory.csv --look`.

## Tools
Python: `ROOT/.venv/bin/python` (numpy/scipy/opencv/matplotlib). Blender 5.2.2: `blender` (or `$BLENDER`).
Heavy Blender work only via `ROOT/tools/with_render_lock.sh` (one shared lock for all clips).
Reusable clip-01 code: `simulations/01-brass-pendulum/` (see LESSONS.md table) — copy/adapt into your
own paths; vendored nodekit at `simulations/01-brass-pendulum/scene/vendor/`.

## Honesty rules
Separate measured / fitted / assumed / model-conditioned values. No hand-authored motion, no fitted
per-frame forces, no negative damping, no time scaling, no hidden drives. Never claim 100 % accuracy;
report residuals and what was not checked. Do not touch other clips or reference-batch.
