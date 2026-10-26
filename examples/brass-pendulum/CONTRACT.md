# Simulation 01 — Brass pendulum: shared contract (v3, frozen by the lead)

Changes to this file are made only by the lead integrator and bump the version. If you need a change,
write the request into your own `NOTES.md` and say so in your final report.

## Project root
`<CASE_ROOT>` (below: `ROOT`).
Workspace root (two levels up): `WS`.

## Reference (immutable — never modify, re-encode in place, or retime)
- `WS/reference-batch/clips/01-brass-pendulum.mp4`
- sha256 `ced22d105d7bf9e93d7c152fab02b62e077cbe54162056114201c50b61f56bae`
- H.264, 1920×1088, 24 fps constant, 241 frames, no audio. Frame `n` (0-based) is at `t = n/24 s`
  (pts verified: 0.000000 … 10.000000). Physical time in every file = this `t`, seconds.
- It is an AI-generated clip, an accepted *visual* reference, **not** a physical measurement.
  Prompted intent was "about 4° initial displacement, gentle planar oscillation, slight damping".
  Earlier QA: amplitude looks ≈ ±6°, period ≈ 1.25 s by eye, no visible decay (unverified).

## Image conventions
- Pixel coordinates `(u, v)`: origin at the top-left corner of the top-left pixel, `u` right, `v` down,
  pixel centres at integer + 0.5. Full resolution 1920×1088. Report subpixel floats.
- Tag every image observation with `frame`, `t_s`, method and a confidence/uncertainty.

## World conventions (SI)
- Metres, kilograms, seconds, radians internally (degrees allowed only in reports, labelled).
- Right-handed, **Z up**. **X** = horizontal direction lying in the swing plane, pointing to image-right.
  **Y** = horizontal, perpendicular to the swing plane, pointing away from the reference camera side
  (X × Y = Z).
- **Origin = the pivot point P**: the point about which the suspension rotates (where the wire leaves
  its pin/loop). P is fixed: (0, 0, 0).
- Gravity: `g = 9.80665 m/s²` along −Z (standard gravity; an assumption, the clip has no gravity
  reference). The tabletop / base top surface is a horizontal plane at `z = −h_pivot`.
- Pendulum angle `θ` is defined *by position* (not by an Euler sign convention):
  bob centre `= P + L_c · (sin θ, 0, −cos θ)` where `L_c` is pivot-to-bob-centre distance.
  So `θ > 0` means the bob is toward +X (image right). `ω = dθ/dt`.
- Blender: 1 Blender unit = 1 m, scene Z up (Blender default), scene frame rate 24 fps,
  `frame_start = 0`, `frame_end = 240`, Blender frame `n` ↔ `t = n/24`. Blender cameras look along
  local −Z with local +Y up.

## Scale
The clip contains no metric reference. Metric scale is **model-conditioned**: it is inferred from the
measured period under the declared pendulum model and the assumed `g`. Anything derived this way must
be labelled "model-conditioned" and must not later be used as independent proof of the same model.
Object-size plausibility (desk-toy scale) may be reported as a sanity range only.

## Parameter files (separate from construction code)
- `params/physics.json` — owned by **dynamics**: model id, L_c, bob radius/mass/density, suspension
  properties, damping coefficients, θ0, ω0, g, integrator settings, plus provenance of each value
  (`measured` | `fitted` | `assumed` | `model-conditioned`) and uncertainty.
- `params/geometry.json` — owned by **geometry/rig**: stand, base, arm, pin, wire, bob dimensions
  expressed in this world frame (pivot at origin).
- `params/camera.json` — owned by **camera/appearance**: intrinsics (sensor, focal, vertical FOV,
  resolution, principal point), extrinsics (world→camera), uncertainty; plus the locked diagnostic
  camera.
- `params/look.json` — owned by **camera/appearance**: lights, materials, world, render settings.
- Fitted values live only in `params/`. Builders read params; they never hard-code fitted numbers.

## Motion data interface (dynamics → rig / verifier)
- `solver/out/trajectory.csv` columns:
  `t_s,theta_rad,omega_rad_s,alpha_rad_s2,x_m,z_m,KE_J,PE_J,E_J,tension_N,P_diss_W` — dense samples
  (at least every reference frame and every 1/4 frame) from the validated solver at the fitted params.
- `solver/out/trajectory_meta.json`: params hash (sha256 of `params/physics.json`), solver id,
  tolerances, git-less code hash (sha256 of solver source files), created_utc.
- Blender motion: a single rig object **`PND_Swing`** (empty) at P, whose rotation about world +Y
  gives the pendulum angle. The rig agent must choose and numerically verify the Euler sign so that
  the evaluated bob world position equals `P + L_c (sin θ, 0, −cos θ)`.
  The suspension and bob are children of `PND_Swing`. Motion is baked from the solver as keyframes
  on every frame with Bézier handles set from ω (cubic Hermite), and the bake is reproducible by a
  command. Baked motion is disclosed as baked.

## Directories and ownership (write only inside your own directory unless listed)
| Role | Owns |
|---|---|
| Lead integrator | `CONTRACT.md`, `config/`, `checkpoints/`, `build_all.sh`, `STATUS.json`, final `review/` page |
| Reference measurement | `measurement/` |
| Physics formulation | `physics/` |
| Dynamics implementation | `solver/`, `params/physics.json` |
| Geometry & rig | `scene/` (builder modules, `scene/build.py`), `params/geometry.json` |
| Camera calibration | `camera/`, `params/camera.json` |
| Look: materials, textures, lighting (v3) | `look/`, `params/look.json`, `scene/look.py` (module imported by `scene/build.py`) |
| Independent physics verifier | `verify/` (must not import from `solver/`) |
| Independent visual reviewer | `visual_review/` |
| Reproducibility & delivery | `delivery/` |

Everyone may *read* everything. Nobody edits another role's files. Each role keeps a `NOTES.md` in its
directory with: what it did, commands to reproduce, outputs (with sha256), open issues, requests to
other roles.

## Tools
- Python with numpy/scipy/opencv/matplotlib: `ROOT/.venv/bin/python` (run scripts with `-I` is not
  required for our own code). Blender: `blender` (or `$BLENDER`) (5.2.2 LTS; bundled numpy 2.3.4).
- Video-to-Blender skill (nodekit, NK library) at `<video-to-blender>/skills/procedural-blender-nodes`
  (read its `SKILL.md` before writing node graphs).
- Heavy Blender work (any render > 960 px wide, any Cycles render, any bake/render longer than ~20 s)
  must run under the shared lock: `ROOT/tools/with_render_lock.sh <command…>`. Keep preview renders
  small (≤ 960×544, EEVEE or Workbench) and few. Never run more than one Blender render at a time
  across the team; queue rather than reduce quality silently.
- Machine: 10 CPU cores, 24 GB RAM, Apple Silicon.

## Honesty rules
- Separate measured image quantities, fitted world quantities and assumptions.
- No hand-authored motion curves, no negative damping, no hidden driving forces, no changing length,
  no time scaling to chase the AI clip. If the AI clip shows non-physical behaviour (e.g. amplitude
  growth), document it as a reference defect.
- Never claim 100 % physical accuracy. Report residuals, tolerances and what was not checked.
- Do not touch `WS/reference-batch`, other clips, or any other simulation. Clip 01 only.

## Standing user instruction (added in contract v2, 2026-10-26): "give visual feedbacks everytime"
Every meaningful checkpoint, specialist handoff, correction and approval request must include actual
visual evidence the user can view (current image, annotated frame, rendered preview, playable video or
comparison) — never text-only milestone reports or bare file paths.
- Measurement: annotated reference frames with pivot/bob tracks and trajectory overlays.
- Geometry/rig: the current blockout from the reference camera.
- Dynamics: a short playable motion preview plus matched turning-point frames (plots supplement it).
- Appearance changes/fixes: reference / previous / current comparisons, same camera and timestamps,
  close-ups where useful.
- Validation: measured-versus-simulated overlays and relevant diagnostic plots.
Each visual carries a short caption: what changed, what the evidence shows, what remains wrong, next fix.
Label every artifact (burned-in or adjacent caption) with version, timestamp/frame, and status
(isolated component | integrated scene; draft | validated). Distinguish observed data, model predictions,
assumptions and uncertainty; visual similarity is not proof of physical accuracy. Never present an old
visual as new; label reused images "unchanged last preview". Prefer cheap previews between milestones.
Specialists: put each handoff visual in your own directory under `<role>/visuals/` (versioned filenames,
e.g. `v003_blockout_f000-240.jpg`) and list them first in your final report with captions. The lead
publishes them on the review page `review/index.html`.

## v3 (2026-10-26): user instruction "you need to match textures etc."
Appearance is split from camera calibration into its own role (Look). Texture/material/lighting match to
the reference is a first-class requirement: walnut grain direction/scale/colour, end-grain, oiled sheen,
polished brass reflections/highlights, steel pin, dark wire, backdrop gradient, tabletop/floor and
shadows — compared region by region against the reference with measured statistics and
reference / previous / current visuals at the same camera and frame.
