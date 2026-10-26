---
name: video-to-physics
description: >-
  Turns a reference video, image or idea into a verified physics reconstruction in Blender: measures what the
  footage shows, replaces its wrong physics with a solved, benchmarked model (rigid bodies and oscillators,
  elastic rods and beams, continuum FEA, incompressible CFD with dye or smoke, fracture), bakes the result into
  the same scene and proves it with an independent verifier, negative controls, convergence studies and
  side-by-side review. Ships a CLI (`scripts/v2p.py`) for case scaffolding, fal image-to-video generation and
  screening, camera stabilisation and rectification, checkpoints, review pages, side-by-sides, release films and
  a watchdog. Use when the user wants an AI-generated or real clip "with correct physics", a physically accurate
  re-simulation of a filmed or imagined event, a benchmarked simulation rendered in Blender, or an audit of what a
  clip gets physically wrong. Not for stylised or art-directed motion with no physical model, real-time game
  physics, generic Blender modelling or node work, or engineering sign-off of real structures.
license: MIT
metadata:
  hermes:
    tags:
      - physics
      - simulation
      - blender
      - verification
      - video
      - cfd
      - fea
    category: science
---

# Video 2 Physics

Keep the scene, replace the wrong physics with verified physics. The reference (if there is one) fixes what the
viewer sees: geometry, camera, materials, light. The motion comes only from a declared physical model, solved to
convergence, checked against published benchmarks by an independent verifier, and baked into Blender so the
render does exactly what the solver says. What the reference does wrong is measured and reported, never copied.

## Input routing

Every input kind ends in the same case layout (`v2p.py new CASE --input ... --kind ...`). What changes is where
the scene comes from and which review steps apply. Details: `references/inputs.md`.

| Input | `--kind` | Scene comes from | Motion reference | Review against reference |
|---|---|---|---|---|
| AI-generated clip (any generator) | `ai-video` | measured from the clip (camera, geometry, look) | measured; anchor only its physical opening | yes; defects become a divergence map |
| Real footage | `footage` | measured from the footage | measured; physically meaningful (still not ground truth for unmeasured parameters) | yes |
| Still image + description | `image` | measured from the image; description fills the rest | none | look only (frame 0) |
| Idea / text only | `idea` | built in Blender as a first frame; optionally `v2p.py generate` + `v2p.py screen` to get an AI clip from that frame | the generated clip, if made | yes, if a clip was generated |
| No reference | `none` | designed from the brief | none | skipped; physics review still applies |

Then route the phenomenon to a model, solver and benchmark set with `references/physics-routing.md`.

## Workflow

Each phase ends with a deliverable and an acceptance check (`references/workflow.md`). Post labelled visuals
(version | frame/time | status) at every checkpoint.

1. **Scope and contract.** `v2p.py new`. Write `CONTRACT.md` (goal, world frame, units, ownership, interfaces,
   honesty rules) and `config/acceptance.json` (tolerances) **before** any measurement or fit. Declare the anchor
   rule, the free choices and what is assumed. Probe the input (`v2p.py probe`).
2. **Measure.** Camera stability (`v2p.py stabilize`), calibration or the known camera, rectification
   (`v2p.py rectify`, with refraction through flat water or glass), tracks with two independent methods and σ,
   and the list of generator or footage defects. Skip for `idea`/`none` without a clip.
3. **Formulate and set benchmark targets.** Governing equations, boundary and initial conditions, the cheapest
   model the closed forms can validate, every parameter labelled measured / assumed / fitted /
   model-conditioned, and numeric benchmark targets with sources and tolerances.
4. **Solve and converge.** Benchmarks first, then production. Three or more resolution levels with observed
   order; energy, mass or equilibrium budgets; settled windows for developed states. Output follows the data
   contracts (`references/solvers.md`).
5. **Bake into Blender.** Keyframes, PC2 mesh cache or image-sequence textures, camera from json; frozen copy;
   `v2p.py checkpoint CASE ckNNN` makes an immutable checkpoint (`references/blender-integration.md`).
6. **Independent verification and negative controls.** A verifier with no shared code re-derives the benchmarks,
   audits the saved .blend against the solver and shows that its checks catch planted errors
   (`references/verification.md`).
7. **Visual and reproducibility review.** Side-by-side at identical t (`v2p.py sbs`), contact sheets, scene
   logistics, clean rebuild from the checkpoint. Fix rounds produce new checkpoints, never edits to old ones.
8. **Final render, report, review page.** One full-resolution render of the accepted checkpoint, `REPORT.md`,
   `v2p.py page CASE`; optional release film (`v2p.py release`) (`references/review-and-delivery.md`).
9. **Human approval gate.** Present the page and stop. Only the user, in chat, approves a case or starts the next.

## Roles

Six specialists and a lead. Each role writes only in its own directory, reads everything, and keeps a heartbeat in
`<role>/PROGRESS.log`. Briefs: `templates/roles/`; guardrails and budgets: `references/agents-and-guardrails.md`.

| Role | Owns | Delivers |
|---|---|---|
| Lead | `CONTRACT.md`, `config/`, `checkpoints/`, `review/`, `STATUS.json` | decisions (logged in `config/CHANGELOG.md`), checkpoints, final render, report, approval request |
| A Measurement | `measurement/` | camera json, drift, tracks with σ, defect list, anchor-window evidence |
| B Formulation | `physics/` | `FORMULATION.md`, `benchmarks.json`, expected-behaviour plot |
| C Solver | `solver/` | benchmark runs, convergence study, production output per data contract |
| D Scene | `scene/` | deterministic Blender build, bake, scene checks, previews |
| E Independent verifier | `verify/` | own reference solver, audit of each checkpoint, negative controls, PASS/FAIL json |
| F Visual / reproducibility reviewer | `visual_review/` | side-by-side critique, logistics and look findings, clean-rebuild check |

**Parallel** (agents that support sub-agents): wave 1 runs A, B, D and E's reference solver at once; C starts
from B's targets and synthetic data, then takes A's anchors; after the first checkpoint, E, F and a preview render
run in parallel. **Sequential** (one agent): play each role in order A → B → C → D → E → F, keep the directories
separate, and write E's reference solver from the formulation without opening `solver/`. Either way, other agents'
messages are never user approval.

## Hard rules

- **Never imitate generator defects.** Teleporting props, energy gain, time compression, impossible deformation
  and drifting cameras are measured and listed, not reproduced.
- **Label every value** as measured, assumed, fitted or model-conditioned, with its source.
- **Anchor only the physically behaving opening** of a reference, ended by a rule declared before any fit. Name
  the anchored choices (for example the shedding phase, the initial angle) and fit nothing else.
- **Benchmarks before production:** closed forms or published values, with tolerances fixed in
  `acceptance.json`. Tolerances are never relaxed after a failure.
- **Convergence on at least three levels** (mesh, time step, or both) with the observed order reported; check
  force amplitudes and fields, not only frequencies.
- **The independent verifier shares no code** with the solver and ships negative controls its checks must catch.
- **Checkpoints are immutable and built from frozen copies** of solver output, never from the working tree.
- **Visual feedback at every checkpoint:** labelled images or video first, numbers second.
- **Preview in the final engine** for metal, glass and other reflective or refractive materials (Cycles at low
  samples), so the look is judged on what will ship.
- **One full-resolution render per case**, after review fixes. Previews ≤ 960 px wide.
- **Frame 0 of any film is a finished thumbnail** (title over a hero frame), never black or a fade-in.
- **Never claim real-world accuracy** from generated footage. The claim is that the render does exactly what the
  declared model says, verified; the model's inputs are declared.
- **Guardrails:** time budgets per role, heartbeats, bounded waits with a failure check, `v2p.py watchdog`, one
  Blender render at a time (`v2p.py lock -- ...`). Paid jobs: save request ids before polling; never resubmit
  a job whose state is unknown.

## Commands

Run from this skill's directory: `python3 scripts/v2p.py <cmd> ...`. Python 3.10+ with numpy, opencv-python,
Pillow (scipy optional); Blender 5.2+ for `scripts/blender/` (`blender -b --factory-startup --python X -- args`).

| Command | Use |
|---|---|
| `new CASE --input PATH_OR_TEXT [--title T] [--kind ai-video\|footage\|image\|idea\|none]` | scaffold a case: contract, config, role dirs, reference copy + sha256 |
| `probe VIDEO` | fps, frame count, size, duration, sha256 |
| `generate CASE --first-frame PNG --prompt TEXT\|--prompt-file F [--n 3] [--seed S] [--endpoint E] [--duration 10] [--dry-run]` | fal image-to-video candidates from your first frame (`FAL_KEY` from env) |
| `screen CASE [--round R]` | rank candidates: frame-0 diff, camera drift, motion timeline; `screen.json` + sheet |
| `stabilize VIDEO --out DIR` | per-frame homography frame 0 → n on static regions, drift plot + json |
| `rectify VIDEO --camera CAM.json --plane z=H [--refract n=1.333,z=Z] --extent x0,x1,y0,y1 --px MM --out DIR` | top-down rectification of a world plane, optional flat-interface refraction |
| `sbs REF OURS OUT [--label L] [--frames N]` | reference / ours / 50-50 blend at true fps, no retiming |
| `sheet VIDEO OUT [--times 0,1,2...] [--label L]` | labelled contact sheet |
| `checkpoint CASE NAME` | run `config/build.json`, freeze outputs into `checkpoints/NAME` with `MANIFEST.sha256` and `READY` |
| `page CASE` | playable review page `review/index.html` from `review/page.json` |
| `watchdog ROOT [--loop-min 12 --job-min 45 --idle-min 25]` | flag stale waits, long jobs, stuck render lock, idle roles |
| `lock -- CMD...` | run CMD under the global render lock |
| `release CONFIG.json OUT.mp4 [--draft]` | release film from a segment config |

Blender scripts: `render_preview.py`, `bake_pc2.py`, `dye_sheet.py`, `camera_from_json.py`, `check_bake.py`.
Library modules (`scripts/v2p/`): `checks.py` (order, Richardson, GCI, energy, periodicity, negative-control
harness), `elastica.py`, `dye.py`, `ode.py`, plus camera, video and composition helpers.

## Data contracts

- Camera json: `{"image_size":[W,H], "intrinsics":{"fx_px","fy_px","cx_px","cy_px"}, "extrinsics":{"R_world_to_cam":3x3,"t":[3]}}`;
  pinhole, OpenCV convention, pixel centres at +0.5.
- Deformation npz: `positions` (N_samples, rows, cols, 3) in world metres, `t` in seconds; optional per-sample arrays.
- Tracer PNGs: 16-bit RGB, channel k = tracer k concentration 0..1; plane extent, px size and pixel-centre rule in `meta.json`.
- Frame n ↔ t = n / fps. Reference and render are compared at the same t; nothing is retimed.

## Reference routing

Read only what the task needs.

| File | Read when |
|---|---|
| `references/workflow.md` | planning a case: each phase's deliverables and acceptance |
| `references/inputs.md` | deciding how to start from a clip, footage, image, idea or nothing; first-frame generation, screening, prompts, fal costs |
| `references/physics-routing.md` | choosing the model, solver, benchmarks and expected defects for a phenomenon |
| `references/measurement.md` | camera drift, calibration, rectification, refraction, tracking, uncertainty, what not to fit |
| `references/formulation-and-benchmarks.md` | writing `FORMULATION.md` and `benchmarks.json`; labelling; anchor rules |
| `references/solvers.md` | practical solver recipes: elastica, generalised-α, CFD windows and inflow, Lagrangian dye, damping |
| `references/verification.md` | the independent verifier, negative controls, convergence/GCI, budgets, PASS/FAIL |
| `references/blender-integration.md` | baking solver output into Blender; props and transfer logistics; scene checks |
| `references/review-and-delivery.md` | visual review, side-by-side, review page, report, release film |
| `references/agents-and-guardrails.md` | running roles as sub-agents or sequentially; budgets, heartbeats, watchdog, approvals |
| `references/lessons.md` | before starting a case, and when something surprising happens |

Templates (`templates/`): `CONTRACT.md`, `acceptance.json`, `build.json`, `page.json`, `release.json`,
`REPORT.md`, `GUARDRAILS.md`, and paste-in role briefs in `templates/roles/`. Worked cases are in the repository's `examples/`.

## Output expectations

- An immutable checkpoint with the saved .blend, frozen solver output, `MANIFEST.sha256`, and the commands to
  rebuild it; one full-resolution render; a side-by-side at true time (when there is a reference).
- `REPORT.md`: what the reference shows (measured), the corrected model with every value labelled, verification
  numbers against predeclared tolerances, scene additions not in the reference, declared deviations, the
  sensitive assumptions, and what was not checked.
- `review/index.html` from `v2p.py page`; the case stops at the human approval gate.
