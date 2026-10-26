# Inputs

Video 2 Physics accepts five kinds of input. They differ in where the scene comes from and how much of the
reference can be trusted. The physics never comes from the reference; at most a few declared initial choices do.

## Contents

- [Choosing the kind](#choosing-the-kind)
- [AI-generated clip](#ai-generated-clip)
- [Real footage](#real-footage)
- [Still image + description](#still-image--description)
- [Idea only: first frame in Blender, then optional generation](#idea-only-first-frame-in-blender-then-optional-generation)
- [No reference](#no-reference)
- [Writing prompts for physical clips](#writing-prompts-for-physical-clips)
- [Screening candidates](#screening-candidates)
- [Costs and safety for paid generation](#costs-and-safety-for-paid-generation)

## Choosing the kind

| Kind | Trust in geometry | Trust in motion | Scale | Main extra work |
|---|---|---|---|---|
| `ai-video` | visual only | opening only, up to a declared rule | model-conditioned unless a metric object is visible | camera drift, defect map |
| `footage` | high after calibration | physical, but parameters still unidentified | from a known object or calibration target | calibration, lens distortion, noise |
| `image` | frame 0 only | none | from known objects or assumed | describe the event; design the forcing |
| `idea` | exact (we built it) | from our generated clip, if any | exact (we chose it) | first-frame build, prompt, screening |
| `none` | exact | none | exact | none of the reference steps |

When the user can choose, prefer `idea` with a Blender first frame: geometry, camera and scale at frame 0 are
known exactly, measurement shrinks to the motion only, and the look already matches.

## AI-generated clip

- Treat the clip as an accepted **visual** reference, not a measurement. It sets the scene; its motion is
  evidence of intent at best.
- Expect: a "locked" camera that zooms, pans or orbits; energy that grows or drifts; a cadence defect (every k-th
  frame step larger); thin parts drawn thicker and inconsistently; props that appear without a cause;
  time compression; exaggerated secondary motion; textures that flicker.
- Fit only inside the anchor window (`formulation-and-benchmarks.md`). The rest becomes a divergence map.
- A motion that looks wrong may be physical (an elastic-pendulum sideways swing, the middle of a slow torsion
  cycle). Test the general physical system before calling a motion non-physical.
- Report: "matches the scene of the reference; physics replaced". Never claim real-world accuracy.

## Real footage

- Calibrate: a checkerboard or known object, or rectilinear static edges; correct lens distortion before
  tracking. Record the camera json in the data-contract format.
- Motion is physical, so the anchor window can be longer, but unmeasured parameters (damping law, material
  constants, mass) remain unidentifiable unless the footage constrains them. Say which.
- Rolling shutter and compression smear fast edges; measure motion blur per clip.
- Prefer footage with a metric reference in the plane of motion.

## Still image + description

- Measure what frame 0 gives: camera (vanishing points or known objects), geometry, materials.
- The description supplies the event. Turn it into a forcing or loading schedule in the contract (who moves what,
  when, at what speed) and check it is mechanically feasible before solving.
- Review against the reference is frame-0 look only.

## Idea only: first frame in Blender, then optional generation

1. **Design for measurability.** Choose a scene where a published benchmark applies and the observable is visible:
   a countable load, a grid or scale in the plane of motion, a locked camera square to the motion, matte
   surfaces where tracking matters, contrast between moving parts and background.
2. **Design the physics numbers first**, then the prompt: dimensions, material, speed, expected time scales, how
   many cycles or events fit in the clip (aim for ≥ 2–3 periods of anything periodic).
3. **Build and render frame 0** in Blender with a camera you record in the camera-json format. Keep the build
   script; it becomes `scene/`. Render frame 0 in the final engine.
4. **Optional: generate a clip** from that frame with `v2p.py generate CASE --first-frame frame0.png
   --prompt-file prompt.txt --n 3`. Use `--dry-run` first to see the request and the estimated cost.
   Give only the first frame; an end frame or keyframes would hide what the generator does on its own.
5. **Screen** with `v2p.py screen CASE` and pick a candidate (below). Approval of the chosen reference comes from
   the user.
6. Measure only the generated motion: where it departs from physics. Frame 0 remains our render; report the
   frame-0 difference between the generated clip and our frame (the generator re-grades slightly).

A look model used only to make frame 0 (for example potential-flow streamlines for dye) can show something the
solved physics only develops later. Declare that deviation; do not fake it in the solved result.

## No reference

Design the scene and the forcing from the brief, write the contract and acceptance, and run phases 2–8. Skip
reference fidelity and the side-by-side; the visual review checks plausibility, logistics and look on its own.

## Writing prompts for physical clips

- Start with "The same scene, one continuous real-time shot from the locked-off <view> camera."
- Describe the event as a sequence with rough timing ("one every second", "slowly", "steadily"), matching the
  physics design numbers.
- State what must not move: "The vise and workbench never move."
- End with exclusions: "No cuts, no camera movement, no slow motion, no text."
- Name causes for every motion if you want them shown (a hand, an arm, a feeder); generators otherwise make props
  appear mid-air.
- Do not put physical claims the generator cannot honour in the prompt (exact loads, frequencies). Those belong to
  the solver.
- Generate several seeds (3 is usually enough); record prompt, seeds and endpoint in the case. There is no default
  endpoint: pass `--endpoint` or set `generation.endpoint` in `config/case.json`; extra model inputs go in
  repeatable `--param K=V`.

## Screening candidates

`v2p.py screen CASE [--round R]` writes `screen.json` and a contact sheet per candidate.

| Metric | Prefer | Reject when |
|---|---|---|
| frame-0 difference vs our first frame | small (re-grade only) | layout changed or objects moved at frame 0 |
| camera drift (ECC on the static mask) | ≤ a few px over the clip | tens of px or more (camera moved) |
| motion timeline | the event happens inside the clip and is visible | nothing moves, or motion is everywhere (re-stylised) |
| contact sheet | event readable, parts intact | parts merge, vanish or change shape |

Pick the least drift with a clear event. If every candidate fails, revise the prompt (not the physics) and run a
new round (`--round 2`).

## Costs and safety for paid generation

- `FAL_KEY` comes from the environment and is never printed, logged or written to the case.
- `--dry-run` prints the request and an estimated cost. State the estimate to the user and get approval before the
  first paid run; rates change, so verify them on the model page.
- The tool saves each request id **before** polling. If polling is interrupted, resume from the saved id; never
  resubmit a job whose state is unknown (it may already be billed and running).
- Polling is bounded; on timeout the job is reported as uncertain with its id, not retried.
- Screening at a lower resolution, then one full-resolution generation of the chosen candidate, is cheaper when
  the endpoint prices by resolution.
- Keep generated clips and their request metadata in the case; they are the reference's provenance.
