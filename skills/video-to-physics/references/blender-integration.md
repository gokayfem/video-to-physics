# Blender Integration

The scene role builds the scene deterministically from parameters and bakes the solver output into it. The render
must reproduce the solver to the audit tolerances: nothing in Blender simulates, eases or retimes the motion.

Run every script as `blender -b --factory-startup --python scripts/blender/<script>.py -- <args>` (arguments
are documented in each script's usage string). Heavy Blender work goes through `v2p.py lock -- <command>` so only one
render or bake runs at a time.

## Contents

- [Scene build](#scene-build)
- [Camera from json](#camera-from-json)
- [Rigid motion: keyframe bakes](#rigid-motion-keyframe-bakes)
- [Deformation: PC2 mesh cache](#deformation-pc2-mesh-cache)
- [Fields: image-sequence tracer textures](#fields-image-sequence-tracer-textures)
- [Props, transfers and logistics](#props-transfers-and-logistics)
- [Look](#look)
- [Checks](#checks)
- [Checkpoints](#checkpoints)
- [Previews](#previews)

## Scene build

- One build script, driven by parameter files (geometry, look, render), that creates the scene from nothing and
  saves the .blend. Two builds must be identical.
- Blender units: 1 unit = 1 m; scene fps = reference fps; frames 0 … N−1; frame n ↔ t = n/fps.
- Publish an **object-name and bake contract** early: object names (with a case prefix), origins, local axes,
  which channels are keyed. The solver output follows it.
- Measured proportions flow to the solver immediately (hinge offsets, radii, tip bodies); placeholder dimensions
  cause intersections in the first bake.
- Build checkpoints from the checkpoint's own frozen copy of solver output, never from an absolute path into the
  working tree.
- Store provenance on the objects: custom properties with the solver output's sha256 and version.

## Camera from json

`scripts/blender/camera_from_json.py` creates the camera of record from a camera json (data contract: pinhole,
OpenCV convention, pixel centres at +0.5). For a moving camera, key focal length and shift per frame (zoom) or
location and rotation (orbit), from the smoothed per-frame solution. Set render resolution to the json's
`image_size`. Lock diagnostic cameras (location, rotation, scale) so they cannot be nudged.

## Rigid motion: keyframe bakes

- Key location and rotation on every frame from the solver samples, with Bézier handles from the velocities
  (cubic Hermite: handle offset = velocity × Δt/3). This reproduces the trajectory between frames.
- Rotational DOFs faster than a few rad/s need subframe keys (for example every 1/(4 fps)) to keep the evaluated
  angle within tolerance between frames.
- Use quaternion rotation mode for 3-D orientations; keep Euler only for single-axis hinges.
- Constraints (pins, hinges) are geometry, not Blender constraints that recompute motion.

## Deformation: PC2 mesh cache

`scripts/blender/bake_pc2.py` converts a deformation npz (`positions` (N_samples, rows, cols, 3) in world metres,
`t` in seconds) into a PC2 file and adds a Mesh Cache modifier to the target mesh.

- The target mesh's vertex order must match the npz grid (rows × cols); build it from the same grid definition.
- Positions are world coordinates; give the object an identity transform or convert in the bake script, never both.
- Sample the npz at frames and subframes; the Mesh Cache modifier interpolates linearly between samples, so the
  sample spacing sets the in-between error.
- Thickness (a strip's 0.5 mm) comes from a Solidify modifier after the cache, or from offset rows in the npz;
  state which.

## Fields: image-sequence tracer textures

`scripts/blender/dye_sheet.py` puts the tracer PNG sequence (16-bit RGB, channel k = tracer k) on a plane with a
world → UV map from `meta.json` (extent, px size, pixel-centre rule, row and column orientation).

- The plane sits at the dye height inside the medium; the look maps each channel to a dye colour by absorption
  (Beer–Lambert-like), not by emission, so overlapping dyes darken as real dye does.
- Image-sequence frame offset: frame n uses `f{n:04d}.png`; check frame 0 and the last frame explicitly.
- Refraction: a flat water surface above the sheet is rendered by the engine; the measurement role's Snell model
  must agree with what the render shows (validate on a grid).
- Occluders (bodies, needles, walls) are real geometry above the sheet.

## Props, transfers and logistics

- Every prop that moves has a visible cause: a hand, an arm, a feeder. Props never appear, vanish or teleport.
- Design the transfer against the load schedule **before** building the mechanism: required peak speed
  ≈ 2 × path length / (cycle time − seat time). With a tight schedule, move the supply to the load point (a feeder
  at the load point instead of a carry from a distant stack).
- Keep carried props at ≤ about 1 m/s; empty returns may be faster but should stay out of frame where possible.
- Stacks shrink as items leave; counts in the scene match the physics load at every time.
- Seat each prop at the physics load point one sample before its scheduled time; the contact ramp is part of the
  solver's load.
- Additions not in the reference (pegs, feeders, arms) are listed in the report as ours.

## Look

- Match the scene of the reference (or our first frame) by region metrics and 2-D luma profiles of large surfaces
  (backdrops, tables), not by a few region means: an upside-down gradient can pass a region metric.
- Fix the backdrop and environment first, then judge metal and glass, which reflect them.
- Light-link strip lights to metal objects to place highlights where the reference has them; disable light probes
  that paint the table onto metal.
- Measure motion blur per case and set the shutter in the render config.
- Post-processing grain is seeded per frame; a frozen grain pattern is visible.
- A look fix re-measures the full profile (both tails, fine and coarse structure), not only the reported metric,
  to avoid over-correction.

## Checks

`scripts/blender/check_bake.py` reopens the saved .blend and evaluates it.

| Check | Typical tolerance |
|---|---|
| evaluated transforms or mesh vs solver at frames + 4 subframes | < 1e-5 m / rad (meshes often 1e-8 m) |
| fixed anchors and clamps | < 1e-6 m |
| declared lengths and attachments | < 1e-6 m (1e-5 m for visual attachments) |
| intersections and clearances | 0 intersections; clearance above declared bound |
| camera vs camera json | reprojection of known points < 0.1 px |
| image-sequence frame mapping | frame n shows tracer frame n |
| logistics | prop speeds, counts, continuity, no visibility toggles |

## Checkpoints

`config/build.json` lists the commands that build the checkpoint and the outputs to keep. `v2p.py checkpoint CASE
ckNNN` runs them, copies declared outputs into `checkpoints/ckNNN`, writes `MANIFEST.sha256` and `READY`. A
checkpoint is never edited; a fix produces ckNNN+1. Include the code needed for a clean rebuild (scene, solver
output, config, verify inputs); exclude scratch, caches and `.blend1` files.

## Previews

- `scripts/blender/render_preview.py` renders frames or crops at ≤ 960 px wide.
- Metal, glass, water and other reflective or refractive materials: preview in the final engine (Cycles at
  ~24 samples with denoising). Fast rasterised engines can render brushed metal nearly black without reflection
  probes and mislead the look review.
- Render only the frames a review needs (three frames, a strip of crops); the full-resolution sequence is rendered
  once, by the lead, at the end.

## Blender gotchas (verified)

- **Mesh Cache timing.** Blender samples the cache at `scene_frame * frame_scale - frame_start`, so `frame_start`
  is in *sample* units. With `spf` samples per frame and the first frame `f0`, set `frame_start = f0 * spf`
  (`scripts/blender/bake_pc2.py` does this).
- **Image sequences in background mode.** `ImageUser.frame_current` is not updated when Blender runs with `-b`, so
  a check that sets the frame and reads it back cannot fail. Read the frame from the depsgraph-evaluated material
  instead (`scripts/blender/dye_sheet.py` does this).
- **Rod discretisation.** Put the unknown tangent angles at nodes `s_i = i·h` with a rigid half-link at the clamp
  and a half link at the tip. Link-midpoint angles are only first-order accurate for eccentric tip loads.
