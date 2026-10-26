# Lessons

Generalised lessons from completed reconstructions (rigid pendulums, an elastic pendulum, a torsion disc, a
buckling strip, a cylinder wake). Each is a rule with its reason. Read before starting a case; append new ones to
the case's `config/CHANGELOG.md` and promote the reusable ones here.

## Contents

- [What references do](#what-references-do)
- [Modelling](#modelling)
- [Measurement](#measurement)
- [Solving and convergence](#solving-and-convergence)
- [Scene, bake and logistics](#scene-bake-and-logistics)
- [Look and review](#look-and-review)
- [Verification](#verification)
- [Process and speed](#process-and-speed)

## What references do

- A "locked" generated camera nearly always moves: zoom of a few percent, pans, sometimes an orbit. Measure it on
  every clip; never bend geometry to explain it.
- Generated motion does not conserve energy: amplitudes grow, periods wander, sometimes gradually. Anchor only the
  opening with a rule declared before fitting, with both a jump and a cumulative-drift criterion.
- Generators have cadence defects (every k-th frame step larger). Disclose; never retime.
- Generated clips can be time-compressed several-fold (fluids especially). Measure front and convection speeds
  against what the flow can carry.
- Props appear mid-air, counts jump, supplies never shrink. These are defects to list, not schedules to copy.
- Generators exaggerate secondary motion and draw thin parts thicker and inconsistently.
- A motion that looks wrong can be physical (an elastic-pendulum swing; the middle of a slow torsion cycle). Fit
  the general physical system before calling it a defect.
- A mode the prompt asked for may be absent; its parameters are then unidentifiable. Assume with a range and start
  it at rest.
- Our own first frame can contain a non-physical look detail (a look model that wraps dye around a body instantly).
  Declare it as a frame-0 deviation; do not fake it in the solved result.

## Modelling

- Pick the cheapest model the closed forms can validate. A discrete elastica beat a shell model for a thin strip.
- Model every load-carrying part: tip bodies, eccentric offsets, self-weight. The idealised formula can miss the
  onset by a whole load step.
- Decide physical hinges by mechanical argument, not by the apparent rotation centre in the image.
- Static asymmetries can be a physical tilt of the scene relative to gravity (an unlevel table) when a camera roll
  would break the static fit.
- Mass, damping and absolute scale are usually not identifiable from kinematics. Label them assumed or
  model-conditioned.
- Name the sensitive assumption up front: damping often sets the sway amplitude and end pose after an event, while
  the event itself (onset load, shape, frequency) does not depend on it.
- Developed states (a wake that has been running) are the physical start; an impulsive start can need far longer
  than the clip.
- Declare the material assumption that keeps the model valid (elastic limit for large twists or bends) and check
  the computed stress against it.
- Decide velocity definitions (where U is measured) in the contract; they determine inflow calibration.

## Measurement

- Track only what the fit and the divergence map need; extra methods and features cost the most time.
- Use the most precise feature (rings, edges, silhouettes), and check it is rigidly attached to the body.
- Specular and flickering textures cannot be tracked.
- When the static-mask homography fails, the camera moved in 3-D: solve pose per frame, then smooth it, and check
  a static-scene jerk metric.
- Measure motion blur per clip; do not carry a shutter value over.
- Refraction through a flat water surface shifts a printed grid by several pixels; a flat-interface Snell model
  fixes it.
- Bootstrap with ≤ 20 resamples while iterating.

## Solving and convergence

- Coarse levels can look converged in frequency while force amplitudes are 10–20 % off. Check amplitudes and
  fields on three levels; cross-check with the independent reference.
- Settle developed flows per band (near, mid, far, outlet); a restart sends a kinetic-energy pulse downstream one
  band per period.
- Calibrate inflow per solver with an empty-channel run; correlation-based values were off enough to move St and
  drag beyond tolerance.
- Eulerian dye smears by millimetres at practical resolution. Lagrangian material lines drawn as mass-conserving
  filaments are sharp, cheap and keep the mass budget checkable.
- Integrate orientations with quaternions or exact projection, with guards on constraint drift and non-finite
  state; a solver that passes nominal checks can still blow up in a parameter variant.
- Control-response variants start from their own re-equilibrated state.
- Run the parameter variants as a self-check early, not first at audit time.

## Scene, bake and logistics

- Agree one npz key contract (and object-name/bake table) between solver and scene at kickoff.
- Bake rotations faster than a few rad/s with subframe keys.
- Design prop transfers against the load schedule before building the mechanism: peak speed ≈ 2 × path /
  (cycle − seat time). A feeder at the load point turned a 14 m/s teleport into a 0.7 m/s carry.
- Give the verifier an explicit logistics item; physics-only audits miss teleporting props.
- Build checkpoints from the checkpoint's own frozen copy of solver data; an absolute path into the working tree
  breaks provenance.
- Checkpoints include the code needed for a clean rebuild and exclude scratch.
- Measured proportions flow to dynamics immediately; placeholder dimensions caused intersections.

## Look and review

- Preview in the final engine for metals and glass; a rasterised preview rendered brushed steel and brass
  near-black.
- Region-mean look metrics miss an upside-down backdrop gradient; compare 2-D luma maps and profiles with the
  camera of record.
- Fix the backdrop first, then metal; light-link highlights to metal objects; disable probes that paint the table
  onto metal.
- Each look fix re-measures the full profile, not only the reported metric, to avoid over-correction.
- Seed film grain per frame.
- Run the reviewer's metrics inside look rounds so the first review finds less.
- Don't re-render at full resolution after every fix. One full-resolution render, at the end.

## Verification

- An independent reference with a different method catches what self-checks miss (under-resolved forces).
- Negative controls prove the checks can fail.
- Re-checks use the verifier's exact recorded arguments; ship a recheck script. Rounded arguments produce false
  FAILs.
- Verify manifests from inside the checkpoint directory.
- Motion-identical checkpoints need only saved-file checks plus a dump comparison.

## Process and speed

- The largest single loss was an unbounded wait on a marker from a command that had already exited. Every wait
  has a deadline and a failure check; the lead runs the watchdog.
- `pgrep -f` on a string that appears in the waiting command matches itself.
- End shell waits with `exit 0`.
- Decide modelling questions within one message; log them.
- Keep a parameter snapshot for every round shown to the user.
- One case at a time when the user wants to approve each; parallelise within a case.
- Typical timings: measurement 15–40 min, formulation 6–30 min, solver 20–60 min plus fix rounds, scene 12–45 min,
  verifier 30–50 min, final Cycles render at 1920 px ≈ 10 s per frame at ~48 samples.
