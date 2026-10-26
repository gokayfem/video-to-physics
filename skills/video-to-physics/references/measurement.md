# Measurement

Measurement turns the reference into a camera of record, tracks with uncertainty, and a list of defects. It does
not produce physics: it produces the few anchors the contract allows and the evidence for the divergence map.

## Contents

- [Order of work](#order-of-work)
- [Camera stability](#camera-stability)
- [Known camera vs calibrated camera](#known-camera-vs-calibrated-camera)
- [Rectification](#rectification)
- [Refraction through water and glass](#refraction-through-water-and-glass)
- [Tracking](#tracking)
- [Uncertainty](#uncertainty)
- [Motion blur and cadence](#motion-blur-and-cadence)
- [What not to fit](#what-not-to-fit)
- [Deliverables](#deliverables)

## Order of work

1. `v2p.py probe` the reference: fps, frame count, size, sha256. Frame n is at t = n / fps.
2. Camera stability (always, even for a "locked" camera).
3. Camera of record: known (our first frame) or calibrated.
4. Rectify planes of motion when the observable lives on a plane.
5. Track only the features the fit and the divergence map need.
6. Write the defect list and the anchor-window evidence.

## Camera stability

- `v2p.py stabilize VIDEO --out DIR` estimates a per-frame homography from frame 0 to frame n on static regions
  and writes residuals and a drift plot. Mask out everything that moves (bodies, fluid, shadows of moving parts).
- Generated clips nearly always drift: zoom of a few percent, small pans, sometimes a full orbit. Real tripods
  drift less but still settle and breathe.
- Model zoom as per-frame focal length and principal point unless parallax proves a dolly or orbit.
- When the homography residual on static regions is large (tens of px), the camera moved in 3-D: solve per-frame
  pose from vanishing points or a small bundle on persistent edges, then **smooth the pose** (penalised spline)
  so the static scene does not shake in the render. Add a static-scene jerk metric (render vs reference).
- Never deform geometry or move scene objects to explain camera motion.

## Known camera vs calibrated camera

| Situation | Camera of record | Uncertainty |
|---|---|---|
| `idea`/`none` with our first frame | the Blender camera, exported as camera json | exact (check: reproject known points, ≤ 0.1 px) |
| generated from our first frame | our camera at frame 0 + measured drift per frame | drift fit residual |
| footage with a target | calibration from the target, lens distortion corrected | target reprojection RMS |
| AI clip or footage without a target | rectilinear static edges, vanishing points, known proportions | focal length typically ±3 %; scale model-conditioned |

The camera json follows the data contract: pinhole, OpenCV convention, pixel centres at +0.5, `R_world_to_cam`
and `t`. `scripts/blender/camera_from_json.py` creates the Blender camera from it (per-frame keys if the camera
moves). Check: static-landmark reprojection RMS ≤ 2 px (≤ 0.1 px for our own render).

## Rectification

`v2p.py rectify VIDEO --camera CAM.json --plane z=H --extent x0,x1,y0,y1 --px MM --out DIR` produces a top-down
image sequence of a world plane at metric pixel size. Use it to measure positions, speeds, wavelengths and dye
fronts in metres instead of pixels. Choose `--px` near the camera's footprint per pixel on that plane; finer adds
nothing.

## Refraction through water and glass

Anything seen through a flat liquid surface or a glass plate is displaced. A pure pinhole model can be off by
several pixels near the edges of a channel.

- Model a flat interface with Snell's law: `--refract n=1.333,z=Z` (water surface at height Z).
- Validate on a known pattern under the interface (a printed grid in our first frame): residual should drop to a
  fraction of a pixel.
- Thick glass, curved surfaces or nested media (acrylic under water) need their own model or a declared error.
  Renderers may not support nested indices of refraction; check what the look model actually does.

## Tracking

- Use two independent methods per feature (for example template matching and edge/centroid fitting) and report the
  per-frame difference as part of σ.
- Track the most precise feature available (a ring, an edge, a silhouette), not the visual centre of a body.
- **Check attachment** before fitting: a feature must translate and rotate with its body. Generators draw marks
  that do not rotate about the body centre; use the silhouette in that case.
- Specular, brushed or flickering textures cannot be tracked reliably; use silhouettes and edges.
- Thin parts (wires, hooks, pins) are drawn thicker and bent inconsistently by generators; measure their width
  at full resolution and keep the physical hinge where mechanics puts it.
- Fluids: measure fronts, vortex positions and periods on rectified frames; per-band intensity timelines give the
  shedding frequency.

## Uncertainty

- σ per frame from the two methods; bootstrap the fit (≤ 20 resamples while iterating, more only for the final
  parameters).
- Report the reprojection RMS of static landmarks separately from the tracking σ.
- State focal-length and scale uncertainty and how it propagates into model-conditioned values.

## Motion blur and cadence

- Measure blur per clip: edge width vs speed gives the effective shutter. Do not carry a shutter value from
  another case. Put it in the render config as a camera property.
- Generators often have a cadence defect (every k-th frame step larger). Detect it from frame-to-frame step
  statistics by n mod k; disclose it; never retime the simulation; report supplementary metrics on the clean
  phase if useful.

## What not to fit

| Do not fit | Because | Instead |
|---|---|---|
| anything after the anchor window | generator drift is not physics | divergence map |
| damping law and value, unless identifiable in the window and plausible | amplitude growth and noise dominate | declared assumption with a range; sensitivity run |
| mass, density, absolute material constants | not observable from kinematics | declared values with sources |
| absolute scale without a metric reference | only ratios are observable | model-conditioned from a period and assumed g, labelled |
| per-frame forces, time scaling, hidden drives | they encode the defect into the model | never |
| a mode the clip does not show | its stiffness is unidentifiable | assume with a range; start it at rest |
| generator-exaggerated secondary motion | defect | report the ratio to the physical value |

## Deliverables

- camera json (and per-frame camera if it moves), drift plot, reprojection RMS
- tracks with σ (csv) and plots
- `summary.json`: events with times, defects with frame ranges and numbers, anchor-window evidence evaluated with
  the predeclared rule
- labelled visuals: contact sheet with overlays (`v2p.py sheet`), drift, tracks
