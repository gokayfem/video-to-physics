# Camera role — NOTES (clip 01, brass pendulum)

Status: **camera calibration delivered (draft, isolated component)**. Appearance work was moved to the Look
specialist (CONTRACT v3), so this role does not own `scene/look.py` or `params/look.json`. My early look draft is
parked in `camera/handover/look_draft_from_camera_role.py` for the Look specialist to reuse or discard.
Measured reference region colours stay in `camera/out/ref_regions.json` (overlay: `camera/out/ref_regions_overlay.jpg`).

## Update v003 (gauge)
`params/camera.json` was re-written with L_eff_m = 0.43852487 m (physics v004 tilt fit). The camera centre is now (0.8764, −1.3473, −0.3057) m and the gauge factor is 1.0. Pixel results are unchanged (frame 0 / 240: 0.46 / 0.57 px RMS), and the Blender projection check passes again. New visuals: `v003_*`.

## Update v002 (gauge)
`params/camera.json` was re-written with L_eff_m = 0.4391130415 m from physics.json. All pixel-space results are unchanged: static RMS 0.27 px; reference-matched camera 0.46/0.41/0.31/0.36/0.57 px at frames 0/60/120/180/240. The Blender projection check still passes (max 0.0019 px). The builder gauge factor is 1.0.
New visuals: `v002_drift_ref_vs_locked_f000_f240.jpg` and `v002_reprojection_f120_full.jpg` / `_zooms.jpg`. They are regenerated with the v002 camera and are pixel-identical in content to v001, as expected.

## Visuals (camera/visuals/, v001, isolated component, draft)
- `v001_reprojection_f120_full.jpg`: frame 120. Cyan = calibrated model silhouette edges, red = measured
  sub-pixel edge points, yellow = residual ×20. Also shows the pivot P_ring (+), P_eff (×), and all 241 bob
  centres mapped to frame 120 (red = measured, cyan = fitted circle model). The static RMS is 0.27 px.
- `v001_reprojection_f120_zooms.jpg`: zooms on the post top and arm, the base corners and the bob arc.
- `v001_drift_ref_vs_locked_f000_f240.jpg`: frames 0 and 240. The reference-matched camera gives 0.46 and 0.57 px RMS.
  The locked camera gives 7.9 and 6.8 px RMS (max 16 px), so the drift is real.
- `v001_drift_similarity_plot.png`: per-frame image scale, translation and residual relative to frame 120.
- `camera/out/landmarks.jpg` is the same as v001_reprojection_f120_full (landmark/edge picks overlay).

## Method
1. **Static landmarks.** I used straight silhouette edges instead of corners, because every edge is rounded.
   `edges.py` takes approximate segments picked by eye on gridded zoom crops. On frame 120 it samples
   sub-pixel gradient maxima across each edge (grey + wood chroma R−B) and fits a robust line.
   There are 15 edges and 969 points: base (left-top, back-top, front-left vertical, front-bottom, back-right vertical),
   post (left, right, top-front, top-right, the two junctions with the base), and arm (top, bottom silhouette, front-bottom,
   end-right). I dropped internal or highlight edges on the base, whose line fits were 2–5 px RMS.
2. **Model** (`calibrate.py`). Three boxes in the stand frame (origin P_ring, Z up, X along the arm): base, post and arm.
   Fillet radii are free. The residual is each point's distance to the image line of the plane through the
   camera centre tangent to the rounded edge (a sharp edge when r=0).
   Camera: pinhole, square pixels, principal point fixed at the centre. A free principal point is degenerate:
   it runs off to u=2057.
   Other observations:
   - P_ring = (1089.5, 127.0) px in frame 120, the wire departure under the loop.
   - P_eff = (1090.0, 180.4) px, the least-squares intersection of 241 straight lower-wire lines (`wire_pivot.py`).
   - The measurement role's bob track (`measurement/bob_track.csv`, mapped to frame 120), used for centres and
     image radius. The bob sits on a circle of radius 1 (gauge L_eff = 1) about P_eff in a vertical plane at
     azimuth φ, with θ_n free per frame.
   Soft priors: post centred on the arm in Y; post and arm cross-sections square (±15 %); fillet ≤ 0.04 L.
3. **Drift** (`track_static.py`, `drift_fit.py`). 59 KLT features on the stand are tracked from frame 120 into every
   frame, and a similarity is fitted per frame. The scale goes from 0.98321 to 1.01565, i.e. +3.30 % over the clip,
   nearly linear. The pan is tx 16.5 → −15.2 px and ty 2.8 → −2.0 px; roll is < 5e-4 rad.
   The similarity fit has 0.3–0.8 px RMS. Affine and homography fits are no better, and there is no depth-dependent
   residual between post/arm and base. I therefore represent the drift as a **zoom**: per-frame focal length s·f
   and principal point s·R·c + t, with the camera centre fixed. A dolly cannot be excluded rigorously, but the
   evidence favours the zoom. The measurement role found the same independently (scale 1.0327).
   The raw per-frame values and cubic-in-t smooth curves are in `camera/drift.csv`; the smooth fit is within
   `smooth_vs_raw_maxdev_px` of the raw.
4. **Projection** (`project.py`, pure numpy). It provides `load_camera(frame=n | kind="locked")`, `project`, `ray`,
   `intersect_plane`, `depth` and `blender_camera_settings`. The Blender check
   (`test/validate_projection_blender.py`, Blender 5.2.2) gives a max error of **0.0019 px** for a synthetic camera and
   for the reference cameras at frames 0/120/240 plus the locked camera (`out/projection_validation.json`).

## Results (params/camera.json)
- f = **2431 px** (45.58 mm on a 36 mm sensor; vertical FOV 25.23°, horizontal FOV 43.1°). The principal point is
  at the centre. The spread over 28 variants has sd 46 px, range 2379–2645 (`out/calib_variants.json`).
  The cost profile in f (`out/f_profile.json`) is shallow. The static RMS is 0.27 px at 2431 px, 0.34 px at 2250/2600
  and 0.45 px at 2749. The bob data slightly favour a longer f.
  **Realistic f range: about 2350–2750 px.** Distance scales with f (dist/f ≈ constant).
- Camera centre in L_eff units is (2.00, −3.07, −0.70). Distance to the pivot is 3.73 L (sd 0.07, range 3.65–4.05).
  In metres with **L_eff_m = 0.43911 m** (physics.json `L_eff_m`, model-conditioned, sd 0.0049 m; the gauge decision is in CHANGELOG 2026-10-25T22:47:40Z): (0.878, −1.349, −0.306), distance 1.638 m. The earlier v001 metres used the placeholder L_c = 0.3867 m and are superseded.
  View pitch is +1.0° (looking very slightly up). The view azimuth is −36.0° from +Y (the camera sits on the +X,−Y side).
  Roll is 0.23°.
- **Reprojection (R4 ≤ 2 px).** Static edges: 0.27 px RMS at frame 120 (969 points).
  With the reference-matched per-frame camera: frames 0/60/120/180/240 give 0.46/0.41/0.31/0.36/0.57 px.
  The locked camera gives 7.9/4.0/0.29/3.5/6.8 px.
  Points within 8 px of the frame border are excluded, because the base bottom edge leaves the frame during the push-in.
- **Swing plane.** The free fit gives φ = +4.3° (sd 1.1°, range −1.3…+5.7°) from the arm axis toward +Y.
  **Adopted φ = 0**: the swing plane is the vertical plane through the arm. This changes the bob RMS only from 0.66 to 0.75 px.
  The swing plane makes **36.0°** with the image plane, and the bob moves toward the camera on the image-right side.
  The bob radius is +5 % at the right turning point, and the fit reproduces it (0.51 px radius RMS).
  Consequence: the true angular amplitude is about **±11°**, not the ≈6° seen by eye in the image.
  θ is in the range −10.9…+11.6° over the clip, and the amplitude grows (AI clip).
- **Pivot.** The wire's straight lower part (and the bob arc) rotates about P_eff, which is 0.081 L_eff below the wire
  departure point P_ring (sd 0.005; 0.10 if the P_eff weight is loosened). Near the pin loop the wire visibly bends
  (`suspension.csv` / measurement summary agree: lower intersection v≈171–180 vs loop v≈125).
- Bob radius from the image: 0.0867 L_eff = 0.0381 m (with L_eff_m = 0.43911 m).
- The implied stand dimensions (in L_eff and metres) are in `params/camera.json` under
  `stand_geometry_implied_by_calibration`. Base: x −0.927…0.497, y −0.372…0.362, z −1.406…−1.209; post top 0.164; arm 0.134 × 0.134.

## Requests / open issues
1. **To the lead / dynamics: pivot definition.** CONTRACT says the origin P is where the wire leaves the loop
   (P_ring), but the observed rotation centre is P_eff = (0, 0, −0.081 L). A model with the bob on
   P_ring + L_c(sinθ, 0, −cosθ) will mismatch the arc. Either put the origin at P_eff (with the pin loop 0.081 L
   above it), or give the fit a free pivot offset. camera.json gives both points; the metres scale assumes L_eff = L_c.
2. **To geometry.** Their joint fit (`scene/measure/fit_joint.json`) has f = 2749 px at frame 120. Mine is 2431 px,
   with a similar camera position (2.17, −3.25, −0.59 L) and the same ~36° yaw. 2749 px fits my edges at 0.45 px RMS
   (still within R4), so f is weakly determined. The base dimensions differ by about 10 % between the two fits
   (W 1.30 vs 1.42 L). I suggest the verifier arbitrates, using camera.json + project.py as the single camera.
   I have not edited geometry.
3. φ = 0 is an assumption. Data prefer +4 ± 1°, but that sits within systematic AI-clip inconsistencies.
4. No lens distortion is modelled. The principal point is assumed at the centre. Projected-sphere centre offset
   (≈0.2 px) is ignored.
5. When `params/physics.json` L_eff_m changes, re-run `camera/write_params.py`. It reads `L_eff_m` (falls back to L_c only for legacy files). Only metres change; L units and pixels do not. Builder gauge factor = physics L_eff_m / camera.json L_eff_m = 1.0 as of v002.

## Commands (from ROOT)
```
.venv/bin/python camera/extract.py          # median plate + grey stack (500 MB, deleted after use; regenerate)
.venv/bin/python camera/edges.py camera/frames/f_120.png f120   # needs frames/f_120.png (extract via OpenCV, frame 120)
.venv/bin/python camera/track_static.py && .venv/bin/python camera/drift_fit.py
.venv/bin/python camera/bob_quick.py && .venv/bin/python camera/wire_pivot.py && .venv/bin/python camera/bob_to_f120.py
.venv/bin/python camera/calibrate.py --phi0
.venv/bin/python camera/calib_uncertainty.py; .venv/bin/python camera/f_profile.py
.venv/bin/python camera/write_params.py      # -> params/camera.json, camera/drift.csv
blender --background --factory-startup --python camera/test/validate_projection_blender.py
.venv/bin/python camera/reproject_check.py   # -> camera/visuals/v001_*, out/reprojection_by_frame.json
.venv/bin/python camera/look_measure.py      # reference region colours (for the Look specialist)
```
Auxiliary/early scripts: `drift_check.py`, `drift_lines.py` (superseded by the KLT similarity fit), `crop.py` (zoom-grid helper).

## sha256
- `params/camera.json` b8f51e1de285563db20dbfc056ee6bac6bce6e46efa565a2f780df549d223bd3
- `camera/drift.csv` fa398d6fcfebaa42a86dca174fdb39053b005431dd0982cbe49e09a460f9eca5
- `camera/project.py` 2d65216b1730e31937d2d23be49ee165b8a49375cd2d094d603bc0ffb636545e
- `camera/calibrate.py` 2595e06727e5edbc41f53ceb67a43a29570d212b392d55a46784746a0674458e
- `camera/edges.py` 55aec6f3b4f9468c4d7a3e3f79ef300fd60fa455d11eed6c779308b224b38d60
- `camera/write_params.py` b50dc3946089b8cec3f03baedf6978231635f00d4acd23262a0f9d1647185e97
- `camera/drift_fit.py` 388f3fb4d5375dcdd95ddaf6b78759d3aa74fb517bbe1034b64cb76da4cc9d3a
- `camera/track_static.py` 4c8e51500636f49fdab8713804ca58cc418e2502e3b7b04028ed115c2356e390
- `camera/wire_pivot.py` f46c440a6fadaec2dfff1d6e03c4a035eb0c89daf781f94eb7f6ce2964e9239d
- `camera/bob_to_f120.py` 856167860203f3c4c4709035abdf2c9c7643ecef694aa770ceb4adda37dbd9b0
- `camera/reproject_check.py` 5b692f08d9613bc89f2215c2e8b65a1dd60fa044480e968ed20004a98d318f34
- `camera/test/validate_projection_blender.py` 79fcd5a33596f2524117d4884ad355535ae7484049f4a221ecec30fa7923a018
- `camera/out/projection_validation.json` 9a6a1ab5396461bc666d391a1feb6d5c871ebdf1269c5b615ef9f1f0ced71767
- `camera/out/reprojection_by_frame.json` bdda2db913e9d1cc9afcf58e4f6bca8a04832dbc3bcecf56d85204c6350a3368
- `camera/out/ref_regions.json` c930065a2eac5d609512756c49a86fb1ef8aec1b91ecae0ba74840ce804eb25f
- `camera/visuals/v001_drift_ref_vs_locked_f000_f240.jpg` 98c705dd3b95da133bcf0b29488ead528aa44f70f3fe1fc996e62858fcfed384
- `camera/visuals/v001_drift_similarity_plot.png` 50d86c91627e7bcbe715d07066418e246511ac2495f45e020be0c511951987c1
- `camera/visuals/v001_reprojection_f120_full.jpg` ef40d2a71fd5ca291ec2d6044ca2c1c071f112997bda309cd82f5bc037479e5c
- `camera/visuals/v001_reprojection_f120_zooms.jpg` 56ae74f0c70dee0c0317e8929d95949a8ac52f789149384969f18d1a1b21abdf
- `camera/visuals/v002_drift_ref_vs_locked_f000_f240.jpg` 32bd49dabe6fef93a09b232eff08517b91a3addf2a2e99819356ae8a00ee0b38
- `camera/visuals/v002_reprojection_f120_full.jpg` 73de6dcf5067f406245a40dd824899aa222037b7a7cdcabe15ef6a290aad6a69
- `camera/visuals/v002_reprojection_f120_zooms.jpg` 3de28ddd08faab3a8856e67790786eac9a977fcb1c2e17d6513dc401cfc1f2b0
- `camera/visuals/v003_drift_ref_vs_locked_f000_f240.jpg` 93ab7841a91c7e1c6a928205914a89909303eac6a5bba9cad961cf00983f8b97
- `camera/visuals/v003_reprojection_f120_full.jpg` 859bd74de2ade20da540a202917bc1e55249ba49626b69504aa6ee64012359da
- `camera/visuals/v003_reprojection_f120_zooms.jpg` 2791aa2ed054e9cd066690504d1d9be0480cfa12471de0d96e2480015d9a75e5
