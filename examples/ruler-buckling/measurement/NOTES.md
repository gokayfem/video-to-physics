# Measurement v001 (role A) — reference.mp4 (241 f, 24 fps)
Code: code/track.py (camera ORB+RANSAC homography vs f0), code/track3.py (centreline: diff vs inpainted f0 background plate,
sub-pixel row edges), code/recon.py (back-projection with config/camera_ff.json). Scratch caches are not kept (frames.npy).

## Camera — MEASURED
Static: max homography displacement 1.48 px over the frame, clamp point drift max 0.27 px, >=671 inliers/frame. Our camera_ff applies directly.

## Model assumption for 3-D (MODEL-CONDITIONED)
Centreline rays intersected with plane x=0 (perpendicular to broad face, through clamp). Tip = point 280 mm arclength from jaw top.
delta = world Y of that point minus f0-24 baseline (4.7 mm offset removed). Known bias: segmented centreline is jagged, inflating
arclength, so the "280 mm" marker sits ~10-15 mm below the platform (visible in annot images) -> |delta| underestimated by ~5-10 %.
Noise pre-buckle 1.9 mm (one-frame spikes at f24, f48, f70 = floating weight disc contaminating the segmentation).

## Out-of-plane evidence
Row-width test (x=0 bending keeps the strip's width vector = world X -> horizontal row width constant ~39 px; edgewise y=0 bending
gives width ~44/sin(tangent angle)). f160 (left lean): 38-45 px -> consistent with weak-axis bending. f200-240 (right lean): 47->61->68->85 px
as tangent tilts, matching the edgewise prediction (47/53-59/71) -> the right lean is largely in-plane (strong-axis) bending = non-physical,
and its x=0 delta (+130..+170 mm) is not a valid weak-axis deflection. If read as edgewise: tip X ~ +100..+130 mm.

## Weights (MEASURED by eye from v001_peg_contact_f020-128.jpg)
Seated count: 0 until f35; 1 from f36; 3 from f60; 5 from f84; 7 from f112 (counts jump by 2 while only one disc is seen in flight).
Each disc materialises mid-air beside the strip (f24, f48, f72, f100), floats over the peg and descends; no hand ever. Bench stack beside
the vise stays ~4 discs throughout (never depletes). Stack height at onset 7 x 4.5 = 31.5 mm (nominal disc thickness).

## Buckling
Onset f126 (t=5.25 s; first of >=5 frames |delta|>8 mm; slow lean visible from ~f121) with 7 weights = 140 g (+~7.5 g platform/peg)
vs predicted P_cr ~161 g -> 9th weight. Left (-Y) lean grows to -81 mm at f158, plateau to ~f176, then the strip swings back through
vertical (~f184) and over to the right, reaching +170 mm (x0-plane metric) by f230-240 with NO change in load. Left-lean shape is a
single clamped-free arc (curvature max at clamp, near-straight upper third) — qualitatively elastica-like.

## AI defects
1. Weights teleport/materialise mid-air, float without a hand, count jumps by 2 per visible disc (1->3->5->7).
2. Bench supply stack never depletes.
3. Buckles early (7 weights vs ~9 predicted) and at constant load reverses direction (left -> right snap-through) — non-physical for a dead load.
4. Right-side lean is edgewise (strong-axis) bending of a 25x0.5 mm strip — physically impossible at these loads.
5. Weights become ~horizontal-axis stack tilted with tip (fine) but stack appears to lengthen/shift between f180-240 (not quantified).
