# Independent visual and temporal review: ck002 (v001)

Status: **independent review of ck002** (integrated scene, draft). Reviewer: independent visual reviewer.
Contract v3; acceptance `config/acceptance.json` V1–V4. Machine-readable version: `out/review.json`.

Frames reviewed: reference 0–240; `ck002_eevee50` 0–240; `ck002_final100` 0–240 (reviewed while it rendered, then re-run
on all 241 frames after it finished); `ck002_diag_side` 0–240. The reference sha256 was re-verified (`ced22d10…f56bae`).

Labels: **[obs]** measured or seen in pixels, **[hyp]** an interpretation that has not been tested, **[untested]** not checked.

## Visuals (all in `visuals/`, with labels burned in)
| file | what it shows |
|---|---|
| `v001_tp_zc_crop_grid_ck002.jpg` | 23 key frames: turning points (13, 29, 30, 46, 62, 63, 96, 129, 163, 179, 180, 213, 229, 231), zero crossings (5, 21, 38, 55, 104, 171, 237) and the end frames. Each row shows reference and render side by side: pin/wire departure at 4×, wire–bob attachment at 4×, and bob plus contact shadow at 1× in the same window. |
| `v001_dense_attach_f010-016`, `f019-023`, `f027-033`, `f093-099_ck002.jpg` | The same crops for every consecutive frame around two turning points, one zero crossing and the end of the anchor window. |
| `v001_dense_bob_strip_f000-240_ck002.jpg` | Every 2nd frame, reference (top) and render (bottom) in the same window. Yellow ticks mark the tracked reference u and the predicted render u. |
| `v001_bob_track_comparison_f000-240.png` | u(t) for the reference, the render and the projected solver; render minus reference; half-cycle amplitudes; second differences against the expected values. |
| `v001_flicker_heatmap_ck002.png` | Temporal second-difference luma maps for the reference, eevee50 and final100, with the pendulum sweep excluded. |
| `v001_material_regions_f000_ck002.jpg` | 11 regions, reference and render, with Oklab L, C, hue and the grain-detail ratio. |
| `v001_silhouette_iou_f000_120_240_ck002.jpg` | Stand mask overlap at frames 0, 120 and 240, with IoU and boundary distances. |

## Verdict against V1–V4
- **V1 pass [obs].** No changes in geometry shape, no attachment sliding, no length change, no flicker and no popping (VR-01…04).
- **V2 pass with minor detail findings.** The stand, base, arm, pin, wire and bob are all present and readable. The walnut, steel and brass read correctly. The pin/loop proportions and the missing bob eyelet differ from the reference (VR-11, VR-12).
- **V3 pass.** The side-by-side compares identical frame times (VR-18). Stand IoU is 0.937, 0.940 and 0.942 at frames 0, 120 and 240 (VR-13).
- **V4.** Each finding below is tagged.

## Findings
| id | sev | finding | tag | owner |
|---|---|---|---|---|
| VR-05 | **major** | The render has no motion blur (`use_motion_blur=False`), but the reference smears at speed. Edge width going from slow to fast frames: ref 5.8 → 8.7 px, render 5.5 → 6.0 px. The render looks stroboscopically crisp near zero crossings. | obs | lead (user decision, look frozen) |
| VR-06 | minor | Over 0–4 s the 2D RMS is 3.29 px. The residual is a 4-frame sawtooth (f0–f3: +3 → +9.3 px, f4: +0.7), which matches the documented reference cadence defect. R5 as declared fails (18.3 px over 0–10 s). | obs + hyp (attribution) | lead / measurement (reporting) |
| VR-08 | minor | In image space the render amplitude looks undamped (+1.2 %). The Q=1000 decay (−1.9 %, measured in the side view) is cancelled by the +3.3 % push-in. Physics is correct, but "slight damping" will not show. | obs | dynamics + lead |
| VR-09 | minor | Known: base top, table and bob are rendered brighter (dL +0.054…0.059, +0.046…0.057, +0.03…0.04). | obs, known/accepted | lead |
| VR-10 | minor | New: arm, post and base front are also brighter (dL +0.02…0.055), and the wood hue is ~5° more yellow. The backdrop matches. This is probably the same cause as VR-09. | obs + hyp | lead (look) |
| VR-11 | minor | New: the pin is thinner and satin rather than mirror-polished, and the loop hangs larger and looser than in the reference. DOF is off, so it is not defocus. | obs | geometry/rig (+ look) |
| VR-12 | minor | New: the brass eyelet where the wire meets the bob is missing. | obs | geometry/rig |
| VR-15 | minor | The render bob is ~2 % larger (+1.14 px radius). This is within the L_c/r uncertainty, and tracker bias on the darker reference rim is possible. | obs + hyp | geometry (info) |
| VR-01 | note/pass | Render motion matches the projected solver: u RMS 0.066 px, f100 ≡ e50 to ≤ 0.21 px. The second-difference residual is 0.062 px/frame² RMS, which is at the noise floor, so there are no kinks. | obs | – |
| VR-02 | note/pass | The tilt sign is consistent. In the side view the swing centre sits 0.705–0.735° from the stand vertical (expected 0.722° in magnitude). The rigid arc residual is 0.019 px. | obs (side-view sign untested) | – |
| VR-03 | note/pass | The wire is straight (sagitta ≤ 0.067 px). It passes within 0.085 px of the pivot and within 0.17 px of the bob centre. Bob radius follows depth to 0.031 px. | obs | – |
| VR-04 | note/pass | Static flicker is 0.034 luma (f100) against 0.20 in the reference, and region means pop by at most 0.018 luma. | obs | – |
| VR-07 | note | After 4 s the reference amplitude grows from 66.9 to 110.1 px while the render stays at 70.5–71.3 px. This divergence is the documented AI defect, not a simulation artifact. | obs | – |
| VR-13 | note/pass | Silhouette IoU is ≥ 0.937. The reference outline sits ~3–5 px outside the render on the post's left edge, the arm top and the base right end. The result is sensitive to the mask method (a chroma-only mask gives 0.90). | obs | geometry (info) |
| VR-14 | note/pass | The push-in is reproduced: scale 1.0331 in the reference, 1.0329 in the render. | obs | – |
| VR-16 | note | The contact shadow is present and follows the bob (slope 0.41 vs 0.36). The automatic metric has low confidence, so the crops are the primary evidence. | obs | – |
| VR-17 | note | Reviewer-method artifact: on full-res frames my edge tracker jumps up to ~1.5 px where the base's back edge meets the bob rim. This is not in the render. | obs | visual review |
| VR-18 | note/pass | The v002 SBS is frame-aligned at 241 frames and 24 fps. | obs | – |

## Untested
Cycles renders; perceived judder at playback speed by a person; subframe motion (only integer frames were rendered); grain
direction and scale, which were judged by eye only; regions and frames other than 0/120/240; the tilt sign convention in the side camera.

## Reproduce
`visual_review/scripts/run_all.sh`. It first needs the reference decoded to `cache/ref/f%04d.png`; the decode one-liner is in NOTES.md. My code is my own: it reads other roles' CSV and JSON outputs and imports nothing from them.
