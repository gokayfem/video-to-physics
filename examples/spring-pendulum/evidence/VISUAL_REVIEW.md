# Independent visual and temporal review: ck002 (v002, current)

Status: **independent review of ck002**.

- **Render reviewed:** `review/renders/ck002_eevee50` (50 %, EEVEE 32 spp, motion blur shutter 0.09).
- **Side-by-side:** `review/visuals/v002_sbs_ck002_f000-240.mp4`.
- **Method:** the same as v001, with two additions:
  - a weight-steel profile (`weight_steel.py`);
  - a motion-blur metric on the eye-ring edges (`blur_ring.py`).
- **Machine-readable version:** `out/review.json` (v002). The v001 review is embedded under `history` and kept verbatim in `out/review_v001_ck001.json`.
- **History:** the v001 review of ck001 is kept unchanged below.

## v002 visuals (`visuals/v002_*`)
| file | what it shows |
|---|---|
| `v002_weight_steel_profile_ck002.png` | **New.** Mean L profile across the cylinder, plus the brightness histogram, for the reference, ck001 and ck002. |
| `v002_backdrop_brightness_profiles_f000_120_240_ck002.png` | Backdrop L maps, difference map and profiles. |
| `v002_material_regions_f000_ck002.jpg` | Region metrics at frame 0. Numbers for frames 120 and 240 are in `out/ck002/material_stats.json`. |
| `v002_tp_zc_crop_grid_ck002.jpg` | Crop grid at turning points and zero crossings. |
| `v002_dense_attach_f012-018`, `f020-026`, `f230-236_ck002.jpg` | Consecutive-frame crops. |
| `v002_dense_strip_a/b_*_ck002.jpg` | Every 4th frame of the spring and weight. |
| `v002_track_comparison_f000-240.png` | Track comparison. The weight-tracker lines are unreliable on this material (VR-19). |
| `v002_flicker_heatmap_ck002.png` | Flicker heat map. |
| `v002_silhouette_iou_f000_120_240_ck002.jpg` | Static silhouette overlap. |
| `v002_motion_blur_ring_edges_ck002.png`, `v002_motion_blur_edge_width_ck002.png` | Motion-blur edge metrics. |

## v002 verdict
- **R5 / VR-01 pass.**
  - Eye ring: render − solver u sd is 0.115 px at full res (0.06 px at 50 %), max 0.34 px. The v bias is +0.12 px.
  - Turning points agree to within 0.11 frame.
  - In the saved blend the weight origin matches the trajectory H to 1.5e-8 m, and its axis matches n to 1.6e-8 rad.
  - ck002's projected motion differs from ck001's by ≤ 0.014 px.
- **V1 pass.**
  - The spring keeps 21 turns, with min coil gap 2.10 × wire diameter.
  - The lower loop sits 5.238 mm from H, constant to 0.2 µm.
  - Flicker is 0.079 luma.
- **V3 pass.**
  - IoU is 0.967, 0.965 and 0.958; push-in 1.0478 against 1.0484.
  - The SBS has 241 frames at 24 fps and is aligned.
- **V2 partial.** The backdrop, walnut hue, table and S-hook are now close. **The weight steel is still major**, and the lower loop is too thick (new, minor).

## v002 findings
| id | sev (v001→v002) | finding | tag | owner |
|---|---|---|---|---|
| VR-05 | major → **major** | **The weight now reads darker and mirror-like; the reference is bright satin.** This confirms the lead's impression.<br>• **Reference:** brightness is unimodal around L 0.46, with two vertical bright bands (L 0.73 / 0.70 at ±0.35 of the width) and a 0.45 centre.<br>• **ck002:** brightness is bimodal (0.27 / 0.67), with one broad central bright lobe at ~0.60 (a blurred beige-white reflection) and near-black flanks (0.29 at \|x\| 0.6–0.75).<br>• **Summary numbers:** 37 % of side pixels are below L 0.35 (reference 16 %). Mean L is 0.454 against 0.485. Chroma is 0.018 against 0.029. Fine detail is 0.019 against 0.029 (ck001 was 0.011).<br>**Fix:** use satin roughness and keep the anisotropy. Place two light-linked vertical strip reflections at ±0.35 of the width (~±20° off the camera axis). Lift the flanks to L 0.35–0.45. Suppress the central lobe and add slight warmth. | obs; cause of the lobe [hyp] | C |
| VR-04 | major → minor | **The backdrop is largely fixed: the pool is now low behind the weight.** Residuals:<br>• The peak is ~65 px too high: (1075–1077, 679–690) against the reference's (1087–1089, 746–758).<br>• The upper backdrop is too dark: opening top −0.045 L, above the bar −0.08 L.<br>• The lower outer corners are +0.03–0.05 too bright.<br>The low opening now matches (+0.017…+0.023). | obs | C |
| VR-06 | minor → minor | **The walnut hue is fixed:** now +3…7°, was +9…13°. The left post is still +0.04…+0.05 L lighter. Figure detail is 0.53–0.75 of the reference at 50 %. | obs | C |
| VR-07 | minor → minor | **The table left is fixed** (dL −0.005…+0.003). The base front right is still +0.064…+0.070. **New:** the crossbar is +0.025…+0.051 lighter and neutral/greenish (hue 66° against 34°). | obs | C |
| VR-18 | – → **minor (new)** | **The lower loop is too thick:** 14–15 px against 9–11 px in the reference. It reads as a rod in front of the eye ring ending in a dark stub, rather than a wire hooked through the ring. **Fix:** use the spring wire diameter. | obs | geometry C |
| VR-08 | minor → note | **The hook is resolved:** the shank is now 14–17 px against 14–15 px. Residual details: the reference hook top shows ~10 px above the bar and has a twisted end. The eye ring is 34.6 px in the render against 33.7 px in the reference (same tracker). | obs | C (optional) |
| VR-14 | note → note | **Shutter 0.09 is active.** There is no strobing or smear in the zero-crossing crops. The eye-ring edge widths show no measurable speed dependence in either clip: reference 5.95 − 0.034·speed, render 5.71 − 0.007·speed. A 0.09 shutter adds at most 1.4 px at 16 px/frame. Whether the shutter matches the reference is **untested**. | obs + untested | review (final render) |
| VR-19 | – → note (new, method) | **My weight side-edge tracker fails on the mirror flanks:** left fit rms 5.4 px, axis sd 1.84 px, tilt slope 0.91. The saved-blend check proves the weight transform is exact, so this is a measurement limitation, not a render defect. | obs | review |
| VR-20 | – → note (new) | The spring band is +0.02…+0.03 L brighter than the reference. | obs | C (optional) |
| VR-02, 03, 13, 15 | pass → pass | Continuity, flicker, IoU and SBS alignment: see the verdict. | obs | – |
| VR-10, 11, 12 | unchanged | The motion is identical to ck001, so the documented AI defects are unchanged: weight over-tilt 1.34 against 1.05; first-cycle mismatch and amplitude/period drift; ring v range 19.17 px against 19.07. | obs/hyp | lead |
| VR-09, 16, 17 | unchanged | Not re-tested. VR-09 cannot be re-tested on ck002 because of VR-19. | – | – |

---

# History — independent visual and temporal review: ck001 (v001)

Status: **independent review of ck001** (preview `review/renders/ck001_eevee50`, 50 %, EEVEE 32 spp, frames 0–240).
Contract v1; acceptance `config/acceptance.json` V1–V4 and R5. Machine-readable version: `out/review_v001_ck001.json` (also embedded in `out/review.json` under `history`).
The reference sha256 was re-verified (`fee726c2…6865cd`). Every crop compares identical frame numbers; nothing is retimed.

Labels: **[obs]** measured or seen in pixels, **[hyp]** an interpretation that has not been tested, **[untested]** not checked.

## Visuals (`visuals/`, labels burned in)
| file | what it shows |
|---|---|
| `v001_backdrop_brightness_profiles_f000_120_240_ck001.png` | Smoothed backdrop L maps (frame 0) for the reference and the render, the difference map, and horizontal and vertical L profiles at frames 0 and 240. |
| `v001_track_comparison_f000-240.png` | Eye-ring u for the reference, the render and the projected solver; render − solver; reference − solver; weight tilt; swing half-amplitude and turning-point lag. |
| `v001_tp_zc_crop_grid_ck001.jpg` | 18 frames: turning points 1/15/31/46/62/108/155/186/233, zero crossings 8/23/38/116/194/225, and 0/120/240. Each row shows the S-hook/top coil at 2×, the eye ring/lower loop at 3× and the weight at 1×, reference and render side by side. |
| `v001_dense_attach_f012-018`, `f020-026`, `f230-236_ck001.jpg` | The same crops on consecutive frames around a turning point, a zero crossing and a late turning point. |
| `v001_dense_strip_a_f000-120`, `b_f124-240_ck001.jpg` | Every 4th frame of the spring and weight. Reference on top, render below, same pixels. |
| `v001_flicker_heatmap_ck001.png` | Temporal second-difference luma maps, with the moving sweep excluded. |
| `v001_material_regions_f000_ck001.jpg` | 13 regions with Oklab L/C/h, contrast ratio and a fine-detail ratio. The detail ratio is also given against the reference degraded to 50 % for a fair comparison. |
| `v001_silhouette_iou_f000_120_240_ck001.jpg` | Static-frame mask overlap and the push-in check. |
| `v001_motion_blur_edge_width_ck001.png` | Weight side-edge width against speed. |

## Verdict
- **R5 pass at preview scale [obs].** In the render, the eye ring and weight sit where the projected checkpoint trajectory puts them.
  - Ring u sd is 0.17 px at full-res scale (0.09 px at 50 %), with a maximum of 0.50 px.
  - Weight-axis u sd is 0.05 px and weight tilt sd is 0.07°.
  - Turning points agree to within 0.1 frame.
  - There is no mod-4 pattern.
  - The constant +1.6 px offset in ring v comes from my tracker (VR-01).
- **V1 pass [obs].**
  - The spring keeps 21.0000004 turns at all 481 frame and half-frame samples, with no coil self-intersection (min gap 2.1 × wire diameter).
  - The lower loop stays on the hinge (constant to 0.1 µm) and the S-hook/top coil are static.
  - There is no flicker: 0.09 luma against 0.28 in the reference.
- **V2: partial fail on materials and lighting.** All parts are present, but:
  - the backdrop gradient is inverted (major);
  - the weight steel reads differently (major);
  - the walnut, table and hardware thickness differ (minor).
- **V3 pass.** Static-frame IoU is 0.956, 0.957 and 0.958 at frames 0, 120 and 240. The SBS is frame-aligned. The push-in scale is 1.0484 in the reference against 1.0477 in the render.
- **Motion vs reference.** All differences match documented AI defects: amplitude wander, period drift and excess weight tilt (VR-10, VR-11). The largest visible gap inside the fit window is the first cycle.

## Findings
| id | sev | finding | tag | owner |
|---|---|---|---|---|
| VR-04 | **major** | **The backdrop gradient is inverted vertically inside the frame opening.** This confirms what the lead noticed. The reference is brightest LOW, at L 0.751 around (1087, 746) px, behind the weight just above the table. The render is brightest HIGH, at 0.770 around (902, 179) px under the crossbar. Low-opening dL is −0.173, −0.165 and −0.167 at frames 0, 120 and 240. The top of the opening is +0.10 and above the bar is +0.14. At u = 700 the reference rises from 0.58 to 0.70 down the frame while the render falls from 0.71 to 0.49. Outside the frame the levels agree (dL −0.01…−0.02), but next to the posts at v 500–700 the reference is brighter (~0.55 vs 0.47). **Fix:** lower and aim the backdrop wash to about image (1000–1090, 700–760). Reduce the upper-centre level by 0.10–0.14 L. Spread the low band outward to the posts. | obs; cause [hyp] | scene+look C |
| VR-05 | **major** | **The weight steel reads differently.** The render is darker (dL −0.03…−0.05) and its hue is 15° more beige. It shows a large diffuse beige blotch, probably a table reflection. It lacks the reference's crisp vertical light/dark highlight bands, the bright top rim and the visible top-face ellipse. Detail is 0.78–0.86 of the reference at 50 %. **Fix:** lower roughness or use anisotropic circumferential brushing, add vertical strip reflections light-linked to the weight, reduce the table reflection and brighten the rim. Re-judge after VR-04, because the weight reflects the backdrop [hyp]. | obs + hyp | scene+look C |
| VR-06 | minor | **The walnut posts don't match.** They are +0.05…+0.06 L lighter, with hue 9–13° more yellow. The render grain is fine and straight, while the reference shows warm red-brown cathedral figure. Detail is 0.66–0.81 of the reference at 50 %. | obs | C |
| VR-07 | minor | The table left is darker (dL −0.06…−0.07), while the table right matches. The base-plate front right is brighter (+0.06…+0.07). This probably has the same cause as VR-04. | obs + hyp | C |
| VR-08 | minor | **The hardware is too thin.** At frame 0, rows v 140–150, the S-hook shank is 9 px wide in the render against 14–15 px in the reference. The reference also has a twisted, crimped wire end at the bar, and its hook top is ~10 px higher. The reference eye-ring wire is thicker and reads more 3D. | obs (exact diameters untested) | geometry C |
| VR-09 | minor | The render weight may be wider. My tracker gives 144.2 px for the render against 134.7 ± 5.4 px for the reference; the measurement role gives 140.4 px. This could be tracker bias on the reference's highlights. | obs + hyp | C / review |
| VR-11 | minor | **Fit-window tracking is ~5 px.** Ring u RMS is 5.13, 4.92 and 13.4 px over 0–4, 0–7.5 and 7.5–10 s; the R2 fail is already documented.<br>• **First cycle:** the reference is +11 px off at frame 0 and −14 px at frame 8. Its first half-amplitude is 7.71° against 8.24° for the solver.<br>• **Amplitude:** the reference then grows to 8.24° at ~5 s and drops to 7.6°, while the solver (Q 500) decays monotonically from 8.24° to 7.59°.<br>• **Timing:** turning points agree to within ±15 ms through 7.5 s, then the reference lags by up to 79 ms at frame 235. Reference half-periods run 0.623 → 0.671 s; the solver's are 0.648 s.<br>All of this matches the documented amplitude wander and period drift. | obs; attribution [hyp] | lead / B |
| VR-10 | note | **The reference weight over-tilts.** The tilt/swing slope is 1.34 in the reference against 1.05 for the solver and 1.06 for the render. This shows at every turning point: ±10.5–11.9° in the reference against ±8.5° in the render. It is a documented AI defect; do not copy it, and disclose it with the SBS. | obs; attribution [hyp] | lead |
| VR-01 | note/pass | **Render vs solver.** The figures are in the R5 verdict. Ring v and radius offsets (+1.6 / +0.45 px) move between −1.7 and +1.5 px when I change the tracker's exclusion cones, so they can't be attributed to the render. Second-difference residuals are at tracker noise: ring u 0.30 px/frame² RMS, weight axis 0.11. | obs | – |
| VR-02 | note/pass | Spring continuity in the saved blend: see V1. Static parts moved 0 m and the eye stays rigid on the weight to ≤ 9e-8 m. | obs | – |
| VR-03 | note/pass | Flicker: the worst render frame is 0.15 luma. Only the moving table shadow shows up, which is expected. | obs | – |
| VR-12 | note | The radial (ring v) range matches: reference 19.07 px, solver 19.08, render 19.48. | obs | – |
| VR-13 | note/pass | Silhouette IoU is ≥ 0.956, with median boundary distance 0 px and p95 6–12 px. The reference-only blob at the top of the left post is the lighter render walnut dropping out of the mask (VR-06). | obs | – |
| VR-14 | note | **Motion blur could not be confirmed.** My edge metric is dominated by the cylinder's shading: reference 6.9 + 0.009·speed, reference at 50 % 7.1 + 0.006·speed, render 8.3 + 0.034·speed. Render edges are ~1.2 px softer at every speed, which is preview softness rather than blur. There is no strobing or smear in the zero-crossing crops. The shutter value is **untested**. | obs + untested | review (final render) |
| VR-15 | note/pass | The SBS has 241 frames at 24 fps and is frame-aligned. 14 frames near turning points can't be told apart from their neighbours by thumbnail matching. | obs | – |
| VR-16 | note | The tight coil section is slightly more open: centreline gap 8.25 mm (~13 px) against the reference's dense pitch of 7.75–11.9 px. | obs + hyp | C (optional) |
| VR-17 | note | The reference backdrop has concentric compression banding. Don't copy it. | obs | – |

## Untested
- Final full-resolution or Cycles render; R5 at full scale.
- Subframe motion and perceived judder at playback speed.
- Exact wire and hook diameters.
- The motion-blur shutter value.
- Out-of-plane motion (the trajectory is planar).
- Region metrics at frames other than 0, 120 and 240.

## Reproduce
- Decode the reference to `cache/ref` as described in NOTES.md, then run `scripts/run_all.sh`.
- The Blender geometry probe (`scripts/blend_spring_frames.py`) is read-only, renders nothing and runs under `tools/with_render_lock.sh`.
- I wrote my own trackers. The only external code I use is the checkpoint's `camera/project.py`, which is the camera of record.
