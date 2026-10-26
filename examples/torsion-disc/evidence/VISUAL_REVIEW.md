# Clip 03 torsion disc: independent visual & temporal review v001 (ck001)

Reviewer E (wave 2). I built nothing in the pipeline. I wrote my own trackers, which are applied the same way to the reference and to the render. The only shared code is the camera of record, `checkpoints/ck001/camera/project.py`.

Inputs:
* reference mp4: sha256 verified;
* ck001 `torsion.blend`, probed read-only, no render;
* `review/renders/ck001_eevee50`: 50 % EEVEE preview;
* ck001 `trajectory.csv` and `params`;
* `measurement/*.csv`.

Labels: **[obs]** measured here, **[hyp]** hypothesis, **[untested]** not checked. Machine-readable findings are in `out/review.json`.

## Visuals (visual_review/visuals/, all labelled "independent review of ck001")
| file | what it shows |
|---|---|
| v001_track_comparison_f000-240_ck001.png | Reference (own tracker + measurement ψ), render and projected solver. Panels: twist, twist residuals, twist rate, highlight-lobe axis vs camera azimuth, disc silhouette centre u/v, centre and wire residuals, sway envelope |
| v001_rectified_twist_strip_f000-240_ck001.jpg | Disc rectified top-down through the camera of record every 20 frames, reference vs render, with the measured line and the solver mark |
| v001_crop_grid_disc_mark_ck001.jpg | Disc + mark crops every 30 frames, reference over render, same window |
| v001_crop_grid_clamp_wire_top_ck001.jpg | Clamp, screws, collet and wire exit every 30 frames |
| v001_dense_attach_f008-015_ck001.jpg | Dense strip at the first sway turning point: wire → disc attachment |
| v001_silhouette_iou_f000_120_240_ck001.jpg | Static walnut frame + clamp silhouette overlay and IoU |
| v001_backdrop_table_profiles_f000_120_240_ck001.png | Backdrop/table Oklab-L maps, difference map, row/column profiles, horizon |
| v001_material_regions_f000_ck001.jpg | Region metrics (walnut, clamp, screw, table, wall, disc face, rim) |
| v001_flicker_map_ck001.png | Temporal 2nd-difference heat maps and per-frame series |
| v001_camera_jitter_ck001.png | Frame-to-frame image motion of a right-post window: reference vs render vs camera of record |

## Answers to the brief

1. **Render vs simulation: PASS (R5).** [obs]
   * The saved blend matches the solver projection to 3e-4 px; wire ends are exact to 2e-10 m.
   * Rendered pixels against the projected solver, over 241 frames:

     | quantity | rms | max |
     |---|---|---|
     | twist | 0.073° | 0.41° |
     | silhouette centre du | 0.11 px | 0.28 px |
     | silhouette centre dv | 0.59 px (constant −0.59 bias) | 0.71 px |
     | wire u | — | 0.32 px |
     | wire tilt | — | 0.13° |

   * [hyp] The −0.59 px dv bias and the 0.994 outline scale come from the rounded rim edge and anti-aliasing.
2. **The rendered twist follows the LARGE rotation.** It is not a "small twist". [obs]
   * Render: −389.7° over 10 s (solver −389.7°). Over 0–7.5 s: render −291.8°, reference −289.4°.
   * Initial rate: −24.6 °/s (solver −24.7; fitted ψ̇0 = −23.5 °/s at t = 0).
   * ψ0 = 221.3° is the angle from the torsion equilibrium. The mark's world azimuth is ψ + μ, which gives −168.1° at frame 0. The measurement gives −168.4°.
   * My own reference tracker agrees with measurement ψ to 0.21° rms.
3. **Temporal stability.** [obs]
   * The wire is continuous A → disc at every frame. The mark stays rigid with the disc. The disc shape is constant (outline rms 0.44 px). No popping of the disc or wire, and no intersection in the crops.
   * **But** the camera of record carries per-frame pose jitter, so the static scene shakes in the render and not in the reference (VR03-04, high):
     * right-post window jerk: render median 0.39 px, max 3.6 px; reference 0.11 / 0.82 px;
     * render flicker spikes at frames 38 and 58–63 = the largest jerks of the camera path.
4. **Scene match.** [obs]
   * Static silhouette IoU: 0.916 / 0.939 / 0.947 at f0 / f120 / f240 (V3 PASS); boundary median 1–2 px.
   * Brushed-brass highlight lobes stay with the camera, as they should:
     * lobe axis: render −23.7°, reference −17.5°;
     * camera azimuth −23.4°, while the mark turns −390°.
   * Mismatches, in priority order: clamp hardware (VR03-06), brass saturation and ring banding (VR03-07), backdrop/table (VR03-09), mark contrast (VR03-08), walnut tone (VR03-10).
   * Shadows: seen in crops only. The render's shadow is softer and lighter [untested quantitatively].
5. **Motion inside and after the windows.** [obs]
   * Twist: reference − solver is 1.1° rms in 0–7.5 s, then grows to +11.1° at 10 s. The AI slows faster: −33.5 vs −36.8 °/s in 9–10 s.
   * Sway: the AI envelope bumps at ~5 s and collapses after 6 s, and its phase drifts after ~3 s. Physics stays ~constant (Q 500).
   * Both are the documented AI defects and are correctly not copied.

## Findings
| id | sev | finding | observed / hypothesis | fix | owner |
|---|---|---|---|---|---|
| VR03-01 | pass | Render reproduces the solver | twist max 0.41°, centre ≤ 0.71 px, wire ≤ 0.32 px / [hyp] bevel causes the −0.59 px dv | — | — |
| VR03-02 | pass | The large rotation is rendered | −389.7° (render) vs −389.7° (solver) vs −378° (AI) | correct the "small twist" description | lead |
| VR03-03 | info | AI twist slows after 7.5 s | +11.1° at 10 s | disclose | lead |
| VR03-04 | **high** | Camera path jitter → render shake/flicker | right-post jerk 3.6 px (render) vs 0.8 px (ref); spikes at 38 and 58–63 / [hyp] pose noise in the per-frame bundle | smooth the camera path, then re-check R4 and the fit | A → lead |
| VR03-05 | medium | Disc outline ~7 px off the reference silhouette in the sway window; wire top 8.9 px right of the AI wire exit | B was fitted to Q (mark centre). Q − P = −0.031 D is world-fixed at every frame, while measurement's P pose reproduces the silhouette to < 1 px / [hyp] systematic along the view direction (rim thickness / f) | choose the observable (silhouette vs Q); refit B and the pivot | A + B, lead |
| VR03-06 | medium | Front screw and collet lack detail | screw contrast ×0.02; the reference has a chrome hex-socket head + washer and a chrome collet | model and material | C |
| VR03-07 | medium | Brass desaturated, ring banding, rim highlight | C 0.034 vs 0.048, lobe amplitude ×0.8 | saturation ↑, ring streaks ↓, wider rim strip | C |
| VR03-08 | low | Mark lighter and wider | dip ×0.78; FWHM 0.015 vs 0.010 D (part is the 50 % preview) | darker cavity; re-measure at 100 % | C |
| VR03-09 | medium | Backdrop/table too bright and neutral; the gradient does not turn with the orbit; horizon off | wall +0.03 L, chroma ×0.6; right wall +0.10 at f240; horizon 629 vs 615 (f0), 551 vs 578 (f240); table micro-detail ×3 | exposure, tint, wash placement, soften the crease | C |
| VR03-10 | low | Walnut lighter/coarser on the crossbar and feet | foot +0.12 L, crossbar +0.03..0.06 | tone, streak frequency | C |
| VR03-11 | pass | Integrity of the moving parts | continuity, rigidity and shape all OK | — | — |
| VR03-12 | info | Sway diverges after 2.5 s as documented | envelope and phase | disclose | lead |
| VR03-13 | untested | Not checked | full-res render, shadows quantitatively, effect of camera smoothing on R4, disc tilt in pixels | next checkpoint | E |

## Not checked [untested]
* Final full-resolution or final-engine render. Everything here is from the 50 % EEVEE preview.
* Quantitative shadow shape.
* check_rig intersections: I did not re-run them. I only looked at crops.
* Disc tilt in pixels: the silhouette registration covers shift and scale only.
* The motion-blur decision: taken from config/render.json and scene v008 evidence.

Reproduce with `scripts/run_all.sh`; NOTES.md has the setup and sha256s.

---
# v002 re-check of ck002 (smoothed camera, P-silhouette refit, look v025): focused pass/fail
Outputs: `out/ck002/review.json` and `visuals/v002_*` (labelled "independent review of ck002"). Run with `VR_CK=ck002 scripts/run_all.sh`; then run `write_review_json_v002.py` for the review JSON. All checks use the 50 % EEVEE preview.

| id | verdict | evidence (observed) | fix / owner |
|---|---|---|---|
| VR03-01 render = solver | **PASS** | twist rms 0.073° / max 0.25°; centre ≤ 0.78 px (dv −0.63 px bias as before); wire ≤ 0.33 px; −389.9° total | — |
| VR03-04 camera jitter | **PASS (fixed)** | right-post jerk median/max: render 0.07 / 0.18 px, reference 0.11 / 0.82 (ck001 render 0.39 / 3.58); no flicker spikes; reference-to-camera velocity rms 0.16 px (was 0.54), so the orbit is preserved | — |
| VR03-05 disc outline | **PASS**; wire top open (low) | reference silhouette offset in 0–2.5 s: 1.71 px rms (was 7.2); wire mid-span −1.7 px (was −6.4); the reference wire still exits the clamp 6.1 px left of the rendered wire top | disclose, or move the collet with the pivot (B/C) |
| VR03-06 screw/collet | **PARTIAL** | the screw is now a bright chrome dome (ΔL +0.11…+0.21, brighter toward f240, bright washer ring); the reference is a darker hex-socket button head; the collet is still a dark nub | flatter hex-socket head, less specular; chrome collet ring (C) |
| VR03-07 brass | **FAIL: over-corrected (new)** | ring banding gone and rim band continuous (good); but the cones are now hard-edged 4-sector: face chroma p05–p95 0.021–0.082 vs reference 0.038–0.059; angular high-harmonic/h2 ratio 0.26 vs 0.63; lobes still camera-locked | lower chroma in bright sectors, soften cone edges (softer light, more anisotropic roughness/brushing noise) (C) |
| VR03-08 mark | **FAIL (unchanged)** | dip ×0.76 (was ×0.78); FWHM 0.016 vs 0.010 D; [hyp] partly the 50 % preview | darker, narrower cavity; check a full-res crop (C) |
| VR03-09 backdrop/table | **MOSTLY PASS** | wall gradient now flips with the orbit (L left−right: render −0.21→+0.09, reference −0.22→+0.10); horizon within 1 px at f0/f240 (6 px at f120); wall chroma 0.011 vs 0.012; residuals: wall +0.03 L (top rows +0.05), table +0.03 L, table micro-detail ×3 | lower exposure on the wall top and table; less table micro-texture (C) |
| VR03-10 walnut | **PASS (minor)** | IoU 0.918 / 0.936 / 0.943; foot ΔL +0.07 (was +0.12); crossbar +0.02 | — |

**Untested:** full-res render, quantitative shadow comparison, and whether VR03-08 still fails at 100 %.
