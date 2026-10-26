# Concept A (ruler buckling): first-frame scene notes

**Deliverable:** `firstframe/frame0.png`, 1920×1088, rendered in Cycles (Metal GPU, 192 adaptive samples, OIDN, AgX Base Contrast, exposure −0.5). About 37 s under the render lock.

## Rebuild
```
python scene/make_textures.py          # numpy/opencv venv: scene/tex/ruler_ink.png, card_ink.png, card_lines.json
../_tools/with_render_lock.sh blender -b --factory-startup -P scene/build.py -- --render firstframe/frame0.png
#   add --preview for a 50 % render at 48 samples
```
- The build is deterministic. Every number comes from `config/scene_params.json`, and the washer jitter uses a fixed seed.
- Each build writes `scene/out/ruler_ff.blend` and `config/camera_ff.json`.
- The textures are loaded by absolute path. If the folder moves, rebuild.

## Geometry (metres, Z up, bench top at z = 0, camera on −Y)
- **Ruler `RUL_Ruler`:** 300 × 25 × 0.5 mm, z 0.075 to 0.375.
  - The vise jaw tops are at z = 0.095, so 20 mm is clamped and 280 mm is free.
  - Mesh: a mid-surface grid of 10 × 300 segments, solidified (applied). It carries UVs.
  - Object origin: bottom-centre of the mid-surface. FEA displacements can be baked onto this mesh as a mesh cache.
- **Ruler markings:** 1 mm ticks on the −X edge, plus cm numbers 1 to 29 reading bottom to top, with 0 at the clamped end.
- **Platform:** a 25 × 25 × 3 mm aluminium plate on the top end. Two 25 × 12 × 2 mm tabs hold it (the front tab carries two pan-head screws). Its top is at z = 0.378.
- **Washers:** 12 × ISO 7089 M8 brass (8.4/16/1.6 mm) stacked at (0.115, −0.045). The stack is 19.2 mm tall.
- **Grid card:** 600 × 460 mm at y = 0.15, which puts its front face 149 mm behind the ruler.
  - 10 mm lines are 0.5 mm wide. Every 50 mm the line is 0.9 mm wide.
  - Line origin: x = 0 at the ruler centre line and z = 0 at the bench top. Lines run from x −0.29 to 0.29 and z 0.01 to 0.45.
- **FEA material (assumed):** E = 193 GPa, ν = 0.29, ρ = 7900 kg/m³.
- **Hand check of the critical load** (FEA owns the real value):
  - I = b·t³/12 = 2.60e-13 m⁴, L = 0.280 m.
  - P_cr = π²EI/(4L²) ≈ 1.58 N, about 161 g.
  - That is roughly 75 M8 washers at 2.1 g each (washer mass assumed). So 12 washers will **not** buckle this ruler. See open issues.

## Camera (camera_ff.json)
- 85 mm lens, 36 mm sensor (horizontal fit), no shift. fx = fy = 4533.33 px, cx = 960, cy = 544.
- Pitch: 3° down. Distance: 1.80 m to the target (0, 0, 0.19). The camera is locked.
- The ruler measures 2.52 px/mm in its own plane. The full ruler is 69.8 % of frame height, and the visible free length is 706 px.
- **Projection test:**
  - The pinhole model matches Blender's `world_to_camera_view` to within 1.3e-4 px.
  - Rendered grid lines sit within 1 px of the predicted positions at 4 corners. Top corners are checked at z = 0.40 because the card runs past the top of the frame.

## Look decisions (LESSONS: matte surfaces, light linking)
- Lighting: soft area key (front-left) plus fill (right) plus top light.
- A backdrop wash is light-linked to the backdrop and the card only.
- A large "ruler_reflect" area light behind the camera is light-linked to the ruler and platform only. It gives the satin steel an even sheen without lighting anything else, so the ink stays readable. Its energy was tuned from 90 W to 12 W.
- Ruler steel: anisotropic GGX brushed along its length, roughness about 0.3, no mirror streaks.
- Region luma (0–255): ruler 183, card 209, vise 54, backdrop 175, bench 182.
- Backdrop vertical profile: smooth, with no inversions.

## Open issues
1. **Load mismatch.** With these dimensions, 12 M8 washers (about 25 g) are about 6× below P_cr. Options:
   - a thinner or longer ruler (0.3 mm × 300 mm free gives P_cr ≈ 0.34 N, about 16 washers);
   - heavier washers (M16 is about 11 g);
   - a prompt that adds a heavier weight.

   This needs a decision before generation and FEA. The prompt can still show buckling, but the real physics would not.
2. At full resolution the 1 mm ticks are 2.5 px apart, which risks moiré after video compression. The cm numbers are small, about 9 px.
3. The vise is a simplified blocky model (no casting draft or serrations). It reads as a bench vise, but it is somewhat clean and CG-looking.

## Update: lead fix round (supersedes the items above where they conflict)
- **Washers → slotted weights.** The M8 washers are replaced by 12 slotted brass weights (`RUL_Weight00..11`), stacked at (0.18, −0.06) m.
  - Each is Ø25 × 4.5 mm with a Ø5.5 mm hole and a 5 mm radial slot. Nominal mass: 0.020 kg each.
  - Buckling should start at the 9th weight (180 g > P_cr ≈ 161 g).
- **Peg.** `RUL_PlatformPeg`: Ø4 × 20 mm, steel, centred on the platform top (top at z = 0.401 m).
- **Ruler markings.** Only 5 mm ticks (short) and 10 mm ticks (long), plus cm numbers 1–28.
  - The number for 29 was dropped because the platform tab covered it.
  - Numbers are 5.3 mm tall, which is about 12.3 px.
- **Camera reframed** so a full 12-weight stack on the peg stays in frame: distance 1.96 m, target z = 0.22 m.
  - Ruler tip at (960, 184.5) px; jaw-top line at (960, 831.8) px.
  - Scale: 2.31 px/mm. The full ruler is 64 % of frame height (down from 70 %, to leave this headroom).
  - The top of a 12-weight stack projects to y ≈ 51 px.
- **Textures** are now saved relative to the .blend (`//../tex/*.png`).
- The superseded camera numbers and open issues 1, 2 and 4 above no longer apply.

## Update: photoreal pass, per the user's "no lines or measurements" (supersedes the look items above)
- **Removed:** the grid card (`grid_card.enabled = false`) and all ruler markings (`ruler.markings.enabled = false`). The ruler is now a plain brushed stainless strip.
- **Wall:** a plain warm-grey plaster wall at y = 1.6 m, with large-scale mottling and a fine bump. It is out of focus: Cycles depth of field at f/5.6, focused at 1.96 m on the ruler plane.
- **Bench:** grey, with faint wipe/scratch roughness streaks. Its back edge is at y = 0.60 m.
- **Vise:** black cast iron with a two-scale casting bump and sparse bare-metal paint chips. The jaw inserts have serrations (wave-texture bump).
- **Brass weights:** tarnish tint and smudge-scale roughness variation.
- **Lights:** `ruler_reflect` is now 5 W.
- **Post-processing** (`scene/post.py`): mild vignette (corner gain 0.90) and seeded grain (σ 1.6 levels).
  - The raw render is `scene/out/frame0_raw.png`; `post.py` writes `firstframe/frame0.png` from it.
- **Camera and dimensions are unchanged:** `config/camera_ff.json` is byte-identical (md5 4bbffe8e…, snapshot in `scene/out/camera_ff_before_photoreal.json`).
- **Pipeline:** build/render with build.py, writing to `scene/out/frame0_raw.png`, then run `python scene/post.py scene/out/frame0_raw.png firstframe/frame0.png`.
- **Known limits:**
  - The vise is still a blocky proxy (no casting draft or rounded body).
  - The bench has no wood grain or dents.
  - The scene is sparse; there are no props.

## Update: 3/4 colourful workshop (current; the front version is kept as firstframe/frame0_v003_front.png)
- **Camera** (orbit mode): azimuth 40° (front-right), pitch 20° down, 60 mm lens, 1.62 m to target (0.06, 0, 0.17).
  - Depth of field: f/4, focused on the strip mid-point.
  - `camera_ff.json` is rewritten. Pinhole vs Blender agree to 5e-5 px.
  - Projection test includes "bend_room" points at ±150 mm tip deflection carrying 12 weights; all are in frame.
- **Previous version kept:**
  - frame: `firstframe/frame0_v003_front.png`
  - camera: `config/camera_ff_v003_front.json`
  - params: `scene/out/scene_params_v003_front.json`
- **Look:**
  - oak bench with procedural grain and tool-mark bump;
  - terracotta pegboard at y = 0.56 m;
  - teal enamel vise with chips;
  - warm key light, cool rim light, light-linked strip fill;
  - out-of-focus props: screwdriver, mallet, cable coils, jar.
- Physics objects and dimensions are unchanged.
- **Limits:**
  - The props and vise are simple primitives, so the look is closer to a stylised "toy" render than true photoreal.
  - The wood grain reads as regular stripes.
  - The pegboard holes are a regular pattern, though not measurement marks.

## Animation (role D, v001) — scene/build_anim.py
- **Build:** `../_tools/with_render_lock.sh blender -b --factory-startup --python scene/build_anim.py -- --out scene/out/ruler_anim.blend --deformation solver/out/deformation.npz [--render-frames 0,46,114,240 --render-dir scene/visuals --render-prefix v00N --samples 32]`
  Opens `scene/out/ruler_ff.blend` (rebuild with build.py first if scene_params changed). Writes `<out>.blend`, `<out>_strip.pc2`, `<out>_checks.json`.
- **Test input (analytic, NOT solver):** `python scene/anim/make_test_deformation.py` → `scene/anim/test_deformation.npz`.
- **npz interface read:** `mid_surface_m` (N,301,11,3) world m, node order [s from bottom end, width −X→+X] (also accepts `positions`/`mid`/`x`, or (N,3311,3));
  `t_s` (N,) (default k/120 s; resampled linearly onto the 1201-sample grid 0..10 s if different); optional `weight_times_s`, `tip_load_N`, `buckled`, `source`.
  Schedule priority: npz `weight_times_s` → `meta.json` (`weight_times_s` | `loading_schedule_s` | `weight_place_times_s` | `loading.weight_times_s`) → default 1.0+0.6k s, k<7.
- **Bake contract:** scene frames 0..240, frame n ↔ t=n/24 s; 5 samples/frame. `RUL_Ruler` Mesh Cache modifier `RUL_StripCache` (PC2, FRAME mode, frame_scale 5, linear).
  Solid vertex = mid node ± t/2 · deformed unit normal (thickness rigid). `RUL_TipFrame` empty = top-centre mid node, axes (width, normal, tangent); straight strip ↔ identity at (0,0,0.375).
  `RUL_Platform*` parented to it. Weights `RUL_Weight11..05` transferred (top first); world keys while on stack/in gripper, then Child Of `RUL_RideTip` (influence 0→1 CONSTANT at seat time) with basis = seat pose in tip frame (50 µm gaps, evaluated bevel extent).
  Arm `RUL_Arm*`: column at (0.25,0.11), 2×0.27 m SCARA, quill, wrist, side-clamp gripper (palm 34 mm beside the weight so the 40 mm peg passes the weight centre); all keys computed (no hand animation).
- **Peg:** 40 mm (lead decision; scene_params updated, build.py re-run, camera_ff.json camera block unchanged).
- **v002 arm relocation:** the SCARA is now ceiling-mounted at xy (0.25, 0.11). The column runs from z = 0.66 to the mount at 1.6, and the links sit in the z = 0.66 plane. The camera frame top is at z ≈ 0.58 over the working area, so the column, carriage and links are out of frame; this is checked every 4th frame (`arm_column_links_out_of_frame`).
  - The only parts in frame are a Ø6 mm telescoping quill (`RUL_ArmQuill`, Z-scale keyed), the wrist, the gripper and the weight.
  - Finish is dark satin.
  - The 0.6 s cycle is unchanged, with smoothstep acceleration and deceleration on every phase.
- **v003 (built against the real solver output `solver/out/deformation.npz` + `meta.json`):**
  - **Feeder:** `RUL_Feeder*` is a dark-satin ring of r = 55 mm at z = O + 7 mm.
    - It carries 7 pin-and-disc pedestals at 0–240° (40° apart), placed in the order [0, 2, 4, 6, 1, 3, 5], so consecutive weights are at least 80° apart.
    - The ring is open toward −Y (the sway side and the camera side).
    - An L-bar runs to a post at (0, 0.12).
    - `RUL_FeedWeight0..6` (no bevel, exact 4.5 mm) sit on the pedestals. Each pedestal top is the weight's seat bottom + 4 mm.
    - Each weight slides radially, at constant height, and its slot passes over the peg; it is then lowered 4 mm onto the stack.
    - The bench stack is 12 untouched spares.
  - **Seat poses:** the undeformed seat bottom is O_z + slot_offset_j + 10 µm·(j+1), from meta.json.
    - `RUL_TipFrame` is keyed with `tipbody_matrix_world`, so platform objects and seated weights use undeformed-world bases.
  - **Gripper program:** waypoints in world or tip-relative coordinates, with smoothstep between them.
    - Carry: close at t0−0.20 s, slide in to t0−0.045 s, lower to t0, hold to t0+0.15 s, then open.
    - The return is reverse slide → rise → chord above the remaining weights → descend.
  - **Render setting:** motion blur on, shutter 0.5, fed by the PC2 subframes.
- **v004 polish:**
  - **Gripper exit:** after the 7th release the gripper rises straight up to z = 0.70 between 4.87 and 5.42 s (peak about 0.7 m/s). It is fully out of frame from 5.42 s (`arm_out_of_frame_after_exit`).
  - **Link plane:** raised to z = 0.80.
  - **Final lowering:** stretched to 0.10 s, running t0 − 0.10 → t0. The carry starts at t0 − 0.24.
  - **Empty chord:** about 3.2 m/s. The 1.5 m/s cap does not fit the 0.6 s cycle, so it was left as is.
