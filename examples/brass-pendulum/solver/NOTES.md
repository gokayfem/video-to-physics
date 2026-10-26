# solver/ NOTES (dynamics implementation role)

Status (v004, CURRENT): **FITTED, ANCHORED on 0–4.0 s, with a fitted level tilt of the static scene.** See CHANGELOG
2026-10-25T23:06:55Z and 23:14:43Z. Gravity points exactly along −Z_world, and θ is measured from TRUE vertical.
`params/physics.json` sha256 dc20e99c…, `trajectory.csv` 8ff8bb05…. Regenerate with `solver/regen_fitted_params.sh`; the output is byte-reproducible.
The v003 and v002 sections below are superseded and kept for history.

## v004: anchored fit with level tilt
**Tilt convention** (stored verbatim in physics.json `level_tilt_rad.convention`):
* The static scene (table, stand, pivot hardware, backdrop and ALL cameras, i.e. everything in the camera.json/geometry.json stand frame) is
  rotated by **+level_tilt_rad about world +Y through P** (right-hand rule). In Blender: a parent empty at P with `rotation_euler = (0, +β, 0)`.
* PND_Swing is NOT tilted; it keeps `rotation_euler[1] = −θ` with θ measured from gravity.
* With β > 0, the stand's up axis in world is (0.01260, 0, 0.99992). It leans toward +X.
  In the stand/camera frame the bob's rest direction therefore sits at +β, toward image right.

**Sign verification** (`solver/tilt_check.py` → `fit_tilt/Ta_4.0/tilt_check.json`):
* R_y(β)ᵀ·bob(θ) = bob_stand(θ+β) to 1.1e-16.
* Blender `Euler((0,β,0)).to_matrix()` = R_y(β) to 6.4e-09.
* Anchor turning points in the stand frame (right mean / left mean / asymmetry):
  * data: +7.42 / -5.92 / **+1.50°**
  * projected model with the tilt: +7.49 / -6.04 / **+1.45°**
  * same model without the tilt: asymmetry +0.01°
* Mean u residual in the anchor: +0.18 px with the tilt, -7.35 px without (RMS 3.24 vs 8.12 px).

**Production values** (T_a = 4.0 s; uncertainties from the moving-block bootstrap, 40 resamples; the formal value is in brackets):
| quantity | value | sd |
|---|---|---|
| level_tilt_rad (β) | **0.01260 rad (+0.722°)** | 0.00099 (95 % 0.01103–0.01413) [0.00045] |
| L_eq | 0.47510 m | 0.00260 [0.00084] |
| model period | 1.3842 s | |
| L_c (ratio 12.30) | 0.47385 m | |
| r | 38.52 mm | |
| k (L_c in camera units) | 1.08055 | 0.00030 |
| **L_eff_m** | **0.43852 m** | |
| θ_init (from gravity) | -0.08447 rad (-4.84°) | 0.0027 |
| ω_init | 0.3793 rad/s | 0.0122 |
| Q | 1000 (**assumed**) | see damping |
* Correlations: L_eq–ω0 +0.73, L_eq–θ0 +0.64, ω0–β −0.52, L_eq–β −0.34. No flags. Reduced χ² 89 (it was 457 without the tilt).
* T_a sensitivity: β = 0.01164 / 0.01260 / 0.01327 rad and L_eq = 0.47353 / 0.47510 / 0.47303 m for T_a = 3.6 / 4.0 / 4.5 s.
* **Anchor RMS 3.24 px** (u 3.16, v 0.70; n ≡ 0 mod 4 frames 3.67 px).
* **R2** 13.93 px: FAIL. **R3** 27.27 px: FAIL. **Phase at 10 s** -0.082 cycles: FAIL (the 0.05 target).
* Supplementary n ≡ 0 mod 4: R2 13.52, R3 25.41 px. With the drift-free camera: 14.94/28.34 px.
* Production trajectory vs the fit model: 5.3e-11 rad. Tension 19.82–20.24 N; E(10 s)/E0 = 0.955. Self-checks pass.

**Damping**, fitted jointly with the tilt (`solver/tilt_damping_evidence.py` → `fit_tilt/damping_evidence.json`):
| T_a | β | γ (formal) | Q | ΔBIC | boot 95 % γ | boot 95 % Q | Q, n≡0 mod 4 | L_eq (MQ) | anchor | R2 | R3 | phase |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3.6 | 0.01164 | 0.0174 ± 0.0045 | 131 | 1159 | [0.0000, 0.0400] | [57, inf] | 123 | 0.47353 | 3.14 | 13.92 | 30.05 | -0.092 |
| 4.0 | 0.01260 | 0.0222 ± 0.0040 | 102 | 2488 | [0.0065, 0.0492] | [46, 349] | 99 | 0.47510 | 3.24 | 13.93 | 27.27 | -0.082 |
| 4.5 | 0.01327 | 0.0164 ± 0.0037 | 139 | 1763 | [0.0066, 0.0424] | [54, 345] | 172 | 0.47303 | 3.40 | 14.40 | 30.94 | -0.095 |
* M1 production candidate at T_a 4.0 s (`fit_tilt/Ta_4.0_prod`, second bootstrap): Q 102, boot 95 % γ [0.0000, 0.0359], so Q is [63, inf).
  γ is flagged UNIDENTIFIED (bootstrap relative sd 0.51). Its metrics: anchor 2.94, R2 16.43, R3 31.05 px, phase -0.088.
* **Decision: keep Q = 1000 (declared assumption).**
  * The point estimates (Q 131 / 102 / 139) are consistent and inside 100–6000.
  * But the bootstrapped Q does not stay inside the range: the 95 % lower bounds are 46–63, and γ = 0 is not excluded for T_a 3.6 s or the second 4.0 s bootstrap.
  * The formal 4–6σ overstates the evidence because the residuals are strongly correlated.
  * The cadence check (n ≡ 0 mod 4 frames: Q 123 / 99 / 172) agrees, so this is not a cadence artifact.
  * Switching to the fitted Q = 102 is one argument change in regen_fitted_params.sh, using the `Ta_4.0_prod` values.

**Divergence map** (corrected physics vs AI motion; turning points parabola-refined on the triangulated angle, measured from gravity):
| t_data [s] | A_data [°] | A_model [°] | ΔA [°] | Δt (model − data) [ms] | px | segment |
|---|---|---|---|---|---|---|
| 0.525 | +6.89 | +6.80 | +0.09 | -14 | 0.7 | anchor |
| 1.204 | -6.96 | -6.79 | +0.17 | +4 | 2.0 | anchor |
| 1.897 | +6.84 | +6.78 | +0.07 | -1 | 0.7 | anchor |
| 2.593 | -6.60 | -6.77 | -0.17 | +1 | 2.1 | anchor |
| 3.289 | +6.37 | +6.75 | -0.39 | -8 | 4.4 | anchor |
| 3.985 | -6.36 | -6.74 | -0.38 | -6 | 3.6 | anchor |
| 4.684 | +7.47 | +6.73 | +0.74 | -17 | 8.6 | divergence |
| 5.378 | -8.43 | -6.72 | +1.70 | -13 | 17.2 | divergence |
| 6.073 | +9.29 | +6.71 | +2.58 | -21 | 28.5 | divergence |
| 6.765 | -10.76 | -6.70 | +4.06 | -25 | 43.4 | divergence |
| 7.490 | +10.04 | +6.69 | +3.35 | -53 | 37.3 | divergence |
| 8.196 | -8.54 | -6.68 | +1.85 | -71 | 22.6 | divergence |
| 8.901 | +7.90 | +6.67 | +1.23 | -78 | 19.4 | divergence |
| 9.621 | -7.44 | -6.66 | +0.78 | -111 | 16.6 | divergence |

### Commands (v004)
```
for T in 3.6 4.0 4.5; do .venv/bin/python solver/fit_projected.py --t-anchor $T --fit-tilt --damping auto --nboot 40 --out solver/out/fit_tilt/Ta_$T; done
.venv/bin/python solver/fit_projected.py --t-anchor 4.0 --fit-tilt --damping fitted --nboot 40 --out solver/out/fit_tilt/Ta_4.0_prod
.venv/bin/python solver/tilt_damping_evidence.py
.venv/bin/python solver/tilt_check.py --fit-dir solver/out/fit_tilt/Ta_4.0
.venv/bin/python solver/make_fit_provenance.py --fit-dir solver/out/fit_tilt/Ta_4.0 --evidence solver/out/fit_anchored/anchor_evidence.json \
     --damping-evidence solver/out/fit_tilt/damping_evidence.json --tilt-check solver/out/fit_tilt/Ta_4.0/tilt_check.json
./solver/regen_fitted_params.sh
.venv/bin/python solver/run.py && .venv/bin/python solver/selfcheck.py
.venv/bin/python solver/visuals/make_fit_visuals.py --version v004 --fit-dir solver/out/fit_tilt/Ta_4.0 --status "fitted on opening 0–4.0 s; correct physics afterwards; table tilt fitted"
.venv/bin/python solver/visuals/make_visuals.py --version v004
```

### v004 visuals
| file | shows | still wrong / assumed |
|---|---|---|
| `v004_fit_overlay_f000-240.mp4` | Reference with the projected corrected physics (tilted stand frame) drawn over it. Labels: anchor/corrected tag, table tilt +0.72°, Q = 1000 assumed, θ relative to gravity. | Anchor 0.6–4 px. After 4 s the AI amplitude growth separates the two by up to 43 px (frame 163). |
| `v004_fit_residuals.png` | u/v residuals with the anchor window shaded, and measured vs model θ (both from gravity). | |
| `v004_fit_overlay_frames.png` | Turning-point crops 12/29/46/96 (0.6/2.0/0.7/3.6 px), growth peaks 146/163/179, and 212/228/240. | |
| `v004_solver_preview_f000-240.mp4`, `v004_turning_points.png` | Solver side view (gravity frame). | Not tilted, not projected. |

## v003 (superseded)
Status (v003, superseded by v004): **FITTED, ANCHORED on the opening window 0–4.0 s.** This follows the user decision recorded in CHANGELOG 2026-10-25T23:06:55Z.
Correct passive physics then runs unchanged to 10 s, and the motion after T_a is reported as a divergence map, not fitted.
`params/physics.json` sha256 1bb8e14e…, `trajectory.csv` sha256 db502b2b…. Regenerate with `solver/regen_fitted_params.sh`.
The v002 section below (fit on t < 7.5 s) is superseded and kept for history.

## v003 anchored fit
**Choice of T_a.** The criterion was fixed before looking at R2/R3. Take the first half-cycle whose amplitude rises more than 3σ above the
previous one (image half-cycle amplitudes from measurement/summary.json).
* Half-cycle amplitudes fall from 5.77° (0.86 s) to 5.45° (3.64 s). The next one is 5.93° (4.33 s), a **14σ** jump.
* That growing half-cycle starts at the turning point at **3.987 s**, so T_a = 4.0 s (frame 96).
* By turning point, the right side goes 5.86 → 5.78 → 5.50 and then 6.46° at 4.68 s. The left side goes −5.67 → −5.48 → −5.40 and then −6.99° at 5.38 s.
* Period wander is already present inside the window (full periods 1.370 / 1.396 / 1.417 / 1.397 s), and no window can exclude it.
Visual: `visuals/v003_anchor_window_evidence.png`.

**T_a sensitivity** (anchor RMS on t ≤ T_a; R2 and R3 computed exactly as declared, on t < 7.5 and t ≥ 7.5 s):
| T_a [s] | L_eq [m] (boot sd) | T_model [s] | anchor RMS px | R2 px | R3 px | phase@10 s [cyc] | damping (M1 in window) |
|---|---|---|---|---|---|---|---|
| 3.5 | 0.47684 ± 0.0045 | 1.3869 | 8.06 | 13.80 | 25.49 | -0.077 | γ=0.0222, Q=102; boot95 γ [1.16e-13, 0.0773]; adopted: False |
| 3.6 | 0.47751 ± 0.0045 | 1.3879 | 8.07 | 13.77 | 24.25 | -0.073 | γ=0.0203, Q=112; boot95 γ [7.01e-11, 0.0811]; adopted: False |
| 4.0 | 0.48297 ± 0.0062 | 1.3958 | 8.41 | 14.94 | 16.35 | -0.038 | γ=0.0433, Q=52; boot95 γ [0.0374, 0.1]; adopted: False |
| 4.5 | 0.47320 ± 0.0036 | 1.3815 | 8.97 | 14.90 | 32.61 | -0.101 | γ=0.0518, Q=44; boot95 γ [0.0551, 0.118]; adopted: False |
R3 happens to be lowest at 4.0 s, but T_a was **not** chosen by R3. L_eq has a T_a systematic of ±1 % (0.4732–0.4830 m), similar to the bootstrap sd.

**Production values (T_a = 4.0 s, Q = 1000 assumed).**
* L_eq 0.48297 ± 0.0062 m (formal 0.0017). Model period 1.3958 s.
* L_c 0.48169 m; r 39.16 mm (ratio 12.30 measured). k = 1.08154 ± 0.00058 L_eff, so **L_eff_m = 0.44538 m**.
* θ0 -0.08403 ± 0.0069 rad (-4.81°); ω0 0.4299 ± 0.0257 rad/s.
* Correlations: L_eq–θ0 +0.70, L_eq–ω0 +0.66. No flags. Reduced χ² 457.
* **Anchor RMS 8.41 px** (u 8.27, v 1.52; n ≡ 0 mod 4 frames 8.80 px).
* **R2** 14.94 px: FAIL. **R3** 16.35 px: FAIL. Phase at 10 s -0.038 cycles: PASS (< 0.05).
* Supplementary n ≡ 0 mod 4: R2 15.49 px, R3 15.58 px. With the drift-free camera: 15.81/18.94 px. Cadence offsets are disclosed and nothing is retimed.
* Production trajectory vs the fit model: 5.4e-11 rad. Tension 20.80–21.31 N; E(10 s)/E0 = 0.955. Self-checks pass.

**What the anchor misfit is.** Almost all of the 8.4 px comes from a **left/right swing asymmetry**. In the world frame, the right turning points
are about 7.6° and the left ones about −5.6 to −6.2°. Diagnostic D3, an equilibrium-direction offset th_off, is non-physical with gravity vertical
and is **not adopted**. It gives **+0.72°** (0.0126 rad) and brings the anchor RMS to **3.24 px**. A free pivot dz or swing-plane azimuth does not help
(8.41 / 8.47 px). Possible readings, for the lead and camera role to decide:
(a) the calibrated vertical (camera roll 0.23°, taken from the stand edges) is off by about 0.7°;
(b) the AI scene is internally inconsistent: the bob's rest direction is not along the stand's vertical. Realising it in the scene would mean
tilting gravity relative to the stand, i.e. a stand leaning about 0.7°.

**Damping (linear, free ≥ 0 in the window).**
* In the production geometry it is **not robust**.
  * T_a 3.5 / 3.6 s: the BIC prefers M1, but γ = 0.020 ± 0.009 and the bootstrap 95 % range includes 0 (Q ≈ 100–110, interval up to ∞).
  * T_a 4.0 s: identified (boot95 γ 0.037–0.100), but Q = 52 (95 % 22–60). That is below the plausible floor of 100. It is driven by the left-side amplitude dropping faster than the right.
  * n ≡ 0 mod 4 frames only (cadence check): γ = 0.048 ± 0.015, Q ≈ 47. This agrees, so the result is not a cadence artifact.
  * Decision: **Q = 1000 kept as the declared assumption.**
* Conditional (diagnostic) result: with th_off free, damping becomes consistent across windows.
  * T_a 3.6 s: Q 131 (γ 0.0174 ± 0.0045)
  * T_a 4.0 s: Q 102 (γ 0.0222 ± 0.0040)
  * T_a 4.5 s: Q 139 (γ 0.0164 ± 0.0037)
  * This sits at the low end of the plausible range (FORMULATION: desk toys 10²–10³). It matches the image half-cycle decay 5.77 → 5.45° over 2.8 s (γ ≈ 0.02 /s).
  * If the lead adopts the equilibrium correction, Q ≈ 100–140 can be re-evaluated as fitted (it would need a bootstrap). Not adopted here.

**Divergence map** (measured turning points, parabola-refined on the triangulated angle, vs corrected physics; per-frame distances are in
`fit_anchored/Ta_4.0/fit_projected.json` under `divergence_map`):
| t_data [s] | A_data [°] | A_model [°] | ΔA (data − model) [°] | Δt (model − data) [ms] | px at frame | segment |
|---|---|---|---|---|---|---|
| 0.525 | +7.61 | +7.28 | +0.33 | -25 | 3.8 | anchor |
| 1.204 | -6.24 | -7.27 | -1.03 | -6 | 10.9 | anchor |
| 1.897 | +7.57 | +7.26 | +0.31 | -1 | 3.3 | anchor |
| 2.593 | -5.88 | -7.24 | -1.37 | +1 | 14.1 | anchor |
| 3.289 | +7.09 | +7.23 | -0.14 | +3 | 2.8 | anchor |
| 3.985 | -5.64 | -7.22 | -1.58 | +4 | 16.1 | anchor |
| 4.684 | +8.20 | +7.21 | +0.99 | +4 | 12.0 | divergence |
| 5.378 | -7.70 | -7.20 | +0.51 | +8 | 5.1 | divergence |
| 6.073 | +10.01 | +7.19 | +2.83 | +11 | 30.9 | divergence |
| 6.765 | -10.04 | -7.18 | +2.86 | +16 | 29.8 | divergence |
| 7.490 | +10.77 | +7.16 | +3.60 | -11 | 40.6 | divergence |
| 8.196 | -7.81 | -7.15 | +0.66 | -19 | 6.6 | divergence |
| 8.901 | +8.62 | +7.14 | +1.48 | -26 | 17.0 | divergence |
| 9.621 | -6.71 | -7.13 | -0.42 | -48 | 2.6 | divergence |
After T_a the AI amplitude overshoots the physical one by up to +3.6° (40 px at 7.5 s) and then falls back. The timing stays within ±20 ms
until 9.6 s (−48 ms).

### Commands (v003)
```
for T in 3.5 3.6 4.0 4.5; do .venv/bin/python solver/fit_projected.py --t-anchor $T --nboot 40 --out solver/out/fit_anchored/Ta_$T; done
.venv/bin/python solver/arc_shape_check.py
.venv/bin/python solver/anchor_window_evidence.py --version v003
.venv/bin/python solver/make_fit_provenance.py --fit-dir solver/out/fit_anchored/Ta_4.0
./solver/regen_fitted_params.sh            # byte-reproducible (argv stored in physics.json generator.argv)
.venv/bin/python solver/run.py && .venv/bin/python solver/selfcheck.py
.venv/bin/python solver/visuals/make_fit_visuals.py --version v003 --fit-dir solver/out/fit_anchored/Ta_4.0 --status "fitted on opening 0–4.0 s; correct physics afterwards"
.venv/bin/python solver/visuals/make_visuals.py --version v003
```

### v003 visuals
| file | shows | still wrong / assumed |
|---|---|---|
| `v003_fit_overlay_f000-240.mp4` | Reference with the projected corrected-physics model over it, labelled "fitted on opening 0–4.0 s; correct physics afterwards". Each frame is tagged [ANCHOR (fitted)] or [corrected physics]. | Left turning points about 10–16 px off inside the anchor (asymmetry); 30–40 px divergence at the AI growth peaks around 6–7.5 s. Q assumed. |
| `v003_fit_residuals.png` | u/v residuals and θ (measured vs model), with the anchor window shaded. | |
| `v003_fit_overlay_frames.png` | Turning-point crops: anchor frames 12/29/46/96, AI growth peaks 146/163/179, and 213/230/240. | |
| `v003_anchor_window_evidence.png` | Amplitude onset (T_a), period wander, and T_a sensitivity. | |
| `v003_solver_preview_f000-240.mp4`, `v003_turning_points.png` | Solver side view at the v003 params. | Side view only. |

## v002 (superseded) 
Status (v002): **FITTED production params, generated by `solver/regen_fitted_params.sh`.** These come from the full projected
fit (`solver/fit_projected.py`) to `measurement/bob_track.csv` through `params/camera.json`, using camera/project.py
per-frame cameras. The production trajectory (`solver/out/trajectory.csv`) was produced at those params. The solver self-checks
pass. **Reference fidelity R2/R3 FAIL.** The failure is attributed to reference defects, and the model was not bent to match them.
The v001 sections below describe the earlier placeholder stage and are kept for history.

## v002 projected fit: results (fit t < 7.5 s, hold-out t >= 7.5 s; pivot fixed at P_ring)
Model: bob = k (sin th, 0, -cos th) in camera units (L_eff), with θ(t) from the production RK4 solver. Free parameters:
L_eq, θ0, ω0, k, γ ≥ 0. Weights come from the per-frame sigma_u/v (method A). The swing plane is at azimuth 0 (assumption).
| quantity | value | sd (bootstrap, 40 moving blocks) | sd (Jacobian, scaled by sqrt(chi2_nu)) | provenance |
|---|---|---|---|---|
| L_eq | 0.47564 m | 0.00527 (95 % 0.4663–0.4864) | 0.00110 | model-conditioned (period, g assumed) |
| period (model, finite amplitude) | 1.3852 s | | | model |
| L_c = L_eq/(1+0.4/12.30²) | 0.47438 m | 0.00526 (L_eq) ⊕ 9.2e-05 (ratio) | | model-conditioned |
| r = L_c/12.30 | 38.57 mm | | | model-conditioned |
| k = L_c in L_eff units | 1.08032 | 0.00079 | 0.00069 | fitted (image geometry) |
| L_eff_m = L_c/k | 0.43911 m | ≈ 0.0049 | | model-conditioned |
| θ0 (t = 0) | -0.09170 rad (-5.25°) | 0.0086 rad | 0.0020 | fitted |
| ω0 (t = 0) | 0.4362 rad/s | 0.0469 | 0.0077 | fitted |
| damping | γ = 0 at the bound (M1 BIC 277551 vs M0 277545): **not identifiable** | | | → Q = 1000 **assumed** (range 300–6000) |
* Correlations (Jacobian): L_eq–θ0 +0.72, L_eq–ω0 +0.67, θ0–ω0 +0.48, k is uncorrelated with the others. No parameter is flagged.
  Reduced χ² = 795, so the model is inadequate for this reference. The bootstrap sd is therefore the honest uncertainty;
  it is about 5× the formal sd.
* With Q = 1000 fixed, the production fit re-estimates L_eq, θ0, ω0 and k. Q sensitivity: Q = 300 gives RMS 14.03/27.57 px,
  Q = 6000 gives 13.39/27.16 px. L_eq changes by < 2e-4.
* **R2** fit RMS **13.55 px** (u 13.31, v 2.57) against a target of 2.0 px: **FAIL**. **R3** hold-out RMS **27.25 px**
  (u 27.01, v 3.65) against 3.0 px: **FAIL**. Phase at 10 s is **-0.084 cycles** against 0.05: **FAIL**.
* Supplementary metrics (CHANGELOG): residuals use the per-frame camera, so the measured drift is removed. With the drift-free camera instead, RMS is
  14.28/29.02 px. On n ≡ 0 mod 4 frames only, RMS is 13.06 px (fit, 45 frames) and
  25.01 px (hold-out, 16 frames). Mean u residual per cadence phase (n mod 4 = 0..3): -6.65, -7.14, -6.86, -6.66 px.
  The measured cadence offsets are disclosed (+0.277/+0.113/−0.094/−0.296 frames). The simulation is **not** retimed.
* Attribution: in the first 3.6 s the projected model follows the reference to about 1 px at the right turning points. The left turning points
  are about 1.5° short in the reference (asymmetric swing; diagnostic D3, a free equilibrium offset, gives +0.60°, RMS 12.25 px). After that,
  the reference amplitude grows +62 % (3.6 → 7.1 s) and its period wanders (1.357–1.434 s). A passive pendulum cannot do either.
* **Arc-shape (apparent rotation point) effect**, dynamics-free (`arc_shape_check.py`, free θ per frame):
  hinge at P_ring gives RMS 1.29 px (radial 1.22, max 3.75 px, k = 1.0804 ± 0.0001).
  A free hinge gives 1.15 px (dz = -0.243 ± 0.014 L_eff, which is degenerate with k on this short arc).
  The arc-shape defect therefore costs about 0.6 px RMS in quadrature (max 3.8 px radial), against a 13.6 px dynamic misfit.
  k = 1.080 is slightly above the lead's expected 1.05–1.07.
* Diagnostics (not production): D1 free pivot dz gives dz = -0.260 L_eff and RMS 13.55/27.18 px.
  D2 (+φ) gives φ = 0.019 rad and RMS 13.54 px. Freeing the pivot does not fix the misfit.
* Production trajectory at the fitted params: θ matches the fit model to 5.9e-11 rad at frames, tension 19.86–20.38 N, max per-sample
  ΔE/E0 −2.2e-8 (monotone decay), E(10 s)/E0 = 0.955.
* Camera role: re-run `camera/write_params.py` with **L_eff_m = 0.439113 m**, which is now in params/physics.json `L_eff_m`.

### Fitted-params regeneration (byte-reproducible; delivery F1)
```
.venv/bin/python solver/fit_projected.py --nboot 40      # -> solver/out/fit_projected/fit_projected.json (+ series .npz)
.venv/bin/python solver/arc_shape_check.py               # -> solver/out/fit_projected/arc_shape.json
.venv/bin/python solver/make_fit_provenance.py           # -> params_fit_provenance.json (deterministic; hashed in physics.json generator.inputs_sha256)
./solver/regen_fitted_params.sh                          # exact make_params.py argv (also stored in physics.json "generator.argv")
.venv/bin/python solver/run.py && .venv/bin/python solver/selfcheck.py
.venv/bin/python solver/visuals/make_fit_visuals.py --version v002 && .venv/bin/python solver/visuals/make_visuals.py --version v002
```
`generated_utc` is no longer inside physics.json. It is in the sidecar `solver/out/make_params_run.json`, and two consecutive
regenerations give an identical physics.json (sha256 86ba7d90…). Rerunning the whole chain (fit → arc → provenance → params) also reproduces it bit-for-bit (provenance sha256 806f4d0b…; trajectory f18d7894…).

## Visuals (CONTRACT v2), in solver/visuals/
| file | what it shows | what's still wrong / assumed |
|---|---|---|
| `v002_fit_overlay_f000-240.mp4` (960x544, 24 fps, 241 frames) | The reference with the fitted passive model projected over it through the per-frame camera (magenta bob outline and wire from P_ring), the measured bob centre (green x), and P_ring (yellow +). Labelled "fitted model projected – draft" with frame, t, θ and the model-to-measured distance. | Good to about 1 px early on. It drifts apart after 3.6 s because the reference amplitude grows and its period wanders (reference defects). Q is assumed. |
| `v002_fit_overlay_frames.png` | Full-resolution crops at the model turning points (frames 12/29/46/62), frame 171 (reference amplitude peak), and the hold-out frames 212/228/240, with the per-frame distance. | Up to 34 px at frame 240 (hold-out). |
| `v002_fit_residuals.png` | u/v residuals vs t with the fit/hold-out line, n ≡ 0 mod 4 frames marked, the ±3σ band, and measured θ (ray ∩ swing plane) vs model θ. | Shows the growth-driven misfit directly. |
| `v002_solver_preview_f000-240.mp4`, `v002_turning_points.png` | Same as v001, at the FITTED params. | Side view, not projected. |
| `v001_solver_preview_f000-240.mp4` (960x544, 24 fps, 241 frames, sha256 below) | Solver trajectory at true time (t = n/24, no retiming). Side view in world X–Z, with θ(t) and E/E0 strips and a moving cursor. Burned-in labels: version, t, frame, θ, ω, status "isolated component – draft", params hash, "PLACEHOLDER". | The params are placeholders: release from rest at +6°, T ≈ 1.25 s, Q = 1000 assumed. The view is a side view, not the reference camera. Nothing is compared to the reference yet. |
| `v001_turning_points.png` | Top row: reference frames (observed, unprocessed crop) at the frames nearest the solver's turning points (t = 0, 0.625, 1.25, 1.875, 8.75, 9.375, 10 s). Bottom row: the solver pose at the same times. | **Phase mismatch is visible.** At frame 0 the reference bob is left of the pivot and moving right, so the clip does not start at a +6° turning point. In the reference, frames 15/30/45 are not clean turning points of the placeholder model either. This is expected for placeholders. The fit (θ_init, ω_init free) fixes it. Amplitudes are not comparable (no camera projection). |
| `v001_informal_bobquick_fit.png` | INFORMAL fit of the camera role's quick bob track (u, v vs model, residuals, hold-out). | The model fails (12 px RMS). It shows a period ≈ 1.40 s, amplitude growth or modulation, and camera drift. Not validated data, not used for params. |
| `../out/selfcheck.png` | P1 step-halving ladder (slope 4), conservative energy drift (~1e-14), damped energy balance, θ(t). | Developer check only. |
| `../out/fit_tests/fit_*.png` | Synthetic fit tests: data, model, residuals, hold-out split at 7.5 s. | Synthetic data, not the reference. |

## Model (matches physics/FORMULATION.md, model id `planar_compound_sphere_rigid_massless_rod_v1`)
`I_P θ'' = −m g L_c sin θ − c θ' [− k_q|θ'|θ' − τ_c tanh(θ'/ω_s)]`, with `I_P = m L_c² + 0.4 m r²` and
`c = I_P ω0/Q`. The k_q and Coulomb terms are implemented but OFF (value 0) in params.
The bob centre is `P + L_c (sin θ, 0, −cos θ)`. Tension on the bob along the wire is `T = m (g cos θ + L_c ω²)`.
`PE = m g L_c (1 − cos θ)`, `P_diss = c ω² (+ k_q|ω|³ + τ_c ω tanh)`.
`pendulum.py` also accepts a flat legacy dict (point_mass / solid_sphere / explicit, plus wire linear density).
Tests use it for generality.

## Integrator
Classical RK4 with a fixed step. Production h = 1/3072 s, which is 32 substeps per 1/96 s output, so every
output sample is an integration node. Why this method: a known order (4), deterministic and bitwise
repeatable, no dense-output interpolation, and the error sits at round-off level.
Dissipated work W(t) is integrated as a third state, so `E − E0 + W` tests the integrator itself.
**Non-smooth terms** (|ω|ω, Coulomb): when ω changes sign inside a step, the step is split at the turning
point, located by Illinois regula falsi on the RK4 sub-step. Without the split, the k_q case converged at
order ~3.4. With it, the order is 4.00.

## Commands (from ROOT)
```
.venv/bin/python solver/make_params.py --status placeholder ...   # regenerates params/physics.json (full command below)
.venv/bin/python solver/run.py                                    # -> solver/out/trajectory.csv, trajectory_meta.json
.venv/bin/python solver/selfcheck.py                              # -> solver/out/selfcheck.json/.png (exit 0 = all pass)
blender -b --factory-startup --python solver/test/make_test_scene.py -- --L_c <L_c> --r <r> --out $PWD/solver/test/stand_in.blend
blender -b solver/test/stand_in.blend --python solver/bake_blender.py -- --trajectory solver/out/trajectory.csv \
        --bob-object TEST_Bob --out $PWD/solver/test/stand_in_baked.blend --report solver/out/bake_check.json
blender -b solver/test/stand_in_baked.blend --python solver/bake_blender.py -- --trajectory solver/out/trajectory.csv \
        --bob-object TEST_Bob --check-only --report solver/out/bake_check_reopened.json
.venv/bin/python solver/test/test_fit.py --nboot 20                # synthetic fit tests -> solver/out/fit_synthetic_tests.json
.venv/bin/python solver/fit.py --mode angle|track|projected [--nboot 40] [--pivot-px U V] [--ratio R]
.venv/bin/python solver/visuals/make_visuals.py --version v001
```
Placeholder params command:
```
.venv/bin/python solver/make_params.py --status placeholder --status-note "..." --T-meas 1.25 --sigma-T 0.02 --prov-T assumed \
  --source-T "eye estimate, CONTRACT (unverified)" --theta-amp-deg 6 --sigma-amp 0.035 --prov-amp assumed \
  --source-amp "eye estimate, CONTRACT; prompt intent ~4 deg" --ratio 13 --sigma-ratio 2.6 --prov-ratio assumed \
  --source-ratio "physics/FORMULATION.md s6 rough sheet estimate (+/-20%), awaiting measurement" --Q 1000 \
  --theta-init-deg 6 --omega-init 0 --prov-init assumed --source-init "placeholder: release from rest at t=0"
```
For the rig role: run `bake_blender.py` on the real scene with `--bob-object <bob name>` and `--out`/`--save`.
It refuses a scene whose timing is not (24 fps, 0..240) unless `--fix-scene-timing` is given. It also refuses a
parented PND_Swing.

## Results (placeholder params, params sha256 dd7e90e9…)
* Derived: L_eq 0.387602 m, L_c 0.386686 m, r 29.75 mm, m 0.9370 kg (assumed brass), I_P 0.140442 kg m²,
  c 7.064e-4 N m s (Q = 1000).
* **P1**: step-halving ladder h = 1/192 … 1/3072 s, max |Δθ| from 2.9e-7 down to 4.5e-12 rad. Observed order is
  4.00 in all four test models (production, c+k_q, compound with wire mass, smoothed Coulomb).
  The production pair h/2 vs h/4 differs by max 1.7e-14 rad, against a tolerance of 1e-7.
* **P3**: period from refined zero crossings vs the elliptic `4 sqrt(I_P/(m g d)) K(sin²(θ0/2))`, at
  0.5/2/6/15/60°, point-mass and compound models: max relative error 6.0e-14 (tolerance 1e-7).
* **P4**: conservative drift max 1.9e-14 E0 (tolerance 1e-8). Damped balance |E − E0 + ∫P| max 1.8e-13 E0
  (tolerance 1e-7). Analytic dE/dt ≤ 0 at all samples. For production params E(10 s)/E0 = 0.951.
* **P7**: tension 9.139–9.289 N (> 0). **P8**: θ(0), ω(0) exact (0 error). Grid t = k/96 exact. Bitwise repeatable.
* Linear damping theory (0.5°, γ = 0.05/s) vs the closed form: 9e-5 of amplitude. This is the expected
  sin θ nonlinearity, not solver error.
* **Bake / P6 developer check (stand-in scene)**: on the reopened .blend, 961 samples (every frame plus subframes
  0.25/0.5/0.75): max angle error 5.25e-7 rad (Hermite estimate ≈ 5e-7; tolerance 1e-5). On integer frames the
  error is 8.3e-9 rad. Bob position error 2.1e-7 m, |L − L_c| 3.8e-8 m, out of plane 0, pivot drift 0,
  θ0 error 1.0e-9 rad. Euler sign verified: `rotation_euler[1] = −θ`. Keyframe arrays sha256 is identical
  across two bakes (fb910b77…). Handles are written as FREE and read back bit-identical (no auto
  recalculation).
* Custom props on PND_Swing: `sim01_params_sha256`, `sim01_trajectory_sha256`, `sim01_solver_source_sha256`,
  `sim01_bake`, `sim01_bake_sha256`.

## fit.py (implemented, synthetic-tested; NOT yet run on the reference)
See the docstring. Fit on t < 7.5 s, hold out t ≥ 7.5 s. Nested M0/M1/M2 (none / γ / γ+β), chosen by BIC
with a 2-BIC preference for simpler models. Damping is bounded ≥ 0. γ = c/(2 I_P) (FORMULATION convention).
Uncertainty comes from a Jacobian covariance (formal and residual-scaled) and a moving-block bootstrap
(block = 1 period). A parameter is flagged when it is AT_BOUND (an approximate one-sided 95 % upper limit is
given), UNIDENTIFIED (relative sd > 0.5), or STRONGLY_CORRELATED (|r| > 0.95). Growth is flagged from an
unconstrained damped sinusoid (λ < −3σ) and from the peak envelope. Phase error is reported per crossing
and averaged over the last period.

**Synthetic tests** (`solver/test/test_fit.py --nboot 20`, 75 s; all verdicts true; `solver/out/fit_synthetic_tests.json`):
| case | truth | chosen | recovered (est ± sd, z) | fit / hold-out RMS | end phase error |
|---|---|---|---|---|---|
| A track similarity, Q = 160, σ 0.4 px | L_eq 0.38760, θ0 5.5°, ω0 0.03, γ 0.01591 | M1 | L_eq 0.387564 ± 5.1e-5 (−0.7), θ0 0.09204 ± 0.0027 (−1.5), γ 0.01558 ± 3.1e-4 (−0.5) | 0.55 / 0.61 px | −1.5e-4 cyc |
| B angle, Q = 1000, σ 1 mrad | γ 0.002515 | M1 | L_eq 0.38766 ± 8e-5 (0.7), γ 0.00185 ± 5.7e-4 (−1.2); ω0 flagged UNIDENTIFIED (true 0) | 1.04 / 1.01 mrad | 7e-4 cyc |
| D projected through a synthetic camera, Q = 300 | L_eq 0.38760, γ 0.008386 | M1 | L_eq 0.387595 ± 2.2e-5 (−0.2), θ0 (0.7σ), γ 0.00827 ± 4.2e-4 (−0.3); no flags | 0.59 / 0.54 px | 1.3e-4 cyc |
| C fake growing amplitude (analytic e^{+0.03t}) | non-physical | M0 | γ = 0 in M0/M1/M2 (constraint active); flagged `AMPLITUDE_GROWTH` (λ = −0.0301 ± 0.0003) and envelope slope +0.0315/s | 3.3 / 8.2 px (expected misfit) | n/a |
| E undamped truth | γ = 0 | M0 (BIC 392.1 vs 397.9) | damping not reported as a value | - | - |
In track mode θ0, R_px and c_v are flagged STRONGLY_CORRELATED (|r| ≈ 0.999), as expected from arc-curvature
identification. Truth still lies within 3σ.

**Informal run on `camera/out/bob_quick.csv`** (the camera role's quick track, NOT the validated measurement track;
not used for params; `solver/out/informal/`, visual `visuals/v001_informal_bobquick_fit.png`):
the static-similarity rigid-pendulum fit **fails** (RMS 12.3 px fit, 23.8 px hold-out, M0 chosen, damping at 0).
Observations:
(i) Zero-crossing period ≈ **1.40 s** (1.37–1.41 per cycle), not the 1.25 s eye estimate. If confirmed,
L_eq ≈ 0.48 m (model-conditioned).
(ii) The pixel amplitude is not monotonic and grows strongly around t ≈ 6–7.5 s (unconstrained sinusoid
λ = −0.076 ± 0.004 /s, i.e. growth).
(iii) v drifts by about +20 px over 10 s (camera push-in).
The zoom alone (≈ 5–6 % per 5 s) cannot explain (ii). This points to a probable reference defect, still to be
confirmed on a drift-compensated track.

## Open issues / requests
1. **Camera moves in the reference.** Geometry notes a push-in or zoom (focal 2830 → 2990 px, f0 → f120).
   `track` mode assumes a static similarity map. Measurement should deliver a drift-compensated track (the
   static landmarks suggest this exists) or a per-frame camera. Otherwise the zoom will leak into the
   amplitude and damping estimates. A slow zoom makes the apparent amplitude GROW. Do not read that as physics.
2. **Request to camera role**: params/camera.json layout. fit.py accepts `K` or `fx_px/fy_px/cx_px/cy_px`
   (or `focal_mm` + `sensor_width_mm` + `resolution`) and `R_world_to_cam` + `t_world_to_cam` (or a 4x4
   `world_to_camera`), with `camera_convention` "opencv" or "blender". Pixel coordinates follow the CONTRACT
   corner-origin convention. A per-frame camera would need an extension (not implemented).
3. **Ratio L_c/r**: placeholder 13 (physics). geometry.json measures bob_radius/L_c = 0.0812 ± 0.003, so the
   ratio is 12.3. Once that is confirmed and tagged as measured, regenerate params with `--ratio 12.3`.
   L_c changes by about 1e-4 relative.
4. In `track` mode, θ0 is strongly correlated with R_px and c_v (amplitude only comes through arc curvature).
   The flags say this. A measured pivot pixel (`--pivot-px`, e.g. the wire exit (1087,124) from geometry)
   breaks the degeneracy.
5. The reference starts mid-swing (bob left of the pivot at frame 0, moving right, per the turning-point
   visual). θ_init/ω_init must come from the fit. The placeholder release from rest at +6° is wrong in phase.
6. fit.py never writes params/physics.json. After a well-conditioned fit, params are regenerated with
   `make_params.py --status fitted` from the fitted T, amplitude, θ_init, ω_init (and Q only if decay is
   significant), after the lead approves.
7. Coulomb friction uses tanh smoothing (no stick phase). It is not used in production.

## sha256 (at v004 hand-off)
```
dceda20a31b426ab695bac71e7761acf63c10ac6784f09ec2123844eb0f31f7f  anchor_window_evidence.py
1fe50532f7179d39ba8b445c9937bc3abe1bc73a99b4a2b7df39507fc7c091ef  arc_shape_check.py
5546c76b7012965aefaa92b429c48432fd78618cdfa18d6af6f9086c24337b4a  bake_blender.py
b7f7c7bb54914aa3ebb9f0118d5388bef114277235aa6fdc463460ce4da573ec  fit.py
68d4ee64b8778f44e453bc71475d2b2f786f1c25274960e6555af77264ff8ce5  fit_projected.py
793889b2e56cf5de04a3eba71f469711ad1bf42cb2213a8bb4aa43d38aba5bfe  make_fit_provenance.py
26413761c3f780a557246818c0ac490cb778558878b8793f6b0a1f297d43ec3c  make_params.py
21f44cb6a96854185daf2a8cacc6a2f2d0c8bca14fb62fec9900624df43884be  pendulum.py
bb1ecd576fd8ad48f1fb248112d7e73d24e8be7c255a41372251f734f2b59e1c  run.py
42a0852894d65c0b1d2203ae077c5bdf1ec96ea0f750795a9608bea67d10f0f4  selfcheck.py
ed4053e5ed33935fc0e5049f878d2c4b1e3ec978065de2f58bb3c774b81714c1  tilt_check.py
b9ecdd21b1bba02518bae34163b10099942f388331462ac5e01e51ee8ead6d2a  tilt_damping_evidence.py
736e654c6db772ff858c21d66c5114f887c4d834644b78cc92ce7d9687e47174  regen_fitted_params.sh
34e43f0b72c75edac0f77037c5a8a3f0706705bb2252336df27c03c828bc8dff  visuals/make_fit_visuals.py
ce2ba312af229b4e42015ca1e9177743c8e0ed16e76e0e9a4e991d9869a97367  visuals/make_visuals.py
6f75520ba51e4fff72e0e956b46588e04793c81b89089c09c68b69f453d163d3  visuals/v004_fit_overlay_f000-240.mp4
80a405fc673b34cc502a7e312231434601853a813c9826ecacd50a177c662d3f  visuals/v004_fit_overlay_frames.png
e42a14eb98ca57af8acc8497092bb54af811eeb10fc1c04e15b10d49817681ab  visuals/v004_fit_residuals.png
6fb763d83ad17c59b64cee39eb358ac85a3a2ed220b2a03e8487c662e9911558  visuals/v004_solver_preview_f000-240.mp4
7f6447e6adb7e52b0f6820378dc974e80d83c47a0d07a2e7e298499760dc8c5d  visuals/v004_turning_points.png
a036ee312229ef2c6510ef6490dc4199bfc7ce9665c5e965a46bc527ed32a452  out/bake_check.json
816d9f4951795b4827ab1791c0ac9f5db31a20d3e6c28879c46abb9855354f2a  out/bake_check_reopened.json
e0b5b6fadf73f127e06a6875fe5a23e955d6d98a91948ae57dfd38721a1d77b8  out/fit_synthetic_tests.json
628b9463251e90613cae38137f75891a49c95009a03f9faf727b16e039a678d9  out/make_params_run.json
4f6949d03a4382c47b5e3d2dd2b7458939dc523dadf8da648221be7f6d4cb1e4  out/selfcheck.json
3c9ddc7c896b262b12239f0271998bb007394663a540b2df47af02f077907cef  out/trajectory_meta.json
8ff8bb053a9a093d2f9ca97e842c890452a350491a7bf0ec06c1c98f044649ba  out/trajectory.csv
71c2c3ad76634288bc8fba00b72e1f552dc99c3420f66e2f25d28d0317e74d5e  out/fit_tilt/damping_evidence.json
0cc9d241af86bb4d06f7c45896ea4d0a4dc2821b70fe816ad614fd10a125f2c8  out/fit_tilt/Ta_4.0/fit_projected.json
f6f92921804b700c71403de1568ea57078d00e654f3e4a226b2432695e6472a1  out/fit_tilt/Ta_4.0/params_fit_provenance.json
69a9627ab0e40f93993b939a9cb8368b0e9a148a258b1702c4824d87fc0f27ba  out/fit_tilt/Ta_4.0/tilt_check.json
dc20e99c7119379f640eee4fb22b7cb405cc9495b0b659785e1742b37f093ee7  ../params/physics.json
```
(the NOTES.md hash is not listed, since it would change on edit; earlier visuals are unchanged history)
