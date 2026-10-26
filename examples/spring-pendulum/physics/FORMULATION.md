# Clip 02, vertical spring: physics formulation (role B, v1)

Conventions: `CONTRACT.md` v1 and `config/coordinates.md`. SI units, Z up, gravity `g = 9.80665 m/s²` along −Z
(assumed). Origin **A** = where the top hook bears on the crossbar. X is horizontal toward image right, Y points away from
the camera. Production solver: `solver/spring_pendulum.py`, model id **`elastic_pendulum_hinged_body_v1`**.

## 0. What is visible, and what the physics must explain
* There is a rigid frame. An S-hook hangs from the crossbar and holds a close-wound helical extension spring. The spring's
  lower end is a hook, and that hook passes through the eye-ring on top of a brushed-steel cylinder.
* The AI clip shows a **lateral swing** of roughly ±10° of the whole spring + weight, plus a **tilt of the weight**, and possibly a
  small vertical bob. The prompt asked for vertical-only bobbing. The CHANGELOG (kickoff) frames the correct physics as the
  3D elastic pendulum: a spring-mass system released with a lateral offset does swing. The amplitude and
  frequency are fitted on the anchor window, and nothing is forced to be vertical.

## 1. Model choice (simplest model that is defensible for what is visible)

| candidate | DOF | verdict |
|---|---|---|
| (a) 1D vertical mass-spring | 1 | **rejected**: it cannot represent the visible swing, and suppressing the swing would mean hand-authoring the motion |
| (b) 3D elastic pendulum, point mass on a massless linear spring from A | 3 | the limit `b, I⊥ → 0`. Kept as a solver option (`body: "point"`) and as the P3 limit |
| (c) **elastic pendulum + rigid weight hinged at the lower hook** (spring + hinged rigid cylinder) | 5 | **chosen**. The ring rides on the hook, which is a ball-joint-like contact that cannot carry a bending moment. A rigid cylinder whose CoM sits `b ≈ 4 cm` below that contact is a second pendulum. Its tilt relative to the spring axis is real physics, and it is exactly what the clip's "tilting weight" can be checked against |
| (d) + spin about the own axis | 6 | not needed. For an axisymmetric body hinged on its own axis, the torques about the axis vanish, so `ω₃` is conserved. With zero spin at release, `ω₃ ≡ 0`. The geometric twist (holonomy) is still integrated (auxiliary quaternion), so the ring's orientation is exact |
| (e) continuum spring (surge / transverse string modes) | ∞ | rejected. The lowest internal modes are ≫ 10 Hz (see §2), so the Rayleigh correction (m_s/3) captures them to O((m_s/m)²) |

The weight is modelled as a uniform solid cylinder (radius r, height h). The CoM is at the cylinder centre, a distance `b` below H.
The ring/eye mass (~1 %) is neglected and declared.

**Rigid hooks.** The top S-hook and the spring's lower hook are rigid wire. They swing with the spring axis A→H and are included in
the effective free length `L0`. The top hook pivots in the crossbar eye (moment-free).

## 2. Spring: what it is and is not
* **Axial law.** The spring is linear: `F = k(|PH| − L0)` along P→H (v2: P = P_eff, §8; A in v1). A close-wound extension spring has an initial
  tension `F_i`. Above `F_i` the law is `F = F_i + k x`, which is identical to a linear law with a shorter effective free length.
  `L0` is therefore an *effective* length and absorbs `F_i` and the rigid hooks. Validity needs `T > F_i` at all times, so the
  coils stay open. The weight hangs visibly stretched, and the fitted tension range is reported (P7).
* **Lateral stiffness / bending.** Both ends are moment-free: the hook sits in an eye, and the lower hook sits in the ring.
  A massless member loaded only at two moment-free ends is a **two-force member**. It stays straight along A→H whatever
  its bending stiffness, so bending stiffness cannot change the dynamics. Spring mass causes a transverse inertial load, which
  bows the spring. The bow is of order `(m_s/m)·(B/(T ℓ²))·ℓ`. Estimate: d = 1.2 mm wire, D = 12 mm, N = 30, ℓ = 0.3 m
  give `B ≈ (E d⁴/32)(2G/(E+2G))(ℓ/(πDN)) ≈ 2.3e-3 N m²` and `T ℓ² ≈ 0.86 N m²`. That is a bow of ~1e-2 × 1e-2 × ℓ ≈ 30 µm,
  far below a pixel (~0.3 mm/px). **The spring behaves as an elastic string under tension for the pendulum mode.**
* **Spring mass (Rayleigh).** With a linear velocity profile along the spring, `T_s = ½(m_s/3)|Ḣ|²` and `V_s = m_s g H_z/2`.
  This gives the classical `m_eff = m + m_s/3` for the axial mode, and the same 1/3 (rigid-rod moment of inertia) for the swing.
  The error is O((m_s/m)²) ~ 1e-4 relative in frequency. Assumed m_s ≈ 10 g against m ≈ 1 kg (`make_params.py` defaults:
  steel wire 1.2 mm, coil 12 mm, 30 turns). With a heavy end mass, the first surge mode is at `ω_surge ≈ π sqrt(k/m_s) ≈ 340 rad/s` for k ≈ 120 N/m and m_s = 10 g.
  That is ~30× the axial mode, so it is quasi-static. The first transverse string mode, `(π/ℓ) sqrt(T ℓ/m_s)`, is ≈ 180 rad/s
  for T ≈ 9.5 N and ℓ = 0.3 m: also quasi-static.

## 3. Governing equations
Generalised coordinates `q = (X, n)`: X is the weight CoM, n is the unit body axis from the hinge H to the CoM, and
`H = X − b n` (the spring's lower end). The constraint is `|n| = 1`.

```
T = ½ m |Ẋ|² + ½ I⊥ |ṅ|² + ½ μ |Ẋ − b ṅ|²,            μ = m_s/3,  I⊥ = m(3r² + h²)/12 (about the CoM)
V = ½ k (|H| − L0)² + m g X_z + ½ m_s g H_z
D = ½ c_s (ŝ·Ḣ)² + ½ c_x |Ẋ|² + ½ c_r |ṅ|²               ŝ = H/|H|,  all c ≥ 0 (Rayleigh dissipation)
```
Lagrange with multiplier λ for `|n| = 1`, using `a = m + μ`, `c = −μ b`, `d = I⊥ + μ b²`, `det = a d − c²`:
```
a Ẍ + c n̈ = F_X        F_X = −(T + c_s ṡ) ŝ − (m + m_s/2) g e_z − c_x Ẋ
c Ẍ + d n̈ = F_n + λ n  F_n = +b (T + c_s ṡ) ŝ + b (m_s/2) g e_z − c_r ṅ
T = k(|H| − L0)   (spring tension, > 0 when stretched),   ṡ = ŝ·Ḣ
λ = (−|ṅ|² det + c n·F_X − a n·F_n)/a     ⇒  n·n̈ = −|ṅ|²  (|n| = 1 and n·ṅ = 0 are preserved exactly)
```
* **Force on the weight at the hinge** = the spring force `−(T + c_s ṡ) ŝ` (the hook is massless).
* **Orientation.** `ω = n × ṅ` (with ω₃ = 0) and `q̇ = ½ (0, ω) ⊗ q`, where q(0) is the minimal rotation from local −Z to n₀.
* **Energy** (zero at static equilibrium): `E = T + V − V_eq`, and `dE/dt = −P_diss`, with
  `P_diss = c_s ṡ² + c_x |Ẋ|² + c_r |ṅ|² ≥ 0`. W(t) = ∫P_diss is integrated as a state.
* **Damping parameterisation.** `c_x = 2 m γ_sw` (swing), `c_s = 2(m+μ) γ_ax − c_x ≥ 0` (axial),
  `c_r = 2 I⊥ γ_rock`.
  * γ values are fitted only if identifiable inside the anchor window and ≥ 0. Otherwise they are declared assumptions.
  * Air drag on a 4 cm cylinder at ~0.1 m/s is mostly quadratic, but its Q is ≫ 100. The linear law is the declared model.
  * Hinge rubbing (ring on hook) is the most plausible real loss for the rocking mode. The rocking-mode Q is assumed (γ_rock = 0.5 s⁻¹, Q ≈ 30) because rocking is not resolvable in the clip.

## 4. Analytic limits (P3, all checked in `solver/selfcheck.py` to < 1e-7)
* **Static stretch:** `δ = (m + m_s/2) g / k`, and the equilibrium is `|PH| = ℓ = L0 + δ`, with H straight below the pivot P and the CoM a distance b below H.
* **Axial mode** (body hanging straight, exactly linear): `ω_ax = sqrt(k/(m + m_s/3))` and `γ = (c_s + c_x)/(2(m+μ))`. The exact solution is
  `z(t) = z_eq − z₀ e^{−γt}(cos ω_d t + (γ/ω_d) sin ω_d t)`.
* **Swing, point mass, massless spring:** `ω_sw = sqrt(g/(L0 + δ))`. The stretch-dependent length is the only "spring" effect at linear
  order.
* **Swing + rocking (rigid hinged body), linearised in-plane:**
  `M = [[(m+μ)ℓ², m ℓ b], [m ℓ b, m b² + I⊥]]`, `K = diag((m + m_s/2) g ℓ, m g b)`, and the mode frequencies solve `det(K − ω²M) = 0`.
  For ℓ ≈ 0.3 m and b ≈ 4 cm, the swing mode is at ≈ 0.85 Hz and the rocking mode at ≈ 5 Hz.
  * In the swing mode the weight tilts **slightly more than the spring**: ψ/φ ≈ 1 + O(b/ℓ).
  * A large, slow tilt of the weight relative to the spring axis is **not** a normal mode.
  * The fit tests whether the measured tilt is consistent with this.
* **Autoparametric (1:2) resonance.**
  * The spring pendulum exchanges energy between the bob and the swing when `ω_ax ≈ 2 ω_sw` (Vitt–Gorelik).
  * The exchange period scales as `~ 1/(ε ω)`, where ε is the amplitude of the parent mode.
  * The model keeps the full nonlinear coupling, so whatever the fitted ratio `ω_ax/ω_sw` implies is *predicted*, not suppressed.
  * The fitted ratio is reported. If it lies within ~10 % of 2, the swing amplitude modulation after T_a is genuine physics and is labelled as such in the divergence map.

## 5. Effects bounded and excluded
| effect | size | treatment |
|---|---|---|
| Foucault precession | Ω sinλ·10 s ≈ 5e-4 rad of swing-plane rotation | excluded (< 0.1 px) |
| Air buoyancy / added mass | ρ_air/ρ_steel ≈ 1.5e-4 relative frequency | absorbed in the model-conditioned scale |
| Frame compliance | frame is rigid by prompt; measured static landmarks | excluded |
| Hook sliding in the eye / ring | would show as attachment jumps | excluded; such a jump is an AI defect (a T_a trigger) |
| Ring/hook contact geometry (rolling radius ~ 2 mm) | changes b by ≲ 2 mm | b is defined from the image as hinge → cylinder centre |
| Spring torsion–extension coupling (Wilberforce) | needs a torsion mode tuned to ω_ax; ω₃ = 0 at release | excluded and declared. Visible twisting in the clip would be an AI defect |

## 6. Identifiability (single view, g assumed)
* Only **frequencies** carry metric information.
  * `ω_sw` sets ℓ (model-conditioned: given the measured image ratios b/ℓ, r/ℓ, h/ℓ, the two-body swing frequency fixes ℓ in metres).
  * `ω_ax` sets k/(m + m_s/3), which with g gives the static stretch `δ = g(1 + m_s/2m)/ω_ax²`.
  * Absolute scale is therefore **model-conditioned** (the clip-01 precedent).
  * Camera units per metre are a fitted output for the camera role.
* **Mass is unidentifiable.** The kinematics depend only on k/m, m_s/m, I⊥/(m b²), and c/m. m comes from an assumed steel density × the
  cylinder volume (7900 kg/m³). k and the tensions scale with it and are labelled model-conditioned.
* **L0 vs δ.** Only ℓ and ω_ax are kinematic. L0 = ℓ − δ follows. If no axial bob is visible inside the anchor window, ω_ax is
  **not identified**. It then becomes a declared assumption, with the range implied by a plausible spring.
* **Out-of-plane (Y) motion.** It is observable only through perspective (size/vertical changes) with one camera. It is fitted only if the BIC demands it.
  Otherwise it is fixed at 0 (a planar release) and flagged.
* **Damping** is fitted only if it is identifiable in the anchor window (bootstrap 95 % lower bound > 0 and a plausible Q). Otherwise it is a declared assumption.
  Amplitude growth in the AI clip is never fitted as negative damping or as a drive.
* **Static tilt β** of the frame relative to gravity (clip-01 precedent). It is fitted only if the swing's left/right asymmetry demands it.

## 7. params/physics.json schema (written only by `solver/make_params.py`)
* `model_id`, `body`.
* `parameters`: {g, ell_eq, omega_axial, k, L0, m, m_s, b, I_perp, damping{gamma_*, c_s, c_x, c_r}}, each with `value/unit/provenance`.
* `geometry_m` (for the scene: ℓ, cylinder r/h, H→cylinder centre, ratios).
* `camera_scale` and `level_tilt_rad` (if fitted).
* `initial_state` {X0, V0, n0, u0}.
* `derived` (equilibrium, mode frequencies, ratio ω_ax/ω_sw).
* `integrator`.
* `generator` (exact argv and input hashes).
* `fit_provenance`.

## 8. v002 decisions after measurement (lead, CHANGELOG; measurement/summary.json)
* **Swing pivot P = P_eff.** P is the spring's top coil on a FIXED S-hook, 0.225071 W_in below A (measured; the reference S-hook
  is static to 0.64 px). The solver origin is P, and the trajectory is written in world coordinates (`pivot_world_m`).
  A freely hinged S-hook at A would swing with the spring (it is a massless two-force link), so the static hook is a
  modelling choice by the lead. Measurement lists it among the AI defects.
* **Geometry in camera units**, measured by role C: b = 0.228077, r = 0.114416, h = 0.286985, eye centre e = 0.03474 below H,
  wire d = 0.009682, coil D = 0.107988, 22 turns. Metres = camera units / λ, where λ = Lcam/ℓ is fitted from the swing frequency.
* **k is a declared assumption** from the coil formula `k = G d⁴/(8 D³ N)` with G = 79.3 GPa (spring steel), giving k = 1285 N/m at the fitted
  scale. No axial mode is observed, so k cannot be identified from a frequency. The axial and rocking modes start at rest:
  * de0 is the quasi-static stretch for the initial swing state, ded0 = 0;
  * (ψ0, ψ̇0) lie on the linear swing-mode shape, ψ = κφ with κ = 1.058.
* **Sensitivity of the predicted axial motion to k** (v003, range of |PH| over 10 s):

  | k | ω_ax/ω_sw | range of |PH| |
  |---|---|---|
  | × 0.5 | 2.31 | 15.3 mm (~25 px) |
  | × 1 (1336 N/m) | 3.26 | 2.65 mm (~4 px) |
  | × 2 | 4.61 | 0.9 mm |

  * The reference's ≤ ±1.2 px rms radial motion disfavours soft springs near the 2:1 autoparametric band.
  * A diagnostic fit of rax to the swing-locked 2ω stretch ripple gives k = 1491 N/m (1.12× the coil value). That is consistent with the coil value, and it is evidence from an AI clip.
* **Weight rocking.** Physics allows the weight to tilt only κ = 1.055 times the swing angle in the swing mode (v003). The reference shows 1.3–1.4×:
  * turning-point tilt ≈ 10.5–11.9° against 8.5° from the model;
  * the excess is an AI defect and is reported in the divergence map.
* **Marker nuisance dw.**
  * With the scene v001 geometry, the measured weight-centre marker sat 0.0138 W_in (≈ 9 px) off the cylinder centre.
  * The scene v002 refit, which removes the top-face ellipse from h, brings it to 0.0009 ± 0.0014 W_in, consistent with 0. It is therefore dropped in v003.
* **v002 variant diagnostics** (scene v001 geometry, 0–7.5 s, bootstrap 0; not adopted):

  | variant | fit-window RMS | ΔBIC | note |
  |---|---|---|---|
  | free rocking IC | 6.94 px | −2160 | excites a 4 Hz rocking mode that is absent in the reference |
  | static tilt | 6.40 px | −835 | β = 0.08° |
  | body-axis offset | 6.56 px | −263 | α = −0.6° |

  * The production baseline was 6.84 px. A negative ΔBIC means the variant is preferred.
  * All are sub-degree, or they chase AI systematics.
