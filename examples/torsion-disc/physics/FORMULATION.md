# Clip 03, torsion disc: physics formulation (role B, v1)

Conventions: `CONTRACT.md` v1 and `config/coordinates.md`. SI units, Z up, gravity `g = 9.80665 m/s²` along −Z (assumed).
Origin **A** = where the wire leaves the black clamp on the crossbar. X horizontal toward image right (≈ along the
crossbar), Y away from the camera. Production solver: `solver/torsion_disc.py`, model id **`disc_on_wire_rigid6_v1`**.

## 0. What is visible, and what the physics must explain
* A thin polished brass disc (prompt: "about ten centimetres across") hangs level from a single thin steel wire attached
  at the centre of its top face. The wire rises to a fixed black clamp A on a walnut crossbar.
* The AI clip (CHANGELOG kickoff, `reference-batch/qa.json`): the disc rotates about the wire by **tens of degrees**
  (prompt: a few degrees), with **little damping**, a full-diameter mark instead of a radius, and a **small sideways sway**.
* The corrected model is the general rigid-body-on-wire system: torsion + swing (spherical pendulum) + disc tilt. Its
  amplitude, period and phase are fitted on the anchor window. Sway is modelled if measured, never suppressed or invented.

## 1. Model choice

| candidate | DOF | verdict |
|---|---|---|
| (a) 1-DOF torsion pendulum `I_z ψ'' = −κψ − c ψ'` | 1 | the P3 limit (exact for pure twist, see §4). Cannot represent the visible sway |
| (b) torsion + spherical pendulum with a freely hinged point body | 3 | rejected: the wire is soldered/clamped into the disc, so the disc's tilt is elastically tied to the wire (§2) |
| (c) **rigid disc (6 DOF) on an inextensible wire with torsion + end-bending springs** | 5 (6 − 1 constraint) | **chosen**. Twist ψ, two swing angles of the wire, two tilt angles of the disc. All gyroscopic and geometric couplings are kept |
| (d) continuum Kirchhoff rod | ∞ | rejected: its internal modes are ≫ the visible ones (§2) and the boundary-layer springs capture the static bending to O(λ/ℓ) |

The disc is a uniform solid cylinder (radius R, thickness t). The wire attaches at B = centre of the top face; the CoM
X is `t/2` below B along the disc normal. Wire mass (~6e-5 kg, 3e-4 of the disc) is neglected and declared.

## 2. Wire: what it is and is not
* **Translation (swing): an inextensible string under tension.** Axial stiffness `EA/ℓ ≈ 200e9·9e-8/0.09 ≈ 2e5 N/m`
  gives an axial mode at `sqrt(EA/(ℓm)) ≈ 1000 rad/s`, 100× the swing. The static stretch `mg ℓ/EA ≈ 1e-5 m` is far below
  a pixel (~0.2 mm/px). The wire is therefore an inextensible constraint `|B − A| = ℓ` with tension `T ≥ 0` (P7).
* **Bending: boundary layers only.** For a wire of bending stiffness EI under tension T, bending is confined to layers of
  length `λ = sqrt(EI/T)` at the clamped ends. With the placeholder numbers (d ≈ 0.34 mm, T ≈ 2 N) λ ≈ 8 mm against
  ℓ ≈ 90 mm. Outside the layers the wire is straight. The layer at each end acts on the string direction s as a
  rotational spring of stiffness `k = sqrt(EI·T)` (classical cable-with-bending-stiffness result, error O((λ/ℓ)²)):
  * at the clamp, between s and the vertical: `V_A = k_A (1 + s_z)` (= ½ k_A φ² for small φ). This is equivalent to
    moving the swing pivot λ below A;
  * at the disc, between s and the disc's downward normal n: `V_B = k_B (1 − s·n)`. This ties the disc's tilt to the wire.
  * T varies by < 0.1 % during the motion, so `k = sqrt(EI·mg)` is held constant (declared).
* **Torsion: linear, κ = G π d⁴/(32 ℓ).** Justification for tens of degrees: the largest shear strain is
  `γ = (d/2) ψ/ℓ ≈ 1.7e-4 · 0.52/0.09 ≈ 1e-3` at 30°. That is linear elastic and ≥ 5× below the shear yield strain of hard
  steel wire (τ_y/G ≈ 5e-3–1e-2). Geometric non-linearities are O(γ²) ≈ 1e-6:
  * outer-fibre helix shortening `½γ²`, Poynting torsion–extension coupling;
  * the tension-induced torsional stiffness `T d²/(8ℓ) ≈ 3e-7 N m`, which is 3e-4 of κ. It is absorbed in the fitted κ.
  The wire's own torsional inertia (~1e-11 kg m²) is negligible against I_z ≈ 2.5e-4 kg m².
* **Wire diameter is derived, not measured.** The twist period fixes κ/I_z. With I_z from the assumed brass density and
  the model-conditioned R, t, the wire diameter follows: `d = (32 ℓ κ/(π G))^{1/4}` (G = 79.3 GPa assumed). Then
  `EI = E π d⁴/64 = (1+ν) ℓ κ` and `λ/ℓ = R ω_t sqrt((1+ν)/(2 g ℓ))` are mass-independent.
  The ~1 px wire in the reference (QA: "very faint") is consistent with a 0.2–0.5 mm wire.

## 3. Governing equations
State: CoM X, velocity V, orientation quaternion q (body → world), world angular velocity ω, dissipated work W.
Body frame: z = disc up-normal, x = zero-twist reference direction. Definitions:
```
r = B − X = R(q)(0, 0, t/2),   B = X + r,   s = B/|B|,   n = −R(q) e_z  (disc downward normal)
I_z = m R²/2,   I_⊥ = m(3R² + t²)/12,   I_world = I_⊥ 1 + (I_z − I_⊥) n nᵀ
```
### 3.1 Energy, dissipation
```
T_kin = ½ m |V|² + ½ ωᵀ I_world ω
V     = m g X_z + ½ κ ψ² + k_A (1 + s_z) + k_B (1 − s·n)          (zero at the hanging equilibrium after the constant m g (ℓ + t/2))
D     = ½ c_t ψ̇² + ½ c_x |V|² + ½ c_r |ω_⊥|²,   ψ̇ = G·ω,   ω_⊥ = ω − (ω·n) n,   all c ≥ 0
P_diss = c_t ψ̇² + c_x |V|² + c_r |ω_⊥|² ≥ 0,   dE/dt = −P_diss   (the constraint force does no work: s·Ḃ = 0)
```
### 3.2 Twist angle
ψ is the swing-twist decomposition about the disc normal: `q = q_swing ⊗ q_twist(ψ, e_z)`, where q_swing is the minimal
rotation of body z. In closed form `ψ = 2 atan2(q_z, q_w)`. For a world-frame rotation increment δθ:
```
G = ∂ψ/∂θ = [w y + z x,  z y − w x,  w² + z²] / (w² + z²)      ⇒  torque −(κψ + c_t ψ̇) G
```
* The wire really twists about its own axis s. The decomposition through the bent ends differs from this one by the
  holonomy of the spherical triangle (vertical, s, n). That is ≈ ½|φ||θ| ≤ 1e-3 rad for the observed sway, below the
  fit uncertainty, and is declared.
* ψ = 2 atan2(q_z, q_w) of the continuously integrated q is single-valued on (−2π, 2π): a guard keeps |ψ| < 1.9π. The decomposition is singular only for a 180° tilt (w = z = 0).
### 3.3 Newton–Euler with the wire tension (index-1)
```
m V̇ = F₀ − T s,                 F₀ = −m g e_z − ∂V_bend/∂B − c_x V
I ω̇ + ω × Iω = τ₀ − T (r × s),  τ₀ = r × (−∂V_bend/∂B) + k_B n × s − (κψ + c_t ψ̇) G − c_r ω_⊥
∂V_bend/∂B = [k_A (e_z − s_z s) − k_B (n − (s·n) s)] / |B|        (perpendicular to s)
T (1/m + uᵀI⁻¹u) = s·a₀ + u·α₀ + s·(ω × (ω × r)) + |Ḃ|²/|B|,  u = r × s, a₀ = F₀/m, α₀ = I⁻¹(τ₀ − ω × Iω)
q̇ = ½ (0, ω) ⊗ q
```
* Integration: classical RK4 with a fixed step `h = 1/3072 s`, which divides 1/96 s exactly.
* After every step there is an **exact projection**:
  * q ← q/|q|;
  * X shifted along s so that |B| = ℓ;
  * the radial part of Ḃ removed from V.
  The projection is O(h⁵) per step, so the 4th order is kept.
* **Guards** raise on:
  * a non-finite state;
  * a post-projection constraint error > 1e-12;
  * a per-step drift > 1e-8;
  * tension < 0 (a wire cannot push);
  * |ψ| > 1.9π.
  This is the clip-02 lesson.

## 4. Analytic limits (P3, checked in `solver/selfcheck.py` to < 1e-7)
* **Pure twist (exact at any amplitude):** with the wire and disc vertical and ω ∥ e_z, G = e_z and all bending and
  gravity torques vanish. Then `I_z ψ'' + c_t ψ' + κ ψ = 0` exactly:
  ```
  ω_t = sqrt(κ/I_z),   T_t = 2π sqrt(I_z/κ),   ψ(t) = e^{−γt}[ψ₀ cos ω_d t + (ψ̇₀ + γψ₀)/ω_d sin ω_d t],  γ = c_t/(2I_z)
  ```
  This is checked at 30° and at 60° with an initial rate.
* **Static equilibrium:** X = (0, 0, −ℓ − t/2), level disc, T = m g.
* **Linear swing + tilt (in-plane, per axis):** with φ the wire angle and θ the disc-normal angle,
  ```
  M = [[m ℓ², m ℓ h], [m ℓ h, m h² + I_⊥]],   K = [[m g ℓ + k_A + k_B, −k_B], [−k_B, m g h + k_B]],   h = t/2,   det(K − ω²M) = 0
  ```
  * Placeholder numbers (R = 5 cm, t = 3 mm, ℓ = 9.1 cm, T_t = 3 s) give a swing mode at 9.8 rad/s (T = 0.64 s) and a tilt mode at 13.6 rad/s.
  * In the swing mode the disc tilts **more** than the wire (θ/φ = κ₁ ≈ 3). The tilt mode lies just above the swing mode, so the disc's own inertia lags the wire.
  * This is a real, testable prediction: it is the expected "disc stays level?" signature of sway.
  * Without end bending (k = 0) the two modes reduce to the hinged-disc limit.
* **Planar pendulum, finite amplitude:** with an ideal string (k = 0) and t = 0, the period is `T = 4 sqrt(ℓ/g) K(sin²(φ₀/2))`. This is checked at φ₀ = 0.3 rad.
* **Energy:** conservative drift < 1e-8, and the damped balance `E − E₀ + W` < 1e-7 E₀ (P4).

## 5. Damping (declared unless identifiable)
| mode | mechanism | estimate | treatment |
|---|---|---|---|
| torsion | air shear in the oscillating Stokes layer on both faces | torque ≈ μ π R⁴ ψ̇/δ, δ = sqrt(2ν/ω) ≈ 4 mm, so Q ≈ 5000 | **Q_t fitted** if identifiable in the anchor window (BIC + bootstrap 95 % lower bound of γ > 0, Q in [100, 2e4]). Otherwise **Q_t = 1000** declared (range 300–5000) |
| torsion | internal friction of the steel wire (loss factor ~1e-3–1e-4) | Q ≈ 1e3–1e4 | included in Q_t |
| swing | edgewise air drag on a thin disc | Q ≫ 100 | Q_sw = 500 declared |
| tilt | face-normal air drag (rim speed R θ̇) and bending hysteresis | Q ≈ 20–100 | Q_tilt = 50 declared |
* All laws are linear Rayleigh; the coefficients are c_t = I_z ω_t/Q_t, c_x = m ω₁/Q_sw and c_r = I_⊥ ω₂/Q_tilt.
* With Q_t = 1000 the twist amplitude decays by only 1 % in 10 s (T_t ≈ 3 s). "Little damping" in the AI clip is therefore physically right.
* Amplitude **growth** is never fitted as negative damping or as a drive.

## 6. Identifiability (single view, g assumed)
* **κ/I_z (ω_t)** comes from the twist period. It is scale- and mass-free and is the best-determined quantity.
* **Mark offset μ, ψ₀, ψ̇₀** come from the mark azimuth: the mark sits at μ + ψ(t), and ψ oscillates about 0.
* **Scale.** There are two modes, and the mode used is recorded in `geometry_m.scale_mode`.
  * `swing`: if the sway shows a clear period inside the anchor window, the swing frequency fixes ℓ in metres. The ratios R/ℓ and t/ℓ are measured, and λ/ℓ follows from §2. Scale = model-conditioned, from g.
  * `prompt`: otherwise R = 5.0 cm is declared from the prompt ("about ten centimetres across") and ℓ = Lcam·R/R_cam.
  * The other is reported as a consistency check: an AI sway at a non-pendulum frequency is an AI defect.
* **Mass is unidentifiable.** m = ρ π R² t with brass ρ = 8500 kg/m³ (assumed). κ, k and T scale with it and are labelled model-conditioned. The kinematics depend only on ω_t, ℓ, R, t, λ/ℓ and Q.
* **Disc tilt.** It is fitted only if the BIC demands free tilt initial conditions. Otherwise the tilt mode starts at rest: θ = κ₁φ per axis, the swing-mode shape (`tilt_mode = mode1`).
* **Out-of-plane sway (Y)** is seen only through perspective with one camera. It is fitted only if the BIC demands it.
* **Static tilt β** of the frame relative to gravity follows the clip-01 precedent. It is fitted only if a static asymmetry demands it: the wire not vertical at rest.

## 7. params/physics.json schema (written only by `solver/make_params.py`)
* `model_id`, `status`, `primary` (fit parameters with provenance and σ).
* `parameters`: {g, ell, R, t, m, I_z, I_perp, omega_torsion, kappa, k_A, k_B, damping{Q_*, c_t, c_x, c_r}}, each as `value/unit/provenance`.
* `wire` {d, EI, boundary layer λ, max shear strain}.
* `geometry_m`.
* `camera_scale`.
* `mark.mu_body_rad`.
* `initial_state` {X0, V0, q0 (w,x,y,z), w0 (world)}.
* `derived` (modes).
* `integrator`.
* `generator` (exact argv + input hashes).
* `fit_provenance`.

## 8. v002 decisions after measurement (lead CHANGELOG 2026-10-26T03:42:09Z; measurement/NOTES.md)
* **What the reference shows.**
  * The twist is a one-way −379° rotation in 10 s, with the rate rising and then falling.
  * Sway: T ≈ 0.95 s, 0.026/0.040 D, with energy injections (+50 % at 0.65 s, +45 % at 3–5.3 s).
  * The disc stays perpendicular to the wire (tilt about Y vs swing x: gain −0.96).
  * The camera orbits.
* **Twist = the middle of one slow torsion half-swing.** The linear oscillator `I_z ψ'' = −κψ − c ψ'` reproduces the rising-then-falling rate. The fitted period is T_t ≈ 36 s and the amplitude ≈ 221°.
  * Linear torsion still holds at 221°, provided the wire is hard-drawn spring (music) steel:
    * the shear strain is γ = (d/2) ψ/ℓ ≈ 3.2e-3 (d ≈ 0.32 mm from κ, ℓ ≈ 0.19 m);
    * the shear stress is ≈ 260 MPa and the tension stress ≈ 280 MPa, giving a von Mises stress ≈ 520 MPa.
  * That is elastic for music wire (torsional yield ≈ 1 GPa). It would **not** be elastic for mild steel (τ_y ≈ 150–200 MPa), which is a declared assumption.
  * Geometric non-linearity is O(γ²) ≈ 1e-5.
  * The guard on ψ was relaxed to |ψ| < 1.9π (ψ = 2 atan2(q_z, q_w) on the continuous q is single-valued on (−2π, 2π)).
* **Rigid wire–disc joint** (`bend_ends = rigid_disc`; lead decision).
  * The reference disc stays perpendicular to the wire. A boundary-layer joint `k_B = sqrt(EI·mg)` with the torsion-derived wire cannot do that:
    * with d ≈ 0.32 mm, k_B ≈ 0.05 N m, so the disc's own tilt mode (~4 rad/s) lies *below* the swing (6.9 rad/s);
    * the disc would then tilt *against* the swing (the clamped variant's mode shape θ/φ < 0, or ≫ 1 near resonance).
  * A rigid hub/collet joint is therefore modelled as a stiff joint spring `k_B = I_⊥ (15)² g/ℓ`, so the joint mode is ~110 rad/s. The disc follows the wire to (1/15)² ≈ 0.4 %.
  * Rigid limit, informational: the compound pendulum `ω² = (m g (ℓ+h) + k_A)/(m (ℓ+h)² + I_⊥)`. Here I_⊥ = m R²/4 lengthens the period by ~3 % against a point mass.
  * The tilt drag is then referenced to the swing mode (c_r = I_⊥ ω₁/Q_tilt, Q_tilt = Q_sw = 500). Referencing it to the stiff joint frequency would have turned it into a spurious swing Q ≈ 14 (caught in v002 development).
* **Per-mode anchor windows** (lead, disclosed deviation): ψ is fitted on 0–7.5 s (hold-out 7.5–10 s); B and the tilt are fitted on 0–2.5 s. The model and its integration are joint.
  * **Coupling, quantified on the fitted model.** Removing the sway initial state changes the twist by ≤ 0.14° over 10 s, so sway feeds into the twist negligibly. This justifies the longer twist window.
  * Removing the twist changes B by up to ~3.7 mm (~9 px) over 10 s, but ≲ 1 mm inside 0–2.5 s. This is the gyroscopic precession of the swing ellipse by the spinning disc (I_z ψ̇ ≈ 0.01 N m s against a swing inertia ≈ 0.09 kg m²).
  * The coupling is real physics and is kept. It is the reason the fit is joint and not mode-by-mode.
* **Scale** is model-conditioned from the sway period through the compound-pendulum model: ℓ ≈ 0.19 m and D ≈ 0.22 m, not the prompt's 0.10 m.
* **Suspension point.** The camera role placed A by assumption vertically above the frame-0 disc centre (reported miss 0.037/0.010 D). The fit therefore frees a horizontal pivot offset (Ax, Ay) in camera units (ΔBIC ≈ −18 000).
  * Ax ≈ +0.018 ± 0.003 D is identified. Ay (along the view) is not identified, with σ 0.012 D.
  * The trajectory is written in world coordinates with `pivot_world_m` added. The scene must put TOR_Anchor at the pivot.
* **Not reproduced (AI defects, divergence map).**
  * The sway energy injections, and the sway period drift from 0.914 s (fitted 0–2.5 s) to the AI's 0.95 s average.
  * The tilt/swing gain 0.64 about X: the physical value is 1.00.
  * The twist deceleration after 7.5 s, −11° at 10 s.
  * The mark kink and the full-diameter mark (scene).
