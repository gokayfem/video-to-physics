# Clip 01, brass pendulum: physics formulation (physics/ role, v1)

Conventions follow `CONTRACT.md` v1: SI units, Z up, pivot P at the origin, bob centre
`= P + L_c (sin θ, 0, −cos θ)`, `ω = dθ/dt`, `g = 9.80665 m/s²` (assumed). Numbers quoted below come from
`physics/sanity_calc.py`, saved as `physics/sanity_calc_output.json`. Machine-readable formulas are in
`physics/formulas.json`.

**Recommended model id: `planar_compound_sphere_rigid_massless_rod_v1`**, with the option
`damping = linear_viscous_pivot` (the coefficient is a declared assumption, see §5).

---

## 1. Model choice

### Symbols
| symbol | meaning | unit |
|---|---|---|
| `r` | bob (sphere) radius | m |
| `l` | wire length, from pivot P to the wire's attachment point on the top of the bob | m |
| `L_c = l + r` | distance from pivot to bob centre (`d` in the compound-pendulum formula) | m |
| `m` | bob mass, `= (4/3)π r³ ρ_b` | kg |
| `I_P` | moment of inertia of the swinging body about the pivot axis (world Y through P) | kg m² |
| `L_eq = I_P/(m L_c)` | equivalent simple-pendulum length | m |
| `ω₀ = sqrt(m g L_c / I_P) = sqrt(g/L_eq)` | small-angle angular frequency | rad/s |
| `θ₀` | amplitude, the turning-point angle (conservative case) | rad |

### (a) Point mass on a massless inextensible string
`θ̈ + (g/L_c) sin θ = 0`. This model ignores the bob's rotational inertia `(2/5) m r²`. The error in length
or period is set by `L_eq/L_c − 1 = (2/5)(r/L_c)²`:

* r = 1.5 cm, L_c ≈ 0.387 m gives 6.0e-4 relative in L (3e-4 in period).
* r = 3 cm, the image ratio L_c/r ≈ 13 (rough, see §6), gives 2.4e-3 relative in L (1.2e-3 in period).

These errors are not negligible compared with the P3 tolerance (1e-7). They may be comparable to the
period-measurement precision (about 1e-3). Adding the term costs nothing. **Rejected as the
production model**; it is kept only as a limit (`r → 0`).

### (b) Compound pendulum: solid sphere rigidly attached to a rigid wire (CHOSEN)
`I_P = m L_c² + (2/5) m r² + I_wire`, where `I_wire = m_w l²/3` for a uniform wire of mass `m_w`.
The gravitational moment arm is `m L_c + m_w l/2`.

* **Massless wire (recommended):** `L_eq = L_c + (2/5) r²/L_c`. Inverting for `L_c` gives
  `L_c = [L_eq + sqrt(L_eq² − 1.6 r²)]/2` (take the root with `L_c > r`).
* **Wire mass:** steel, 0.3 mm diameter, about 0.37 m long gives `m_w ≈ 2.1e-4 kg`, against
  `m ≈ 0.12 kg` for r = 1.5 cm. At fixed `L_c`, `L_eq` changes by −3.0e-4 relative. For r = 3 cm
  (m ≈ 0.96 kg) the change is −3.7e-5. The dynamics stay exact for any value of this term, because it
  only changes `I_P` and the moment arm. The only effect is on the mapping from the fitted `L_eq` to the
  geometric `L_c`. Wire diameter and bob size are unmeasurable at this precision, so I recommend a
  massless wire and recording `I_wire` as a bounded systematic (≤ 3e-4 in L_c). Dynamics may add the
  term if the geometry role fixes `d_wire`. The formulas are in `formulas.json`.
* **Rigid versus flexible attachment.** A real thin wire cannot carry a bending moment, so the sphere
  could rock about its attachment point. That makes a 2-DOF double pendulum (wire angle φ and sphere
  angle ψ). I linearised it. For r = 1.5 cm, the low-mode period differs from the rigid compound period
  by **4.5e-6** relative; for r = 3 cm, by 3.5e-5. The second mode (the bob rocking) is at 6.6 Hz and
  4.7 Hz respectively and is not excited by a release from rest in the rigid configuration. The contract
  rig (bob is a child of `PND_Swing`) is already rigid. **The rigid model is therefore consistent with
  the rig, and the difference is far below the measurement precision.**

### (c) Slack string, or elastic stretch of the steel wire
Assumed values: wire diameter 0.3 mm (plausible range 0.2–0.5 mm), `E = 200 GPa` (range 190–210),
brass density 8500 kg/m³.

| | r = 1.5 cm (m = 0.120 kg) | r = 3 cm (m = 0.961 kg) |
|---|---|---|
| static tension `m g` | 1.178 N | 9.43 N |
| tension range over the swing (θ₀ = 6°) | 1.1720 – 1.1913 N | 9.376 – 9.530 N |
| peak stress | 17 MPa | 135 MPa |
| static strain | 8.3e-5 | 6.7e-4 |
| static stretch (absorbed into `L_c`) | 31 µm | 0.24 mm |
| **dynamic length variation over the swing** | **0.51 µm** | **3.9 µm** |
| axial (bounce) mode frequency | 89 Hz | 32 Hz |

* The static stretch is part of the fitted length and is not observable separately.
* The dynamic stretch is ~1e-6 of L. It changes the period by a relative amount of order
  `ΔL/L × θ₀²`, which is less than 1e-8.
* The axial mode is about 100× and 40× the pendulum frequency, so it follows the slow motion
  adiabatically.
* For r = 3 cm with a 0.2 mm wire, the dynamic length variation becomes about 9 µm. That is still
  invisible, but it exceeds the P5 tolerance `|L(t) − L_c| < 1e-6 m`. P5 is about the rig, which is
  inextensible by construction, so there is no conflict.
* **Slack:** tension stays above zero (`T > 0`) whenever `cos θ ≥ (2/3) cos θ₀` (point-mass form) for
  θ₀ < 90°. At 6° the minimum tension is `m g cos θ₀ = 0.9945 m g`. Slack is impossible in this regime.
* **Conclusion: an inextensible, always-taut suspension is justified.** P7 still checks tension at run time.

### (d) Spherical versus planar pendulum
The spherical pendulum reduces exactly to the planar one when the initial out-of-plane angle and angular
velocity are both zero, because planar motion is an invariant subspace. The perturbations that could
break planarity:

* **Earth rotation (Foucault).** The swing plane precesses at `Ω_E sin λ ≈ 7.29e-5 × sin 40° ≈ 4.7e-5 rad/s`.
  Over 10 s that is 4.7e-4 rad. The resulting out-of-plane bob displacement is about
  `L_c θ₀ × 4.7e-4 ≈ 2e-5 m`, which is less than 0.05 px.
* **Airy precession** of an elliptical orbit is proportional to the product of its semi-axes. It is zero
  for a planar release.
* **The reference itself.** Out-of-plane motion would appear as a periodic change in apparent bob size or
  vertical image wobble at the swing frequency. Measurement should report any out-of-plane component. If
  it is present, it is a reference property and does not justify adding a DOF unless it is large.

**Conclusion: planar, 1 DOF (θ).**

### Other effects considered and bounded (all excluded from the production model)

| effect | size (r = 1.5 cm) | treatment |
|---|---|---|
| Air buoyancy (reduces `m g` by a factor ρ_air/ρ_b = 1.4e-4) and added mass (½ ρ_air V) | period +1.06e-4 relative | Absorbed into the fitted `L_eq` (scale is model-conditioned anyway). Declared as a ≤ 1.1e-4 systematic on the L mapping. |
| Wire wrapping around a pin of radius r_p (moving contact point) | period change of order (r_p/L) θ₀², about 4e-5 for r_p = 1.5 mm | Excluded: the contract fixes P. If the geometry role sees the wire *looped over* the pin, it is noted as a systematic. |
| Stand or arm compliance, base rocking | unknown, small for a solid walnut arm | Excluded |
| Earth rotation | see (d) | Excluded |

---

## 2. Governing equations (chosen model)

Lagrangian, with the potential energy measured from the pivot height:

```
T_kin = ½ I_P ω²                         [J]
V     = − m g L_c cos θ                  [J]      (massless wire)
I_P θ̈ = − m g L_c sin θ − τ_d(θ, ω)      [N m]    (τ_d = dissipative torque, ≥ 0 in the sense τ_d ω ≥ 0)
```

which is equivalent to `θ̈ + ω₀² sin θ + τ_d/I_P = 0`, with `ω₀² = m g L_c / I_P = g / L_eq` in s⁻².

* State: `(θ, ω)`. Constraints: `|bob − P| = L_c`, `y_bob = 0`, `P` fixed.
* Initial conditions: `θ(0) = θ_init`, `ω(0) = ω_init`. Both are fitted, because the clip does not start
  at a turning point necessarily. Equivalently, fit an amplitude and a phase.
* **Energy**, with the zero of potential energy at the bottom position (as in the contract CSV,
  `PE_J = m g L_c (1 − cos θ)`):
  `E = ½ I_P ω² + m g L_c (1 − cos θ)`.
  Then `dE/dt = −τ_d ω = −P_diss ≤ 0`.
  * Linear viscous damping: `τ_d = c ω`, so `P_diss = c ω²` (W).
* **Bob kinematics:**
  * Position `x = L_c sin θ`, `z = −L_c cos θ`.
  * Speed `v = L_c |ω|`.
  * Acceleration of the bob centre: tangential `L_c θ̈`, centripetal `L_c ω²` (pointing toward P).

### Suspension tension
This is the axial force along the wire acting on the bob, positive when the wire is in tension. It comes
from the radial equation for the bob centre, which moves on a circle of radius `L_c`:

```
T_wire(θ, ω) = m (g cos θ + L_c ω²)                                   [N]   (general, holds with damping)
conservative: ω² = (2 m g L_c / I_P)(cos θ − cos θ₀)
  ⇒ T_wire = m g [cos θ + 2 (m L_c²/I_P)(cos θ − cos θ₀)]
point-mass limit (I_P = m L_c²): T = m g (3 cos θ − 2 cos θ₀)
```

* In the rigid model the wire also carries a small shear force, `F_⊥ = m g sin θ (I_cm/I_P)` where
  `I_cm = (2/5) m r²` (undamped). This is the force needed to spin the sphere. Its maximum is 7.4e-5 N,
  which is 6e-5 of the tension. With damping, add `−m L_c τ_d/I_P` and the corresponding bending moment.
* A real flexible wire cannot carry this shear; that is exactly the 4.5e-6 double-pendulum difference
  in §1(b).
* If wire mass is included, the tension at the pivot end exceeds the tension at the bob end by about
  `m_w g`.

### Conservative period: exact, valid for any θ₀ < π with no damping and a fixed frictionless axis
```
T(θ₀) = 4 sqrt(I_P / (m g L_c)) · K(k),   k = sin(θ₀/2),
K(k) = ∫₀^{π/2} dφ / sqrt(1 − k² sin² φ)        (complete elliptic integral of the 1st kind, modulus k)
```
**Library warning:** `scipy.special.ellipk` takes the *parameter* `m = k²`, not the modulus k. Use
`ellipk(sin(θ₀/2)**2)`.

Series: `T/T₀ = 1 + θ₀²/16 + 11 θ₀⁴/3072 + …`, where `T₀ = 2π sqrt(L_eq/g)`.

Representative values of the correction factor `T/T₀ = 2K/π`:

| θ₀ | T/T₀ |
|---|---|
| 4° | 1.000305 |
| 6° | 1.000686 |
| 15° | 1.00430 |

Small-angle limit: `θ(t) = θ₀ cos(ω₀ t + φ)`.

The ODE cross-check (DOP853, rtol 1e-12) reproduces the elliptic period to 8e-14 relative.

### Small-angle damped limit (linear viscous)
```
θ̈ + 2γ θ̇ + ω₀² θ = 0,     γ = c / (2 I_P) [s⁻¹]
θ(t) = θ₀ e^{−γ t} cos(ω_d t + φ),   ω_d = sqrt(ω₀² − γ²)
Q = ω₀ / (2γ) = I_P ω₀ / c,    logarithmic decrement per period δ = γ T_d = 2π γ/ω_d ≈ π/Q
E(t) ≈ E₀ e^{−2γ t}
```
For finite amplitude with `sin θ`, the amplitude envelope still decays as `e^{−γ t}` to first order in γ
(the energy loss per cycle is `c ∮ ω dθ`). Corrections of order `γ θ₀²` are below 1e-6 for these
parameters. Verifiers should therefore compare P9 envelopes at θ₀ = 2°, or with an energy-based
envelope.

---

## 3. Scale inference (model-conditioned)

1. Measure the period `T_meas` (zero crossings or a sinusoid fit to θ(t) in image angle) and the amplitude
   `θ₀` (needs the camera's projection of the swing plane).
2. Remove the finite-amplitude correction: `T₀ = T_meas / (2K(sin(θ₀/2))/π)`.
   * Ignoring it at 6° biases L by +1.37e-3.
   * At 4° the bias is +6.1e-4.
   * The 4° prompt versus the 6° eyeball difference alone is 7.6e-4 in L.
3. `L_eq = g (T₀/2π)²`.
4. Use the image ratio `ρ_img = L_c/r`, measured in the swing plane after camera correction: invert
   `L_eq = L_c (1 + (2/5)/ρ_img²)` to get `L_c = L_eq / (1 + 0.4/ρ_img²)`, then `r = L_c/ρ_img` and
   `l = L_c − r`. With an assumed bob radius instead, use `L_c = [L_eq + sqrt(L_eq² − 1.6 r²)]/2`.
5. Error propagation: `δL/L = 2 δT/T ⊕ δg/g`. g is an assumption fixed by the contract. Locally g varies
   by ±0.3 %, which gives ±0.3 % in L. That is a systematic of the declared assumption, not of the fit.

All of these values are `model-conditioned`. They are not independent evidence that the model is right.

Representative case: `T_meas = 1.25 s`, `θ₀ = 6°` gives `T₀ = 1.249143 s` and **`L_eq = 0.387602 m`**.

| | r = 1.5 cm | r = 3 cm (L_c/r ≈ 13) |
|---|---|---|
| L_c | 0.387369 m | 0.386670 m |
| l (wire) | 0.372369 m | 0.356670 m |

---

## 4. Kinematics for the representative case (T = 1.25 s, θ₀ = 6°, r = 1.5 cm)

| quantity | value |
|---|---|
| ω₀ | 5.030 rad/s |
| ω_max | 0.5265 rad/s |
| v_max (bob centre) | 0.204 m/s |
| Re_max = v·2r/ν | 406 (6° amplitude); 271 at 4°; 811 for r = 3 cm |
| Cd (steady sphere, Morrison 2013 correlation) at Re_max | 0.60 (0.69 at Re 271; 0.50 at Re 811) |
| Keulegan–Carpenter number KC = v_max T/(2r) | 8.5 |
| Stokes layer thickness sqrt(2ν/ω₀) | 2.4 mm (r/δ ≈ 6) |

Re is a few hundred, so the flow is neither in the Stokes regime (Re ≪ 1) nor at the textbook Cd ≈ 0.47
plateau (Re ≈ 1e3–2e5). The steady correlation gives Cd ≈ 0.5–0.7. The flow is also oscillatory with a
low KC number, so a quasi-steady Cd is only an order-of-magnitude tool here. Within the swing, the
instantaneous Re runs from 0 to 406.

---

## 5. Damping

### Candidate dissipative torques τ_d(θ, ω) and their per-cycle amplitude loss
These hold for small damping and nearly harmonic motion. `ΔA/A` per cycle equals δ ≈ π/Q.

| model | torque τ_d | ΔA per cycle | ΔA/A per cycle (representative case) |
|---|---|---|---|
| Linear viscous (pivot or air, linearised) | `c ω` | `A (1 − e^{−γT})` | `γ T = c T/(2 I_P)` |
| Quadratic drag on the sphere | `½ ρ_a C_d (π r²) L_c³ |ω| ω` | `(4/3) ρ_a C_d π r² L_c³ A² / I_P` | **2.3e-4** |
| Quadratic drag on the wire (cylinder, `Cd ≈ 1 + 10 Re^{-2/3}`, Re ≈ 2, so Cd ≈ 7) | `½ ρ_a C_dw d_w (l⁴/4) |ω| ω` | same form | **9.7e-5** |
| Oscillatory Stokes drag on the sphere (Stokes 1851; Landau & Lifshitz §24), force in phase with velocity `6π μ r (1 + r/δ_s) U` | `6π μ r (1 + r/δ_s) L_c² ω` | linear | **1.9e-4** |
| Coulomb pivot friction | `τ_f sgn ω` | `4 τ_f/(m g L_c)` (linear decay in A, stops in finite time) | `τ_f ≈ 1.2e-4 N m` would give 1 %/cycle |

A note on the quadratic-drag derivation: with SHM `θ = A cos ω₀t`, the energy lost per cycle is
`ΔE = k_q ∮|ω|³ dt = (8/3) k_q A³ ω₀²`, with `k_q = ½ ρ_a C_d π r² L_c³`. Then `ΔA/A = ΔE/(2E)` with
`E = ½ I_P ω₀² A²`. For a point mass this reduces to `ΔA/A = ρ_a C_d L_c A / (ρ_b r)`.

#### Air-only estimate
Adding the three air terms gives a deliberately generous estimate that double-counts a little, because
the steady Cd already contains viscous drag.

* r = 1.5 cm: `ΔA/A ≈ 5.2e-4` per cycle, so **Q_air ≈ 6000**. After 8 cycles the amplitude is
  0.9959 × θ₀ (6.000° becomes 5.975°).
* The ODE with quadratic terms only gives 5.984° after 8 cycles.
* r = 3 cm: Q_air ≈ 16 000 (ratio 0.9984 after 8 cycles).
* So **air alone would make decay invisible in 10 s**. At 6° with L_c ≈ 720 px in the image, the
  amplitude is ~75 px. A 0.4 % loss is 0.3 px.

#### Real desk toys
Real desk toys lose more energy at the support: wire bending hysteresis at the clamp, pin/loop rubbing,
and stand or arm flex. Typical Q is 10²–10³ (this is my order-of-magnitude judgement, not a cited
measurement).

| Q | amplitude ratio after 8 cycles | 6° becomes |
|---|---|---|
| 100 | 0.778 | 4.67° (≈ −17 px at a 75 px amplitude) |
| 300 | 0.920 | 5.52° |
| 1000 | 0.975 | 5.85° |
| 3000 | 0.992 | 5.95° |

The prompt asks for "slight damping", which corresponds to Q ≈ 300–3000.

### Recommendation
* Use **linear viscous pivot damping** `τ_d = c ω`, with `c = I_P ω₀ / Q`.
* It is the simplest model with an exact closed-form envelope (P9 friendly) and exact energy accounting
  (`P_diss = c ω²`). It never produces growth and has no stick-slip discontinuity.
* At this Re and amplitude, the quadratic drag fraction of the loss is small and indistinguishable from
  linear damping over 8 cycles. The envelopes differ by less than 1e-4 of the amplitude for Q ≥ 1000.
* Coulomb friction is rejected: it is unidentifiable here and it creates a stick discontinuity.
* **How c is set:**
  * If measurement detects a decay with significance: fit `γ` from the amplitude envelope over
    t < 7.5 s (`fitted`).
  * Otherwise: `damping.Q` is `assumed` with nominal **Q = 1000**, range [300, 6000]. The upper end is
    the air-only limit; the lower end is the most decay that would be "slight" and still roughly
    consistent with "no clear visible decay" (−8 % over 8 cycles, about 6 px).
  * Q = ∞ (conservative) must not be presented as physical, because a real pendulum always dissipates.
    It may be used as a validation case (P3/P4).
* **Growth.** If the clip shows amplitude growth beyond noise, it is a reference defect. Never fit
  negative damping.

---

## 6. Identifiability from the clip

| quantity | identifiable? | how / why not | provenance tag |
|---|---|---|---|
| period T, θ(t) phase | **yes** | image time is exact (t = n/24) | measured |
| amplitude θ₀ (true angle) | yes, given the camera | needs the swing-plane orientation from the camera solve; foreshortening otherwise | fitted (measured + camera) |
| θ_init, ω_init | yes | fit to the track | fitted |
| L_eq, hence metric scale | **only model-conditioned** | `L_eq = g(T₀/2π)²` holds *if* the clip obeys a pendulum with standard g; an AI clip need not | model-conditioned |
| ratio L_c/r | yes, roughly | image measurement in the swing plane; a rough sheet read is ~13 (wire ≈ 670 px, bob radius ≈ 54 px at full resolution, sheet-thumbnail estimate, ±20 %) | measured (via the measurement role) |
| r, l in metres | model-conditioned | from L_eq and the ratio | model-conditioned |
| mass m, density | **no** | kinematics are invariant to m (all forces scale with m); declared from brass density × volume | assumed |
| wire diameter, E | no (sub-pixel) | only affects stretch (µm) | assumed |
| damping γ / Q | only if the decay exceeds noise | with σ_amp ≈ 0.3 px on a ~75 px amplitude, a 1σ detection over 8 cycles needs ΔA/A ≳ 0.4 % (Q ≲ 6000); a firm detection needs Q ≲ 1000. If there is no decay, it is unidentifiable and set by assumption. | fitted or assumed |
| damping *law* (linear, quadratic, Coulomb) | no | needs many decay times and an amplitude range | assumed |
| g | no | the clip has no independent time–length reference | assumed (contract) |

**Notable consistency flag.** If the sheet-based ratio L_c/r ≈ 13 is right, then L_c ≈ 0.387 m (from the
period) implies r ≈ 3 cm, a 6 cm diameter bob of about 0.96 kg brass. The stand is then about 45–50 cm
tall. That is larger than the "2.5–3.5 cm bob desk toy" assumption. Both are physically possible; the
period decides the scale. Alternatively, a 1.5 cm bob with the same ratio would need L ≈ 0.19 m and
T ≈ 0.88 s. **The measurement role must give the ratio. Physics should not assume r.**

---

## 7. Parameter schema for `params/physics.json` (proposal for the dynamics role)

See `formulas.json → physics_json_schema_proposal` for the machine-readable version. Each value carries
`{value, unit, provenance, range|sigma, source}`.

```json
{
  "model_id": "planar_compound_sphere_rigid_massless_rod_v1",
  "g":            {"value": 9.80665, "unit": "m/s^2", "provenance": "assumed", "source": "CONTRACT v1"},
  "T_meas":       {"value": null, "unit": "s", "provenance": "measured", "sigma": null},
  "theta_amp":    {"value": null, "unit": "rad", "provenance": "fitted"},
  "L_eq":         {"value": null, "unit": "m", "provenance": "model-conditioned", "sigma": null,
                   "derivation": "g*(T0/2pi)^2, T0 = T_meas/(2K(sin(theta_amp/2))/pi)"},
  "ratio_Lc_over_r": {"value": null, "provenance": "measured", "sigma": null},
  "L_c":          {"value": null, "unit": "m", "provenance": "model-conditioned"},
  "bob_radius":   {"value": null, "unit": "m", "provenance": "model-conditioned"},
  "wire_length":  {"value": null, "unit": "m", "provenance": "model-conditioned", "note": "L_c - r"},
  "rho_bob":      {"value": 8500, "unit": "kg/m^3", "provenance": "assumed", "range": [8400, 8730]},
  "bob_mass":     {"value": null, "unit": "kg", "provenance": "assumed", "derivation": "4/3 pi r^3 rho_bob"},
  "I_P":          {"value": null, "unit": "kg m^2", "provenance": "model-conditioned", "derivation": "m L_c^2 + 0.4 m r^2"},
  "wire": {"diameter": {"value": 3e-4, "unit": "m", "provenance": "assumed", "range": [2e-4, 5e-4]},
           "E": {"value": 2.0e11, "unit": "Pa", "provenance": "assumed", "range": [1.9e11, 2.1e11]},
           "rho": {"value": 7850, "unit": "kg/m^3", "provenance": "assumed"},
           "mass_included": false, "inextensible": true},
  "damping": {"law": "linear_viscous_pivot",
              "Q": {"value": 1000, "provenance": "assumed", "range": [300, 6000]},
              "c": {"value": null, "unit": "N m s", "derivation": "I_P*omega0/Q"},
              "note": "replace with 'fitted' only if measured decay is significant; never negative"},
  "theta_init":   {"value": null, "unit": "rad", "provenance": "fitted"},
  "omega_init":   {"value": null, "unit": "rad/s", "provenance": "fitted"},
  "air": {"rho": 1.204, "mu": 1.813e-5, "T_K": 293.15, "p_Pa": 101325, "provenance": "assumed",
          "used_for": "plausibility only (Q bound); not in production EOM"},
  "systematics_on_L_mapping": {"buoyancy_added_mass_rel_T": 1.1e-4, "wire_mass_rel_Leq": 3e-4,
          "flexible_wire_rel_T": 5e-6, "g_local_rel": 3e-3}
}
```
