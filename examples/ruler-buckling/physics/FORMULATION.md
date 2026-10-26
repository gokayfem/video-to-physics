# Ruler buckling — mechanical formulation (role B, v002)

Version v002 uses the lead's decisions logged at 2026-10-26T09:22:56Z:
- the peg is 40 mm tall;
- the strip is spring-temper AISI 301 (full hard);
- the schedule places 7 weights only, seated at 1.0, 1.6, 2.2, 2.8, 3.4, 4.0 and 4.6 s;
- the strip buckles toward −Y.

`physics/derive.py` (sha256 4822ca92…, run time about 12 s) writes `physics/benchmarks.json` (4de3883f…) and `physics/visuals/v002_load_deflection.png`.
`config/scene_params.json` still says the peg is 20 mm, so `derive.py` overrides the peg height to 40 mm in code. All values are SI.
Labels: **[A]** assumed, **[D]** declared scene geometry, **[M]** model-conditioned result.

v001 is superseded. Its numbers were the same except for the 20 mm peg (n_cr was 6.32).

![v002](visuals/v002_load_deflection.png)

## 0. Data
| quantity | value | |
|---|---|---|
| E, ν, ρ (AISI 301 full hard) | 193 GPa, 0.29, 7900 kg/m³ | [A] |
| b × t, b/t | 25 × 0.5 mm, 50 | [D] |
| I = b t³/12 | 2.604e-13 m⁴ → **EI = 0.05026 N·m²** | |
| strong-axis I_s = t b³/12 | 6.51e-10 m⁴ (2500 × I) | |
| self-weight q = ρ b t g | 0.9688 N/m (strip from jaw to top, 0.280 m, weighs 27.7 g) | |
| L (jaw top z = 0.095 to strip top z = 0.375) | 0.280 m | [D] |
| platform tabs (front and back, 12 mm tall) | they clamp the top 12 mm of the strip → **flexible length L_f = 0.268 m** | [D] (scene/build.py) |
| rigid tip body at n = 0, m₀ | **13.94 g** = plate 5.06 + tabs 2 × 1.62 + strip inside the tabs 1.19 + screws 0.48 (est.) + peg Ø4×40 3.97 | [D] + [A] densities (Al 2700, steel 7900) |
| weights | n × 20 g, Ø25 × 4.5 mm, stacked from 15 mm above O | [D] |

O is the strip mid-surface at the bottom of the tabs (s = L_f). The tip body's centre of mass (CoM) lies on the strip tangent, at height e(n) above O.
e(n) is computed from the part list. Its values are e₀ = 17.0 mm, e(6) = 27.3 mm and e(7) = 29.5 mm.
The rotary inertia J(n) about that CoM is 2.8e-6 kg·m² at n = 0 and 2.25e-5 kg·m² at n = 7.
The front and back tabs are symmetric, so there is no out-of-plane CoM offset.

## 1. Model and boundary conditions
**Model: a 1-D, geometrically exact, planar Kirchhoff rod (the elastica), bending about the weak axis in the Y–Z plane.**

Why this is sufficient:
- **Slenderness.** L/t = 560. Shear and axial strain are below 1e-5.
- **Strong-axis bending** has 2500 × the stiffness, so it does not occur.
- **Torsional buckling** would need GJ/r_p² ≈ 1500 N, about 1000 × the actual load.
- **Lateral-torsional buckling** would need strong-axis bending, which is absent.
- **Plate effects.** At bifurcation the Searle parameter β = b²/(Rt) is 0 and the edges are free, so the bending stiffness is EI. The upper bound EI/(1−ν²) = 1.092 EI is case E. After buckling, β at the root is 6.5 at n = 7, where the strip partly stiffens toward 1.092 EI. Use EI as the baseline and run 1.092 EI as a sensitivity case.

Boundary conditions and equations:
- **Base.** At s = 0 the strip is clamped: w = 0, θ = 0. A rigid jaw is assumed [A].
- **Tip.** At s = L_f the strip carries a rigid body of mass m(n) with its CoM at offset e(n) along the tangent and inertia J(n). It is a dead vertical load W = m·g, giving **M(L_f) = W·e·sin θ_tip**. Self-weight q is distributed along the strip.
- **Equations.** x' = sin θ, z' = cos θ, EI(θ' − κ₀) = M, M' = −N sin θ, with N(s) = W + q(L_f − s).

Linear checks (for q = 0 the critical condition is kL_f·tan(kL_f) = L_f/e, with k² = W/EI). The finite-element model reproduces the Euler case, the rigid-offset case and Greenhill (qL³/EI = 7.837) to better than 1e-5. Mesh convergence of W_cr is 1e-6 at 80 elements.

| case | W_cr [N] | total mass [g] | n_cr (continuous) | buckles at weight # |
|---|---|---|---|---|
| A: naive π²EI/4L², L = 0.280 | 1.5818 | 161.2 | 7.37 | 8 |
| B: L_f = 0.268, no offset e, no self-weight q | 1.7266 | 176.0 | 8.10 | 9 |
| C: B + rigid-body offset e(n) | 1.4135 | 144.1 | 6.51 | 7 |
| **D: corrected (L_f, e(n), self-weight)** | **1.3580** | **138.4** | **6.22** | **7** |
| E: D with plate stiffness EI/(1−ν²) (upper bound) | 1.4756 | 150.4 | 6.82 | 7 |

Results:
- E ± 2.5 % gives n_cr between 6.05 and 6.40. Including the plate bound (case E), the range is 6.05–6.82.
- **The 7th weight causes buckling.** At n = 6, W/W_cr = 0.968. At n = 7, W/W_cr = 1.112.

## 2. Post-buckling benchmarks
**Ideal closed form (case A).** With p = sin(α/2), where α is the tip angle:
- P/P_cr = (2K(p)/π)²
- δ/L = 2p/K(p)
- drop/L = 2 − 2E(p)/K(p)

| p | α [deg] | P/P_cr | δ/L | drop/L |
|---|---|---|---|---|
| 0.1 | 11.5 | 1.0050 | 0.1270 | 0.0100 |
| 0.3 | 34.9 | 1.0480 | 0.3731 | 0.0911 |
| 0.5 | 60.0 | 1.1517 | 0.5932 | 0.2590 |
| 0.7 | 88.9 | 1.3806 | 0.7585 | 0.5310 |
| 0.9 | 128.3 | 2.1079 | 0.7893 | 0.9724 |

The shooting solver matches this table to better than 4e-11. The solver (role C) must match it to 0.5 %.

**Corrected static equilibria (case D)** are in `benchmarks.json → nonlinear_corrected`. Values are magnitudes; the declared sense is −Y.

| n | 5 | 6 | **7 (final state)** |
|---|---|---|---|
| top-end Y [mm], imperfection a₀ = L/1000 | 1.6 | 8.7 | **150.3** (perfect strip: 149.4) |
| tip angle [deg] | 0.5 | 2.8 | **53.9** |
| top-end drop [mm] | 0.006 | 0.17 | **57.4** |
| root curvature [1/m] / σ_max [MPa] | 0.04 / 2 | 0.26 / 13 | **5.21 / 252** |

The root stress at n = 7, 252 MPa, is far below the yield strength of 301 full hard (about 965 MPa or more). The strip stays elastic.
With the stack tilted 54°, weights 1–7 (31.5 mm tall) are all on the 40 mm peg, so they cannot slide off.

## 3. Dynamics
**Weight placement [A].** The contact force ramps as F = m_w·g·S((t − t₀)/τ), with τ = 0.15 s and S the smootherstep 6x⁵ − 15x⁴ + 10x³.
The weight's mass and inertia join the tip body at t₀ + τ. Before that the hand carries the weight, so there is no impact.
Axial effects are negligible: the static shortening is about 0.2 µm.

**Frequencies about the straight state** (finite-element model with preload, tip mass, offset and inertia):

| n | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| f₁ [Hz] | 2.73 | 1.76 | 1.29 | 0.97 | 0.72 | 0.49 | 0.19 |
| f₂ [Hz] | 22.9 | 21.5 | 20.6 | 19.9 | 19.1 | 18.3 | 17.4 |
| f₃ [Hz] | 67.6 | 65.6 | 63.5 | 61.2 | 58.7 | 56.1 | 53.4 |

- Rayleigh check at n = 0 without preload: 3.14 Hz.
- **Post-buckled f₁ at n = 7 ≈ 0.72 Hz** (estimated from the tangent lateral stiffness of 3.28 N/m and the effective mass).
- The weights arrive every 0.6 s. At n ≥ 4 the f₁ periods are 1.4–5 s, longer than that spacing. The pre-buckling creep therefore does not settle between weights, and the solver's dynamics decide the exact onset time.

**Buckling event.** After the 7th weight is seated (ramp 4.45–4.6 s), the deflection grows at s ≈ 7.0·√0.112 ≈ 2.3 s⁻¹, an e-folding time of about 0.43 s.
- The strip topples from about 9 mm to 150 mm in **about 1.2–1.5 s**, finishing around 5.8–6.1 s.
- It then overshoots and sways at about 0.7 Hz until 10 s.
- With Q₁ = 50 the sway decays over τ = Q/(πf) ≈ 22 s, so it is still visible at 10 s. That is physical behaviour.

**Damping [A].** Q₁ = 50, with a range of 20–150. As Rayleigh damping: α = 0.063 s⁻¹ and β = 3.2e-4 s. Add quadratic air drag 0.5·ρ_air·C_d·b·v² per metre, with C_d ≈ 2.

**Solver settings (role C).**
- Implicit time integration: generalised-α with ρ∞ = 0.8–0.9.
- Time step dt = 1/960 s ≈ 1.04 ms, with convergence levels 2.08, 1.04 and 0.52 ms. Output 4 subframes per frame.
- Mesh: at least 40 Hermite elements, or at least 100 rod segments. Convergence levels: 4, 2 and 1 mm segments.
- Checks: W_cr = 1.358 N within 0.5 %, the elastica table, the n = 7 equilibrium, and the energy residual.

## 4. Imperfection
The initial bow is w₀(s) = a₀(1 − cos(πs/2L_f)), **with a₀ = L/1000 = 0.28 mm, in the −Y sense** (lead decision; the benchmark magnitudes mirror).

Sensitivity:

| a₀ | creep at n = 5 [mm] | creep at n = 6 [mm] | final state at n = 7 [mm] |
|---|---|---|---|
| L/2000 | 0.8 | 4.4 | 149.9 |
| L/1000 | 1.6 | 8.7 | 150.3 |
| L/500 | 3.3 | 16.9 | 151.3 |

Below n_cr the creep follows w ≈ a₀/(1 − W/W_cr). The bifurcation is a stable pitchfork, so the imperfection sets only the direction and the creep before buckling. It does not change the critical weight count or the final state (within ±1 %).

## 5. Identifiability vs the AI reference clip
**What the clip can tell us (timing and appearance only):**
- the placement rhythm;
- the lean direction (−Y);
- scale.

**What it cannot tell us:**
- E, temper or yield strength, damping, or masses;
- the critical weight count. The AI clip shows the onset at 7 discs appearing mid-air, without a physical cause. Our 7 comes from the declared geometry and the assumed E;
- post-buckling amplitudes or frequencies.

All results are model-conditioned. None is a real-world measurement.
