# physics/ NOTES (physics formulation role)

## Summary
- **Model:** `planar_compound_sphere_rigid_massless_rod_v1`. The bob is a solid sphere rigidly attached to
  a massless, inextensible rod (the wire). It swings in a plane about a fixed, frictionless axis. Dissipation
  is linear viscous pivot damping, `τ = c ω`.
- **Equation of motion:** `I_P θ̈ = −m g L_c sin θ − c ω`, with `I_P = m L_c² + (2/5) m r²`.
- **Rejected alternatives, with quantified reasons** (details in FORMULATION.md §1):
  - Point mass: 6e-4 error in L for r = 1.5 cm, 2.4e-3 for r = 3 cm.
  - Flexible-wire double pendulum: changes the period by only 5e-6.
  - Elastic wire: 0.5 µm of dynamic stretch, axial mode at 89 Hz.
  - Spherical pendulum: Foucault drift gives about 2e-5 m out of plane in 10 s.
  - Wire mass: ≤ 3e-4 in L_eq; recorded as a bounded systematic.
  - Buoyancy and added mass: 1.1e-4 in period, absorbed into the fitted L_eq.
- **Damping estimates for the representative bob.**
  - Air alone gives Q ≈ 6000. Over 8 cycles that is about 0.4 % amplitude loss, roughly 0.3 px, which is
    invisible.
  - A real desk toy is likely Q ~ 10²–10³, set by the support, not by air.
  - If no decay is measured, Q is an assumption: nominal 1000, range 300–6000.

## Reproduce
```
cd ROOT && .venv/bin/python physics/sanity_calc.py > physics/sanity_calc_output.json
```
Outputs (sha256):
- `physics/sanity_calc.py` 68d995de194f4738467d2e490ed438f3394652b0e4e69c69133572fccec168f7
- `physics/sanity_calc_output.json` b9667c330834c2a2b301907c14f88de30ad6f217abcff760f351416dc2a0b225
- `physics/formulas.json` and `physics/FORMULATION.md`: hashes change on edit, so recompute them with
  `shasum -a 256`.

The ODE cross-check inside the script reproduces the elliptic period to 8e-14 relative (DOP853,
rtol 1e-12). The flexible-wire comparison is a separate inline generalised-eigenvalue calculation,
documented in FORMULATION.md §1(b).

## Sources

**Pendulum theory**
- Exact pendulum period with the complete elliptic integral, standard result. K(k) definition: NIST DLMF
  §19.2 https://dlmf.nist.gov/19.2. Implementation note: SciPy's `ellipk(m)` uses the parameter m = k²,
  see https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.ellipk.html

**Air properties**
- U.S. Standard Atmosphere, 1976 (NOAA-S/T-76-1562, NASA-TM-X-74335), https://ntrs.nasa.gov/citations/19770009539.
  It supplies the Sutherland viscosity law `μ = 1.458e-6 T^1.5/(T + 110.4)` and the air gas constant
  287.053 J/(kg K).
- At 20 °C and 101 325 Pa these give ρ = 1.2041 kg/m³ and μ = 1.8134e-5 Pa s, so ν = 1.506e-5 m²/s.
- Cross-check: the Omni Calculator table lists 1.204 kg/m³ and 1.813e-5 Pa s at 20 °C (secondary source):
  https://www.omnicalculator.com/physics/kinematic-viscosity-of-air

**Sphere drag**
- Morrison, F. A., "Data Correlation for Drag Coefficient for Sphere", Michigan Tech, 2016:
  https://pages.mtu.edu/~fmorriso/DataCorrelationForSphereDrag2016.pdf. This is an explicit Cd(Re)
  correlation fitted to Schlichting's data, valid up to Re 1e6.
- NASA Glenn, "Shape Effects on Drag": https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/shape-effects-on-drag/.
  It gives a sphere Cd of "from .07 to .5" and says Cd is strongly Re-dependent.
- The commonly quoted Cd ≈ 0.47 is the subcritical plateau value, about Re 1e3–2e5. **At our Re ≈ 270–810
  the correlation gives Cd ≈ 0.5–0.7, so 0.47 is not appropriate here.**

**Oscillatory sphere drag and cylinder drag**
- Oscillating-sphere (Stokes) drag: G. G. Stokes (1851), "On the effect of the internal friction of fluids
  on the motion of pendulums", Trans. Camb. Phil. Soc. 9, 8. Also Landau & Lifshitz, *Fluid Mechanics*,
  §24, problem 5.
- Cylinder drag fit `Cd ≈ 1 + 10 Re^{-2/3}`: F. M. White, *Viscous Fluid Flow* / *Fluid Mechanics*. I
  recalled this from memory and did not re-fetch it; it is used only for an order-of-magnitude wire-drag
  bound.

**Brass density**
- C26000 cartridge brass about 8.53 g/cm³, C36000 free-cutting brass about 8.50 g/cm³.
- These come from supplier and metallography datasheets:
  https://www.metallographic.com/materials/c26000-cartridge-brass and
  https://fiskalloy.com/product/ma36b/
- The Copper Development Association pages (https://alloys.copper.org/alloy/C26000) returned HTTP 403 to
  my fetcher, so the CDA value is **not verified first-hand**.
- The adopted range 8400–8730 kg/m³ covers the common brasses. Mass is unidentifiable from the clip
  anyway.

**Steel**
- Steel E = 190–210 GPa and ρ = 7850 kg/m³ are standard handbook values, assumed and not fetched. They
  matter only for µm-level stretch.

## Open questions and requests
1. **To measurement:** report the ratio L_c/r (pivot-to-bob-centre over bob radius) in the swing plane,
   after camera correction, with an uncertainty.
   - A rough thumbnail read gives about 13. With T = 1.25 s that implies a bob of about 6 cm diameter
     (about 0.96 kg), not 3 cm.
   - The period sets the scale, so this changes the mass and the Q estimate, not the kinematics.
   - Also report: any out-of-plane motion, an amplitude envelope with σ (to decide whether damping is
     fitted or assumed), and whether amplitude *grows*. Growth would be a reference defect; never fit
     negative damping.
2. **To camera:** the true θ amplitude needs the swing-plane orientation. Image-plane angles are
   foreshortened unless the plane is fronto-parallel.
3. **To geometry:** is the wire looped over the pin (rolling contact, so a moving pivot) or clamped at a
   point? The contract fixes P. A looped wire is a systematic of about 4e-5 in period for a 1.5 mm pin.
   Report the pin radius if it is visible.
4. **To dynamics:**
   - Use `T_wire = m (g cos θ + L_c ω²)` for the CSV `tension_N` (general form, valid with damping).
   - Use `PE = m g L_c (1 − cos θ)` and `P_diss = c ω²`.
   - Take c from `c = I_P ω₀ / Q`.
5. **To verifier:**
   - Run P9 damping checks at small amplitude, or compare energy envelopes, because the closed form
     `exp(−γt)` is a first-order small-damping result.
   - P3 uses `T = 4 sqrt(I_P/(m g L_c)) K(sin(θ₀/2))`. Pass m = k² if using SciPy.
6. **No contract change requested.** One note for the lead: P5's `|L(t) − L_c| < 1e-6 m` is a rig check.
   The physical elastic stretch variation of a real wire (0.5–9 µm depending on r and d_w) would exceed
   it, but the model is inextensible by declaration, so there is no conflict.
