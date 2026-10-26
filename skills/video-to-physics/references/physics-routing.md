# Physics Routing

Start here once the phenomenon is known. Each section gives the model to start from, solver options from
cheapest to heaviest, benchmarks with published values and typical tolerances, and the defects generators
usually add. Pick the **cheapest model the benchmarks can validate**; move up only when a stated effect needs it.

Benchmark values below are standard textbook or literature results; cite the source in `benchmarks.json` and
re-derive anything geometry-specific in `physics/derive.py`. Tolerances are typical; the contract fixes the real
ones before data.

## Contents

- [Choosing the model class](#choosing-the-model-class)
- [Solver options](#solver-options)
- [Rigid bodies, pendulums, oscillators](#rigid-bodies-pendulums-oscillators)
- [Contact, rolling, bouncing](#contact-rolling-bouncing)
- [Elastic rods and beams](#elastic-rods-and-beams)
- [Continuum FEA](#continuum-fea)
- [Incompressible CFD with dye or smoke](#incompressible-cfd-with-dye-or-smoke)
- [Free-surface and sloshing](#free-surface-and-sloshing)
- [Fracture](#fracture)
- [Granular and other](#granular-and-other)

## Choosing the model class

Ask in order; the first "yes" picks the class.

| Question | Class | Start with |
|---|---|---|
| Does it move as a few rigid bodies (swing, spin, roll, bounce)? | rigid-body ODE | reduced coordinates, exact constraints, event handling |
| Is it a slender rod, strip, wire or beam that bends (buckling, vibration, sag)? | 1-D elastica / beam | discrete Kirchhoff rod (`v2p/elastica.py`) or Euler–Bernoulli modes |
| Is it a 2-D/3-D solid whose stress or shape field matters (plates, blocks, contact)? | continuum FEA | FEniCSx, linear or hyperelastic |
| Is it a flow made visible by dye, smoke or particles? | incompressible CFD + passive tracer | 2-D if Re and geometry allow; Basilisk or FEniCSx |
| Does a liquid surface move (slosh, waves, drops)? | free-surface flow | linear modes first; VOF only if needed |
| Does something crack or break? | fracture | stress analysis for the origin; phase-field for the path; rigid bodies for the pieces |
| Is it many grains (pour, pile, hourglass)? | granular | continuum laws first (Beverloo, angle of repose); DEM if needed |

Before calling a motion "non-physical", test the general physical system: a sideways swing of a spring-hung
weight is an elastic-pendulum mode; a one-way rotation with rising then falling rate can be half of a slow
torsion cycle.

## Solver options

| Option | Use for | Notes |
|---|---|---|
| Closed form | linear oscillators, elliptic periods, Euler loads, static beams | always the first benchmark |
| ODE (`v2p/ode.py`: RK4, generalised-α; scipy DOP853) | rigid bodies, reduced models, modal models | fixed step for reproducible bakes; adaptive with tight rtol for references |
| Discrete elastica (`v2p/elastica.py`) | inextensible rods and strips, buckling and post-buckling | exact inextensibility, no locking, few hundred DOFs |
| FEniCSx (dolfinx) FEM | plates, solids, hyperelasticity, phase-field fracture, CFD cross-checks (IPCS, Taylor–Hood) | conda-forge install; mesh with gmsh |
| Basilisk | 2-D/3-D incompressible flow, adaptive grids, embedded boundaries, VOF | C, compiles with a C compiler; good for cylinder wakes |
| Other CFD (OpenFOAM, lattice Boltzmann) | larger 3-D flows, cross-checks | heavier setup; same benchmarks apply |
| Rigid-body engines (Bullet, MuJoCo) | many contacts, stacking | only if they pass the same independent checks as a reduced ODE |
| DEM (custom, LIGGGHTS, Yade) | granular flows | expensive; benchmark against continuum laws |

Installs (FEniCSx, Basilisk, others) need the user's approval; keep environments inside the project.

## Rigid bodies, pendulums, oscillators

Model: planar or 3-D rigid bodies in reduced coordinates; quaternions or exact projection for orientations;
damping as a declared law (linear viscous, Q) unless identifiable.

| Benchmark | Value | Typical tolerance |
|---|---|---|
| small-amplitude pendulum | T₀ = 2π√(L/g); compound: T₀ = 2π√(I/(m g d)) | 1e-7 relative |
| finite amplitude | T = (2/π) T₀ K(k), k = sin(θ₀/2), K complete elliptic integral of the first kind | 1e-7 relative |
| damped oscillator | log decrement δ = π/Q; amplitude ∝ e^(−ω₀t/2Q) | 1e-6 relative |
| torsion pendulum | T = 2π√(I/κ), κ = GJ/L, J = πd⁴/32 for a round wire | 1e-7 relative |
| elastic pendulum | spring–swing internal resonance near ω_spring = 2 ω_swing | qualitative + energy exchange |
| conservative energy | drift \|ΔE\|/E₀ | < 1e-8 |

Generator defects: camera zoom or orbit read as motion; amplitude growth; period drift; cadence jitter; pivot or
rotation centre in the wrong place; exaggerated secondary tilt; absent modes the prompt asked for.

## Contact, rolling, bouncing

Model: reduced ODE with exact rolling constraint (angle tied to arc length) and event detection for impacts;
restitution and friction as declared parameters.

| Benchmark | Value | Typical tolerance |
|---|---|---|
| rolling on an incline | a = g sin β / (1 + I/(m r²)) | 1e-7 relative |
| ball in a bowl (radius R, ball r) | ω² = g / ((R − r)(1 + I/(m r²))) for small oscillations | 1e-7 relative |
| bounce height ratio | h_{n+1}/h_n = e² | 1e-7 relative |
| event times | independent solver | < 1e-6 s |
| forces | normal ≥ 0 in contact; friction inside the cone | exact sign |

Generator defects: sliding without rolling, energy gained in bounces, interpenetration, bodies floating after
impact, contact without deformation or sound cue timing.

## Elastic rods and beams

Model: inextensible Kirchhoff rod (elastica) with self-weight, tip bodies and eccentric loads; Euler–Bernoulli
modes for small vibration. Check torsional and lateral-torsional buckling loads are far above the load before
using a planar model; check strong-axis bending is excluded by the section ratio.

| Benchmark | Value | Typical tolerance |
|---|---|---|
| Euler clamped–free | P_cr = π² EI / (4 L²) | ≤ 0.5 % (solver usually 1e-5) |
| Euler pinned–pinned / clamped–clamped | π² EI / L² / 4 π² EI / L² | ≤ 0.5 % |
| eccentric tip load (clamped–free, offset e on the tangent) | k L tan(k L) = L / e, k² = P / EI | 1e-5 |
| heavy column (Greenhill), clamped–free | q L³ / EI = 7.837 | 1e-5 |
| elastica post-buckling, clamped–free | P/P_cr = (2 K(p)/π)², p = sin(α/2), α = tip rotation (α = 90° gives 1.3932) | ≤ 1 % |
| cantilever modes | ω_n = (β_n L)² √(EI / (ρ A L⁴)), β_n L = 1.8751, 4.6941, 7.8548 | ≤ 0.5 % |
| root stress | σ = M c / I; compare with the declared yield | elastic margin stated |

Generator defects: buckling at the wrong load or in an S-shape, strip stretching or changing width, bending about
the strong axis, the bent shape relaxing or reversing under constant load, loads appearing without a carrier.

## Continuum FEA

Model: linear elasticity for small strain; St Venant–Kirchhoff or neo-Hookean for large deformation;
generalised-α or Newmark for dynamics; Rayleigh damping as a declared assumption.

| Benchmark | Value | Typical tolerance |
|---|---|---|
| cantilever tip deflection | δ = P L³ / (3 EI) | ≤ 0.5 % |
| plate / beam modal frequencies | Euler–Bernoulli or Kirchhoff plate tables for the BCs | ≤ 1 % |
| Cook's membrane (plane stress, incompressible variant) | published converged tip displacement for the chosen formulation | ≤ 1 % |
| patch test | exact linear field | machine precision |
| energy balance (dynamics) | work in = strain + kinetic + dissipated | residual ≤ 1e-3 of peak strain energy |
| mesh convergence | ≥ 3 meshes, observed order | order within 0.5 of theory |

Watch for shear and membrane locking with low-order elements on thin parts; prefer a rod or shell model there.

## Incompressible CFD with dye or smoke

Model: incompressible Navier–Stokes; 2-D when Re is below the 3-D transition and the geometry is planar
(cylinder wake: mode-A onset Re ≈ 188–190); passive tracers advected as material lines when Sc ≫ 1 (dye in water).
Shallow layers need a floor-friction term; check the shallow-wake stability number.

| Benchmark | Value | Typical tolerance |
|---|---|---|
| Schäfer–Turek 2D-1 (Re 20, steady) | C_D 5.57–5.59, C_L 0.0104–0.0110 | inside the published band |
| Schäfer–Turek 2D-2 (Re 100, periodic) | C_D,max 3.22–3.24, C_L,max 0.99–1.01, St 0.295–0.305 | inside or within 3 % of the band |
| cylinder St(Re), unbounded (Williamson 1996) | St = −3.3265/Re + 0.1816 + 1.6e-4 Re (49 < Re < 178); ≈ 0.164 at Re 100, ≈ 0.183 at Re 150 | ≤ 1–2 % after blockage correction |
| mean drag, unbounded cylinder, Re 100 | C_D ≈ 1.33–1.35 (Park et al. 1998; Henderson 1995) | ≤ 2–3 % |
| shedding onset | Re ≈ 46–47 (Hopf); control: no shedding below | qualitative control |
| Taylor–Green vortex | exact decay e^(−2νk²t) | ≤ 1e-3 relative |
| Poiseuille / lid-driven cavity | exact profile / Ghia et al. 1982 tables | ≤ 1 % |
| vortex convection speed | ≈ 0.85–0.9 U in the near wake (cylinder) | ≤ 5 % |
| divergence, mass | ‖∇·u‖ small; tracer mass budget | per contract |

Correct for blockage (Allen–Vincenti / Maskell-type effective velocity) when comparing with unbounded data.
Report force amplitudes (C_D, C_L′) as well as St; coarse grids can match St while forces are 10–20 % off.

Generator defects: time compression (dye fronts faster than the flow can carry them), symmetric instead of
alternating vortices, oversized vortex rows, dye appearing from nowhere or changing colour balance, vortices
accelerating downstream or moving upstream, a wake that looks impulsively started.

## Free-surface and sloshing

| Benchmark | Value | Typical tolerance |
|---|---|---|
| linear slosh mode, rectangular tank length L, depth h | ω_n² = g k_n tanh(k_n h), k_n = nπ/L | ≤ 1 % |
| deep-water wave dispersion | ω² = g k | ≤ 1 % |
| capillary–gravity | ω² = (g k + σ k³/ρ) tanh(k h) | ≤ 2 % |

Generator defects: liquid volume not conserved, waves faster or slower than dispersion allows, surface not
level at rest.

## Fracture

Model: stress analysis (beam or FEA) for the load and origin; phase-field fracture (AT1/AT2) for the path; rigid
bodies for the pieces after separation. Brittle crack speeds are far above video rates, so the checkable results
are the origin, the path and the failure load range, not the propagation speed.

| Benchmark | Value | Typical tolerance |
|---|---|---|
| three-point bending, round rod radius r, span L | σ_max = F L / (π r³) at mid-span, tension face | exact by theory |
| failure load | from a declared strength range (brittle strength is statistical) | consistency, not prediction |
| crack origin | maximum tensile stress location | exact by theory |
| phase-field implementation | single-edge-notched tension and shear tests (Miehe et al. 2010) | within published scatter |
| fragment motion | rigid-body checks as above | as rigid bodies |

Generator defects: crack starting on the compression side, several random breaks, excessive bending before
failure, halves floating or bouncing without energy loss.

## Granular and other

| Phenomenon | Model | Benchmark |
|---|---|---|
| hopper or hourglass discharge | Beverloo law W = C ρ_b √g (D − k d)^{5/2}, C ≈ 0.58, k ≈ 1.5 | rate constant in time (independent of fill height) |
| pile formation | angle of repose (material, declared) | slope angle |
| projectile with drag | ODE with quadratic drag | vacuum limit parabola; terminal speed √(2mg/(ρ C_D A)) |
| cloth or hair | XPBD or FEM shell; declared stiffness | static drape vs catenary for a hanging chain y = a cosh(x/a) |

Granular generator defects: discharge rate changing with fill height, grains passing through walls, piles at the
wrong angle. When no benchmark applies, the case needs an independent second implementation and conservation
checks before it can claim verification; say so in the report.
