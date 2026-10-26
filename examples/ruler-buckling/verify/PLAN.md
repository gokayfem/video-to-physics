# verify/ — role E plan (independent physics & convergence validation)

Rule: nothing in solver/ is read or imported. Production outputs (solver/out/*.npz|json, checkpoints/*) are read as data;
solver studies are re-run only through their documented commands. Python: simulations/01-brass-pendulum/.venv/bin/python.

## Independent reference — ref/elastica.py
Planar inextensible Kirchhoff rod, clamped (theta=0) at the jaw line, flexible length L_f = 0.268 m (declared: top 12 mm held by
two symmetric Al tabs), self-weight q = rho*A*g, rigid tip body (plate + 2 tabs + strip-in-tabs + screws + Ø4×40 mm peg = 13.94 g,
CoM 17.0 mm above the tab bottom O) + n × 20 g weights (4.5 mm, stacked from 15 mm above O). Dead load W at the body CoM:
EI θ'' = −(W + q(L−s)) sin θ,  EI θ'(L) = W(e sin θ_L + d cos θ_L). Shooting (DOP853, rtol 1e-13) + Brent.
Bifurcation from the linearised BVP. Validated (ref/validate_reference.py → out/reference.json, visuals/v001_elastica_branch_euler_check.png):
Euler π²EI/4L² (1e-14), eccentric closed form tan kL = 1/(ke) (1e-14), Greenhill heavy column (6e-15), elliptic-integral
elastica to 150° tip angle (8e-12), tip-mass cantilever frequency equation vs independent Hermite FE (1e-7), FE buckling vs shooting (7e-9).

## Checks on production — checks/check_production.py (tolerances)
C1 P_cr (meta) vs actual-BC reference ≤0.5 % · C1b weight count at onset · C2 settled post-buckled tip deflection ≤1 % ·
C3 equilibrium residual (θ vs ∫M/EI and EI κ vs M) ≤1e-3 · C4 centreline arc length ≤1e-6 m · C5 vise nodes fixed & tab
segment rigid ≤1e-6 m · C6 energy: D = work(weights, incl. transfer contact force) − ΔU − ΔV_self − ΔT ≥ −tol and non-decreasing (tol 5e-3·max U).

## Negative controls — checks/negative_controls.py (synthetic, from the reference)
good_static, bad_E_plus1pct, bad_length_plus1pct, bad_stretch_2e-5, good_dynamic_Q50 (energy-consistent 1-DOF Ritz), bad_energy_injection.
Visual: visuals/v001_negative_controls.png.

## Checkpoint audit — checks/ck001_audit.py (+ checks/blend_dump.py run inside Blender -b)
(1) P_cr / n_cr / static n=7 vs independent; (2) quasi-static holds t<4.6 s vs independent statics using the production
imperfection inferred from its own t=0 sample; PC2 + evaluated Blender strip vs npz (≤1e-6 m); length/energy/clamp;
(3) weight schedule in the blend (seat times, ≤7 on peg, bench stack shrinking, speed / visibility / hand); (4) negative controls.
Output out/ck001_audit.json, visuals/v002_ck001_audit.png.

## Convergence (todo when requested)
Production already reports N = 67/134/268/536 and dt = 2.08/1.04/0.52 ms (solver/out/dyn_*.npz). Role E compares those files
with check_production.py metrics (tip trajectory, buckling time) as data; re-runs only via documented commands.
