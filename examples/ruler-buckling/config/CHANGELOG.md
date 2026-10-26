# Changelog (lead)

## 2026-10-26T09:14:19Z — reference approved by user; contract v1; one-at-a-time order ruler→vortex→glass.

## 2026-10-26T09:22:23Z — loading schedule (lead, after measurement v001)
Reference (measured): weights seat at f36/60/84/112 (counts 1/3/5/7 — two per visible disc, appearing mid-air), onset f126 at 7 weights,
non-physical reversal and edgewise lean later. Decision: physical schedule = ONE weight every 0.60 s starting t = 1.00 s
(weights 1..12 seated at 1.0, 1.6, …, 7.6 s), each lowered visibly from the stack (no teleporting; the stack shrinks), contact-force ramp
per FORMULATION. Buckling time/shape then come only from the solver (expected ≈9th weight ≈ 5.8 s). AI defects documented, not copied.

## 2026-10-26T09:22:56Z — lead decisions after formulation v001 (supersede the 12-weight schedule)
Corrected P_cr = 1.358 N (rigid tab length 12 mm, eccentric tip body, self-weight) → buckles while the 7th weight is placed
(ideal Euler would say 8th; the old '9th' note was wrong). Decisions: (1) strip declared spring-temper stainless (AISI 301 full hard,
E 193 GPa assumed, yield ≫ 611 MPa) so it stays elastic; (2) peg height 40 mm (guides ≥ 8 weights; weights cannot slide off at the
post-buckled tilt); (3) schedule: 7 weights only, one every 0.60 s from t = 1.00 s (seated at 1.0, 1.6, 2.2, 2.8, 3.4, 4.0, 4.6 s);
buckling is a slow topple (~1.5 s) then a damped sway (Q1 ≈ 50 assumed) to 10 s; (4) buckling direction toward −Y (camera-left),
matching the reference's visible lean (imperfection L/1000 in that sense); (5) tip body mass at n=0 = 11.96 g (platform+tabs+peg).

## 2026-10-26T09:25:16Z — correction to item (5) above
With the 40 mm peg (3.97 g) the tip body at n = 0 is 13.94 g (not 11.96 g); CoM 17.0 mm above the tab bottom at n = 0, 29.5 mm at n = 7.
P_cr unchanged 1.358 N; n_cr = 6.22 → 7th weight. n = 7 static: 150 mm sideways, 53.9°, root stress 252 MPa (elastic for 301 FH).

## 2026-10-26 ck001 (lead)
- Solver v001 accepted: discrete Kirchhoff elastica (numpy/scipy, not dolfinx; exact inextensibility), N=268, generalized-α ρ∞ 0.85, dt 0.52 ms.
  Validated: Euler P_cr err 2.6e-6; full-model n_cr 6.2229 vs independent shooting 6.2224; static n=7 150.32 mm (bench 150.35).
  Dynamic convergence: mesh 0.30 mm (N134), dt 1.2 µm. Energy residual 1.9e-4 of peak strain energy.
- Assumed damping Q≈50 (Rayleigh + quadratic drag) decides the pose at 10 s (−170 mm; undamped −15.6 mm). Disclosed as assumed.
- Scene v002 (ceiling-mounted SCARA, column out of frame) + solver npz → checkpoints/ck001 via build_all.sh. All scene checks pass.

## 2026-10-26 ck001 audit (lead)
- Verifier E: 6/7 PASS (independent W_cr 1.3575726 N, n_cr 6.2224; n=7 static 150.339 vs 150.316 mm; PC2/evaluated mesh 7.5e-9 m; negative controls 6/6).
- FAIL item 3: weight transfers peak ~14 m/s (teleport), stack pitch 4.70 vs 4.5 mm. → D fix round v003: raised feeder or equivalent, peak ≤1 m/s, seat offsets from meta.json.
- Max tip rotation 75.4° (t 6.84 s) < 90°: weights stay pressed on the plate; rigid-stack assumption holds.
- Post-buckling sway amplitude at 10 s depends on assumed damping (undamped differs by up to 182 mm). Disclose as assumption.
