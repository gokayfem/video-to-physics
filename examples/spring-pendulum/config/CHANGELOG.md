# Changelog (lead only)

## 2026-10-25T23:40:54Z — kickoff
Inherited from clip 01: anchor-window physics, per-frame camera, static tilt option, visual-evidence rule, look approach (procedural walnut/brass/steel, AgX, backdrop sweep + table).

## 2026-10-25T23:41:09Z — anchor-window rule (declared before measurement)
The anchor window is [0, T_a] where T_a is the first turning point (in any measured oscillation mode:
vertical bob, lateral swing) after which that mode's half-cycle amplitude INCREASES by more than 3 sigma
relative to the previous half-cycle, or a visible discontinuity occurs (attachment jump, coil-count change,
shape change). Minimum 2.5 s; if no such event occurs, T_a = 10 s with a held-out last 2.5 s.
Physics framing: the reference shows a lateral swing that the prompt excluded. A spring-mass system CAN swing
laterally (elastic/spring pendulum) if released with lateral displacement; the corrected physics is therefore
the 3D elastic pendulum (plus rigid-body rocking if observed) fitted on the anchor window, not a forced
vertical-only motion. Mode coupling (autoparametric resonance near omega_axial = 2*omega_swing) is real
physics and must be predicted, not suppressed.

## 2026-10-26T00:21:45Z — bake density (lead)
Verifier finding: whole-frame Hermite keys miss rigid rocking modes (15–25 rad/s) by ~4e-4 rad between
frames (P6 tolerance 1e-5 rad). Decision: bake keys at every 1/96 s (4 per frame: subframe keyframes
allowed) for all baked channels; the P6 tolerance is unchanged.

## 2026-10-26T00:52:12Z — lead decisions after measurement v001
1. Anchor rule applied as declared: T_a = 10.0 s (no >3σ half-cycle jump). Per the rule, the production fit
   uses 0–7.5 s with 7.5–10 s held out. The reference also shows GRADUAL non-physical drift (swing amplitude
   +4.5 % to ~4 s then -7.5 %; half-period +7.5 %, 6σ trend), which the declared rule does not catch. Not
   changing the rule after seeing data; instead a clearly labelled DIAGNOSTIC fit anchored on 0–4.0 s is
   reported alongside. (Lesson for clips 03+: declare a cumulative-drift criterion up front.)
2. Hinge: the swing is along the crossbar axis; a hook seated on a bar cannot rotate in that plane, so the
   spring bends at its top coils. Adopt a fixed hook and a swing hinge at P_eff (top coil, 0.225 W_in below A)
   as the physical idealization (bending concentrated at the clamp-like hook). Rigid weight rocking about the
   eye hinge (double-pendulum in-phase mode, tilt ≈1.3–1.4× swing) is in the model.
3. No axial bob is present: spring stiffness k is unidentifiable — declared assumption with range (dynamics
   chooses, e.g. from typical coil-spring stiffness for the visible static stretch), axial mode initially at
   rest; any axial motion is then predicted by coupling only.
4. Length unit W_in (crossbar inner span); metric scale model-conditioned via the swing period and g.

## 2026-10-26T01:53:54Z — P9 failure on ck001 (lead decision)
Verifier: P9 formula checks pass, but the variant-vs-reference sub-check FAILS for k×4 (production NaN at
1.96 s; body-axis |n| grows to 2.6) and L0+50 % (h vs h/2 differ 3.4 mm). Root causes: (1) production solver
does not preserve |n| = 1 (constraint drift) — a solver defect, must be fixed (quaternion/attitude
integration or exact projection, plus a convergence guard); (2) those variants started from the nominal
initial state, compressing the spring (min tension −101/−140 N), which a hooked spring cannot sustain.
Decision: fix (1) in production; define P9 variants with re-equilibrated initial conditions (variant's own
static stretch, same swing amplitude) — this tests control response as intended; the tolerance is NOT
relaxed. The solver must also flag any compressive spring force (P7) instead of silently continuing.

## 2026-10-26T02:49:48Z — ck003 (look v016 + lower-loop wire fix), lead re-check
Trajectory/params identical to verified ck002 (trajectory sha d45be5ab…). Verifier's check_blend on ck003:
P5/P6/P8-blend PASS. Dump comparison ck002 vs ck003: 84/85 arrays bitwise identical (all keyframes, weight
matrix, eye, coil counts, topology); 'spring_bottom' differs by 5.0 mm — expected, because VR-18 changed
the lower-loop wire geometry where that point is measured; the attachment itself passes P5 in ck003.
