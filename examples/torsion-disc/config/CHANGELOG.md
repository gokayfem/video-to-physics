# Changelog (lead only)

## 2026-10-26T03:09:31Z — kickoff
Inherited from clip 01: anchor-window physics, per-frame camera, static tilt option, visual-evidence rule, look approach (procedural walnut/brass/steel, AgX, backdrop sweep + table).

## 2026-10-26T03:09:32Z — anchor-window rule (declared before measurement; includes clip-02 lesson)
T_a = the earliest of: (a) the first turning point after which any measured mode's half-cycle amplitude
INCREASES by > 3σ over the previous half-cycle; (b) a cumulative-drift test: the earliest time at which the
mode's amplitude or half-period trend (linear fit from t=0) departs from a passive-decay model fitted on the
first two full cycles by > 3σ in the non-physical direction (growth, or period change not explained by
amplitude); (c) a visible discontinuity (mark/shape/attachment jump). Minimum 2.5 s; if none, T_a = 10 s with
held-out last 2.5 s. Physics framing: earlier QA saw tens of degrees of twist (prompt: a few degrees), a
full-diameter mark instead of a radius, and small sideways sway. A disc on a wire is a torsion pendulum
coupled with swing (pendulum) modes; the corrected model is that general rigid-body-on-wire system,
fitted on the anchor window; sway is modelled if measured, never suppressed or invented.

## 2026-10-26T03:42:09Z — lead decisions after measurement v001 (one disclosed rule deviation)
Findings: twist is one-way −379° in 10 s with the rate rising then falling (−25 → −47 → −33 °/s): consistent
with the middle of ONE slow torsion half-swing (slow-sinusoid fit T = 31.8 ± 0.2 s, amplitude 226°), i.e.
physically plausible for a heavy disc on a thin wire; no twist defect detected. The sway (disc+wire pendulum,
T ≈ 0.95 s) shows non-physical energy injection (+50 % in 0–0.75 s, +45 % 3–5.3 s). The camera ORBITS (azimuth
+15.4° → −8.0°), modelled as per-frame camera pose; scene camera must be keyed in location AND rotation.
Rule outcome as declared: T_a = 2.5 s (global, triggered by the sway). DEVIATION (lead, disclosed): because
twist and sway are dynamically decoupled to first order (dynamics must quantify the coupling), apply the rule
PER MODE: twist mode — no trigger → fit on 0–7.5 s, hold out 7.5–10 s; sway mode — fit on 0–2.5 s.
Reason: a 2.5 s global window contains only ~75° of a 32 s twist cycle and leaves κ/I unidentifiable, while
the twist shows no defect. The declared global-window fit (all modes 0–2.5 s) is reported alongside as the
rule-compliant diagnostic. Scale: model-conditioned from the sway period (simple-pendulum inference D≈0.26 m;
dynamics to decide with the physical pendulum model) — not from the prompt's 10 cm.

## 2026-10-26T04:37:55Z — visual review of ck001: lead decisions
- VR03-04 (high): camera of record carries per-frame pose jitter (render static-scene jerk median 0.39 / max
  3.6 px vs reference 0.11 / 0.82 px). Smooth the camera path (low-pass/spline over pose parameters, preserving
  the measured orbit), re-check R4 ≤ 2 px and the reference-vs-render jerk.
- VR03-05: fit observable decision — use the disc SILHOUETTE centre P (rigidly attached to the disc), not the
  mark rotation centre Q (constant −0.031 D world-Y offset from P while the disc turns 390°: not attached → AI
  drawing inconsistency). Refit B and the pivot with P after the camera smoothing; report the change.
- VR03-06..10: one bounded look round (screw/collet detail, brass saturation + continuous rim band, ring
  banding, mark cavity, backdrop tint/brightness + softer crease, walnut tone on feet/crossbar).
- Twist after 7.5 s: AI slows faster (+11.1° at 10 s); sway envelope/phase drift: disclosed, not copied.

## 2026-10-26T07:22:38Z — lead: v004 refit collected (stale wait stopped)
The dynamics role's "block until fits finish" loop waited 3.5 h for a 'done' marker from a command that had
exited 144 after its outputs were already written. Verified outputs are complete and consistent (fits 07:44–
07:45, params 07:49 sha 1ca3ae76…, trajectory ea2b7544…, selfcheck all PASS 07:54, v004 visuals 07:54); the
stale wait (pid 97451) was stopped; nothing else touched. v004 (smoothed camera, P silhouette centre): P anchor
1.34 px (target 2.22) PASS; twist anchor 1.04° vs 3σ 0.78° FAIL (reference fidelity; AI mark wobble ~1 Hz);
tilt 0.95°; pivot (3.2, −6.4) mm (Ay weakly identified ±20 mm); T_torsion 36.4 s; Q_t 1000 assumed.

## 2026-10-26T07:43:12Z — ck003 (final look v028), lead re-check
Trajectory identical to verified ck002 (ea2b7544…). Verifier's check_blend on ck003 with the verifier's exact
arguments (--exclude-static TOR_WireEnd, --com-local 0,0,-0.0034307237): P5/P6/P8-blend PASS; dump vs ck002
PASS. (Two earlier lead runs failed only because of lead argument errors — missing --exclude-static, and a
rounded COM offset −0.0034275 that shifted frame-0 C by 3.2 µm > 1 µm tolerance; not scene defects.)
Accepted look residuals: screw washer still catches light (+0.08 L clamp region), brass cone gaps slightly greyer,
fine radial structure ~half the reference; reference wire leaves the clamp ~6 px left of ours.
