# Formulation and Benchmarks

The formulation is the declared physics: the model, why it is sufficient, every parameter with its provenance,
and the numbers the solver must hit before production. It is written before any production run and changed only
by a logged lead decision.

## Contents

- [FORMULATION.md structure](#formulationmd-structure)
- [Labels](#labels)
- [Choosing the model](#choosing-the-model)
- [Initial state and anchoring](#initial-state-and-anchoring)
- [Loading, forcing and logistics](#loading-forcing-and-logistics)
- [Benchmarks](#benchmarks)
- [Sensitive assumptions](#sensitive-assumptions)
- [benchmarks.json](#benchmarksjson)

## FORMULATION.md structure

1. **Decisions table**: one line per choice (velocity definition, boundary conditions, damping law, initial state).
2. **Data**: every parameter with value, unit and label.
3. **Model and why it is sufficient**: governing equations; the effects neglected and the number that shows each
   is small (slenderness, Re vs transition, stiffness ratios, buckling loads of other modes).
4. **Boundary and initial conditions.**
5. **Benchmarks**: closed forms and published results with sources and tolerances.
6. **Expected production behaviour**: the numbers the production run should give (onset time, period, amplitude,
   St, deflection), so a wrong run is visible at once.
7. **Limits**: what the model does not capture, quantified where possible.
8. **Sensitivity**: which results depend on which assumptions.

Keep `physics/derive.py` short and fast; it regenerates `benchmarks.json` and the expectations plot.

## Labels

| Label | Meaning | Example |
|---|---|---|
| measured | read from the reference or our own scene with an uncertainty | ratio of two lengths in the image; geometry of our first frame |
| declared | chosen scene design (for `idea`/`none`, or props we add) | load schedule, dimensions of our first frame |
| assumed | a value not observable here, taken from literature or chosen | Young's modulus, viscosity, damping Q, g |
| fitted | estimated from the reference inside the anchor window | initial angle, period |
| model-conditioned | follows from a fitted value through the model and assumptions | absolute length from a period and assumed g |
| derived | computed from the above | critical load, Reynolds number |

Every value in the formulation, the report and the review page carries one of these labels.

## Choosing the model

- Start from `physics-routing.md`. Use the cheapest model that the closed forms can validate: a discrete elastica
  beat a shell model for a thin strip (exact inextensibility, no locking, hundreds of DOFs).
- Model the general physical system, not the prompt's intended motion. If the reference shows a real mode the
  prompt did not ask for, the model includes it.
- Include every body that carries load or inertia: tip masses, eccentric offsets, self-weight, platforms. An
  idealised closed form can miss the onset by a whole load step.
- For rotational or constrained degrees of freedom, specify quaternions or exact projection, and runtime guards
  on constraint drift and non-finite state.
- For fluids, define where velocities are measured (for example mid-depth centreline at the body station in an
  empty channel) so inflow calibration has a target.
- Declare material assumptions that keep the model valid (for example an elastic limit high enough for the
  computed stress); check them against the computed stress.

## Initial state and anchoring

- The physical state at t = 0 is a decision: at rest, mid-cycle, or a developed periodic state (a wake that has
  been running for a long time). Choose what is physical for the scene, not what is convenient.
- **Anchor rule** (references with motion): fit only on an opening window ended by a rule declared in
  `acceptance.json` before any fit metric exists. Declare both a jump criterion (a half-cycle change > 3σ) and a
  cumulative-drift criterion (amplitude or period trend > 3σ from the first cycles), since drift can be gradual.
- Per-mode windows are allowed only for modes shown to be dynamically decoupled (quantify the coupling); disclose
  as a deviation and report the rule-compliant fit next to it.
- Name the anchored choices in the contract (initial angle and rate, buckling side, shedding phase). Fit nothing
  else. After the window, physics runs unchanged.
- For `idea` with a generated clip, frame 0 is ours; anchor only phase-like choices from the clip's opening.

## Loading, forcing and logistics

- A load or forcing schedule is a scene decision fixed before solving: times, magnitudes, contact ramps.
- Every load has a visible physical cause (an arm, a hand, a feeder). Design the logistics before building:
  required peak speed ≈ 2 × path length / (cycle time − seat time). Keep carried props at walking-speed levels
  (≈ 1 m/s or less for small objects); move the supply close to the load point if the schedule is tight.
- Contact forces of transfers enter the energy budget.

## Benchmarks

| Kind | Purpose | Example |
|---|---|---|
| closed form | proves the discretisation | Euler load, elliptic period, Poiseuille profile |
| published benchmark | proves the solver on a standard problem | Schäfer–Turek 2D-2, Miehe notched tests |
| actual-BC reference | proves the production configuration | critical load with tip body and self-weight by shooting |
| control response | proves parameters act as the equations predict | stiffness × 2 changes the period by 1/√2; no shedding below onset |
| limit case | proves continuity to known regimes | zero damping conserves energy; zero friction recovers the 2-D result |

Every benchmark has a source, a value, a tolerance and the command that reproduces it. Tolerances go in
`acceptance.json` before production and are never relaxed after a failure.

## Sensitive assumptions

Name, up front, the assumption that most changes what the viewer sees. Typical: damping sets the post-event sway
amplitude and the end pose; an imperfection sets the buckling side; a friction law sets the stopping point.
Report a sensitivity run (for example undamped vs declared damping) with the size of the effect.

## benchmarks.json

```json
{
  "version": "v001",
  "generated_by": "physics/derive.py",
  "parameters": {"E_Pa": {"value": 1.93e11, "label": "assumed", "source": "handbook value, spring-temper stainless"}},
  "benchmarks": [
    {"id": "B1_euler", "quantity": "P_cr_N", "value": 1.5818, "formula": "pi^2 EI/(4L^2)",
     "source": "Euler", "tolerance_rel": 0.005}
  ],
  "expected_production": {"onset_load_step": 7, "onset_t_s": 4.66},
  "sensitivity": [{"assumption": "damping Q", "affects": "end pose", "range": "undamped vs Q=50"}]
}
```

The numbers above show the shape only; every case derives its own.
