# Ruler buckling — contract (lead)
Goal: editable, reproducible, physically derived reconstruction (subtle motion) of a 300×25×0.5 mm stainless strip clamped in a
vise, loaded by 20 g slotted brass weights placed one by one on a peg at the top; Euler buckling then post-buckling. The AI
reference buckles early and makes weights appear without a hand — do NOT imitate that.
Frames/units: SI, Blender world from config/scene_params.json + config/camera_ff.json (exact camera of our frame 0; frame n at t = n/24 s,
241 frames). Clamp = vise jaw-top line (z = 0.095 m); free length L = 0.280 m; strip origin bottom-centre mid-surface.
Material: ASSUMED E = 193 GPa, ν = 0.29, ρ = 7900 kg/m³ (stainless). Weights: 20 g each (declared), dead vertical load applied at the
peg (eccentric above the strip tip: include the platform + peg geometry and masses).
Loading: weights are physically transferred (visible path from the stack to the peg, lowered gently — no teleporting); the
loading schedule (times) may follow the reference's visible placement rhythm, but the deformation comes only from the solver.
Physics standard: Euler P_cr for the actual boundary conditions (clamped–free + tip mass/eccentricity) to ≤0.5 %; post-buckling vs
elastica; mesh / load-step / time-step convergence on ≥3 levels; energy/equilibrium residuals; imperfection sensitivity reported.
Interfaces: solver/out/deformation.npz (per frame 0–240 + 4 subframes: strip mid-surface node positions in world coords on the
scene mesh's 10×300 grid, plus tip load, buckling state), solver/out/meta.json (params/solver sha). Scene bakes it as a mesh cache.
Ownership: measurement/ (A), physics/ (B), solver/ + params/physics.json (C), scene/ (D), verify/ (E), visual_review/ (F),
lead: CONTRACT, config/, checkpoints/, review/, STATUS.json. Agent rules: simulations/_kit/GUARDRAILS.md (paste-in rules apply).
Every handoff: labelled visuals first (version | frame/time | status), then numbers. Measured vs fitted vs assumed vs model-conditioned
always distinguished; never claim 100 % real-world accuracy from generated footage.
Tools: FEniCSx env batch2/_env/fenicsx/bin/python (dolfinx 0.11, gmsh, pyvista); generic py: simulations/01-brass-pendulum/.venv/bin/python;
Blender blender (renders via batch2/_tools/with_render_lock.sh).
