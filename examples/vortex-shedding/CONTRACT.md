# Vortex shedding — contract (lead)
Goal: editable, reproducible, physically derived reconstruction of laminar vortex shedding behind a D = 10 mm black cylinder in a
shallow clear-acrylic channel (inner width 0.12 m, length 0.6 m, water depth 0.02 m), visualised by 7 needles of coloured dye
(magenta/cyan/yellow) at mid-depth. Keep the scene of our frame 0; replace the AI's flow with a verified Navier–Stokes solution.
Do NOT imitate the AI's defects (measured by A; e.g. wake filling the channel faster than the flow can carry dye, oversized vortices).
Frames/units: SI, world from config/scene_params.json + config/camera_ff.json (exact camera of frame 0; frame n at t = n/24 s, 241
frames). World: Z up, plate top z = 0, flow +X, channel centreline y = 0, cylinder centre (−0.040, 0), dye sheet z = 0.013, water
surface z = 0.023, needle tips x = −0.093 at y = 0, ±0.008, ±0.016, ±0.024; streak colour index per needle [0,1,2,0,1,2,0].
Physical state (lead decision): the flow has been running for a long time, so at t = 0 the wake is ALREADY in periodic shedding
(an impulsive start at Re 150 needs ≫ 10 s to shed). Dye release started shortly before t = 0 such that at t = 0 the dye front sits
at x_cyl + 0.6 D, as in frame 0. Shedding phase at t = 0 is the one free choice: anchored on the reference's first ~1.5 s (which side
rolls up first), declared as such. Nothing else is fitted to the reference.
Fluid (ASSUMED/declared): water ν = 1.0e-6 m²/s, ρ = 998 kg/m³, mean U = 0.015 m/s → Re = 150. 2-D incompressible Navier–Stokes in
the dye plane (justify: Re < 190 mode-A threshold; floor friction estimate), no-slip channel walls and cylinder, inflow/outflow
per FORMULATION. Dye = passive tracers (Sc ≫ 1), one tracer per colour group, continuous point/line sources at the needle tips.
Physics standard: Schäfer–Turek 2D-2 (Re 100 channel benchmark: St, Cd_max, Cl_max within published ranges); St(Re) vs Williamson
1996 (blockage-corrected) ≤ 2 %; mean Cd vs literature ≤ 3 %; shedding onset near Re ≈ 47 (control); 3 resolution levels with
observed convergence of St, Cd, Cl and dye fields; divergence ≤ tolerance; tracer mass conservation; no energy injection.
Interfaces (C → D/E): solver/out/dye/f0000.png … f0240.png (16-bit RGB, R,G,B = tracer 0,1,2 concentration 0..1, channel
x ∈ [−0.30, 0.30], y ∈ [−0.06, 0.06] at 0.25 mm/px, row 0 = +y edge, col 0 = −x edge; exact map in meta.json);
solver/out/fields.npz (per frame: u, v, ω on a 1 mm grid + probes: Cd(t), Cl(t), wake probe v(t)); solver/out/meta.json (params,
code sha256, levels, St, Cd, Cl, phase anchor). Scene bakes the dye PNGs as an image sequence on the dye sheet.
Ownership: measurement/ (A), physics/ (B), solver/ + params/physics.json (C), scene/ (D), verify/ (E), visual_review/ (F),
lead: CONTRACT, config/, checkpoints/, review/, STATUS.json. Agent rules: simulations/_kit/GUARDRAILS.md (paste-in rules apply).
Every handoff: labelled visuals first (version | frame/time | status), then numbers. Measured vs fitted vs assumed vs model-conditioned
always distinguished; never claim 100 % real-world accuracy from generated footage.
Tools: Basilisk (export BASILISK=…/batch2/_env/basilisk/src; export PATH=$PATH:$BASILISK; qcc -O2 f.c -o f -lm), FEniCSx env
batch2/_env/fenicsx/bin/python, generic py simulations/01-brass-pendulum/.venv/bin/python, Blender blender
(renders only via batch2/_tools/with_render_lock.sh).
