#!/bin/zsh
# Exact regeneration of the FITTED params/physics.json (clip 02, v005: scene geometry v003, solver v2). Run from anywhere.
# Upstream chain (deterministic; bootstrap seeds fixed):
#   .venv/bin/python solver/prepare_tracks.py
#   C=".venv/bin/python solver/fit_projected.py --tracks solver/out/tracks_for_fit.csv --points ring wctr --pivot-dz 0.225071 --pivot-dx -0.002875 \
#      --k-coil --axial-rest --rock-rest --damping auto --nboot 30"
#   $C --t-anchor 7.5 --out solver/out/fit/prod_Ta7.5        (PRODUCTION: rule T_a = 10 s -> fit 0-7.5 s, hold-out 7.5-10 s)
#   $C --t-anchor 4.0 --out solver/out/fit/diag_Ta4.0        (labelled DIAGNOSTIC, lead request)
set -e
cd "${0:A:h}/.."
.venv/bin/python solver/make_params.py --status fitted --fit-json solver/out/fit/prod_Ta7.5/fit_projected.json \
 --status-note "FITTED on 0-7.5 s (CHANGELOG rule: no >3 sigma growth -> T_a = 10 s, last 2.5 s held out). Passive elastic pendulum + hinged rigid weight (in-phase double pendulum), swing pivot P = spring top coil on the fixed S-hook (P_eff, 0.225 W_in below A; lead decision), projected through params/camera.json per-frame cameras. Scale model-conditioned from the swing frequency (g assumed). Geometry from params/geometry.json (scene geometry v003, role C: b_c 0.209616 W_in). k from the coil formula (declared assumption, G 79.3 GPa, 21 active turns), axial and rocking modes start at rest. Q = 500 declared (damping not identifiable: AI amplitude grows 1-4 s then decays). Fit-window RMS 6.72 px (R2 target 3.24 px: FAIL; AI period wander 7.5 % and AI weight rocking 1.3x the physical in-phase tilt), hold-out 15.5 px." \
 --prov-damping assumed --source-damping "Q = 500 for swing and axial (declared; range 100-5000). M1 in the fit window: gamma_sw at the 0 bound, bootstrap 95% includes 0 -> not identified (solver/out/fit/prod_Ta7.5/fit_projected.json model_selection)"
