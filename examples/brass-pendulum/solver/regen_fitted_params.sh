#!/bin/zsh
# Exact regeneration of the FITTED (anchored + level tilt, v004) params/physics.json. Run from anywhere.
# Upstream chain (deterministic):
#   for T in 3.6 4.0 4.5: .venv/bin/python solver/fit_projected.py --t-anchor $T --fit-tilt --damping auto --nboot 40 --out solver/out/fit_tilt/Ta_$T
#   .venv/bin/python solver/fit_projected.py --t-anchor 4.0 --fit-tilt --damping fitted --nboot 40 --out solver/out/fit_tilt/Ta_4.0_prod   (M1 candidate)
#   .venv/bin/python solver/tilt_damping_evidence.py
#   .venv/bin/python solver/tilt_check.py --fit-dir solver/out/fit_tilt/Ta_4.0
#   .venv/bin/python solver/make_fit_provenance.py --fit-dir solver/out/fit_tilt/Ta_4.0 --evidence solver/out/fit_anchored/anchor_evidence.json \
#        --damping-evidence solver/out/fit_tilt/damping_evidence.json --tilt-check solver/out/fit_tilt/Ta_4.0/tilt_check.json
set -e
cd "${0:A:h}/.."
.venv/bin/python solver/make_params.py --status fitted \
 --status-note "FITTED, ANCHORED on the opening window t in [0, 4.0] s (CHANGELOG 2026-10-25T23:06:55Z; T_a from the pre-set 3-sigma growth-onset rule) WITH a fitted level tilt of the static scene relative to true gravity (CHANGELOG 2026-10-25T23:14:43Z). Passive rigid compound pendulum under true gravity, projected through params/camera.json per-frame cameras in the tilted stand frame, hinge at P_ring. Correct physics runs unchanged to 10 s; motion after T_a is a divergence map. Damping: fitted Q ~100-140 is consistent across windows but its bootstrap 95% interval leaves [100, 6000] -> Q = 1000 declared assumption. Anchor RMS 3.24 px; R2/R3 as declared FAIL (13.9/27.3 px); see fit_provenance." \
 --L-eq 0.47510037330783805 --sigma-L-eq 0.002604811846664234 --L-eq-source "fitted on anchor window 0-4.0 s with level tilt: projected pixel fit (period through RK4 model, g assumed); sd = moving-block bootstrap (formal-scaled 0.00084); T_a sensitivity 3.6-4.5 s: 0.4730-0.4751 m" \
 --theta-amp-deg 6.798047048425182 --prov-amp fitted --source-amp "max |theta| (relative to true gravity) of the fitted model over 0-10 s (informational; used only for T_meas = model period)" \
 --prov-T model-conditioned --source-T "model finite-amplitude period at the fitted L_eq (not a raw measurement; measured opening full periods 1.370/1.396/1.417/1.397 s)" \
 --ratio 12.30 --sigma-ratio 0.45 --prov-ratio measured --source-ratio "params/geometry.json (measured image ratio), per lead / CHANGELOG 2026-10-25T22:31:55Z item 2" \
 --Q 1000 --prov-Q assumed --source-Q "FORMULATION s5 declared assumption. Tilt fits: M1 point estimates Q = 131/102/139 (T_a 3.6/4.0/4.5, consistent, in range) but bootstrap 95% lower bounds 46-63 (< 100) and gamma = 0 not excluded for T_a 3.6 and the 4.0 production bootstrap -> rule not met (solver/out/fit_tilt/damping_evidence.json)" \
 --theta-init=-0.08447293248409532 --omega-init 0.37929871118296804 --sigma-init 0.002650872471767581 --prov-init fitted \
 --source-init "anchored tilt fit; relative to TRUE vertical; sd(theta_init)=0.0027 rad, sd(omega_init)=0.012 rad/s (bootstrap)" \
 --k-Leff 1.0805488686020446 --sigma-k-Leff 0.0002970340729121708 --source-k-Leff "anchored tilt fit k = |bob - P_ring| in camera.json units (hinge at P_ring), sd = bootstrap" \
 --level-tilt 0.012595906923464214 --sigma-level-tilt 0.0009913335550363076 --source-level-tilt "anchored projected fit (T_a 4.0 s), sd = moving-block bootstrap (formal-scaled 0.00045); T_a 3.6/4.5: 0.01164/0.01327 rad; sign verified by solver/tilt_check.py (projection reproduces the +1.50 deg right/left asymmetry: model 1.45 deg with tilt, 0.01 deg without)" \
 --extra-json solver/out/fit_tilt/Ta_4.0/params_fit_provenance.json
