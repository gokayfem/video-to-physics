#!/bin/zsh
# Exact regeneration of the FITTED params/physics.json (clip 03). Run from anywhere.
# Upstream chain (deterministic; bootstrap seeds fixed):
#   .venv/bin/python solver/prepare_tracks.py      (v004: P track measurement/disc_P_track.csv, smoothed camera)
#   C=".venv/bin/python solver/fit_projected.py --obs solver/out/tracks_for_fit.csv --channels pt:B psi nrm --geom-json solver/geom_measured.json \
#      --scale-mode swing --bend-ends rigid_disc --fit-pivot --om-t-start 0.15 0.2 0.3"
#   $C --t-anchor 2.5 --t-anchor-channel psi=7.5 --nboot 20 --out solver/out/fit/prod_permode   (PRODUCTION: lead per-mode windows)
#   $C --t-anchor 2.5 --nboot 10 --out solver/out/fit/diag_global2.5                           (rule-compliant global window, DIAGNOSTIC)
set -e
cd "${0:A:h}/.."
.venv/bin/python solver/make_params.py --status fitted --fit-json solver/out/fit/prod_permode/fit_projected.json \
 --status-note "FITTED with per-mode anchor windows (lead decision CHANGELOG 2026-10-26T03:42:09Z, disclosed rule deviation): twist on 0-7.5 s (hold-out 7.5-10 s), sway/tilt on 0-2.5 s. Rigid disc on an inextensible torsion wire, rigid wire-disc joint (disc stays perpendicular to the wire), projected through the SMOOTHED per-frame orbiting camera of record (params/camera.json 50377461); observable = disc silhouette centre P (lead VR03-05), not the mark rotation centre Q. Scale model-conditioned from the sway period (g assumed); suspension point = A + fitted horizontal offset. Brass density assumed (mass unidentifiable). Q_t = 1000 declared (damping not identified)." \
 --prov-damping assumed --source-damping "Q_t = 1000 declared (range 300-5000); M1 in the fit window drives gamma_t to the 0 bound (solver/out/fit/prod_permode/fit_projected.json model_selection)"
