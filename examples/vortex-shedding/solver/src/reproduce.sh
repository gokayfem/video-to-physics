#!/bin/bash
# Full reproduce sequence for the delivered v003 window (run from solver/). Serial; ~1.5 h wall clock in total.
set -e
export BASILISK=${BASILISK:?set BASILISK to your basilisk/src checkout}
export PATH=$PATH:$BASILISK
PY=${PY:-python3}
S=$PWD; RU=$S/runs
qcc -O2 -disable-dimensions src/bench.c -o runs/bench.exe -lm
qcc -O2 -disable-dimensions src/prod.c  -o runs/prod.exe -lm
# 1 benchmark (Schaefer-Turek 2D-2)
mkdir -p runs/bench_run; (cd runs/bench_run; for L in 8 9 10; do ../bench.exe $L 7 > log-$L.txt 2>&1; done); $PY src/post_bench.py
# 2 empty-channel U_in calibration (stop manually once u_cl(x_cyl) is steady, ~56 D/U at L10)
#   mkdir -p runs/empty_L10; (cd runs/empty_L10; NOCYL=1 UIN=0.9249 ../prod.exe 10 1000 0 5.805)
# 3 precursor L9, 0 -> 180 D/U (seeded), dumps 172..179.5 (the run then continues into an L9 window; stop it after dump-179.50)
mkdir -p runs/L9_phi0/raw; (cd runs/L9_phi0; UIN=0.9249 ../prod.exe 9 180 0 5.805 1 || true)
# 4 L11 from the L9 state: 179.5 -> 209.5+15 (dumps 182.5..190.5 + dump-end at 224.5)
mkdir -p runs/L11_v001/raw; (cd runs/L11_v001; UIN=0.9249 ../prod.exe 11 195.5 0 5.805 1 $RU/L9_phi0/dump-179.50)   # stop after dump-190.50 exists
mkdir -p runs/L11/raw runs/L11/vel; (cd runs/L11; TV0=198.0 T0_FIXED=209.5 UIN=0.9249 ../prod.exe 11 190.5 0 4.73 24 $RU/L11_v001/dump-190.50)
# 5 continuation 224.5 -> >= 278.125 (dye cleared; full-channel 0.5 mm velocity history); stop (kill) once t >= 278.125
mkdir -p runs/L11e/vel runs/L11e/raw; (cd runs/L11e; NODYE=1 VFULL=1 TV0=224.5 UIN=0.9249 ../prod.exe 11 1e9 0 5.805 24 $RU/L11/dump-end)
# 6 post: periodicity, Lagrangian dye (t_rel from the outer-streak particle front), outputs, meta
$PY src/ke_v003.py
$PY src/lagr.py runs/L11e 262.125            # v003 renderer (kept for comparison)
LAGDIR=lag4 $PY src/lagr4.py runs/L11e 262.125   # v004 mass-conserving renderer (delivered)
LAGDIR=lag4 VER=v004 $PY src/post_v003.py runs/L11e 262.125
LAGDIR=lag4 $PY src/meta_patch.py v004 runs/L11e 262.125 224.5 runs/L11e/ke_periods.json
# (L10/L9 convergence windows: see out/meta.json 'convergence' and PROGRESS.log for their restart commands)
