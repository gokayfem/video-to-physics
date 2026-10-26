#!/bin/sh
# Reproduce all measurement/ outputs from the reference clip. ~10 min on 10 cores. Decoded frames are cached in measurement/cache (1.4 GB, deletable).
set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python
for s in s1_drift s2_landmarks s3_bob s4_wire s5_analysis s9_cadence s8_control_edge s6_plots s7_visuals s5_analysis s10_finalize; do
  echo "== $s"; $PY measurement/scripts/$s.py > /dev/null
done
