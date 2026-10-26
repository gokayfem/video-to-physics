#!/bin/sh
# Reproduce measurement/ + camera/ + params/camera.json for clip 02 from the reference clip (~15 min).
# Decoded frames are cached in measurement/cache/frames_bgr.npy (1.5 GB, deletable; regenerated automatically).
set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python
for s in s1_drift s1b_parallax s1c_layers_ecc s2_landmarks s4_spring s3_weight s5_ring s6_analysis s7_cadence; do echo "== $s"; $PY measurement/scripts/$s.py > /dev/null; done
$PY -c "import sys; sys.path.insert(0,'measurement/scripts'); from common import *; F=load_frames(); cv2.imwrite('camera/frames/f_000.png', np.ascontiguousarray(F[0]))"
$PY camera/edges.py camera/frames/f_000.png f000 > /dev/null
$PY camera/calibrate.py > /dev/null; $PY camera/calib_variants.py > /dev/null; $PY camera/drift_fit.py > /dev/null
$PY camera/write_params.py; $PY camera/reproject_check.py
tools/with_render_lock.sh ${BLENDER:-blender} --background --factory-startup --python camera/test/validate_projection_blender.py > /dev/null
for s in s8_3d s9_anchor s10_visuals s11_summary; do echo "== $s"; $PY measurement/scripts/$s.py > /dev/null; done
