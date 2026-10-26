#!/bin/zsh
# Lead integration build: build_all.sh <checkpoint_name> [--look]
# Rebuilds trajectory from params/physics.json, builds + bakes the scene, runs rig/camera checks,
# and freezes a checkpoint mirroring ROOT's relative layout.
set -euo pipefail
cd "${0:A:h}"
CK=${1:?checkpoint name}; shift
LOOK=(); [[ "${1:-}" == "--look" ]] && LOOK=(--look)
OUT=checkpoints/$CK
[[ -e $OUT ]] && { echo "checkpoint $OUT exists (immutable)"; exit 1; }
B=${BLENDER:-blender}
.venv/bin/python solver/run.py
.venv/bin/python solver/selfcheck.py
mkdir -p $OUT/scene/out
tools/with_render_lock.sh $B -b --factory-startup --python scene/build.py -- \
  --out $OUT/scene/out/pendulum.blend --bake solver/out/trajectory.csv $LOOK
$B -b $OUT/scene/out/pendulum.blend --python scene/check_rig.py -- --out $OUT/scene/out/check_rig.json
$B -b $OUT/scene/out/pendulum.blend --python scene/check_camera.py -- --out $OUT/scene/out/check_camera.json
# freeze inputs + outputs (relative layout)
rsync -a --exclude '__pycache__' --exclude 'visuals' --exclude 'test' --exclude 'verify/out' --exclude 'delivery/dryrun' --exclude 'delivery/ck*' --exclude 'delivery/out' --exclude '*.blend1' params config solver scene camera measurement physics verify delivery tools CONTRACT.md build_all.sh $OUT/ \
  --exclude 'scene/out' --exclude 'measurement/cache' --exclude 'measurement/crops'
cp $OUT/scene/out/pendulum.blend $OUT/pendulum.blend
(cd $OUT && find . -type f ! -name MANIFEST.sha256 -print0 | sort -z | xargs -0 shasum -a 256 > MANIFEST.sha256)
date -u +%FT%TZ > $OUT/READY
echo "checkpoint $OUT ready"
