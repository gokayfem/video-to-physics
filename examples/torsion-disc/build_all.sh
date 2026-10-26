#!/bin/zsh
# Lead integration build (kit): build_all.sh <checkpoint_name>
# Requires per-clip variables below. Rebuilds trajectory, builds+bakes with look, runs checks, freezes checkpoint.
set -euo pipefail
cd "${0:A:h}"
BLEND_NAME=torsion.blend            # e.g. spring.blend
CK=${1:?checkpoint name}
OUT=checkpoints/$CK
[[ -e $OUT ]] && { echo "checkpoint $OUT exists (immutable)"; exit 1; }
B=${BLENDER:-blender}
.venv/bin/python solver/run.py
.venv/bin/python solver/selfcheck.py
mkdir -p $OUT/scene/out
tools/with_render_lock.sh $B -b --factory-startup --python scene/build.py -- \
  --out $OUT/scene/out/$BLEND_NAME --bake solver/out/trajectory.csv --look
$B -b $OUT/scene/out/$BLEND_NAME --python scene/check_rig.py -- --frames --out $OUT/scene/out/check_rig.json
$B -b $OUT/scene/out/$BLEND_NAME --python scene/check_camera.py -- --out $OUT/scene/out/check_camera.json
rsync -a --exclude '__pycache__' --exclude 'visuals' --exclude 'test' --exclude 'verify/out' --exclude 'cache' \
  --exclude 'crops' --exclude 'frames' --exclude '*.blend1' --exclude 'scene/out' --exclude 'look/out' \
  params config solver scene camera measurement physics verify tools CONTRACT.md build_all.sh $OUT/
cp $OUT/scene/out/$BLEND_NAME $OUT/$BLEND_NAME
(cd $OUT && find . -type f ! -name MANIFEST.sha256 -print0 | sort -z | xargs -0 shasum -a 256 > MANIFEST.sha256)
date -u +%FT%TZ > $OUT/READY
echo "checkpoint $OUT ready"
