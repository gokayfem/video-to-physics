#!/bin/zsh
# look/round.sh <tag> <prev> "<caption>" [extra render args]  — build (with bake), render (posed, crops), metrics, compose
set -e
ROOT=${0:A:h}/..
cd $ROOT
TAG=$1; PREV=$2; CAP=$3; shift 3
rm -f look/out/test.blend
OUT=$(${BLENDER:-blender} -b --factory-startup --python scene/build.py -- --out look/out/test.blend --bake solver/out/trajectory.csv --look 2>&1)
if ! echo "$OUT" | grep -q PND_BUILD; then
  echo "[round] scene/build.py failed:"; echo "$OUT" | grep -A8 Traceback | head -12
  echo "[round] retrying with look/build_shim.py (in-memory workaround, builder file untouched)"
  OUT=$(${BLENDER:-blender} -b --factory-startup --python look/build_shim.py -- --out look/out/test.blend --bake solver/out/trajectory.csv --look 2>&1)
  echo "$OUT" | grep -q PND_BUILD || { echo "$OUT" | grep -A8 Traceback; exit 1; }
fi
echo "$OUT" | grep -E "PND_LOOK " | cut -c1-300
tools/with_render_lock.sh ${BLENDER:-blender} -b look/out/test.blend --python look/render_look.py -- --tag $TAG --theta-deg=-4.6 --crops "$@" 2>&1 | grep -E "Traceback|Error" || true
cp params/look.json look/out/${TAG}_look.json
.venv/bin/python look/metrics.py --tag $TAG
cd look && ../.venv/bin/python compose.py --tag $TAG --prev $PREV --caption "$CAP"
