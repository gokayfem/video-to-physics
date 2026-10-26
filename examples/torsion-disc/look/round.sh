#!/bin/zsh
# look/round.sh <tag> <prev> "<caption>" [frame]  — build --look, render (render lock), metrics, compose; snapshot look.json
set -e
ROOT=${0:A:h}/..
cd $ROOT
TAG=$1; PREV=$2; CAP=$3; FR=${4:-0}
mkdir -p look/out
${BLENDER:-blender} -b --factory-startup --python scene/build.py -- --out look/out/$TAG.blend --look 2>&1 | grep -E "Traceback|Error" | cut -c1-200 || true
tools/with_render_lock.sh ${BLENDER:-blender} -b look/out/$TAG.blend --python look/render_look.py -- --tag $TAG --frame $FR --crops 2>&1 | grep -E "Traceback|Error" | cut -c1-200 || true
cp params/look.json look/out/${TAG}_look.json
.venv/bin/python look/metrics.py --tag $TAG --frame $FR
cd look && ../.venv/bin/python compose.py --tag $TAG --prev $PREV --frame $FR --caption "$CAP"
