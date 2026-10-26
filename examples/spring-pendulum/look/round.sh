#!/bin/zsh
# look/round.sh <tag> <prev> "<caption>" [extra render_look args]
# build with --look (unbaked: weight posed at the measured H for the frame), render (render lock), metrics, compose.
set -e
ROOT=${0:A:h}/..
cd $ROOT
TAG=$1; PREV=$2; CAP=$3; shift 3
mkdir -p look/out
${BLENDER:-blender} -b --factory-startup --python scene/build.py -- --out look/out/$TAG.blend --look 2>&1 | grep -E "VSP_BUILD|Traceback|Error" | cut -c1-200
tools/with_render_lock.sh ${BLENDER:-blender} -b look/out/$TAG.blend --python look/render_look.py -- --tag $TAG --crops "$@" 2>&1 | grep -E "VSP_LOOK_RENDER|Traceback|Error" | cut -c1-200
cp params/look.json look/out/${TAG}_look.json
.venv/bin/python look/metrics.py --tag $TAG
cd look && ../.venv/bin/python compose.py --tag $TAG --prev $PREV --caption "$CAP"
