#!/bin/zsh
# Preview one blockout version through PND_Cam_Reference at frames 0/120/240 (F6: never touches scene/out/pendulum.blend).
#   scene/preview.sh v010 "caption" [--look]
# Builds scene/out/preview/<v>.blend (baked when solver/out/trajectory.csv exists), renders under the render lock,
# computes silhouette IoU and writes labelled visuals to scene/visuals/<v>_camref_fNNN.jpg.
set -e
ROOT=${0:A:h:h}; cd $ROOT
V=$1; CAP=$2; shift 2 || true
B=${BLENDER:-blender}; OUT=scene/out/preview; mkdir -p $OUT
BAKE=(); [[ -f solver/out/trajectory.csv ]] && BAKE=(--bake solver/out/trajectory.csv)
tools/with_render_lock.sh $B -b --factory-startup --python scene/build.py -- --out $OUT/$V.blend $BAKE "$@" | grep -E "PND_BUILD|PND_WARNING"
for f in 0 120 240; do
  F=$(printf %03d $f)
  tools/with_render_lock.sh $B -b $OUT/$V.blend --python scene/preview_render.py -- --frame $f --out $ROOT/$OUT/render_${V}_f$F.png > /dev/null
  tools/with_render_lock.sh $B -b $OUT/$V.blend --python scene/preview_render.py -- --frame $f --mask --out $ROOT/$OUT/mask_${V}_f$F.png > /dev/null
  .venv/bin/python scene/measure/silhouette_iou.py --mask $OUT/mask_${V}_f$F.png --frame $f --out-png scene/visuals/${V}_silhouette_iou_f$F.png --json $OUT/iou_${V}_f$F.json
  .venv/bin/python scene/make_visual.py --render $OUT/render_${V}_f$F.png --frame $f --version $V \
      --status "integrated rig - draft; PND_Cam_Reference (camera of record)" --caption "$CAP" --out scene/visuals/${V}_camref_f$F.jpg
done
rm -f scene/out/*.blend1 $OUT/*.blend1
