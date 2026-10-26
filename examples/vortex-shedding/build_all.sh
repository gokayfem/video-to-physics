#!/bin/zsh
# Lead integration build (vortex): build_all.sh <checkpoint_name>
# Freezes the solver output (dye PNGs, fields, meta) into the checkpoint, bakes the dye sequence into the scene from that copy.
set -euo pipefail
cd "${0:A:h}"
CK=${1:?checkpoint name}
OUT=checkpoints/$CK
[[ -e $OUT ]] && { echo "checkpoint $OUT exists (immutable)"; exit 1; }
mkdir -p $OUT/solver/out
cp -R solver/out/dye solver/out/meta.json solver/out/fields.npz $OUT/solver/out/
python3 ../../skills/video-to-physics/scripts/v2p.py lock -- ${BLENDER:-blender} -b --factory-startup --python scene/build_anim.py -- \
  --dye-dir $PWD/$OUT/solver/out/dye
cp scene/out/vortex_anim.blend scene/out/vortex_anim_state.json $OUT/
rsync -a --exclude '__pycache__' --exclude 'visuals' --exclude 'runs' --exclude 'out' --exclude 'test_dye' --exclude 'anim_renders' \
  params config solver/src scene physics CONTRACT.md build_all.sh $OUT/src_snapshot/
(cd $OUT && find . -type f ! -name MANIFEST.sha256 -print0 | sort -z | xargs -0 shasum -a 256 > MANIFEST.sha256)
date -u +%FT%TZ > $OUT/READY
echo "checkpoint $OUT ready"
