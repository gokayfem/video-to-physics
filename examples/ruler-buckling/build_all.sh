#!/bin/zsh
# Lead integration build (ruler): build_all.sh <checkpoint_name>
# Uses the frozen solver output (solver/out), bakes it into the Blender scene, runs scene checks, freezes checkpoint.
set -euo pipefail
cd "${0:A:h}"
CK=${1:?checkpoint name}
OUT=checkpoints/$CK
[[ -e $OUT ]] && { echo "checkpoint $OUT exists (immutable)"; exit 1; }
B=${BLENDER:-blender}
mkdir -p $OUT/scene/out $OUT/solver/out
cp solver/out/deformation.npz solver/out/meta.json $OUT/solver/out/   # build from the frozen copy (provenance)
python3 ../../skills/video-to-physics/scripts/v2p.py lock -- $B -b --factory-startup --python scene/build_anim.py -- \
  --out $OUT/scene/out/ruler_anim.blend --deformation $OUT/solver/out/deformation.npz
rsync -a --exclude '__pycache__' --exclude 'visuals' --exclude 'test' --exclude 'cache' --exclude '*.blend1' \
  --exclude 'scene/out' --exclude 'dyn_*.npz' --exclude 'static_paths.npz' \
  params config solver scene measurement physics CONTRACT.md build_all.sh $OUT/
cp $OUT/scene/out/ruler_anim.blend $OUT/ruler_anim.blend
(cd $OUT && find . -type f ! -name MANIFEST.sha256 -print0 | sort -z | xargs -0 shasum -a 256 > MANIFEST.sha256)
date -u +%FT%TZ > $OUT/READY
echo "checkpoint $OUT ready"
