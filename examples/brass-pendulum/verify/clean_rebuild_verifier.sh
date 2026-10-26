#!/bin/bash
# Verifier-owned clean rebuild for P10 (does not use delivery/ or write into the checkpoint).
# usage: verify/clean_rebuild_verifier.sh <checkpoint_dir_or_copy> <fresh_work_dir> <ref_trajectory.csv> [extra build.py args, e.g. --look]
# Stages only declared inputs (params, solver code, scene code+vendor, camera/project.py+drift.csv),
# re-runs solver/run.py and scene/build.py --bake, then verify/check_blend.py --dump.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; PY="$ROOT/.venv/bin/python"; B=${BLENDER:-blender}
CK="$(cd "$1" && pwd)"; W="$2"; REF="$(cd "$(dirname "$3")" && pwd)/$(basename "$3")"; shift 3; EXTRA=("$@")
[ -e "$W" ] && { echo "work dir exists: $W"; exit 2; }
mkdir -p "$W"; W="$(cd "$W" && pwd)"; export PYTHONDONTWRITEBYTECODE=1
for f in params/physics.json params/geometry.json params/camera.json params/look.json solver/pendulum.py solver/run.py solver/bake_blender.py \
         scene/build.py scene/pnd_nodes.py scene/look.py camera/project.py camera/drift.csv; do
  mkdir -p "$W/$(dirname $f)"; cp -p "$CK/$f" "$W/$f"; done
cp -Rp "$CK/scene/vendor" "$W/scene/vendor"; find "$W" -name __pycache__ -prune -exec rm -rf {} +
(cd "$W" && find . -type f | sort | xargs shasum -a 256) > "$W/staged_inputs.sha256"
(cd "$W" && "$PY" solver/run.py --params params/physics.json --out solver/out) > "$W/solver.log" 2>&1
(cd "$W" && "$ROOT/tools/with_render_lock.sh" $B -b --factory-startup --python scene/build.py -- --out "$W/scene/out/pendulum.blend" \
   --physics params/physics.json --geometry params/geometry.json --camera params/camera.json --bake solver/out/trajectory.csv ${EXTRA[@]+"${EXTRA[@]}"}) > "$W/build.log" 2>&1
"$ROOT/tools/with_render_lock.sh" $B -b "$W/scene/out/pendulum.blend" --python "$ROOT/verify/check_blend.py" -- --params "$W/params/physics.json" \
   --ref "$REF" --trajectory "$W/solver/out/trajectory.csv" --out "$W/check_blend.json" --dump "$W/check_blend.npz" --label "P10 rebuild $(basename $W)" > "$W/check_blend.log" 2>&1
shasum -a 256 "$W/solver/out/trajectory.csv" "$W/scene/out/pendulum.blend"
