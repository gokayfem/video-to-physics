#!/bin/bash
# Verifier-owned clean rebuild for P10 (never writes into checkpoints/).
# usage: verify/clean_rebuild_verifier.sh <checkpoint_copy> <fresh_work_dir> <ref.csv> <params_rel> <blend_rel> [check_blend args...]
#   env SOLVER_CMD, BUILD_CMD: the checkpoint's DOCUMENTED commands, run with cwd = work dir ({PY} and {B} substituted), e.g.
#   SOLVER_CMD='{PY} solver/run.py --params params/physics.json --out solver/out'
#   BUILD_CMD='{B} -b --factory-startup --python scene/build.py -- --out scene/out/torsion.blend --bake solver/out/trajectory.csv --look'
# Stages the checkpoint's inputs only (outputs, renders, blends, caches excluded), rebuilds, then runs check_blend.py --dump.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; PY="$ROOT/.venv/bin/python"; B=${BLENDER:-blender}
CK="$(cd "$1" && pwd)"; W="$2"; REF="$(cd "$(dirname "$3")" && pwd)/$(basename "$3")"; PREL="$4"; BREL="$5"; shift 5
[ -e "$W" ] && { echo "work dir exists: $W"; exit 2; }
: "${SOLVER_CMD:?set SOLVER_CMD}"; : "${BUILD_CMD:?set BUILD_CMD}"
mkdir -p "$W"; W="$(cd "$W" && pwd)"; export PYTHONDONTWRITEBYTECODE=1
rsync -a --exclude 'solver/out/' --exclude 'scene/out/' --exclude '*.blend' --exclude '*.blend1' --exclude '*.png' --exclude '*.mp4' \
      --exclude '__pycache__/' --exclude 'review/' --exclude 'verify/' --exclude 'visual_review/' "$CK/" "$W/"
(cd "$W" && find . -type f | sort | xargs shasum -a 256) > "$W/staged_inputs.sha256"
S=${SOLVER_CMD//\{PY\}/$PY}; Bc=${BUILD_CMD//\{PY\}/$PY}; Bc=${Bc//\{B\}/$B}
mkdir -p "$(dirname "$W/$BREL")"
(cd "$W" && eval "$S") > "$W/solver.log" 2>&1
(cd "$W" && "$ROOT/tools/with_render_lock.sh" bash -c "$Bc") > "$W/build.log" 2>&1
"$ROOT/tools/with_render_lock.sh" $B -b "$W/$BREL" --python "$ROOT/verify/check_blend.py" -- --params "$W/$PREL" --ref "$REF" \
   --out "$W/check_blend.json" --dump "$W/check_blend.npz" --label "P10 rebuild $(basename "$W")" "$@" > "$W/check_blend.log" 2>&1
shasum -a 256 "$W/$BREL"; grep CHECK_BLEND_SUMMARY "$W/check_blend.log"
