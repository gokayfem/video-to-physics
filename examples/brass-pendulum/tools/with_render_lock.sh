#!/bin/zsh
# Serialize heavy Blender work across the team: with_render_lock.sh <command...>
LOCK="${0:A:h}/../../.render.lock"
until mkdir "$LOCK" 2>/dev/null; do
  echo "[render-lock] waiting (held by: $(cat "$LOCK/owner" 2>/dev/null))" >&2; sleep 5
done
echo "$$ $(date -u +%FT%TZ) $*" > "$LOCK/owner"
trap 'rm -rf "$LOCK"' EXIT INT TERM
"$@"
