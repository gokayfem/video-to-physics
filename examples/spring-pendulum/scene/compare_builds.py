"""Compare two build reports (scene/build.py) for determinism: dims, graph dump, lint, cameras, per-object evaluated
mesh hashes (positions + topology), transforms, bake keys.  .venv/bin/python scene/compare_builds.py A.build.json B.build.json"""
import json, sys
A, B = (json.load(open(p)) for p in sys.argv[1:3])
diff = []
for k in ("S_m", "dims_m", "graph_dump_sha256", "lint", "camera", "notes"):
    if A.get(k) != B.get(k):
        diff.append(k)
for n in sorted(set(A["objects"]) | set(B["objects"])):
    if A["objects"].get(n) != B["objects"].get(n):
        diff.append(f"object {n}")
ka = A.get("bake") and json.load(open(A["bake"]["report"])).get("keyframe_arrays_sha256")
kb = B.get("bake") and json.load(open(B["bake"]["report"])).get("keyframe_arrays_sha256")
if ka != kb:
    diff.append("bake keyframes")
print(json.dumps({"identical": not diff, "differences": diff, "objects_compared": len(A["objects"]), "bake_keyframe_sha256": ka}))
sys.exit(1 if diff else 0)
