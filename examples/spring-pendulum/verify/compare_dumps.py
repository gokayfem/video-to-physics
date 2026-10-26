#!/usr/bin/env python
"""P10 repeatability: compare two check_blend.py --dump .npz files from two clean rebuilds.
  .venv/bin/python verify/compare_dumps.py A.npz B.npz --out verify/out/check_repeat.json
PASS iff every keyframe array (co, handles, interpolation, handle types) and the evaluated eye positions, weight world
matrices, spring end points, coil counts and spring topology are bitwise identical."""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vparams import sha256_file
JUDGED = ("eye", "weight_matrix", "spring_top", "spring_bottom", "coil_counts", "topology")
ap = argparse.ArgumentParser(); ap.add_argument("a"); ap.add_argument("b"); ap.add_argument("--out", required=True)
a = ap.parse_args()
A, B = np.load(a.a), np.load(a.b)
keys = sorted(set(A.files) | set(B.files))
rep = dict(inputs={a.a: sha256_file(a.a), a.b: sha256_file(a.b)}, arrays={})
ok = True
for k in keys:
    judged = "|" in k or k in JUDGED
    if k not in A.files or k not in B.files:
        rep["arrays"][k] = "missing in one dump"; ok &= not judged; continue
    x, y = A[k], B[k]
    same = x.shape == y.shape and x.dtype == y.dtype and x.tobytes() == y.tobytes()
    ent = dict(bitwise_identical=bool(same), judged=judged)
    if not same and x.shape == y.shape and x.dtype.kind in "fi":
        ent["max_abs_diff"] = float(np.max(np.abs(x.astype(float) - y.astype(float))))
    rep["arrays"][k] = ent
    if judged and not same:
        ok = False
rep["criterion"] = "keyframe arrays and evaluated " + ", ".join(JUDGED) + " bitwise identical"
rep["status"] = "PASS" if ok else "FAIL"
json.dump(rep, open(a.out, "w"), indent=2)
print(rep["status"])
