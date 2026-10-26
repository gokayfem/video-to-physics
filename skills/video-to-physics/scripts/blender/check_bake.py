"""Check a baked Mesh Cache against its deformation npz at frames and subframes (Blender 5.2+).

  blender -b --factory-startup --python check_bake.py -- --blend BAKED.blend --object V2P_Strip
          [--frames 0,1.25,3.5] [--npz DEF.npz] [--tol 1e-6] [--json REPORT.json]

For each scene frame f (fractional part = subframe) the evaluated mesh (world space) is compared with the npz
vertices at t = f / fps (linear interpolation between samples, exact at sample times). Uses the v2p_* properties
written by bake_pc2.py. Prints 'V2P_CHECK_BAKE {json}'; exit code 1 if the max error exceeds --tol.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bake_pc2 import expected_positions  # noqa: E402


def _args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--blend", required=True)
    p.add_argument("--object", default="V2P_Strip")
    p.add_argument("--frames", default=None, help="comma list of (fractional) scene frames; default: start, mid, end")
    p.add_argument("--npz", default=None)
    p.add_argument("--tol", type=float, default=1e-6)
    p.add_argument("--json", default=None)
    return p.parse_args(argv)


def main():
    import bpy
    a = _args()
    bpy.ops.wm.open_mainfile(filepath=a.blend)
    sc = bpy.context.scene
    ob = bpy.data.objects[a.object]
    npz = a.npz or ob["v2p_npz"]
    fps, thick = float(ob["v2p_fps"]), float(ob["v2p_thickness"])
    vmap = np.load(ob["v2p_map"]) if ob.get("v2p_map") else None
    frames = ([float(x) for x in a.frames.split(",")] if a.frames
              else [sc.frame_start, 0.5 * (sc.frame_start + sc.frame_end), sc.frame_end])
    rows = []
    for f in frames:
        fi = int(np.floor(f))
        sc.frame_set(fi, subframe=f - fi)
        dg = bpy.context.evaluated_depsgraph_get()
        eo = ob.evaluated_get(dg)
        me = eo.to_mesh()
        co = np.empty(len(me.vertices) * 3)
        me.vertices.foreach_get("co", co)
        eo.to_mesh_clear()
        mw = np.array(eo.matrix_world)
        world = co.reshape(-1, 3) @ mw[:3, :3].T + mw[:3, 3]
        exp, k = expected_positions(npz, thick, f, fps, vmap)
        if world.shape != exp.shape:
            raise SystemExit(f"vertex count mismatch: mesh {world.shape} vs npz {exp.shape}")
        err = float(np.max(np.linalg.norm(world - exp, axis=1)))
        rows.append(dict(frame=f, sample=float(k), max_err_m=err))
    worst = max(r["max_err_m"] for r in rows)
    rep = dict(object=ob.name, npz=npz, frames=rows, max_err_m=worst, tol_m=a.tol, passed=worst <= a.tol)
    print("V2P_CHECK_BAKE " + json.dumps(rep))
    if a.json:
        Path(a.json).write_text(json.dumps(rep, indent=1))
    if not rep["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
