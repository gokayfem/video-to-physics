"""Blender-side dump (role E): blender -b <file.blend> --python verify/checks/blend_dump.py -- <out.npz>
Dumps per frame (0..240, plus 4 subframes): world matrices + render visibility of every object whose name looks like a
weight / platform / peg, and the evaluated world vertices of the strip object (mesh cache applied)."""
import sys, re, json
import bpy
import numpy as np

out = sys.argv[sys.argv.index("--") + 1]
sc = bpy.context.scene
objs = [o for o in sc.objects if re.search(r"weight|platform|peg|tab|strip|ruler|hand|feeder|gripper|arm|ring|finger|jaw", o.name, re.I)]
strip = [o for o in objs if o.type == "MESH" and re.search(r"strip|ruler", o.name, re.I)]
names = [o.name for o in objs]
mats, vis, strips, times = [], [], [], []
dg = bpy.context.evaluated_depsgraph_get()
for f in range(sc.frame_start, sc.frame_end + 1):
    for sub in (0.0, 0.2, 0.4, 0.6, 0.8):
        if f == sc.frame_end and sub > 0:
            continue
        sc.frame_set(f, subframe=sub)
        dg = bpy.context.evaluated_depsgraph_get()
        times.append(f + sub)
        mats.append([np.array(o.evaluated_get(dg).matrix_world) for o in objs])
        vis.append([(not o.hide_render) and o.evaluated_get(dg).visible_get() if hasattr(o, "visible_get") else (not o.hide_render) for o in objs])
        if strip:
            so = strip[0].evaluated_get(dg)
            me = so.to_mesh()
            co = np.empty(len(me.vertices) * 3); me.vertices.foreach_get("co", co)
            M = np.array(so.matrix_world)
            co = co.reshape(-1, 3) @ M[:3, :3].T + M[:3, 3]
            strips.append(co.astype(np.float64))
            so.to_mesh_clear()
np.savez(out, names=np.array(names), times=np.array(times), mats=np.array(mats), vis=np.array(vis),
         strip=np.array(strips) if strips else np.zeros(0), strip_name=strip[0].name if strip else "",
         fps=sc.render.fps, frame_start=sc.frame_start, frame_end=sc.frame_end)
print("DUMP_OK", len(times), names)
