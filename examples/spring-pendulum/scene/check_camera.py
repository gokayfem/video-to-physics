"""Blender render-space projection vs the camera of record (camera/project.py) — or, until it exists, vs the scene
role's approximate frame-0 pinhole (params/geometry.json camera_frame0_approx) — on a saved .blend.

blender -b scene/out/spring.blend --python scene/check_camera.py -- [--frames 0,120,240,60.5] [--out scene/out/check_camera.json]
Points: A, the evaluated frame-part bounding-box corners, the weight origin, and a 5x5x5 grid spanning them.
Blender pixel (contract convention): u = x_ndc * 1920, v = (1 - y_ndc) * 1088 (world_to_camera_view, scene camera
evaluated at that (sub)frame). PASS if max |delta| <= 0.05 px at every frame.
"""
import argparse, itertools, json, pathlib, sys
import bpy
import numpy as np
from mathutils import Vector
from bpy_extras.object_utils import world_to_camera_view as w2c

ROOT = pathlib.Path(__file__).resolve().parents[1]
argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--frames", default="0,120,240,60.5")
ap.add_argument("--out", default=str(ROOT / "scene" / "out" / "check_camera.json"))
a = ap.parse_args(argv)
sc = bpy.context.scene
if bpy.context.window:
    bpy.context.window.scene = sc
assert (sc.render.resolution_x, sc.render.resolution_y, sc.render.pixel_aspect_x, sc.render.pixel_aspect_y) == (1920, 1088, 1.0, 1.0)
rep_build = json.loads(pathlib.Path(bpy.data.filepath).with_suffix(".build.json").read_text())
Sm = rep_build["S_m"]
kind = rep_build["camera"]["kind"]
if kind == "camera_of_record":
    sys.path.insert(0, str(ROOT / "camera"))
    import project as PJ
    G = float(sc.camera.get("vsp_gauge_factor", 1.0))

    def proj(P, f):
        return PJ.project(P / G, PJ.load_camera(frame=f))
    ref_name = "camera/project.py (camera of record)"
else:
    geo = json.loads((ROOT / "params" / "geometry.json").read_text())["camera_frame0_approx"]
    R, C, fpx = np.array(geo["R_world_to_cv"]), np.array(geo["C_S"]) * Sm, geo["f_px"]
    cx, cy = geo["principal_point_px"]

    def proj(P, f):
        Xc = (P - C) @ R.T
        return cx + fpx * Xc[:, 0] / Xc[:, 2], cy + fpx * Xc[:, 1] / Xc[:, 2]
    ref_name = "params/geometry.json camera_frame0_approx (APPROXIMATE scene camera; static)"
dg = bpy.context.evaluated_depsgraph_get()
pts = [(0.0, 0.0, 0.0), tuple(bpy.data.objects["VSP_Weight"].matrix_world.translation)]
for n in ("VSP_Posts", "VSP_Crossbar", "VSP_BasePlate"):
    ob = bpy.data.objects[n].evaluated_get(dg)
    m = ob.to_mesh()
    co = np.array([v.co[:] for v in m.vertices])
    ob.to_mesh_clear()
    lo, hi = co.min(0), co.max(0)
    pts += [tuple(np.where(c, hi, lo)) for c in itertools.product([0, 1], repeat=3)]
lo, hi = np.array(pts).min(0), np.array(pts).max(0)
pts += [tuple(lo + (hi - lo) * np.array(c) / 4) for c in itertools.product(range(5), repeat=3)]
P = np.array(pts, float)
rep = {"blend": pathlib.Path(bpy.data.filepath).name, "reference_projection": ref_name, "camera_kind": kind,
       "scene_camera": sc.camera.name, "n_points": len(P), "frames": {}}
worst = 0.0
for fs in a.frames.split(","):
    f = float(fs); n = int(np.floor(f)); sub = f - n
    sc.frame_set(n, subframe=sub)
    bl = np.array([[q.x * 1920.0, (1.0 - q.y) * 1088.0] for q in (w2c(sc, sc.camera, Vector(p)) for p in P)])
    u, v = proj(P, f)
    d = bl - np.c_[u, v]
    e = float(np.abs(d).max()); worst = max(worst, e)
    rep["frames"][fs] = {"max_abs_px": e, "rms_px": float(np.sqrt((d ** 2).mean())), "A_blender_px": bl[0].tolist(),
                         "A_reference_px": [float(u[0]), float(v[0])], "lens_mm": sc.camera.data.lens}
rep["max_abs_px_all"] = worst
rep["pass_0.05px"] = worst <= 0.05
pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
pathlib.Path(a.out).write_text(json.dumps(rep, indent=1))
print("CHECK_CAMERA", json.dumps({"pass": rep["pass_0.05px"], "max_abs_px_all": worst, "kind": kind,
                                  "frames": {k: round(v["max_abs_px"], 5) for k, v in rep["frames"].items()}}))
