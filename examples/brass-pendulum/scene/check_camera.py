"""Blender render-space projection vs camera/project.py (camera of record), on a saved .blend.

blender -b scene/out/pendulum.blend --python scene/check_camera.py -- [--frames 0,120,240,60.5] [--out scene/out/check_camera.json]
Points: pivot, the evaluated stand mesh bounding-box corners, the pin tip, and a fixed 5x5x5 grid around the stand.
Blender pixel (contract convention, top-left corner origin): u = x_ndc * 1920, v = (1 - y_ndc) * 1088 from
bpy_extras.object_utils.world_to_camera_view with the scene camera evaluated at that (sub)frame. PASS if max |delta| <= 0.05 px.
"""
import argparse, json, pathlib, sys, itertools
import bpy, numpy as np
from mathutils import Vector
from bpy_extras.object_utils import world_to_camera_view as w2c
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "camera"))
import project as PJ
argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--frames", default="0,120,240,60.5")
ap.add_argument("--out", default=str(ROOT / "scene" / "out" / "check_camera.json"))
a = ap.parse_args(argv)
sc = bpy.context.scene
if bpy.context.window: bpy.context.window.scene = sc
assert (sc.render.resolution_x, sc.render.resolution_y, sc.render.pixel_aspect_x, sc.render.pixel_aspect_y) == (1920, 1088, 1.0, 1.0)
dg = bpy.context.evaluated_depsgraph_get()
pts = [(0.0, 0.0, 0.0)]
for n in ("PND_Base", "PND_Post", "PND_Arm", "PND_Pin"):
    ob = bpy.data.objects[n].evaluated_get(dg); m = ob.to_mesh()
    co = np.array([v.co[:] for v in m.vertices]); ob.to_mesh_clear()
    lo, hi = co.min(0), co.max(0)
    pts += [tuple(np.where(c, hi, lo)) for c in itertools.product([0, 1], repeat=3)]
lo, hi = np.array(pts).min(0), np.array(pts).max(0)
pts += [tuple(lo + (hi - lo) * np.array(c) / 4) for c in itertools.product(range(5), repeat=3)]
P = np.array(pts, float)
G = float(sc.camera.get("pnd_gauge_factor", 1.0))
# Static tilt: Blender world = R_tilt (G * X_cam), X_cam = camera.json/project.py world (the stand frame).
# So a Blender world point Pb corresponds to project.py point R_tilt^T Pb / G.
tilt_ob = bpy.data.objects.get("PND_StaticTilt")
RT = np.array(tilt_ob.matrix_world.to_3x3()) if tilt_ob else np.eye(3)
TILT = float(tilt_ob.rotation_euler[1]) if tilt_ob else 0.0
rep = {"gauge_factor_blender_over_camera_json": G, "static_tilt_rad": TILT,
       "mapping": "project.py point = R_tilt^T * P_blender / gauge (R_tilt = PND_StaticTilt world rotation)", "blend": (lambda q: q.relative_to(ROOT).as_posix() if q.is_relative_to(ROOT) else str(q))(pathlib.Path(bpy.data.filepath).resolve()), "scene_camera": sc.camera.name, "n_points": len(P), "frames": {}}
worst = 0.0
for fs in a.frames.split(","):
    f = float(fs); n = int(np.floor(f)); sub = f - n
    sc.frame_set(n, subframe=sub)
    bl = np.array([[q.x * 1920.0, (1.0 - q.y) * 1088.0] for q in (w2c(sc, sc.camera, Vector(p)) for p in P)])
    u, v = PJ.project((P @ RT) / G, PJ.load_camera(frame=f))   # (R_tilt^T Pb) / G, row-vector form
    d = bl - np.c_[u, v]
    e = float(np.abs(d).max()); worst = max(worst, e)
    rep["frames"][fs] = {"max_abs_px": e, "rms_px": float(np.sqrt((d ** 2).mean())),
                         "pivot_blender_px": bl[0].tolist(), "pivot_project_px": [float(u[0]), float(v[0])],
                         "lens_mm": sc.camera.data.lens, "shift": [sc.camera.data.shift_x, sc.camera.data.shift_y]}
rep["max_abs_px_all"] = worst
rep["pass_0.05px"] = worst <= 0.05
pathlib.Path(a.out).write_text(json.dumps(rep, indent=1))
print("CHECK_CAMERA", json.dumps({k: (v if k != "frames" else {f: round(x["max_abs_px"], 5) for f, x in v.items()}) for k, v in rep.items()}))
sys.exit(0 if rep["pass_0.05px"] else 1)
