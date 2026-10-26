"""Camera check (scene role C): TOR_Cam_Reference in the saved .blend vs the camera of record.

blender -b scene/out/torsion.blend --python scene/check_camera.py [-- --out scene/out/check_camera.json]
Projects 3D test points (A, disc rim points, stand corners, table points; metres) with Blender's evaluated camera
(world_to_camera_view, 1920x1088, shifts and lens keyed) and with camera/project.py load_camera(frame=n) (points in
D = metres / gauge) at frames 0, 1, 60.5, 120, 180.25, 239, 240 and every 10th frame; reports max |du, dv| px.
Tolerance 0.05 px. Also verifies the lock flags of the locked diagnostic cameras.
"""
import json, pathlib, sys
import bpy
import numpy as np
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector

ROOT = pathlib.Path(__file__).resolve().parent.parent
argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
OUT = argv[argv.index("--out") + 1] if "--out" in argv else str(ROOT / "scene" / "out" / "check_camera.json")
sys.path.insert(0, str(ROOT / "camera"))
import project as PJ  # noqa: E402

sc = bpy.context.scene
cam = bpy.data.objects["TOR_Cam_Reference"]
g = float(cam.get("tor_gauge_factor", 0.0))
rep = json.loads(pathlib.Path(bpy.data.filepath).with_suffix(".build.json").read_text())
kind = rep["camera"]["kind"]
if kind != "camera_of_record":
    raise SystemExit("file was built with the approximate camera; nothing to compare against camera.json")
g = rep["camera"]["gauge_factor_m_per_unit"]
D = rep["dims_m"]
pts_m = [(0, 0, 0), (0, 0, -D["wire_length_A_to_disc_top"])]
for a in np.linspace(0, 2 * np.pi, 8, endpoint=False):
    pts_m.append((0.5 * D["disc_diameter"] * np.cos(a), 0.5 * D["disc_diameter"] * np.sin(a), -D["wire_length_A_to_disc_top"]))
xs = D["span_inner"] / 2
for x in (-xs - D["post_width_x"], -xs, xs, xs + D["post_width_x"]):
    for z in (D["post_top_z"], D["foot_top_z"]):
        pts_m.append((x + D["frame_centre_x"], -D["post_depth_y"] / 2, z))
for x in (-0.15, 0.15):
    pts_m.append((x, -0.05, D["table_z"]))
P = np.array(pts_m)
frames = sorted({0, 1, 60.5, 120, 180.25, 239, 240} | set(range(0, 241, 10)))
rx, ry = sc.render.resolution_x, sc.render.resolution_y
worst, per = 0.0, {}
for fr in frames:
    n = int(np.floor(fr))
    sc.frame_set(n, subframe=fr - n)
    dg = bpy.context.evaluated_depsgraph_get()
    ce = cam.evaluated_get(dg)
    c = PJ.load_camera(frame=fr)
    u_ref, v_ref = PJ.project(P / g, c)
    err = []
    for p, ur, vr in zip(P, u_ref, v_ref):
        q = world_to_camera_view(sc, ce, Vector(p))
        u, v = q.x * rx, (1 - q.y) * ry
        err.append(max(abs(u - ur), abs(v - vr)))
    per[str(fr)] = round(float(max(err)), 6)
    worst = max(worst, max(err))
locks = {}
for ob in bpy.data.objects:
    if ob.type == "CAMERA" and ob.name != "TOR_Cam_Reference":
        locks[ob.name] = all(ob.lock_location) and all(ob.lock_rotation) and all(ob.lock_scale)
res = {"file": bpy.data.filepath, "camera": kind, "gauge_m_per_unit": g, "n_points": len(P), "frames": frames,
       "max_err_px": round(float(worst), 6), "per_frame_max_px": per, "tolerance_px": 0.05,
       "locked_diagnostics": locks, "pass": bool(worst <= 0.05 and all(locks.values()))}
pathlib.Path(OUT).write_text(json.dumps(res, indent=1))
print("TOR_CHECK_CAMERA", json.dumps({k: v for k, v in res.items() if k != "per_frame_max_px"}))
