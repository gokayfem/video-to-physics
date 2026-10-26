"""Numerical rig check on a saved pendulum .blend (geometry & rig role).

blender -b scene/out/pendulum.blend --python scene/check_rig.py -- [--out scene/out/check_rig.json] [--physics params/physics.json]

For several poses of PND_Swing (animation detached for the check, file not saved) it evaluates world transforms and:
  * wire top endpoint (centroid of the top ring of PND_Wire) at P = origin            tol 1e-6 m
  * wire bottom endpoint (centroid of the bottom ring) on the bob surface              tol 1e-6 m
  * wire bottom endpoint = bob top pole along the suspension direction                 tol 1e-6 m
  * |bob centre - P| = L_c                                                             tol 1e-6 m
  * which Euler axis/sign gives bob = P + L_c (sin th, 0, -cos th)  (all 6 candidates tested)
  * clearance of the bob (and the wire) to base, post, arm, bolt, pin over theta in [-20, +20] deg, step 0.25 deg
    (BVH nearest-surface distance on evaluated meshes; negative = intersection)
Exit code 1 if any check fails.
"""
import argparse, json, math, pathlib, sys
import bpy
import numpy as np
from mathutils import Vector, Euler
from mathutils.bvhtree import BVHTree

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ROOT = pathlib.Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--out", default=str(ROOT / "scene" / "out" / "check_rig.json"))
ap.add_argument("--physics", default=str(ROOT / "params" / "physics.json"))
a = ap.parse_args(argv)

sc = bpy.context.scene
if bpy.context.window:
    bpy.context.window.scene = sc
sw, wire, bob = (bpy.data.objects[n] for n in ("PND_Swing", "PND_Wire", "PND_Bob"))
Lc = float(sw["pnd_L_c_m"])
pj = json.loads(pathlib.Path(a.physics).read_text())
pp = pj.get("parameters", pj)
Lphys = pp.get("L_c", pp.get("L_c_m"))
Lphys = float(Lphys["value"] if isinstance(Lphys, dict) else Lphys)

had_action = None
if sw.animation_data and sw.animation_data.action:
    had_action = sw.animation_data.action.name
    sw.animation_data.action = None          # detach for the check only; the file is not saved
sc.frame_set(0)

wl = np.array([v.co[:] for v in wire.data.vertices])
ztop, zbot = wl[:, 2].max(), wl[:, 2].min()
top_idx, bot_idx = np.where(np.abs(wl[:, 2] - ztop) < 1e-9)[0], np.where(np.abs(wl[:, 2] - zbot) < 1e-9)[0]
bob_r_mesh = float(max(v.co.length for v in bob.data.vertices))
rb = float(bob.get("pnd_radius_m", bob_r_mesh))


def pose(angle_rad, axis=1):
    sw.rotation_mode = "XYZ"
    e = [0.0, 0.0, 0.0]; e[axis] = angle_rad
    sw.rotation_euler = e
    bpy.context.view_layer.update()


def endpoints():
    M = wire.matrix_world
    top = sum((M @ Vector(wl[i]) for i in top_idx), Vector()) / len(top_idx)
    bot = sum((M @ Vector(wl[i]) for i in bot_idx), Vector()) / len(bot_idx)
    return top, bot


report = {"blend": (lambda q: q.relative_to(ROOT).as_posix() if q.is_relative_to(ROOT) else str(q))(pathlib.Path(bpy.data.filepath).resolve()), "L_c_rig_m": Lc, "L_c_physics_json_m": Lphys,
          "L_c_rig_minus_physics_m": Lc - Lphys, "bob_radius_m": rb, "bob_radius_mesh_max_m": bob_r_mesh,
          "animation_detached_for_check": had_action, "tolerances_m": {"pivot": 1e-6, "surface": 1e-6, "length": 1e-6}}

# 1) Euler axis / sign
thetas = [math.radians(d) for d in (-20, -6, -1, 0.5, 4, 6, 15, 20)]
cand = {}
for axis, an in enumerate("XYZ"):
    for sign in (+1, -1):
        err = 0.0
        for th in thetas:
            pose(sign * th, axis)
            want = Vector((Lc * math.sin(th), 0.0, -Lc * math.cos(th)))
            err = max(err, (bob.matrix_world.translation - want).length)
        cand[f"rotation_euler[{axis}] ({an}) = {'+' if sign > 0 else '-'}theta"] = err
best = min(cand, key=cand.get)
report["euler_candidates_max_bob_err_m"] = cand
report["euler_convention"] = best
report["euler_convention_ok"] = cand[best] < 1e-6 and best.startswith("rotation_euler[1] (Y) = -theta")

# 2) attachment / length checks at many poses with the verified convention
rows, worst = [], {"wire_top_to_P": 0.0, "wire_bot_to_surface": 0.0, "wire_bot_to_pole": 0.0, "len_err": 0.0,
                   "bob_pos_err": 0.0, "out_of_plane": 0.0}
for d in np.arange(-30.0, 30.0001, 2.5):
    th = math.radians(d)
    pose(-th, 1)
    top, bot = endpoints()
    c = bob.matrix_world.translation.copy()
    pole = Vector((0.0, 0.0, 0.0)) + (c / c.length) * (Lc - rb)
    want = Vector((Lc * math.sin(th), 0.0, -Lc * math.cos(th)))
    r = {"theta_deg": float(d), "wire_top_to_P": top.length, "wire_bot_to_surface": abs((bot - c).length - rb),
         "wire_bot_to_pole": (bot - pole).length, "len_err": abs(c.length - Lc), "bob_pos_err": (c - want).length,
         "out_of_plane": abs(c.y)}
    for k in worst:
        worst[k] = max(worst[k], r[k])
    rows.append(r)
report["attachment_max_m"] = worst
report["attachment_ok"] = all(v < 1e-6 for v in worst.values())

# 3) clearance over +-20 deg
dg = bpy.context.evaluated_depsgraph_get()
statics = {}
for n in ("PND_Base", "PND_Post", "PND_Arm", "PND_PostBolt", "PND_Pin", "PND_WireLoop"):
    ob = bpy.data.objects.get(n)
    if ob is None:
        continue
    ev = ob.evaluated_get(dg)
    me = ev.to_mesh()
    M = ob.matrix_world
    statics[n] = BVHTree.FromPolygons([M @ v.co for v in me.vertices], [tuple(p.vertices) for p in me.polygons])
    ev.to_mesh_clear()
clear = {n: {"min_bob_clearance_m": float("inf"), "at_theta_deg": None} for n in statics}
wire_clear = {n: float("inf") for n in statics if n not in ("PND_Pin", "PND_WireLoop")}
for d in np.arange(-20.0, 20.0001, 0.25):
    th = math.radians(d)
    c = Vector((Lc * math.sin(th), 0.0, -Lc * math.cos(th)))
    for n, bvh in statics.items():
        hit = bvh.find_nearest(c)
        dist = (hit[0] - c).length - rb if hit[0] is not None else float("inf")
        # inside test: ray from centre straight down/up parity is overkill here; nearest-distance < 0 flags overlap
        if dist < clear[n]["min_bob_clearance_m"]:
            clear[n] = {"min_bob_clearance_m": dist, "at_theta_deg": float(d)}
    for s in np.linspace(0.05, 1.0, 40):       # wire samples from just below P to the bob top
        p = c * (s * (Lc - rb) / Lc)
        for n in wire_clear:
            hit = statics[n].find_nearest(p)
            if hit[0] is not None:
                wire_clear[n] = min(wire_clear[n], (hit[0] - p).length)
report["bob_clearance_pm20deg"] = clear
report["wire_clearance_pm20deg_m"] = wire_clear
report["no_intersection_pm20deg"] = all(v["min_bob_clearance_m"] > 0 for v in clear.values()) and \
    all(v > 0 for v in wire_clear.values())
# analytic cross-check for the base top plane
base = bpy.data.objects["PND_Base"].evaluated_get(dg)
zb = max((base.matrix_world @ v.co).z for v in base.to_mesh().vertices)
base.to_mesh_clear()
report["base_top_z_m"] = zb
report["bob_bottom_clearance_at_rest_m"] = (-Lc - rb) - zb
report["samples"] = rows
# static tilt: angle between the stand vertical (PND_StaticTilt local +Z) and true gravity up (+Z world)
sc.frame_set(0)
tr = bpy.data.objects.get("PND_StaticTilt")
if tr is not None:
    up = tr.matrix_world.to_3x3() @ Vector((0, 0, 1))
    pose(0.0, 1)                                   # bob at rest under true gravity
    rest = (bob.matrix_world.translation).normalized()
    stand_down = -up
    ang = math.atan2(stand_down.cross(rest).y, stand_down.dot(rest))
    report["static_tilt"] = {"tilt_rad": float(tr.rotation_euler[1]),
                             "bob_rest_dir_vs_stand_vertical_rad_about_+Y": ang,
                             "match": abs(abs(ang) - abs(float(tr.rotation_euler[1]))) < 1e-7,
                             "swing_parent": sw.parent.name if sw.parent else None}
report["pass"] = bool(report["euler_convention_ok"] and report["attachment_ok"] and report["no_intersection_pm20deg"]
                      and report.get("static_tilt", {}).get("match", True) and sw.parent is None)
pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
pathlib.Path(a.out).write_text(json.dumps(report, indent=1))
print("CHECK_RIG", json.dumps({k: report[k] for k in ("euler_convention", "euler_convention_ok", "attachment_max_m",
                                                       "attachment_ok", "bob_clearance_pm20deg",
                                                       "wire_clearance_pm20deg_m", "no_intersection_pm20deg",
                                                       "bob_bottom_clearance_at_rest_m", "static_tilt", "pass")}, indent=1))
sys.exit(0 if report["pass"] else 1)
