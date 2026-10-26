"""Deterministic builder for the brass-pendulum scene (geometry & rig role).

blender -b --factory-startup --python scene/build.py -- --out scene/out/pendulum.blend \
    [--physics params/physics.json] [--geometry params/geometry.json] [--look] \
    [--bake solver/out/trajectory.csv] [--report scene/out/build_report.json]

* Reads L_c from params/physics.json (parameters.L_c_m.value) and every other dimension as a ratio to L_c from
  params/geometry.json. Nothing fitted is hard-coded here.
* Collections: PND_Stand (PND_StandControls + PND_Base/PND_Post/PND_Arm/PND_PostBolt), PND_Pivot (PND_Pin,
  PND_WireLoop), PND_Swing (empty PND_Swing at P with identity rest rotation; children PND_Wire, PND_Bob).
* Rig convention: rotation_euler[1] = -theta (XYZ) puts the bob at P + L_c (sin th, 0, -cos th)
  (verified numerically by scene/check_rig.py).
* --look   imports scene/look.py (camera & appearance role) and calls apply_look(scene, params).
* --bake   runs solver/bake_blender.py (dynamics role) in-process on PND_Swing (no copy of its code).
Without them: placeholder materials, a placeholder floor/light, the approximate geometry-role camera, no animation.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
import pathlib
import runpy
import sys

import bpy

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import pnd_nodes  # noqa: E402
from nodekit import scene as nkscene, inspect as nkinspect  # noqa: E402  (path set by pnd_nodes)

FPS, F0, F1 = 24, 0, 240


def rel(p):
    """ROOT-relative POSIX path for reports (F3); absolute only for files outside ROOT."""
    if p is None:
        return None
    p = pathlib.Path(p).resolve()
    try:
        return p.relative_to(ROOT).as_posix()
    except ValueError:
        return str(p)


def sha(p):
    p = pathlib.Path(p)
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None


def parse():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "out" / "pendulum.blend"))
    ap.add_argument("--physics", default=str(ROOT / "params" / "physics.json"))
    ap.add_argument("--geometry", default=str(ROOT / "params" / "geometry.json"))
    ap.add_argument("--look", action="store_true", help="apply scene/look.py (camera & appearance role)")
    ap.add_argument("--bake", default=None, help="trajectory.csv to bake with solver/bake_blender.py")
    ap.add_argument("--camera", default=str(ROOT / "params" / "camera.json"),
                    help="camera of record (camera role); cameras are built from it with camera/project.py")
    ap.add_argument("--report", default=None, help="build report JSON (default: <out>.build.json)")
    ap.add_argument("--render-config", default=str(ROOT / "config" / "render.json"),
                    help="lead-owned render config (motion blur)")
    ap.add_argument("--level-tilt-rad", type=float, default=None,
                    help="TEST override of physics.json level_tilt_rad (recorded in the report)")
    a = ap.parse_args(argv)
    for k in ("out", "physics", "geometry", "bake", "camera", "report", "render_config"):     # resolve against the caller's cwd
        if getattr(a, k):
            setattr(a, k, str(pathlib.Path(getattr(a, k)).resolve()))
    return a


# ---------------------------------------------------------------- parameters
def read_Lc(physics_path):
    pj = json.loads(pathlib.Path(physics_path).read_text())
    params = pj.get("parameters", pj)
    for key in ("L_c_m", "L_c"):
        if key in params:
            v = params[key]
            return float(v["value"] if isinstance(v, dict) else v), pj
    raise SystemExit(f"no L_c_m in {physics_path}")


def read_Leff(pj, Lc):
    """Metres per camera.json length unit (CHANGELOG 2026-10-25T22:47Z). Stand ratios and camera translations
    scale with this, never with L_c. Fallback: L_c (with a warning)."""
    params = pj.get("parameters", pj)
    v = params.get("L_eff_m")
    if v is not None:
        return float(v["value"] if isinstance(v, dict) else v), "physics.json L_eff_m"

    def find(o):
        if isinstance(o, dict):
            for k, x in o.items():
                if k == "camera_metre_scale_L_eff_m":
                    return x
                y = find(x)
                if y is not None:
                    return y
        return None
    v = find(pj)
    if v is not None:
        print("PND_WARNING: physics.json has no top-level L_eff_m; using nested camera_metre_scale_L_eff_m")
        return float(v["value"] if isinstance(v, dict) else v), "physics.json (nested) camera_metre_scale_L_eff_m"
    print("PND_WARNING: physics.json has no L_eff_m; scaling stand and cameras by L_c (fallback)")
    return Lc, "FALLBACK L_c (no L_eff_m in physics.json)"


def dims(geo, Lc, Leff):
    """geometry.json ratios are in camera.json length units (L_eff) -> metres via L_eff_m; except bob_radius,
    which is r/|P_ring-bob| (= r/L_c) and only provenance (the bob radius comes from physics.json)."""
    r = {k: v["value"] for k, v in geo["ratios"].items()}
    D = {k: r[k] * Leff for k in r}
    D["bob_radius"] = r["bob_radius"] * Lc
    D["L_c"], D["L_eff"] = Lc, Leff
    D["wire_radius"] = D["wire_diameter"] / 2
    # loop rests on top of the shaft; the wire leaves the loop bottom at P
    # v4 (VR-11): pin axis = fitted arm centre height; tight loop concentric with the pin head, closed by a short V to P
    D["pin_axis_z"] = D["pin_axis_z_fit"]
    D["wire_loop_radius"] = D["pin_head_radius"] + D["wire_radius"] + 0.0005 * Leff
    D["arm_length"] = D["arm_end_x"] - (D["post_centre_x"] + D["post_size_x"] / 2)
    D["post_height"] = D["post_top_z"] + D["h_pivot"]
    D["base_left_margin"] = (D["post_centre_x"] - D["post_size_x"] / 2) - (D["base_centre_x"] - D["base_width"] / 2)
    D["wire_length"] = Lc - D["bob_radius"]
    return D


# ---------------------------------------------------------------- helpers
def collection(name, parent=None):
    c = bpy.data.collections.new(name)
    (parent or bpy.context.scene.collection).children.link(c)
    return c


def mat(name, rgb, metallic=0.0, rough=0.5):
    m = bpy.data.materials.new(name)
    bsdf = m.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (*rgb, 1.0)
    bsdf.inputs["Metallic"].default_value = metallic
    bsdf.inputs["Roughness"].default_value = rough
    m.diffuse_color = (*rgb, 1.0)
    return m


def gn_object(name, coll, group, inputs, material=None):
    me = bpy.data.meshes.new(name)
    ob = bpy.data.objects.new(name, me)
    coll.objects.link(ob)
    mod = ob.modifiers.new(group.name, "NODES")
    mod.node_group = group
    nkinspect.set_modifier_inputs(mod, inputs)
    if material:
        me.materials.append(material)
    return ob, mod


def wire_mesh(name, radius, length, sides=16):
    """Cylinder from exactly (0,0,0) to exactly (0,0,-length); ring vertices only (ends hidden in loop/bob)."""
    me = bpy.data.meshes.new(name)
    verts, faces = [], []
    for z in (0.0, -length):
        for i in range(sides):
            a = 2 * math.pi * i / sides
            verts.append((radius * math.cos(a), radius * math.sin(a), z))
    for i in range(sides):
        j = (i + 1) % sides
        faces.append((i, j, sides + j, sides + i))
    me.from_pydata(verts, [], faces)
    me.shade_smooth()
    me.update()
    return me


def sphere_mesh(name, radius, u=64, v=32):
    """UV sphere built from explicit vertex/face lists (deterministic order across processes; F2).
    Vertex 0 is the top pole at (0,0,+radius) = the wire attachment; then rings top->bottom; last = bottom pole."""
    verts = [(0.0, 0.0, radius)]
    for i in range(1, v):
        ph = math.pi * i / v
        for j in range(u):
            th = 2 * math.pi * j / u
            verts.append((radius * math.sin(ph) * math.cos(th), radius * math.sin(ph) * math.sin(th), radius * math.cos(ph)))
    verts.append((0.0, 0.0, -radius))
    ring = lambda i, j: 1 + (i - 1) * u + (j % u)
    faces = [(0, ring(1, j), ring(1, j + 1)) for j in range(u)]
    for i in range(1, v - 1):
        faces += [(ring(i, j), ring(i + 1, j), ring(i + 1, j + 1), ring(i, j + 1)) for j in range(u)]
    last = len(verts) - 1
    faces += [(ring(v - 1, j + 1), ring(v - 1, j), last) for j in range(u)]
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], faces)
    me.shade_smooth()
    me.update()
    return me


def eyelet_mesh(name, r, z0, z1, sides=24):
    """Short brass stem from z0 (sunk into the bob) to z1 with a rounded top; deterministic vertex/face order."""
    rings = [(z0, r), (z1 - 0.6 * r, r), (z1 - 0.25 * r, 0.8 * r), (z1, 0.35 * r)]
    verts = [(rr * math.cos(2 * math.pi * j / sides), rr * math.sin(2 * math.pi * j / sides), z) for z, rr in rings
             for j in range(sides)]
    faces = [(i * sides + j, i * sides + (j + 1) % sides, (i + 1) * sides + (j + 1) % sides, (i + 1) * sides + j)
             for i in range(len(rings) - 1) for j in range(sides)]
    top = len(verts)
    verts.append((0.0, 0.0, z1))
    k = (len(rings) - 1) * sides
    faces += [(k + j, k + (j + 1) % sides, top) for j in range(sides)]
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], faces)
    me.shade_smooth()
    me.update()
    return me


def mesh_hash(ob, dg):
    import numpy as np
    ev = ob.evaluated_get(dg)
    m = ev.to_mesh()
    co = np.empty(len(m.vertices) * 3, dtype=np.float32)
    m.vertices.foreach_get("co", co)
    pv = np.empty(len(m.loops), dtype=np.int32)
    m.loops.foreach_get("vertex_index", pv)
    ps = np.empty(len(m.polygons), dtype=np.int32)
    m.polygons.foreach_get("loop_start", ps)
    h = hashlib.sha256(co.tobytes() + pv.tobytes() + ps.tobytes()).hexdigest()   # positions + topology (F2)
    n = (len(m.vertices), len(m.polygons))
    bb = (co.reshape(-1, 3).min(0).tolist(), co.reshape(-1, 3).max(0).tolist()) if len(co) else None
    ev.to_mesh_clear()
    return h, n, bb


# ---------------------------------------------------------------- build
def build(a):
    nkscene.reset()
    sc = bpy.context.scene
    sc.name = "PND_Scene"
    sc.render.fps, sc.render.fps_base = FPS, 1.0
    sc.frame_start, sc.frame_end = F0, F1
    sc.frame_current = 0
    sc.unit_settings.system, sc.unit_settings.scale_length = "METRIC", 1.0
    sc.render.resolution_x, sc.render.resolution_y = 1920, 1088

    Lc, phys = read_Lc(a.physics)
    geo = json.loads(pathlib.Path(a.geometry).read_text())
    Leff, Leff_src = read_Leff(phys, Lc)
    D = dims(geo, Lc, Leff)
    notes = [f"stand and camera gauge L_eff_m = {Leff:.9f} m from {Leff_src}; L_c = {Lc:.9f} m (L_c/L_eff = {Lc / Leff:.6f})"]
    # CHANGELOG #2: L_c AND bob radius come from params/physics.json (dynamics owns them);
    # geometry.json's bob_radius ratio is provenance only.
    pp = phys.get("parameters", phys)
    pb = pp.get("bob_radius", pp.get("bob_radius_m"))
    if pb is None or (isinstance(pb, dict) and pb.get("value") is None):
        raise SystemExit(f"no bob radius in {a.physics} (CHANGELOG #2: physics.json is the single source)")
    r_phys = float(pb["value"] if isinstance(pb, dict) else pb)
    r_geo = D["bob_radius"]
    D["bob_radius_measured_ratio_m"] = r_geo
    D["bob_radius"] = r_phys
    D["wire_length"] = Lc - r_phys
    rdiff = (r_phys - r_geo) / r_geo
    notes.append(f"bob radius {r_phys:.6f} m from physics.json (L_c/r = {Lc / r_phys:.3f}); geometry.json measured "
                 f"ratio gives {r_geo:.6f} m (L_c/r = {1 / geo['ratios']['bob_radius']['value']:.3f}), diff {rdiff:+.2%}")

    # materials (placeholders; look.py replaces/edits them)
    M = {"walnut": mat("PND_Walnut", (0.20, 0.09, 0.045), 0.0, 0.45),
         "steel": mat("PND_Steel", (0.80, 0.80, 0.82), 1.0, 0.12),
         "wire": mat("PND_WireSteel", (0.05, 0.05, 0.055), 1.0, 0.35),
         "brass": mat("PND_Brass", (0.90, 0.62, 0.25), 1.0, 0.10)}

    # node groups
    g_stand, g_part = pnd_nodes.stand(), pnd_nodes.stand_part()
    g_pin, g_loop = pnd_nodes.pin(), pnd_nodes.wire_loop()

    c_stand, c_pivot, c_swing = collection("PND_Stand"), collection("PND_Pivot"), collection("PND_Swing")

    # --- stand: one control object + one object per part
    ctrl, ctrl_mod = gn_object("PND_StandControls", c_stand, g_stand, {
        "Post Height": D["post_height"], "Arm Length": D["arm_length"], "Base Width": D["base_width"],
        "Base Depth": D["base_depth"], "Base Thickness": D["base_thickness"],
        "Base Edge Radius": D["base_top_edge_radius"], "Wood Edge Radius": D["wood_edge_radius"],
        "Pivot Height": D["h_pivot"], "Arm End X": D["arm_end_x"], "Arm Centre Z": D["pin_axis_z"],
        "Arm Height": D["arm_height"], "Arm Depth": D["arm_depth"], "Post Size X": D["post_size_x"],
        "Post Size Y": D["post_size_y"], "Post Centre Y": D["post_centre_y"],
        "Base Left Margin": D["base_left_margin"], "Base Centre Y": D["base_centre_y"],
        "Base Corner Radius": D["base_corner_radius"], "Base Bottom Edge Radius": D["base_bottom_edge_radius"],
        "Bolt Radius": D["bolt_radius"], "Bolt Protrusion": D["bolt_protrusion"], "Bevel Segments": 6})
    ctrl.hide_render = True
    ctrl.display_type = "BOUNDS"
    ctrl["pnd_note"] = "Edit the stand here (modifier 'PND Stand'); PND_Base/Post/Arm/PostBolt read it."
    for name, part, m in (("PND_Base", 0, "walnut"), ("PND_Post", 1, "walnut"), ("PND_Arm", 2, "walnut"),
                          ("PND_PostBolt", 3, "steel")):
        gn_object(name, c_stand, g_part, {"Stand": ctrl.name, "Part": part, "Material": M[m].name}, M[m])

    # --- pivot hardware (static)
    gn_object("PND_Pin", c_pivot, g_pin, {
        "Shaft Radius": D["pin_shaft_radius"], "Shaft Start X": D["arm_end_x"] - 0.5 * D["arm_depth"],
        "Head Start X": D["pin_head_start_x"], "Head Radius": D["pin_head_radius"], "Tip X": D["pin_tip_x"],
        "Axis Z": D["pin_axis_z"], "Resolution": 48, "Material": M["steel"].name}, M["steel"])
    gn_object("PND_WireLoop", c_pivot, g_loop, {
        "Loop Radius": D["wire_loop_radius"], "Axis Z": D["pin_axis_z"], "Wire Radius": D["wire_radius"],
        "Tail Angle": math.radians(25.0), "Resolution": 48, "Material": M["wire"].name}, M["wire"])

    # --- swing rig
    swing = bpy.data.objects.new("PND_Swing", None)
    swing.empty_display_type, swing.empty_display_size = "ARROWS", 0.25 * Lc
    swing.rotation_mode = "XYZ"
    swing.location, swing.rotation_euler = (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
    swing["pnd_convention"] = "rotation_euler[1] = -theta  =>  bob centre = P + L_c (sin th, 0, -cos th)"
    swing["pnd_L_c_m"] = Lc
    c_swing.objects.link(swing)
    wire = bpy.data.objects.new("PND_Wire", wire_mesh("PND_Wire", D["wire_radius"], D["wire_length"]))
    wire.data.materials.append(M["wire"])
    bob = bpy.data.objects.new("PND_Bob", sphere_mesh("PND_Bob", D["bob_radius"]))
    bob.data.materials.append(M["brass"])
    bob.location = (0.0, 0.0, -Lc)
    # VR-12: brass eyelet/stem on the bob top, rigid child of PND_Swing (the wire still ends at the bob's top pole)
    z_top = -Lc + D["bob_radius"]
    eye = bpy.data.objects.new("PND_Eyelet", eyelet_mesh("PND_Eyelet", D["eyelet_radius"],
                                                        z_top - 0.004 * Leff, z_top + D["eyelet_height"]))
    eye.data.materials.append(M["brass"])
    for ob in (wire, bob, eye):
        c_swing.objects.link(ob)
        ob.parent = swing                       # matrix_parent_inverse stays identity
    wire["pnd_endpoints_local"] = [[0.0, 0.0, 0.0], [0.0, 0.0, -D["wire_length"]]]
    bob["pnd_radius_m"] = D["bob_radius"]

    # --- cameras of record (params/camera.json, camera role) ------------------------------------------
    cam_info = make_cameras(sc, a.camera, Leff, notes)

    params = {"root": str(ROOT), "L_c_m": Lc, "dims_m": D, "geometry": geo, "physics": phys,
              "objects": {"base": "PND_Base", "post": "PND_Post", "arm": "PND_Arm", "bolt": "PND_PostBolt",
                          "pin": "PND_Pin", "wire_loop": "PND_WireLoop", "swing": "PND_Swing",
                          "wire": "PND_Wire", "bob": "PND_Bob", "stand_controls": "PND_StandControls"},
              "materials": {k: v.name for k, v in M.items()}}
    look_used = None
    look_py = HERE / "look.py"
    if a.look:
        if not look_py.exists():
            raise SystemExit("--look given but scene/look.py does not exist")
        spec = importlib.util.spec_from_file_location("pnd_look", look_py)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.apply_look(sc, params)
        look_used = {"file": rel(look_py), "sha256": sha(look_py)}
    else:
        placeholder_env(sc, D)

    # --- motion blur (VR-05) from lead-owned config/render.json; applied after look.py so the frozen look is untouched
    mb_info = motion_blur(sc, a.render_config, notes)

    # --- static tilt (CHANGELOG 2026-10-25T23:14Z): everything static rotates about +Y through P; PND_Swing does not
    tilt_info = static_tilt(sc, phys, a.level_tilt_rad, notes)

    bake_used = None
    if a.bake:
        bake_py = ROOT / "solver" / "bake_blender.py"
        bake_report = pathlib.Path(a.out).with_suffix(".bake.json")
        bake_report.parent.mkdir(parents=True, exist_ok=True)
        old = sys.argv
        old_cwd = os.getcwd()
        os.chdir(ROOT)                       # so bake_blender records ROOT-relative paths (F3)
        sys.argv = ["blender", "--", "--trajectory", rel(a.bake), "--bob-object", "PND_Bob", "--report", rel(bake_report)]
        try:
            runpy.run_path(str(bake_py), run_name="__main__")
        except SystemExit as e:
            if e.code not in (None, 0):
                raise
        finally:
            sys.argv = old
            os.chdir(old_cwd)
        bake_used = {"script": rel(bake_py), "script_sha256": sha(bake_py), "trajectory": rel(a.bake),
                     "trajectory_sha256": sha(a.bake), "report": rel(bake_report)}
        sc.frame_set(0)

    # --- evaluate, lint, report
    bpy.context.view_layer.update()
    dg = bpy.context.evaluated_depsgraph_get()
    objs = {}
    for ob in sorted(bpy.data.objects, key=lambda o: o.name):
        rec = {"type": ob.type, "collection": [c.name for c in ob.users_collection],
               "parent": ob.parent.name if ob.parent else None,
               "matrix_world": [list(r) for r in ob.matrix_world]}
        if ob.type == "MESH":
            h, n, bb = mesh_hash(ob, dg)
            rec.update(mesh_sha256=h, verts_faces=n, local_bbox=bb)
            warn = [w.message for md in ob.modifiers if md.type == "NODES" for w in md.node_warnings]
            if warn:
                rec["node_warnings"] = warn
        objs[ob.name] = rec
    dump = pathlib.Path(a.out).with_suffix(".graph.json")
    dump.parent.mkdir(parents=True, exist_ok=True)
    nkinspect.dump_groups(str(dump), prefix="PND ")
    lint = nkinspect.check_groups(prefix="PND ")
    report = {
        "blender": bpy.app.version_string, "out": rel(a.out),
        "inputs": {"physics": rel(a.physics), "physics_sha256": sha(a.physics), "geometry": rel(a.geometry),
                   "geometry_sha256": sha(a.geometry), "camera": rel(a.camera), "camera_sha256": sha(a.camera),
                   "vendor_nodekit": "scene/vendor/VENDORED.json", "vendor_sha256": sha(HERE / "vendor" / "VENDORED.json"),
                   "camera_project_py_sha256": sha(ROOT / "camera" / "project.py"),
                   "camera_drift_csv_sha256": sha(ROOT / "camera" / "drift.csv"),
                   "builder_sha256": {p.name: sha(p) for p in (HERE / "build.py", HERE / "pnd_nodes.py")}},
        "L_c_m": Lc, "dims_m": D, "notes": notes, "look": look_used, "bake": bake_used, "camera": cam_info, "static_tilt": tilt_info, "motion_blur": mb_info,
        "graph_dump": rel(dump), "graph_dump_sha256": sha(dump), "lint": lint, "objects": objs}
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    bpy.context.preferences.filepaths.save_version = 0          # no .blend1 backups (F6)
    bpy.ops.wm.save_as_mainfile(filepath=str(pathlib.Path(a.out).resolve()), compress=False)
    stale = pathlib.Path(a.out + "1")
    if stale.exists():
        stale.unlink()
    rp = pathlib.Path(a.report or pathlib.Path(a.out).with_suffix(".build.json"))
    rp.write_text(json.dumps(report, indent=1, default=str))
    print("PND_BUILD", json.dumps({"out": rel(a.out), "L_c": Lc, "lint": lint, "notes": notes,
                                   "warnings": {k: v.get("node_warnings") for k, v in objs.items()
                                                if v.get("node_warnings")}}))


def motion_blur(sc, cfg_path, notes):
    p = pathlib.Path(cfg_path)
    if not p.exists():
        notes.append("no config/render.json: motion blur left as is")
        return None
    mb = json.loads(p.read_text()).get("motion_blur", {})
    on, sh = bool(mb.get("enabled", False)), mb.get("shutter")
    if on and sh is None:
        notes.append("config/render.json motion_blur enabled but shutter is null: motion blur OFF until the lead sets it")
        on = False
    sc.render.use_motion_blur = on
    if on:
        sc.render.motion_blur_shutter = float(sh)
        sc.render.motion_blur_position = mb.get("position", "CENTER")
        if hasattr(sc, "eevee"):
            sc.eevee.motion_blur_steps = int(mb.get("eevee_steps", 1))
    return {"config": rel(p), "config_sha256": sha(p), "enabled": on, "shutter": sh if on else None,
            "position": sc.render.motion_blur_position}


STATIC_PREFIXES = ("PND_Look_",)
STATIC_NAMES = ("PND_StandControls", "PND_Base", "PND_Post", "PND_Arm", "PND_PostBolt", "PND_Pin", "PND_WireLoop",
                "PND_FloorPlaceholder", "PND_KeyPlaceholder")
STATIC_COLLECTIONS = ("PND_Cameras", "PND_Look", "PND_EnvPlaceholder")


def static_tilt(sc, phys, override, notes):
    """Root empty PND_StaticTilt at P, rotation_euler[1] = level_tilt_rad (right-hand rule about +Y, sign as
    documented by dynamics in physics.json). All static content is parented to it with an identity parent inverse,
    so at tilt 0 every world matrix is unchanged. PND_Swing (pendulum, true gravity) is never parented."""
    pp = phys.get("parameters", phys)
    v = pp.get("level_tilt_rad")
    src = "physics.json level_tilt_rad"
    if override is not None:
        tilt, src = float(override), "TEST OVERRIDE --level-tilt-rad"
        notes.append(f"static tilt {tilt:+.6f} rad from TEST OVERRIDE (not physics.json)")
    elif v is None:
        tilt, src = 0.0, "absent in physics.json -> 0"
        print("PND_NOTE: physics.json has no level_tilt_rad; static tilt = 0")
        notes.append("static tilt = 0 (level_tilt_rad absent in physics.json)")
    else:
        tilt = float(v["value"] if isinstance(v, dict) else v)
        notes.append(f"static tilt {tilt:+.6f} rad ({math.degrees(tilt):+.4f} deg) from physics.json")
    root = bpy.data.objects.new("PND_StaticTilt", None)
    root.empty_display_type, root.empty_display_size = "PLAIN_AXES", 0.1
    root.rotation_mode = "XYZ"
    root.location, root.rotation_euler = (0.0, 0.0, 0.0), (0.0, tilt, 0.0)
    root["pnd_note"] = ("Static scene (stand, pivot hardware, set, lights, cameras) rotated about +Y through P by "
                        "level_tilt_rad relative to true gravity; PND_Swing is not a child (hangs under gravity -Z).")
    sc.collection.objects.link(root)
    kids = []
    for ob in sorted(bpy.data.objects, key=lambda o: o.name):
        if ob is root or ob.name == "PND_Swing" or ob.parent is not None:
            continue
        in_coll = any(c.name in STATIC_COLLECTIONS for c in ob.users_collection)
        if ob.name in STATIC_NAMES or ob.name.startswith(STATIC_PREFIXES) or in_coll:
            ob.parent = root
            ob.matrix_parent_inverse.identity()      # local transform/keys are kept as they were (relative to P)
            kids.append(ob.name)
    unparented = sorted(o.name for o in bpy.data.objects
                        if o.parent is None and o not in (root,) and o.name != "PND_Swing")
    if unparented:
        notes.append(f"not under PND_StaticTilt (by design or unknown): {unparented}")
    bpy.context.view_layer.update()
    return {"tilt_rad": tilt, "source": src, "children": kids, "unparented_top_level": unparented}


def make_cameras(sc, camera_json, Leff, notes):
    """PND_Cam_Reference (measured per-frame zoom/pan/roll keyed on every frame), PND_Cam_Locked, PND_Cam_SideOrtho,
    PND_Cam_Overview, all from params/camera.json via camera/project.py (no camera numbers live in this file)."""
    from mathutils import Matrix
    sys.path.insert(0, str(ROOT / "camera"))
    import project as PJ
    CJ = json.loads(pathlib.Path(camera_json).read_text())
    LE = CJ["scale"]["L_eff_m"]
    g = Leff / LE                                 # camera.json metres were written for its L_eff_m; rescale to ours
    if abs(g - 1.0) > 1e-12:
        notes.append(f"camera.json written for L_eff_m={LE}; physics L_eff_m={Leff}: camera translations scaled by "
                     f"{g:.9f} (gauge only). camera/project.py metres disagree until camera/write_params.py is re-run.")
    coll = bpy.data.collections.new("PND_Cameras")
    sc.collection.children.link(coll)

    def new_cam(name, b, ortho=False):
        cd = bpy.data.cameras.new(name)
        cd.sensor_fit, cd.sensor_width = "HORIZONTAL", b.get("sensor_width_mm", 36.0)
        cd.clip_start, cd.clip_end = 0.01, max(20.0 * Leff, b.get("clip_end_m", 0.0) * g)
        if ortho:
            cd.type, cd.ortho_scale = "ORTHO", b["ortho_scale_m"] * g
        else:
            cd.lens = b["lens_mm"]
        cd.shift_x, cd.shift_y = b.get("shift_x", 0.0), b.get("shift_y", 0.0)
        ob = bpy.data.objects.new(name, cd)
        ob.location = [v * g for v in b["location_m"]]
        ob.rotation_mode = "XYZ"
        ob.rotation_euler = b["rotation_euler_xyz_rad"]
        coll.objects.link(ob)
        return ob

    # reference camera: keys on every frame 0..240 from project.load_camera(frame=n) (+ drift.csv model)
    ref = new_cam("PND_Cam_Reference", CJ["reference_camera"]["blender"])
    ref["pnd_gauge_factor"] = g          # Blender metres = camera.json metres * g
    ref["pnd_disclosure"] = ("MEASURED camera motion: lens (zoom), shift_x/y (pan) and roll keyed per frame from "
                             "params/camera.json drift model (camera/drift.csv, camera/project.py load_camera(frame=n)); "
                             "LINEAR between frames. Camera centre fixed.")
    prev = None
    for n in range(F0, F1 + 1):
        bs = PJ.blender_camera_settings(PJ.load_camera(frame=n))
        ref.location = [v * g for v in bs["location"]]
        e = Matrix(bs["rotation_matrix"]).to_euler("XYZ", prev) if prev else Matrix(bs["rotation_matrix"]).to_euler("XYZ")
        prev = e
        ref.rotation_euler = e
        ref.data.lens, ref.data.shift_x, ref.data.shift_y = bs["lens_mm"], bs["shift_x"], bs["shift_y"]
        ref.keyframe_insert("rotation_euler", frame=n)
        for prop in ("lens", "shift_x", "shift_y"):
            ref.data.keyframe_insert(prop, frame=n)
    _linear_all(ref); _linear_all(ref.data)
    lk = new_cam("PND_Cam_Locked", CJ["locked_cameras"]["locked_reference"]["blender"])
    lk["pnd_note"] = "drift-free diagnostic: frame-120 reference calibration held for all frames"
    so = new_cam("PND_Cam_SideOrtho", CJ["locked_cameras"]["side_ortho"]["blender"], ortho=True)
    so["pnd_note"] = CJ["locked_cameras"]["side_ortho"]["description"]
    ov = new_cam("PND_Cam_Overview", CJ["locked_cameras"]["overview"]["blender"])
    ov["pnd_note"] = CJ["locked_cameras"]["overview"]["description"]
    for ob in (lk, so, ov):                          # G1: fixed diagnostic cameras cannot be nudged by accident
        ob.lock_location = ob.lock_rotation = ob.lock_scale = (True, True, True)
    sc.camera = ref
    sc.frame_set(0)
    return {"camera_json": rel(camera_json), "gauge_factor": g, "objects": [ref.name, lk.name, so.name, ov.name],
            "scene_camera": ref.name, "reference_keys": "frames 0..240, LINEAR"}


def _linear_all(idb):
    """Set LINEAR interpolation on every keyframe of an ID (Blender 5.x layered actions or legacy)."""
    ad = idb.animation_data
    if not ad or not ad.action:
        return
    act = ad.action
    curves = []
    if hasattr(act, "layers") and len(act.layers):
        for layer in act.layers:
            for strip in layer.strips:
                for cb in strip.channelbags:
                    curves += list(cb.fcurves)
    elif hasattr(act, "fcurves"):
        curves = list(act.fcurves)
    for fc in curves:
        for kp in fc.keyframe_points:
            kp.interpolation = "LINEAR"


def placeholder_env(sc, D):
    """Neutral grey studio so previews read; replaced by look.py (camera & appearance role)."""
    c = collection("PND_EnvPlaceholder")
    me = bpy.data.meshes.new("PND_FloorPlaceholder")
    s = 60 * D["L_c"]
    z = -D["h_pivot"] - D["base_thickness"]
    me.from_pydata([(-s, -s, z), (s, -s, z), (s, s, z), (-s, s, z)], [], [(0, 1, 2, 3)])
    fl = bpy.data.objects.new("PND_FloorPlaceholder", me)
    me.materials.append(mat("PND_FloorPlaceholder", (0.30, 0.29, 0.27), 0.0, 0.8))
    c.objects.link(fl)
    ld = bpy.data.lights.new("PND_KeyPlaceholder", "AREA")
    ld.energy, ld.size = 120.0 * D["L_c"] ** 2 / 0.15, 2.0 * D["L_c"]
    lo = bpy.data.objects.new("PND_KeyPlaceholder", ld)
    lo.location = (1.5 * D["L_c"], -2.5 * D["L_c"], 2.0 * D["L_c"])
    lo.rotation_euler = (math.radians(50), 0.0, math.radians(30))
    c.objects.link(lo)
    w = bpy.data.worlds.new("PND_WorldPlaceholder")
    w.color = (0.32, 0.30, 0.28)
    bg = w.node_tree.nodes.get("Background")
    bg.inputs["Color"].default_value = (0.32, 0.30, 0.28, 1.0)
    bg.inputs["Strength"].default_value = 1.0
    sc.world = w


if __name__ == "__main__":
    build(parse())
