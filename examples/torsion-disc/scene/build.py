"""Deterministic, params-driven builder for clip 03 (torsion disc) — scene role (C).

blender -b --factory-startup --python scene/build.py -- --out scene/out/torsion.blend \
    [--geometry params/geometry.json] [--physics params/physics.json] [--camera params/camera.json] \
    [--look] [--bake solver/out/trajectory.csv] [--pose "psi,dx,dy,dz,tx,ty"] [--report <out>.build.json]

Objects (prefix TOR_; scene/NOTES.md has the bake contract)
  TOR_Frame collection  TOR_FrameControls (GN 'TOR Frame', hidden) + TOR_Posts / TOR_Crossbar / TOR_Feet / TOR_Clamp /
                        TOR_Screws / TOR_Collet (GN 'TOR Frame Part'); static.
  TOR_Rig collection    TOR_Anchor  empty at the suspension point (physics.json pivot_world_m when fitted, else A = world
                                    origin = wire exit at the collet bottom), never animated
                        TOR_Disc    brass disc mesh; ORIGIN = B = centre of the TOP face (wire attachment); local +Z = disc
                                    normal (up), the engraved diameter runs along local X rotated by the custom property
                                    tor_mark_azimuth_rad (physics.json mark.mu_body_rad, default 0); rotation_mode
                                    QUATERNION; the dynamics bake keys location (= B) and rotation_quaternion (wxyz)
                        TOR_WireEnd empty at B: COPY_LOCATION -> TOR_Disc (bake may key it; the constraint wins, equal values)
                        TOR_Wire    GN 'TOR Wire' between TOR_Anchor and TOR_WireEnd (exact ends for any pose)
  TOR_Cameras           TOR_Cam_Reference keyed per frame (camera of record via camera/project.py when present, else the
                        scene role's APPROXIMATE fitted cameras) + locked diagnostic cameras (lock flags set)
Lengths: params/geometry.json ratios (unit D = disc diameter) x D_m (metres per D; source recorded in the report).
--look runs scene/look.py apply_look(scene, params). --bake runs solver/bake_blender.py in-process (runpy).
--pose: static TEST pose of TOR_Disc: twist psi (deg, about world Z), B offset from the rest point (dx, dy, dz in mm)
and tilt (tx, ty deg about world X, Y), applied as R = Ry(ty) Rx(tx) Rz(psi).
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
import tor_nodes  # noqa: E402
from nodekit import scene as nkscene, inspect as nkinspect  # noqa: E402  (path set by tor_nodes)

FPS, F0, F1 = 24, 0, 240


def rel(p):
    if p is None:
        return None
    p = pathlib.Path(p).resolve()
    try:
        return p.relative_to(ROOT).as_posix()
    except ValueError:
        return str(p)


def sha(p):
    p = pathlib.Path(p) if p else None
    return hashlib.sha256(p.read_bytes()).hexdigest() if p and p.exists() else None


def parse():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "out" / "torsion.blend"))
    ap.add_argument("--geometry", default=str(ROOT / "params" / "geometry.json"))
    ap.add_argument("--physics", default=str(ROOT / "params" / "physics.json"))
    ap.add_argument("--camera", default=str(ROOT / "params" / "camera.json"))
    ap.add_argument("--look", action="store_true")
    ap.add_argument("--bake", default=None)
    ap.add_argument("--pose", default=None, help="TEST pose 'psi_deg,dx_mm,dy_mm,dz_mm,tilt_x_deg,tilt_y_deg'")
    ap.add_argument("--approx-camera", action="store_true", help="force the scene role's approximate cameras")
    ap.add_argument("--report", default=None)
    a = ap.parse_args(argv)
    for k in ("out", "geometry", "physics", "camera", "bake", "report"):
        if getattr(a, k):
            setattr(a, k, str(pathlib.Path(getattr(a, k)).resolve()))
    return a


# ---------------------------------------------------------------- parameters
def _num(v):
    return float(v["value"] if isinstance(v, dict) else v)


def gauge(geo, phys):
    """Metres per D. physics.json geometry_m.disc_radius x 2 (dynamics' scale; status recorded) else geometry.json
    D_m_assumed."""
    if phys is not None:
        gm = phys.get("geometry_m", {})
        if gm.get("disc_radius"):
            return 2.0 * float(gm["disc_radius"]), (f"physics.json geometry_m.disc_radius x 2 (status '{phys.get('status')}', "
                                                    f"scale_mode '{gm.get('scale_mode')}')")
    return _num(geo["D_m_assumed"]), "ASSUMED geometry.json D_m_assumed"


def dims(geo, Dm):
    D = {k: v["value"] * Dm for k, v in geo["ratios"].items() if isinstance(v, dict) and "value" in v}
    D["D"] = Dm
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


def disc_mesh(name, R, t, e, seg=256, rings=24):
    """Disc of radius R, thickness t, top face at z = 0 (origin = top-face centre), both rims rounded with radius e.
    Explicit vertex/face lists (deterministic). The top face has concentric rings (for smooth anisotropic shading)."""
    prof = [(0.0, 0.0)]
    for i in range(1, rings + 1):                                   # top face rings
        prof.append(((R - e) * i / rings, 0.0))
    n = 4
    for i in range(1, n + 1):                                       # top rim fillet: centre (R - e, -e)
        a = math.pi / 2 * (1 - i / n)
        prof.append((R - e + e * math.cos(a), -e + e * math.sin(a)))
    for i in range(0, n + 1):                                       # bottom rim fillet: centre (R - e, -t + e)
        a = -math.pi / 2 * i / n
        prof.append((R - e + e * math.cos(a), -t + e + e * math.sin(a)))
    for i in range(rings - 1, -1, -1):
        prof.append(((R - e) * i / rings, -t))
    verts, ids = [], []
    for r, z in prof:
        if r <= 1e-12:
            ids.append([len(verts)])
            verts.append((0.0, 0.0, z))
        else:
            ring = []
            for j in range(seg):
                a = 2 * math.pi * j / seg
                ring.append(len(verts))
                verts.append((r * math.cos(a), r * math.sin(a), z))
            ids.append(ring)
    faces = []
    for A, B in zip(ids[:-1], ids[1:]):
        if len(A) == 1:
            faces += [(A[0], B[j], B[(j + 1) % seg]) for j in range(seg)]
        elif len(B) == 1:
            faces += [(A[j], B[0], A[(j + 1) % seg]) for j in range(seg)]
        else:
            faces += [(A[j], B[j], B[(j + 1) % seg], A[(j + 1) % seg]) for j in range(seg)]
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], faces)
    for p in me.polygons:
        p.use_smooth = True
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
    h = hashlib.sha256(co.tobytes() + pv.tobytes() + ps.tobytes()).hexdigest()
    n = (len(m.vertices), len(m.polygons))
    bb = (co.reshape(-1, 3).min(0).tolist(), co.reshape(-1, 3).max(0).tolist()) if len(co) else None
    ev.to_mesh_clear()
    return h, n, bb


def anim_hash(ob):
    """sha256 over all keyframe co/handles of the object and its data (cameras: lens/shift keys)."""
    import numpy as np
    h = hashlib.sha256()
    found = False
    for idb in (ob, ob.data):
        ad = getattr(idb, "animation_data", None) if idb is not None else None
        if not ad or not ad.action:
            continue
        act = ad.action
        curves = []
        if hasattr(act, "layers") and len(act.layers):
            for layer in act.layers:
                for strip in layer.strips:
                    for cb in strip.channelbags:
                        curves += list(cb.fcurves)
        for fc in sorted(curves, key=lambda c: (c.data_path, c.array_index)):
            kp = fc.keyframe_points
            a = np.empty(len(kp) * 6, dtype=np.float32)
            co = np.empty(len(kp) * 2, dtype=np.float32); kp.foreach_get("co", co)
            hl = np.empty(len(kp) * 2, dtype=np.float32); kp.foreach_get("handle_left", hl)
            hr = np.empty(len(kp) * 2, dtype=np.float32); kp.foreach_get("handle_right", hr)
            h.update(f"{fc.data_path}[{fc.array_index}]".encode() + co.tobytes() + hl.tobytes() + hr.tobytes())
            found = True
    return h.hexdigest() if found else None


def pose_quat(psi_deg, tx_deg, ty_deg):
    from mathutils import Matrix
    Rm = Matrix.Rotation(math.radians(ty_deg), 3, "Y") @ Matrix.Rotation(math.radians(tx_deg), 3, "X") @ \
        Matrix.Rotation(math.radians(psi_deg), 3, "Z")
    return Rm.to_quaternion()


# ---------------------------------------------------------------- build
def build(a):
    nkscene.reset()
    sc = bpy.context.scene
    sc.name = "TOR_Scene"
    sc.render.fps, sc.render.fps_base = FPS, 1.0
    sc.frame_start, sc.frame_end = F0, F1
    sc.frame_current = 0
    sc.unit_settings.system, sc.unit_settings.scale_length = "METRIC", 1.0
    sc.render.resolution_x, sc.render.resolution_y = 1920, 1088

    geo = json.loads(pathlib.Path(a.geometry).read_text())
    phys = json.loads(pathlib.Path(a.physics).read_text()) if pathlib.Path(a.physics).exists() else None
    camj = json.loads(pathlib.Path(a.camera).read_text()) if pathlib.Path(a.camera).exists() else None
    Dm, Dm_src = gauge(geo, phys)
    D = dims(geo, Dm)
    notes = [f"gauge D_m = {Dm:.6f} m per D from {Dm_src}"]

    # rest wire length / disc thickness / suspension point: physics.json when FITTED (a baked build must be exact), else
    # geometry.json (reference-matched); differences are reported
    ell = D["wire_length_A_to_disc_top"]
    t_disc = D["disc_thickness"]
    pivot = (0.0, 0.0, 0.0)
    fitted = phys is not None and str(phys.get("status", "")).startswith("fitted")
    if phys is not None:
        gm = phys.get("geometry_m", {})
        used = {}
        for key, mine, label in (("ell", ell, "wire length"), ("disc_thickness", t_disc, "disc thickness")):
            if gm.get(key):
                d = float(gm[key]) - mine
                src = "physics.json (fitted)" if fitted else "geometry.json"
                msg = (f"{label}: uses {src}; geometry.json {mine:.6f} m vs physics.json {gm[key]:.6f} m "
                       f"(diff {d:+.6f} m, {d / mine:+.1%})")
                if abs(d) > 0.02 * mine:
                    msg = "WARNING " + msg + " -> simulated and reference-fitted proportions disagree"
                    print("TOR_WARNING", msg)
                notes.append(msg)
                used[key] = float(gm[key]) if fitted else mine
        if fitted:
            ell = used.get("ell", ell)
            t_disc = used.get("disc_thickness", t_disc)
            pv = phys.get("pivot_world_m")
            if pv is not None:
                pivot = tuple(float(x) for x in (pv["value"] if isinstance(pv, dict) else pv))
                notes.append(f"suspension point (TOR_Anchor, wire top) at physics.json pivot_world_m {pivot} m; the clamp mesh stays "
                             f"at the camera's A (origin); offset {math.hypot(pivot[0], pivot[1]) * 1000:.2f} mm, collet radius "
                             f"{D['collet_radius'] * 1000:.2f} mm")
                if math.hypot(pivot[0], pivot[1]) > D["collet_radius"]:
                    print("TOR_WARNING pivot outside the collet footprint")
                    notes.append("WARNING pivot outside the collet footprint")
    mu = 0.0
    if phys is not None and isinstance(phys.get("mark"), dict) and "mu_body_rad" in phys["mark"]:
        mu = _num(phys["mark"]["mu_body_rad"])

    M = {"walnut": mat("TOR_Walnut", (0.16, 0.075, 0.04), 0.0, 0.45),
         "blacksteel": mat("TOR_BlackSteel", (0.02, 0.02, 0.022), 0.6, 0.35),
         "screw": mat("TOR_ScrewSteel", (0.5, 0.5, 0.52), 1.0, 0.2),
         "wire": mat("TOR_WireSteel", (0.6, 0.6, 0.61), 1.0, 0.25),
         "collet": mat("TOR_ColletSteel", (0.8, 0.8, 0.81), 1.0, 0.1),
         "brass": mat("TOR_Brass", (0.8, 0.6, 0.3), 1.0, 0.25)}

    g_frame, g_part, g_wire = tor_nodes.frame(), tor_nodes.frame_part(), tor_nodes.wire()
    c_frame, c_rig = collection("TOR_Frame"), collection("TOR_Rig")

    ctrl, _ = gn_object("TOR_FrameControls", c_frame, g_frame, {
        "Span": D["span_inner"], "Frame Centre X": D["frame_centre_x"], "Post Width": D["post_width_x"],
        "Post Depth": D["post_depth_y"], "Post Top Z": D["post_top_z"], "Bar Height": D["bar_height_z"],
        "Bar Depth": D["bar_depth_y"], "Foot Top Z": D["foot_top_z"], "Foot Height": D["foot_height"],
        "Foot Width": D["foot_width_x"], "Foot Front Y": D["foot_front_y"], "Foot Back Y": D["foot_back_y"],
        "Clamp Width": D["clamp_width_x"], "Clamp Depth": D["clamp_depth_y"], "Clamp Top Z": D["clamp_top_z"],
        "Clamp Bottom Z": D["clamp_bottom_z"], "Clamp Sheet": D["clamp_sheet"], "Clamp Bend Radius": D["clamp_bend_radius"],
        "Top Screw Radius": D["clamp_top_screw_radius"], "Top Screw Height": D["clamp_top_screw_height"],
        "Top Screw Y": D["clamp_top_screw_y"], "Front Screw Radius": D["clamp_front_screw_radius"],
        "Front Screw Z": D["clamp_front_screw_z_above_bottom"], "Front Screw Height": D["clamp_front_screw_height"],
        "Washer Radius": D["clamp_front_washer_radius"], "Washer Height": D["clamp_front_washer_height"],
        "Socket Radius": D["clamp_front_socket_radius"],
        "Collet Radius": D["collet_radius"], "Wood Edge Radius": D["wood_edge_radius"], "Bevel Segments": 3})
    ctrl.hide_render = True
    ctrl.display_type = "BOUNDS"
    ctrl["tor_note"] = "Edit the stand here (modifier 'TOR Frame'); TOR_Posts/Crossbar/Feet/Clamp/Screws/Collet read it."
    for name, part, m in (("TOR_Posts", 0, "walnut"), ("TOR_Crossbar", 1, "walnut"), ("TOR_Feet", 2, "walnut"),
                          ("TOR_Clamp", 3, "blacksteel"), ("TOR_Screws", 4, "screw"), ("TOR_Collet", 5, "collet"),
                          ("TOR_ScrewSockets", 6, "blacksteel")):
        gn_object(name, c_frame, g_part, {"Frame": ctrl.name, "Part": part, "Material": M[m].name}, M[m])

    # --- rig
    anchor = bpy.data.objects.new("TOR_Anchor", None)
    anchor.empty_display_type, anchor.empty_display_size = "SPHERE", 0.003
    anchor["tor_note"] = ("suspension point = wire top end: physics.json pivot_world_m when fitted (else A = world origin = wire exit "
                          "at the clamp collet bottom); never animated")
    anchor.location = pivot
    anchor["tor_pivot_m"] = list(pivot)
    c_rig.objects.link(anchor)

    disc = bpy.data.objects.new("TOR_Disc", disc_mesh("TOR_Disc", D["disc_diameter"] / 2, t_disc, D["disc_edge_radius"]))
    disc.data.materials.append(M["brass"])
    c_rig.objects.link(disc)
    disc.rotation_mode = "QUATERNION"                       # dynamics bake contract: location + rotation_quaternion (wxyz)
    rest_B = (pivot[0], pivot[1], pivot[2] - ell)
    disc.location = rest_B
    disc["tor_origin"] = "B = centre of the top face (wire attachment); local +Z = disc normal; mark along local X rotated by tor_mark_azimuth_rad"
    disc["tor_mark_azimuth_rad"] = float(mu)
    disc["tor_mark_width_m"] = D["mark_width"]
    disc["tor_mark_depth_m"] = D["mark_depth"]
    disc["tor_radius_m"], disc["tor_thickness_m"] = D["disc_diameter"] / 2, t_disc
    disc["tor_rest_location_note"] = "unbaked builds hang B straight below A at geometry.json wire_length_A_to_disc_top"

    wend = bpy.data.objects.new("TOR_WireEnd", None)
    wend.empty_display_type, wend.empty_display_size = "PLAIN_AXES", 0.003
    c_rig.objects.link(wend)
    wend.rotation_mode = "QUATERNION"
    wend.location = rest_B
    con = wend.constraints.new("COPY_LOCATION")
    con.target = disc
    con.name = "TOR_WireEnd_follows_disc_origin"
    wend["tor_note"] = ("wire lower end B; COPY_LOCATION -> TOR_Disc origin, so the attachment is single-sourced and exact; the "
                        "dynamics bake may key its location (bake contract) but the constraint result equals TOR_Disc.location")
    wire, _ = gn_object("TOR_Wire", c_rig, g_wire, {"Top": anchor.name, "Bottom": wend.name, "Radius": D["wire_radius"],
                                                    "Segments": 16, "Profile Resolution": 8, "Material": M["wire"].name}, M["wire"])
    wire["tor_note"] = "GN 'TOR Wire': straight tube A (TOR_Anchor) -> B (TOR_Disc origin); point attribute tor_end"

    pose_info = None
    if a.pose:
        vals = [float(x) for x in a.pose.split(",")] + [0.0] * 6
        psi, dx, dy, dz, tx, ty = vals[:6]
        disc.location = (rest_B[0] + dx / 1000, rest_B[1] + dy / 1000, rest_B[2] + dz / 1000)
        disc.rotation_quaternion = pose_quat(psi, tx, ty)
        pose_info = {"psi_deg": psi, "B_offset_mm": [dx, dy, dz], "tilt_deg": [tx, ty], "status": "TEST POSE (not physics)"}
        notes.append(f"TEST POSE {pose_info}")

    cam_info = make_cameras(sc, camj, geo, Dm, phys, notes, a)

    params = {"root": str(ROOT), "D_m": Dm, "dims_m": D, "geometry": geo, "physics": phys,
              "objects": {"posts": "TOR_Posts", "bar": "TOR_Crossbar", "feet": "TOR_Feet", "clamp": "TOR_Clamp",
                          "screws": "TOR_Screws", "collet": "TOR_Collet", "wire": "TOR_Wire", "disc": "TOR_Disc",
                          "frame_controls": "TOR_FrameControls"},
              "materials": {k: v.name for k, v in M.items()}}
    look_used = None
    if a.look:
        look_py = HERE / "look.py"
        spec = importlib.util.spec_from_file_location("tor_look", look_py)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.apply_look(sc, params)
        look_used = {"file": rel(look_py), "sha256": sha(look_py),
                     "look_json": rel(ROOT / "params" / "look.json"), "look_json_sha256": sha(ROOT / "params" / "look.json")}
    else:
        placeholder_env(sc, D)
    mb_info = motion_blur(sc, notes)

    bake_used = None
    if a.bake:
        bake_py = ROOT / "solver" / "bake_blender.py"
        bake_report = pathlib.Path(a.out).with_suffix(".bake.json")
        bake_report.parent.mkdir(parents=True, exist_ok=True)
        old, old_cwd = sys.argv, os.getcwd()
        os.chdir(ROOT)
        sys.argv = ["blender", "--", "--trajectory", rel(a.bake), "--report", rel(bake_report)]
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

    bpy.context.view_layer.update()
    dg = bpy.context.evaluated_depsgraph_get()
    objs = {}
    for ob in sorted(bpy.data.objects, key=lambda o: o.name):
        rec = {"type": ob.type, "collection": [c.name for c in ob.users_collection],
               "parent": ob.parent.name if ob.parent else None,
               "matrix_world": [[round(x, 9) for x in r] for r in ob.matrix_world]}
        ah = anim_hash(ob)
        if ah:
            rec["anim_sha256"] = ah
        if ob.type == "MESH":
            h, n, bb = mesh_hash(ob, dg)
            rec.update(mesh_sha256=h, verts_faces=n, local_bbox=bb)
            warn = [w.message for md in ob.modifiers if md.type == "NODES" for w in md.node_warnings]
            if warn:
                rec["node_warnings"] = warn
        objs[ob.name] = rec
    dump = pathlib.Path(a.out).with_suffix(".graph.json")
    dump.parent.mkdir(parents=True, exist_ok=True)
    nkinspect.dump_groups(str(dump), prefix="TOR ")
    lint = nkinspect.check_groups(prefix="TOR ")
    report = {
        "blender": bpy.app.version_string, "out": rel(a.out),
        "inputs": {"geometry": rel(a.geometry), "geometry_sha256": sha(a.geometry),
                   "physics": rel(a.physics) if phys is not None else None, "physics_sha256": sha(a.physics),
                   "camera": rel(a.camera) if camj is not None else None, "camera_sha256": sha(a.camera),
                   "camera_project_py_sha256": sha(ROOT / "camera" / "project.py"),
                   "vendor_sha256": sha(HERE / "vendor" / "VENDORED.json"),
                   "builder_sha256": {p.name: sha(p) for p in (HERE / "build.py", HERE / "tor_nodes.py")}},
        "D_m": Dm, "D_m_source": Dm_src, "dims_m": D, "notes": notes, "pose": pose_info, "look": look_used,
        "motion_blur": mb_info, "bake": bake_used, "camera": cam_info, "graph_dump": rel(dump),
        "graph_dump_sha256": sha(dump), "lint": lint, "objects": objs}
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    bpy.context.preferences.filepaths.save_version = 0
    bpy.ops.wm.save_as_mainfile(filepath=str(pathlib.Path(a.out).resolve()), compress=False)
    stale = pathlib.Path(a.out + "1")
    if stale.exists():
        stale.unlink()
    rp = pathlib.Path(a.report or pathlib.Path(a.out).with_suffix(".build.json"))
    rp.write_text(json.dumps(report, indent=1, default=str))
    print("TOR_BUILD", json.dumps({"out": rel(a.out), "D_m": Dm, "lint": lint, "notes": notes, "camera": cam_info.get("kind"),
                                   "warnings": {k: v.get("node_warnings") for k, v in objs.items() if v.get("node_warnings")}}))


def motion_blur(sc, notes):
    """config/render.json (lead): {"motion_blur": {"enabled", "shutter", "position", "eevee_steps"?}}. OFF unless enabled
    and shutter set."""
    p = ROOT / "config" / "render.json"
    if not p.exists():
        sc.render.use_motion_blur = False
        notes.append("motion blur OFF (config/render.json absent)")
        return {"enabled": False, "source": "config/render.json absent"}
    mb = json.loads(p.read_text()).get("motion_blur", {})
    if mb.get("enabled") and mb.get("shutter"):
        sc.render.use_motion_blur = True
        sc.render.motion_blur_shutter = float(mb["shutter"])
        sc.render.motion_blur_position = mb.get("position", "CENTER")
        if hasattr(sc.eevee, "motion_blur_steps"):
            sc.eevee.motion_blur_steps = int(mb.get("eevee_steps", 1))
        notes.append(f"motion blur ON: shutter {mb['shutter']} frame, {sc.render.motion_blur_position} (config/render.json)")
        return {"enabled": True, "shutter": float(mb["shutter"]), "position": sc.render.motion_blur_position, "source": rel(p), "sha256": sha(p)}
    sc.render.use_motion_blur = False
    notes.append("motion blur OFF (config/render.json: not enabled or shutter 0/null)")
    return {"enabled": False, "source": rel(p), "sha256": sha(p)}


# ---------------------------------------------------------------- cameras
def _cv_to_blender(Rcv, C):
    """world->OpenCV camera rotation + centre -> Blender camera matrix_world (camera looks along its local -Z)."""
    from mathutils import Matrix
    Rm = Matrix([list(r) for r in Rcv]).transposed() @ Matrix(((1, 0, 0), (0, -1, 0), (0, 0, -1)))
    M = Rm.to_4x4()
    M.translation = C
    return M


def approx_camera(geo, n):
    """Scene-role APPROXIMATE camera at frame n (float): keys at the fitted frames, rotation slerp, centre and f linear.
    Returns (R_world_to_cv 3x3 list, C in D units, f_px, (cx, cy))."""
    from mathutils import Matrix
    fr = geo["camera_approx"]["frames"]
    keys = sorted(int(k) for k in fr)
    n = min(max(float(n), keys[0]), keys[-1])
    i = max(j for j in range(len(keys)) if keys[j] <= n) if n < keys[-1] else len(keys) - 2
    k0, k1 = keys[i], keys[i + 1]
    s = (n - k0) / (k1 - k0)
    c0, c1 = fr[str(k0)], fr[str(k1)]
    q0 = Matrix(c0["R_world_to_cv"]).to_quaternion()
    q1 = Matrix(c1["R_world_to_cv"]).to_quaternion()
    R = q0.slerp(q1, s).to_matrix()
    C = [(1 - s) * a + s * b for a, b in zip(c0["C_D"], c1["C_D"])]
    f = (1 - s) * c0["f_px"] + s * c1["f_px"]
    pp = c0["principal_point_px"]
    return [list(r) for r in R], C, f, pp


def _approx_settings(geo, n, g):
    R, C, f, pp = approx_camera(geo, n)
    M = _cv_to_blender(R, [c * g for c in C])
    return {"matrix": M, "lens_mm": f * 36.0 / 1920.0, "shift_x": -(pp[0] - 960.0) / 1920.0, "shift_y": (pp[1] - 544.0) / 1920.0}


def make_cameras(sc, camj, geo, Dm, phys, notes, a):
    coll = bpy.data.collections.new("TOR_Cameras")
    sc.collection.children.link(coll)
    proj_py = ROOT / "camera" / "project.py"
    if camj is not None and proj_py.exists() and not a.approx_camera:
        try:
            return _cameras_of_record(sc, coll, camj, Dm, phys, notes, a.camera)
        except Exception as ex:                                   # camera role mid-update: fall back, loudly
            notes.append(f"WARNING camera of record failed ({type(ex).__name__}: {ex}); APPROXIMATE cameras used")
            print("TOR_WARNING camera of record failed:", ex)
            for ob in list(coll.objects):
                bpy.data.objects.remove(ob)
    return _approx_cameras(sc, coll, geo, Dm, notes)


def _new_cam(coll, name, lens=50.0):
    cd = bpy.data.cameras.new(name)
    cd.sensor_fit, cd.sensor_width = "HORIZONTAL", 36.0
    cd.clip_start, cd.clip_end = 0.005, 50.0
    cd.lens = lens
    ob = bpy.data.objects.new(name, cd)
    coll.objects.link(ob)
    return ob


def _approx_cameras(sc, coll, geo, Dm, notes):
    from mathutils import Vector
    ref = _new_cam(coll, "TOR_Cam_Reference")
    ref.rotation_mode = "XYZ"
    prev = None
    for n in range(F0, F1 + 1):
        st = _approx_settings(geo, n, Dm)
        M = st["matrix"]
        ref.location = M.translation
        e = M.to_3x3().to_euler("XYZ", prev) if prev else M.to_3x3().to_euler("XYZ")
        prev = e
        ref.rotation_euler = e
        ref.data.lens, ref.data.shift_x, ref.data.shift_y = st["lens_mm"], st["shift_x"], st["shift_y"]
        ref.keyframe_insert("location", frame=n)
        ref.keyframe_insert("rotation_euler", frame=n)
        for prop in ("lens", "shift_x", "shift_y"):
            ref.data.keyframe_insert(prop, frame=n)
    _linear_all(ref)
    _linear_all(ref.data)
    ref["tor_disclosure"] = ("APPROXIMATE scene-role camera: joint landmark+silhouette fit at reference frames 0/120/240 "
                             "(scene/measure/fit_geometry.py), slerp/linear between; the camera of record (params/camera.json) "
                             "replaces it automatically")
    objs = [ref.name]
    # locked diagnostics: frame-0 approximate camera, a frontal and a high three-quarter overview
    st = _approx_settings(geo, 0, Dm)
    lk = _new_cam(coll, "TOR_Cam_Locked", st["lens_mm"])
    lk.matrix_world = st["matrix"]
    lk.data.shift_x, lk.data.shift_y = st["shift_x"], st["shift_y"]
    for name, loc, tgt, lens in (("TOR_Cam_Front", (0.0, -7.0, -0.6), (0.0, 0.0, -0.6), 60.0),
                                 ("TOR_Cam_Overview", (2.5, -4.5, 2.5), (0.0, 0.0, -0.8), 45.0),
                                 ("TOR_Cam_DiscTop", (0.0, -0.6, 1.2), (0.0, 0.0, -0.9), 50.0)):
        ob = _new_cam(coll, name, lens)
        ob.location = Vector(loc) * Dm
        ob.rotation_mode = "QUATERNION"
        ob.rotation_quaternion = (Vector(tgt) * Dm - ob.location).to_track_quat("-Z", "Y")
        objs.append(name)
    for name in objs[1:] + ["TOR_Cam_Locked"]:
        ob = bpy.data.objects[name]
        ob.lock_location = ob.lock_rotation = ob.lock_scale = (True, True, True)
        ob["tor_note"] = "locked diagnostic camera (scene role)"
    objs.append("TOR_Cam_Locked")
    sc.camera = ref
    sc.frame_set(0)
    notes.append("camera: APPROXIMATE scene-role cameras (params/camera.json or camera/project.py absent or forced)")
    return {"kind": "approximate_scene_fit", "objects": sorted(set(objs)), "scene_camera": ref.name, "reference_keys": "frames 0..240, LINEAR"}


def _cameras_of_record(sc, coll, CJ, Dm, phys, notes, camera_path):
    """TOR_Cam_Reference keyed on every frame from camera/project.py: blender_camera_settings(load_camera(frame=n));
    camera.json lengths are camera units -> metres via camera.json scale.metres_per_unit, else (camera unit D) D_m, else
    physics.json camera_scale.camera_units_per_m. Locked diagnostic cameras with lock flags."""
    from mathutils import Matrix
    sys.path.insert(0, str(ROOT / "camera"))
    import project as PJ
    sc_ = CJ.get("scale") or {}
    if sc_.get("metres_per_unit"):
        g, gsrc = float(sc_["metres_per_unit"]), "camera.json scale.metres_per_unit"
    elif str(sc_.get("unit", "")).strip().startswith("D"):
        g, gsrc = Dm, "camera.json unit = D (disc diameter) -> D_m (same gauge as the geometry)"
    elif phys is not None and isinstance(phys.get("camera_scale"), dict) and phys["camera_scale"].get("camera_units_per_m"):
        g, gsrc = 1.0 / float(phys["camera_scale"]["camera_units_per_m"]), "physics.json camera_scale.camera_units_per_m"
    else:
        g, gsrc = Dm, "camera unit assumed = D (geometry unit) x D_m"

    def new_cam(name, bs):
        cd = bpy.data.cameras.new(name)
        cd.sensor_fit, cd.sensor_width = "HORIZONTAL", bs.get("sensor_width_mm", 36.0)
        cd.clip_start, cd.clip_end = 0.005, 50.0
        if bs.get("type") == "ORTHO":
            cd.type, cd.ortho_scale = "ORTHO", bs["ortho_scale"] * g
        else:
            cd.lens = bs["lens_mm"]
        cd.shift_x, cd.shift_y = bs["shift_x"], bs["shift_y"]
        ob = bpy.data.objects.new(name, cd)
        ob.location = [v * g for v in bs["location"]]
        ob.rotation_mode = "XYZ"
        ob.rotation_euler = Matrix(bs["rotation_matrix"]).to_euler("XYZ")
        coll.objects.link(ob)
        return ob
    ref = new_cam("TOR_Cam_Reference", PJ.blender_camera_settings(PJ.load_camera(frame=0)))
    ref["tor_gauge_factor"] = g
    ref["tor_disclosure"] = ("MEASURED camera motion keyed on every frame from params/camera.json via camera/project.py "
                             "blender_camera_settings(load_camera(frame=n)); LINEAR between frames")
    prev = None
    for n in range(F0, F1 + 1):
        bs = PJ.blender_camera_settings(PJ.load_camera(frame=n))
        ref.location = [v * g for v in bs["location"]]
        e = Matrix(bs["rotation_matrix"]).to_euler("XYZ", prev) if prev else Matrix(bs["rotation_matrix"]).to_euler("XYZ")
        prev = e
        ref.rotation_euler = e
        ref.data.lens, ref.data.shift_x, ref.data.shift_y = bs["lens_mm"], bs["shift_x"], bs["shift_y"]
        ref.keyframe_insert("location", frame=n)
        ref.keyframe_insert("rotation_euler", frame=n)
        for prop in ("lens", "shift_x", "shift_y"):
            ref.data.keyframe_insert(prop, frame=n)
    _linear_all(ref)
    _linear_all(ref.data)
    objs = [ref.name]
    for kind, name in (("locked", "TOR_Cam_Locked"), ("top_ortho", "TOR_Cam_TopOrtho"), ("overview", "TOR_Cam_Overview")):
        try:
            ob = new_cam(name, PJ.blender_camera_settings(PJ.load_camera(kind=kind)))
        except Exception as ex:
            notes.append(f"diagnostic camera {kind} not built: {type(ex).__name__} {ex}")
            continue
        ob.lock_location = ob.lock_rotation = ob.lock_scale = (True, True, True)
        ob["tor_note"] = f"locked diagnostic camera from camera/project.py load_camera(kind='{kind}')"
        objs.append(name)
    sc.camera = ref
    sc.frame_set(0)
    notes.append(f"camera: camera of record (gauge {g:.6f} m per camera unit from {gsrc})")
    return {"kind": "camera_of_record", "camera_json": rel(camera_path), "gauge_factor_m_per_unit": g, "gauge_source": gsrc,
            "objects": objs, "scene_camera": ref.name, "reference_keys": "frames 0..240, LINEAR"}


def _linear_all(idb):
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
    c = collection("TOR_EnvPlaceholder")
    z = D["table_z"]
    s = 40 * D["D"]
    yw = 4.0 * D["D"]
    me = bpy.data.meshes.new("TOR_FloorPlaceholder")
    me.from_pydata([(-s, -s, z), (s, -s, z), (s, yw, z), (-s, yw, z)], [], [(0, 1, 2, 3)])
    me.materials.append(mat("TOR_FloorPlaceholder", (0.42, 0.38, 0.42), 0.0, 0.8))
    c.objects.link(bpy.data.objects.new("TOR_FloorPlaceholder", me))
    me2 = bpy.data.meshes.new("TOR_WallPlaceholder")
    me2.from_pydata([(-s, yw, z), (s, yw, z), (s, yw, z + s), (-s, yw, z + s)], [], [(0, 1, 2, 3)])
    me2.materials.append(mat("TOR_WallPlaceholder", (0.45, 0.43, 0.46), 0.0, 0.9))
    c.objects.link(bpy.data.objects.new("TOR_WallPlaceholder", me2))
    ld = bpy.data.lights.new("TOR_KeyPlaceholder", "AREA")
    ld.energy, ld.size = 30.0 * (D["D"] / 0.1) ** 2, 1.5 * D["D"] * 5
    lo = bpy.data.objects.new("TOR_KeyPlaceholder", ld)
    lo.location = (1.0 * D["D"], -3.0 * D["D"], 3.0 * D["D"])
    lo.rotation_mode = "QUATERNION"
    from mathutils import Vector
    lo.rotation_quaternion = (Vector((0, 0, -0.8 * D["D"])) - lo.location).to_track_quat("-Z", "Y")
    c.objects.link(lo)
    w = bpy.data.worlds.new("TOR_WorldPlaceholder")
    bg = w.node_tree.nodes.get("Background")
    bg.inputs["Color"].default_value = (0.3, 0.29, 0.3, 1.0)
    sc.world = w


if __name__ == "__main__":
    build(parse())
