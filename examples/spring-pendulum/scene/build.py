"""Deterministic, params-driven builder for clip 02 (vertical spring) — scene role (C).

blender -b --factory-startup --python scene/build.py -- --out scene/out/spring.blend \
    [--geometry params/geometry.json] [--physics params/physics.json] [--camera params/camera.json] \
    [--look] [--bake solver/out/trajectory.csv] [--pose "dx,dy,dz[,rx,ry,rz]"] [--report <out>.build.json]

Objects (prefix VSP_; see scene/NOTES.md for the bake contract)
  VSP_Frame collection   VSP_FrameControls (GN 'VSP Frame', hidden) + VSP_Posts / VSP_Crossbar / VSP_BasePlate /
                         VSP_Screws (GN 'VSP Frame Part'); static.
  VSP_Rig collection     VSP_Anchor   empty at A = world origin, never animated
                         VSP_Weight   mesh (brushed-steel cylinder); ORIGIN = hinge H (eye inner top = lower-loop
                                      inner bottom); body axis local -Z from H to the cylinder centre; the dynamics
                                      bake keys its location (= H) and rotation_euler (XYZ)
                         VSP_WeightEye child of VSP_Weight (eye ring, local XZ plane), identity parent inverse
                         VSP_HookLower empty, Copy-Location constraint -> VSP_Weight (so H is single-sourced;
                                      location keys on it are allowed but overridden)
                         VSP_Spring   mesh at the origin carrying GN 'VSP Spring' (Lower End = VSP_HookLower)
  VSP_Cameras            VSP_Cam_Reference (+ diagnostics when params/camera.json exists)
Lengths: params/geometry.json ratios (unit S) x S_m (metres per S; source recorded in the report).
--look runs scene/look.py apply_look(scene, params). --bake runs solver/bake_blender.py in-process (runpy).
--pose sets a static TEST pose of VSP_Weight (H offset from the rest point, metres; Euler XYZ degrees).
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
import vsp_nodes  # noqa: E402
from nodekit import scene as nkscene, inspect as nkinspect  # noqa: E402  (path set by vsp_nodes)

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
    ap.add_argument("--out", default=str(HERE / "out" / "spring.blend"))
    ap.add_argument("--geometry", default=str(ROOT / "params" / "geometry.json"))
    ap.add_argument("--physics", default=str(ROOT / "params" / "physics.json"))
    ap.add_argument("--camera", default=str(ROOT / "params" / "camera.json"))
    ap.add_argument("--look", action="store_true")
    ap.add_argument("--bake", default=None)
    ap.add_argument("--pose", default=None, help="TEST pose 'dx,dy,dz[,rx,ry,rz]' (m, deg) of VSP_Weight vs rest")
    ap.add_argument("--report", default=None)
    a = ap.parse_args(argv)
    for k in ("out", "geometry", "physics", "camera", "bake", "report"):
        if getattr(a, k):
            setattr(a, k, str(pathlib.Path(getattr(a, k)).resolve()))
    return a


# ---------------------------------------------------------------- parameters
def _num(v):
    return float(v["value"] if isinstance(v, dict) else v)


def _find(o, keys):
    if isinstance(o, dict):
        for k, x in o.items():
            if k in keys:
                return x
        for x in o.values():
            y = _find(x, keys)
            if y is not None:
                return y
    return None


def gauge(geo, phys, cam):
    """Metres per W_in (= S). Order (lead, 2026-10-26): physics.json S_m / metres_per_unit;
    physics.json camera_scale.camera_units_per_m (W_in per metre, dynamics; status recorded);
    camera.json scale.metres_per_unit; physics.json ell_eq / measured |A-H|; geometry.json S_m_assumed."""
    st = phys.get("status", "?") if phys is not None else None
    if phys is not None:
        v = _find(phys, ("S_m", "metres_per_unit", "metres_per_W_in"))
        if v is not None and _num(v):
            return _num(v), f"physics.json metres per W_in (status '{st}')"
        cs = phys.get("camera_scale")
        if isinstance(cs, dict) and cs.get("camera_units_per_m"):
            return 1.0 / float(cs["camera_units_per_m"]), (f"physics.json camera_scale.camera_units_per_m = {cs['camera_units_per_m']} "
                                                           f"(provenance '{cs.get('provenance')}', status '{st}')")
    if cam is not None and cam.get("scale", {}).get("metres_per_unit"):
        return float(cam["scale"]["metres_per_unit"]), "camera.json scale.metres_per_unit"
    if phys is not None:
        v = _find(phys.get("parameters", {}), ("ell_eq",)) or _find(phys.get("geometry_m", {}), ("ell_eq",))
        if v is not None:
            ah = geo["ratios"]["spring_length_ref_AH"]["value"]
            return _num(v) / ah, f"physics.json ell_eq ({_num(v)} m, status '{st}') / measured |A-H| = {ah} W_in"
    return _num(geo["S_m_assumed"]), "ASSUMED geometry.json S_m_assumed"


def dims(geo, Sm):
    r = {k: v["value"] for k, v in geo["ratios"].items() if isinstance(v, dict) and "value" in v and k != "loop_gap_A_px"}
    D = {k: (v if k.startswith(("coil_count", "hook_turns")) else v * Sm) for k, v in r.items()}
    D["S"] = Sm
    D["H_to_cylinder_centre"] = geo["derived"]["H_to_cylinder_centre"] * Sm
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


def revolve(name, profile, seg):
    """Surface of revolution about local Z from an (r, z) profile listed top -> bottom; explicit vertex/face lists
    (deterministic). Points with r == 0 become single pole vertices."""
    verts, rings = [], []
    for r, z in profile:
        if r <= 1e-12:
            rings.append([len(verts)])
            verts.append((0.0, 0.0, z))
        else:
            ids = []
            for j in range(seg):
                a = 2 * math.pi * j / seg
                ids.append(len(verts))
                verts.append((r * math.cos(a), r * math.sin(a), z))
            rings.append(ids)
    faces = []
    for A, B in zip(rings[:-1], rings[1:]):
        if len(A) == 1 and len(B) > 1:
            faces += [(A[0], B[j], B[(j + 1) % seg]) for j in range(seg)][::-1] if False else \
                     [(A[0], B[(j + 1) % seg], B[j]) for j in range(seg)]
        elif len(B) == 1 and len(A) > 1:
            faces += [(A[j], A[(j + 1) % seg], B[0]) for j in range(seg)]
        elif len(A) > 1 and len(B) > 1:
            faces += [(A[j], A[(j + 1) % seg], B[(j + 1) % seg], B[j]) for j in range(seg)]
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], faces)
    me.update()
    return me


def weight_mesh(name, R, h, z_top, r_top, c_bot, seg=96):
    """Cylinder radius R, height h, top face at z_top (local), rounded top rim r_top, bottom chamfer c_bot."""
    prof = [(0.0, z_top)]
    n = 6
    for i in range(n + 1):                                          # top rim fillet: centre (R - r_top, z_top - r_top)
        a = math.pi / 2 * (1 - i / n)
        prof.append((R - r_top + r_top * math.cos(a), z_top - r_top + r_top * math.sin(a)))
    zb = z_top - h
    prof += [(R, zb + c_bot), (R - c_bot, zb), (0.0, zb)]
    me = revolve(name, prof, seg)
    for p in me.polygons:
        p.use_smooth = True
    # sharp at the chamfer and the bottom face; smooth on the side and the rounded rim
    me.update()
    return me


def torus_mesh(name, R, r, seg=64, pseg=16):
    """Torus in the local XZ plane (axis = local Y), centre at the origin; explicit lists."""
    verts, faces = [], []
    for i in range(seg):
        a = 2 * math.pi * i / seg
        cx, cz = R * math.cos(a), R * math.sin(a)
        for j in range(pseg):
            b = 2 * math.pi * j / pseg
            rr = R + r * math.cos(b)
            verts.append((rr * math.cos(a), r * math.sin(b), rr * math.sin(a)))
    for i in range(seg):
        for j in range(pseg):
            i2, j2 = (i + 1) % seg, (j + 1) % pseg
            faces.append((i * pseg + j, i * pseg + j2, i2 * pseg + j2, i2 * pseg + j))
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


# ---------------------------------------------------------------- build
def build(a):
    nkscene.reset()
    sc = bpy.context.scene
    sc.name = "VSP_Scene"
    sc.render.fps, sc.render.fps_base = FPS, 1.0
    sc.frame_start, sc.frame_end = F0, F1
    sc.frame_current = 0
    sc.unit_settings.system, sc.unit_settings.scale_length = "METRIC", 1.0
    sc.render.resolution_x, sc.render.resolution_y = 1920, 1088

    geo = json.loads(pathlib.Path(a.geometry).read_text())
    phys = json.loads(pathlib.Path(a.physics).read_text()) if pathlib.Path(a.physics).exists() else None
    camj = json.loads(pathlib.Path(a.camera).read_text()) if pathlib.Path(a.camera).exists() else None
    Sm, Sm_src = gauge(geo, phys, camj)
    D = dims(geo, Sm)
    notes = [f"gauge S_m = {Sm:.6f} m per S from {Sm_src}"]

    # weight: H -> cylinder centre from geometry.json (measured); physics.json's value is compared and reported
    b_c = D["H_to_cylinder_centre"]
    if phys is not None:
        v = _find(phys, ("H_to_cylinder_centre",))
        if v is not None:
            b_phys = _num(v)
            msg = (f"H_to_cylinder_centre: mesh uses geometry.json {b_c:.6f} m; physics.json has {b_phys:.6f} m "
                   f"(diff {b_phys - b_c:+.6f} m, {(b_phys - b_c) / b_c:+.1%})")
            if abs(b_phys - b_c) > 0.02 * b_c:
                msg = "WARNING " + msg + " -> simulated CoM and rendered body disagree; see scene/NOTES.md requests"
                print("VSP_WARNING", msg)
            notes.append(msg)
    D["H_to_cylinder_centre_used"] = b_c

    M = {"walnut": mat("VSP_Walnut", (0.16, 0.075, 0.04), 0.0, 0.45),
         "blacksteel": mat("VSP_BlackSteel", (0.02, 0.02, 0.022), 0.6, 0.35),
         "barsteel": mat("VSP_CrossbarSteel", (0.02, 0.019, 0.018), 0.5, 0.3),
         "screw": mat("VSP_ScrewSteel", (0.03, 0.03, 0.032), 0.8, 0.3),
         "spring": mat("VSP_SpringSteel", (0.55, 0.55, 0.56), 1.0, 0.3),
         "weight": mat("VSP_BrushedSteel", (0.62, 0.62, 0.63), 1.0, 0.25),
         "eye": mat("VSP_EyeSteel", (0.6, 0.6, 0.61), 1.0, 0.22)}

    g_frame, g_part, g_spring = vsp_nodes.frame(), vsp_nodes.frame_part(), vsp_nodes.spring()
    c_frame, c_rig = collection("VSP_Frame"), collection("VSP_Rig")

    ctrl, _ = gn_object("VSP_FrameControls", c_frame, g_frame, {
        "Span": Sm, "Post Width": D["post_width_x"], "Post Depth": D["post_depth_y"], "Post Top Z": D["post_top_z"],
        "Base Top Z": D["base_top_z"], "Bar Height": D["bar_height_z"], "Bar Depth": D["bar_depth_y"],
        "Frame Centre X": D["frame_centre_x"], "Plate Width": D["plate_width_x"], "Plate Depth": D["plate_depth_y"],
        "Plate Centre Y": D["plate_centre_y"], "Plate Thickness": D["plate_thickness"],
        "Plate Corner Radius": D["plate_corner_radius"], "Plate Edge Radius": D["plate_top_edge_radius"],
        "Wood Edge Radius": D["wood_edge_radius"], "Screw Radius": D["screw_head_radius"],
        "Screw Height": D["screw_head_height"], "Screw Inset X": D["screw_inset_x"], "Screw Inset Y": D["screw_inset_y"],
        "Bevel Segments": 5})
    ctrl.hide_render = True
    ctrl.display_type = "BOUNDS"
    ctrl["vsp_note"] = "Edit the frame here (modifier 'VSP Frame'); VSP_Posts/Crossbar/BasePlate/Screws read it."
    for name, part, m in (("VSP_Posts", 0, "walnut"), ("VSP_Crossbar", 1, "barsteel"),
                          ("VSP_BasePlate", 2, "blacksteel"), ("VSP_Screws", 3, "screw")):
        gn_object(name, c_frame, g_part, {"Frame": ctrl.name, "Part": part, "Material": M[m].name}, M[m])

    # --- rig
    anchor = bpy.data.objects.new("VSP_Anchor", None)
    anchor.empty_display_type, anchor.empty_display_size = "SPHERE", 0.01
    anchor["vsp_note"] = "A = top-loop bearing point on the crossbar top; world origin; never animated"
    c_rig.objects.link(anchor)

    R_e, r_e = D["eye_ring_radius"], D["eye_wire_radius"]
    Rw, hw = D["weight_radius"], D["weight_height"]
    z_top = -(b_c - hw / 2)                                   # cylinder top in weight-local coordinates
    weight = bpy.data.objects.new("VSP_Weight", weight_mesh("VSP_Weight", Rw, hw, z_top, D["weight_top_edge_radius"],
                                                            D["weight_bottom_chamfer"]))
    weight.data.materials.append(M["weight"])
    c_rig.objects.link(weight)
    weight.rotation_mode = "XYZ"
    P_eff = (D["hinge_P_eff_x"], 0.0, D["hinge_P_eff_z"])
    rest_H = (P_eff[0], 0.0, P_eff[2] - D["spring_length_ref_PH"])     # hanging straight below the hinge
    weight.location = rest_H
    weight["vsp_origin"] = "hinge H = eye inner top (lower-loop inner bottom); body axis local -Z towards the CoM"
    weight["vsp_cylinder_centre_local_m"] = [0.0, 0.0, -b_c]
    weight["vsp_cylinder_radius_m"], weight["vsp_cylinder_height_m"] = Rw, hw
    weight["vsp_rest_location_note"] = "unbaked builds place H at |A-H| measured at reference frame 120, straight below A"
    eye = bpy.data.objects.new("VSP_WeightEye", torus_mesh("VSP_WeightEye", R_e, r_e))
    eye.data.materials.append(M["eye"])
    c_rig.objects.link(eye)
    eye.parent = weight
    eye.location = (0.0, 0.0, -(R_e - r_e))                   # inner top of the ring at the origin (H)
    eye["vsp_ring_radius_m"], eye["vsp_wire_radius_m"] = R_e, r_e
    gap = (-(b_c - hw / 2)) - (-(2 * R_e))
    if abs(gap) > 1e-9:
        notes.append(f"eye bottom vs cylinder top: {(-(2 * R_e)) - z_top:+.6f} m (positive = gap, negative = overlap)")

    hook = bpy.data.objects.new("VSP_HookLower", None)
    hook.empty_display_type, hook.empty_display_size = "PLAIN_AXES", 0.01
    c_rig.objects.link(hook)
    con = hook.constraints.new("COPY_LOCATION")
    con.target = weight
    hook.location = rest_H
    hook["vsp_note"] = "spring lower end H; copies VSP_Weight's location (weight origin = H), so attachment is exact"

    spring, _ = gn_object("VSP_Spring", c_rig, g_spring, {
        "Lower End": hook.name, "Hinge": P_eff, "Coil Radius": D["coil_mean_radius"], "Wire Radius": D["spring_wire_radius"],
        "Hook Turns": int(D["hook_turns"]), "Hook Turn Pitch": D["hook_turn_pitch"],
        "Open Coils": int(D["coil_count_open"]), "Tight Coils": int(D["coil_count_tight"]),
        "Open Length Ref": D["coil_body_open_length_ref"], "Tight Length Ref": D["coil_body_tight_length_ref"],
        "Bend Length": D["bend_length"], "Resolution": 32, "Profile Resolution": 12, "Hook Wire Radius": D["hook_wire_radius"],
        "Bottom Hardware Length": D["bottom_hardware_length"],
        "Top Loop Inner Height": D["top_loop_inner_height"], "Top Loop Inner Width": D["top_loop_inner_width_y"],
        "Top Loop Corner Radius": 0.8 * (D["top_loop_inner_width_y"] - D["bar_depth_y"]) / 2,
        "Lower Loop Inner Width": D["lower_loop_inner_width_y"], "Lower Loop Corner Radius": 0.35 * D["lower_loop_inner_width_y"],
        "Lower Loop Gap": 0.02 * Sm, "Lower Loop Wire Radius": D["spring_wire_radius"], "Material": M["spring"].name}, M["spring"])
    hinge_ob = bpy.data.objects.new("VSP_HingePeff", None)
    hinge_ob.empty_display_type, hinge_ob.empty_display_size = "SPHERE", 0.005
    hinge_ob.location = P_eff
    hinge_ob["vsp_note"] = "P_eff: swing hinge of the spring body (top coil), static; lead CHANGELOG 2026-10-26T00:52Z"
    c_rig.objects.link(hinge_ob)
    spring["vsp_note"] = ("GN 'VSP Spring': static top loop + S-bend + 2 hook turns down to P_eff; moving body "
                          "P_eff->H(VSP_HookLower) rotated about P_eff (ramped over the bend length); fixed coil counts; "
                          "uniform extension per turn; point attributes vsp_part / vsp_cl / vsp_cl_rest / vsp_w")

    pose_info = None
    if a.pose:
        vals = [float(x) for x in a.pose.split(",")]
        d = vals[:3]
        rdeg = vals[3:6] if len(vals) >= 6 else [0.0, 0.0, 0.0]
        weight.location = (rest_H[0] + d[0], rest_H[1] + d[1], rest_H[2] + d[2])
        weight.rotation_euler = [math.radians(x) for x in rdeg]
        pose_info = {"H_offset_m": d, "rotation_euler_deg": rdeg, "status": "TEST POSE (not physics)"}
        notes.append(f"TEST POSE {pose_info}")

    cam_info = make_cameras(sc, camj, geo, Sm, notes, a.camera)

    params = {"root": str(ROOT), "S_m": Sm, "dims_m": D, "geometry": geo, "physics": phys,
              "objects": {"posts": "VSP_Posts", "bar": "VSP_Crossbar", "plate": "VSP_BasePlate", "screws": "VSP_Screws",
                          "spring": "VSP_Spring", "weight": "VSP_Weight", "eye": "VSP_WeightEye",
                          "frame_controls": "VSP_FrameControls"},
              "materials": {k: v.name for k, v in M.items()}}
    look_used = None
    if a.look:
        look_py = HERE / "look.py"
        spec = importlib.util.spec_from_file_location("vsp_look", look_py)
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
        if ob.type == "MESH":
            h, n, bb = mesh_hash(ob, dg)
            rec.update(mesh_sha256=h, verts_faces=n, local_bbox=bb)
            warn = [w.message for md in ob.modifiers if md.type == "NODES" for w in md.node_warnings]
            if warn:
                rec["node_warnings"] = warn
        objs[ob.name] = rec
    dump = pathlib.Path(a.out).with_suffix(".graph.json")
    dump.parent.mkdir(parents=True, exist_ok=True)
    nkinspect.dump_groups(str(dump), prefix="VSP ")
    lint = nkinspect.check_groups(prefix="VSP ")
    report = {
        "blender": bpy.app.version_string, "out": rel(a.out),
        "inputs": {"geometry": rel(a.geometry), "geometry_sha256": sha(a.geometry),
                   "physics": rel(a.physics) if phys is not None else None, "physics_sha256": sha(a.physics),
                   "camera": rel(a.camera) if camj is not None else None, "camera_sha256": sha(a.camera),
                   "camera_project_py_sha256": sha(ROOT / "camera" / "project.py"),
                   "vendor_sha256": sha(HERE / "vendor" / "VENDORED.json"),
                   "builder_sha256": {p.name: sha(p) for p in (HERE / "build.py", HERE / "vsp_nodes.py")}},
        "S_m": Sm, "S_m_source": Sm_src, "dims_m": D, "notes": notes, "pose": pose_info, "look": look_used, "motion_blur": mb_info,
        "bake": bake_used, "camera": cam_info, "graph_dump": rel(dump), "graph_dump_sha256": sha(dump), "lint": lint,
        "objects": objs}
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    bpy.context.preferences.filepaths.save_version = 0
    bpy.ops.wm.save_as_mainfile(filepath=str(pathlib.Path(a.out).resolve()), compress=False)
    stale = pathlib.Path(a.out + "1")
    if stale.exists():
        stale.unlink()
    rp = pathlib.Path(a.report or pathlib.Path(a.out).with_suffix(".build.json"))
    rp.write_text(json.dumps(report, indent=1, default=str))
    print("VSP_BUILD", json.dumps({"out": rel(a.out), "S_m": Sm, "lint": lint, "notes": notes,
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
    if mb.get("enabled") and mb.get("shutter") is not None:
        sc.render.use_motion_blur = True
        sc.render.motion_blur_shutter = float(mb["shutter"])
        sc.render.motion_blur_position = mb.get("position", "CENTER")
        if hasattr(sc.eevee, "motion_blur_steps"):
            sc.eevee.motion_blur_steps = int(mb.get("eevee_steps", 1))
        notes.append(f"motion blur ON: shutter {mb['shutter']} frame, {sc.render.motion_blur_position} (config/render.json)")
        return {"enabled": True, "shutter": float(mb["shutter"]), "position": sc.render.motion_blur_position, "source": rel(p), "sha256": sha(p)}
    sc.render.use_motion_blur = False
    notes.append("motion blur OFF (config/render.json: not enabled or shutter null)")
    return {"enabled": False, "source": rel(p)}


# ---------------------------------------------------------------- cameras
def _cv_to_blender(Rcv, C):
    """world->OpenCV camera rotation + centre -> Blender camera matrix_world (camera looks along its local -Z)."""
    from mathutils import Matrix
    Rm = Matrix([list(r) for r in Rcv]).transposed() @ Matrix(((1, 0, 0), (0, -1, 0), (0, 0, -1)))
    M = Rm.to_4x4()
    M.translation = C
    return M


def make_cameras(sc, camj, geo, Sm, notes, camera_path):
    coll = bpy.data.collections.new("VSP_Cameras")
    sc.collection.children.link(coll)
    proj_py = ROOT / "camera" / "project.py"
    if camj is not None and proj_py.exists():
        return _cameras_of_record(sc, coll, camj, Sm, notes, camera_path)
    # approximate scene-role camera (frame-0 fit, static)
    ap = geo["camera_frame0_approx"]
    cd = bpy.data.cameras.new("VSP_Cam_Reference")
    cd.sensor_fit, cd.sensor_width = "HORIZONTAL", 36.0
    cd.lens = ap["f_px"] * 36.0 / 1920.0
    cd.shift_x = -(ap["principal_point_px"][0] - 960.0) / 1920.0
    cd.shift_y = (ap["principal_point_px"][1] - 544.0) / 1920.0
    cd.clip_start, cd.clip_end = 0.01, 100.0
    ob = bpy.data.objects.new("VSP_Cam_Reference", cd)
    ob.matrix_world = _cv_to_blender(ap["R_world_to_cv"], [c * Sm for c in ap["C_S"]])
    coll.objects.link(ob)
    ob["vsp_disclosure"] = ("APPROXIMATE scene-role camera: static frame-0 pinhole fit (scene/measure/fit_frame0.py). "
                            "The reference camera drifts (push-in ~4 % by frame 240); camera of record pending.")
    sc.camera = ob
    notes.append("camera: APPROXIMATE frame-0 fit (params/camera.json or camera/project.py absent)")
    return {"kind": "approximate_frame0", "objects": [ob.name], "scene_camera": ob.name, "lens_mm": cd.lens}


def _cameras_of_record(sc, coll, CJ, Sm, notes, camera_path):
    """VSP_Cam_Reference keyed on every frame from camera/project.py: blender_camera_settings(load_camera(frame=n))
    (camera.json lengths are W_in -> metres x S_m); locked diagnostic cameras (locked reference, front ortho, overview)
    with lock_location/rotation/scale set."""
    from mathutils import Matrix
    sys.path.insert(0, str(ROOT / "camera"))
    import project as PJ
    g = Sm                                    # Blender metres per camera.json unit (W_in)

    def new_cam(name, bs):
        cd = bpy.data.cameras.new(name)
        cd.sensor_fit, cd.sensor_width = "HORIZONTAL", bs.get("sensor_width_mm", 36.0)
        cd.clip_start, cd.clip_end = 0.01, 100.0
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
    ref = new_cam("VSP_Cam_Reference", PJ.blender_camera_settings(PJ.load_camera(frame=0)))
    ref["vsp_gauge_factor"] = g
    ref["vsp_disclosure"] = ("MEASURED camera motion: lens (zoom), shift (pan) and roll keyed on every frame from params/camera.json "
                             "via camera/project.py blender_camera_settings(load_camera(frame=n)); LINEAR between frames; centre fixed")
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
    for kind, name in (("locked", "VSP_Cam_Locked"), ("front_ortho", "VSP_Cam_FrontOrtho"), ("overview", "VSP_Cam_Overview")):
        try:
            ob = new_cam(name, PJ.blender_camera_settings(PJ.load_camera(kind=kind)))
        except Exception as ex:                                  # a missing diagnostic camera is not fatal
            notes.append(f"diagnostic camera {kind} not built: {ex}")
            continue
        ob.lock_location = ob.lock_rotation = ob.lock_scale = (True, True, True)
        ob["vsp_note"] = f"locked diagnostic camera from camera/project.py load_camera(kind='{kind}')"
        objs.append(name)
    sc.camera = ref
    sc.frame_set(0)
    return {"kind": "camera_of_record", "camera_json": rel(camera_path), "gauge_factor_m_per_Win": g, "objects": objs,
            "scene_camera": ref.name, "reference_keys": "frames 0..240, LINEAR"}


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
    c = collection("VSP_EnvPlaceholder")
    S = D["S"]
    z = D["base_top_z"] - D["plate_thickness"]
    s = 40 * S
    me = bpy.data.meshes.new("VSP_FloorPlaceholder")
    me.from_pydata([(-s, -s, z), (s, -s, z), (s, D["table_back_edge_y"], z), (-s, D["table_back_edge_y"], z)], [], [(0, 1, 2, 3)])
    me.materials.append(mat("VSP_FloorPlaceholder", (0.45, 0.36, 0.28), 0.0, 0.7))
    c.objects.link(bpy.data.objects.new("VSP_FloorPlaceholder", me))
    yw = D["table_back_edge_y"]
    me2 = bpy.data.meshes.new("VSP_WallPlaceholder")
    me2.from_pydata([(-s, yw, z), (s, yw, z), (s, yw, z + s), (-s, yw, z + s)], [], [(0, 1, 2, 3)])
    me2.materials.append(mat("VSP_WallPlaceholder", (0.3, 0.29, 0.28), 0.0, 0.9))
    c.objects.link(bpy.data.objects.new("VSP_WallPlaceholder", me2))
    ld = bpy.data.lights.new("VSP_KeyPlaceholder", "AREA")
    ld.energy, ld.size = 60.0 * (S / 0.25) ** 2, 1.5 * S
    lo = bpy.data.objects.new("VSP_KeyPlaceholder", ld)
    lo.location = (0.8 * S, -3.0 * S, 1.5 * S)
    lo.rotation_euler = (math.radians(60), 0.0, math.radians(15))
    c.objects.link(lo)
    w = bpy.data.worlds.new("VSP_WorldPlaceholder")
    bg = w.node_tree.nodes.get("Background")
    bg.inputs["Color"].default_value = (0.3, 0.29, 0.28, 1.0)
    sc.world = w


if __name__ == "__main__":
    build(parse())
