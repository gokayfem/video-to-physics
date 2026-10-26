"""Batch-2 ruler buckling: animated, editable reconstruction around the first-frame scene (role D).

  blender -b --factory-startup --python scene/build_anim.py -- --out scene/out/ruler_anim.blend \
          --deformation solver/out/deformation.npz [--meta solver/out/meta.json] [--render-frames 0,184,240 --render-dir DIR]
          [--checks-out scene/out/anim_checks.json]

Opens scene/out/ruler_ff.blend (look + camera untouched; config/camera_ff.json is only READ), then
  * strip: deterministic importer npz -> PC2 point cache (scene/out/<blend stem>_strip.pc2) on a Mesh Cache modifier
    (frame n <-> t = n/24 s, scene frames 0..240, 5 samples per frame = frame + 4 subframes). Each solid vertex =
    mid-surface node + side * t/2 * unit normal of the deformed mid-surface -> thickness rigid through the cross-section.
  * RUL_TipFrame (empty): keyed per sample from the top node row (origin = top-centre mid-surface node; axes = width,
    normal, tangent). Platform plate/tabs/screws/peg are parented to it.
  * pick-and-place arm (RUL_Arm*): column + Z carriage + 2-link SCARA + vertical quill + wrist + 2-finger gripper.
    Each weight travels stack -> peg, is lowered along the peg axis and seated at its schedule time; from the seat time
    a Child Of constraint (influence keyed 0 -> 1) parents it to RUL_TipFrame so it rides with the deformed tip.
  * checks (anim_checks.json): jaw attachment, length, thickness, intersections (BVH), camera, tip-follow, cache readback.
No hand animation: every key is computed from the npz + schedule.
"""
import hashlib
import json
import math
import pathlib
import struct
import sys

import bmesh
import bpy
import numpy as np
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Matrix, Quaternion, Vector
from mathutils.bvhtree import BVHTree

ROOT = pathlib.Path(__file__).resolve().parents[1]
P = json.loads((ROOT / "config/scene_params.json").read_text())
argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


def arg(name, default=None):
    return argv[argv.index(name) + 1] if name in argv else default


def rp(p):
    p = pathlib.Path(p); return p if p.is_absolute() else (ROOT / p).resolve()


OUT = rp(arg("--out", "scene/out/ruler_anim.blend"))
DEF = rp(arg("--deformation", "solver/out/deformation.npz"))
META = rp(arg("--meta", str(DEF.parent / "meta.json")))
CHECKS = rp(arg("--checks-out", str(OUT.with_name(OUT.stem + "_checks.json"))))
RFR = [int(x) for x in arg("--render-frames", "").split(",") if x]
RDIR = rp(arg("--render-dir", "scene/visuals"))
RPREFIX = arg("--render-prefix", "v001")
NOCHECK = "--no-checks" in argv
SCHED_META = {}
FPS, SPF, NFR = 24, 5, 241            # frame n <-> t = n/24 s ; 5 samples per frame (frame + 4 subframes)
R = P["ruler"]; W = P["slotted_weights"]; PL = P["platform"]
TH, NX, NS = R["thickness_m"], R["mesh_segments_width"], R["mesh_segments_length"]
JAW_Z = P["vise"]["jaw_top_z_m"]


# ------------------------------------------------------------------ deformation input
def load_deformation():
    d = np.load(DEF, allow_pickle=False)
    key = next(k for k in ("mid_surface_m", "positions", "mid", "x") if k in d.files)
    mid = np.asarray(d[key], np.float64)
    if mid.ndim == 3:
        mid = mid.reshape(mid.shape[0], NS + 1, NX + 1, 3)
    assert mid.shape[1:] == (NS + 1, NX + 1, 3), f"npz {key} shape {mid.shape}: expected (N,{NS + 1},{NX + 1},3) node order [s,width]"
    n = mid.shape[0]
    t = np.asarray(d["t_s"] if "t_s" in d.files else d["t"] if "t" in d.files else np.arange(n) / (FPS * SPF), np.float64)
    tgt = np.arange((NFR - 1) * SPF + 1) / (FPS * SPF)
    if n != len(tgt) or np.max(np.abs(t - tgt)) > 1e-9:  # resample linearly onto the sample grid (exact if already on it)
        idx = np.clip(np.searchsorted(t, tgt, "right") - 1, 0, n - 2); w = np.clip((tgt - t[idx]) / (t[idx + 1] - t[idx]), 0, 1)
        mid = mid[idx] * (1 - w)[:, None, None, None] + mid[idx + 1] * w[:, None, None, None]; t = tgt
    sched = None
    if META.exists():
        m = json.loads(META.read_text()); sm = m.get("schedule", {})
        if "slot_offset_above_tab_bottom_m" in sm:
            SCHED_META["offsets"] = [float(x) for x in sm["slot_offset_above_tab_bottom_m"]]
        if "origin_undeformed_world" in m.get("tipbody", {}):
            SCHED_META["O_z"] = float(m["tipbody"]["origin_undeformed_world"][2])
        if "weight_contact_start_s" in sm:
            SCHED_META["times"] = [float(x) for x in sm["weight_contact_start_s"]]
    if "weight_times_s" in d.files:
        sched = [float(x) for x in d["weight_times_s"]]
    elif META.exists():
        m = json.loads(META.read_text())
        for k in ("weight_times_s", "loading_schedule_s", "weight_place_times_s"):
            if k in m:
                sched = [float(x) for x in m[k]]; break
        if sched is None and isinstance(m.get("loading"), dict):
            sched = [float(x) for x in m["loading"].get("weight_times_s", [])] or None
    if sched is None and "times" in SCHED_META:
        sched = SCHED_META["times"]
    if sched is None:
        sched = [1.0 + 0.6 * k for k in range(7)]          # lead default (CHANGELOG): 7 weights, 0.6 s apart
    sched = [s for s in sched if s <= t[-1]][:W["count"]]
    src = str(d["source"]) if "source" in d.files else str(DEF)
    extra = {k: np.asarray(d[k]) for k in ("tip_load_N", "buckled", "tipbody_matrix_world") if k in d.files}
    return t, mid, sched, src, extra


def solid_positions(mid, rest_local):
    """mid (N,S,X,3) world -> solid vertex positions (N,V,3) world. rest_local (V,3) of RUL_Ruler (origin bottom-centre)."""
    i = np.rint((rest_local[:, 0] + R["width_m"] / 2) / (R["width_m"] / NX)).astype(int)
    j = np.rint(rest_local[:, 2] / (R["length_m"] / NS)).astype(int)
    side = np.sign(rest_local[:, 1])
    assert np.all(side != 0) and np.allclose(np.abs(rest_local[:, 1]), TH / 2, atol=1e-9)
    ds = np.gradient(mid, axis=1); dx = np.gradient(mid, axis=2)
    nrm = np.cross(ds, dx); nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True)   # rest: +Z x +X = +Y
    return mid[:, j, i] + side[None, :, None] * (TH / 2) * nrm[:, j, i], (j, i, side)


def tip_frames(mid):
    o = mid[:, -1, NX // 2]
    ex = mid[:, -1, -1] - mid[:, -1, 0]; ex /= np.linalg.norm(ex, axis=-1, keepdims=True)
    et = mid[:, -1, NX // 2] - mid[:, -2, NX // 2]
    et -= (et * ex).sum(-1, keepdims=True) * ex; et /= np.linalg.norm(et, axis=-1, keepdims=True)
    ey = np.cross(et, ex)
    M = np.zeros((len(o), 4, 4)); M[:, :3, 0], M[:, :3, 1], M[:, :3, 2], M[:, :3, 3], M[:, 3, 3] = ex, ey, et, o, 1
    return M


def write_pc2(path, pos_local):
    n, v, _ = pos_local.shape
    with open(path, "wb") as f:
        f.write(b"POINTCACHE2\0" + struct.pack("<iiffi", 1, v, 0.0, 1.0 / SPF, n))
        f.write(np.ascontiguousarray(pos_local, "<f4").tobytes())


# ------------------------------------------------------------------ math helpers
def T(x=0, y=0, z=0):
    m = np.eye(4); m[:3, 3] = (x, y, z); return m


def Rz(a):
    m = np.eye(4); c, s = math.cos(a), math.sin(a); m[:2, :2] = [[c, -s], [s, c]]; return m


def ss(x):
    x = min(max(x, 0.0), 1.0); return x * x * (3 - 2 * x)


def blend(A, B, u):
    """Rigid interpolation (lerp position, slerp rotation)."""
    qa = Matrix(A[:3, :3].tolist()).to_quaternion(); qb = Matrix(B[:3, :3].tolist()).to_quaternion()
    M = np.eye(4); M[:3, :3] = np.array(qa.slerp(qb, u).to_matrix()); M[:3, 3] = A[:3, 3] * (1 - u) + B[:3, 3] * u
    return M


def key_matrices(ob, times, mats, constant_after=None):
    """Bake world/basis matrices as loc + quaternion keys at frame = t*FPS (linear)."""
    ob.rotation_mode = "QUATERNION"
    ad = ob.animation_data_create(); act = bpy.data.actions.new(ob.name + "Action"); ad.action = act
    frames = np.asarray(times) * FPS
    loc = np.array([m[:3, 3] for m in mats]); q = []
    prev = None
    for m in mats:
        qq = Matrix(m[:3, :3].tolist()).to_quaternion()
        if prev is not None and prev.dot(qq) < 0:
            qq.negate()
        q.append(tuple(qq)); prev = qq
    q = np.array(q)
    for path, data in (("location", loc), ("rotation_quaternion", q)):
        for c in range(data.shape[1]):
            fc = ob.keyframe_insert(path, index=c, frame=0.0) and None
            fc = next(f for f in fcurves_of(ob) if f.data_path == path and f.array_index == c)
            fc.keyframe_points.clear(); fc.keyframe_points.add(len(frames))
            co = np.empty(2 * len(frames)); co[0::2] = frames; co[1::2] = data[:, c]
            fc.keyframe_points.foreach_set("co", co)
            fc.keyframe_points.foreach_set("interpolation", [bpy.types.Keyframe.bl_rna.properties["interpolation"].enum_items["LINEAR"].value] * len(frames))
            if constant_after is not None:
                fc.keyframe_points[constant_after].interpolation = "CONSTANT"
            fc.update()


def fcurves_of(ob):
    act = ob.animation_data.action
    if hasattr(act, "fcurves"):
        try:
            return list(act.fcurves)
        except Exception:
            pass
    out = []  # Blender 5 layered actions
    for layer in act.layers:
        for strip in layer.strips:
            for cb in strip.channelbags:
                out += list(cb.fcurves)
    return out


# ------------------------------------------------------------------ arm geometry
def mat(name, col, rough=0.35, metal=0.0):
    m = bpy.data.materials.new(name); m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]; b.inputs["Base Color"].default_value = (*col, 1)
    b.inputs["Roughness"].default_value = rough; b.inputs["Metallic"].default_value = metal
    return m


def boxmesh(name, sx, sy, sz, center=(0, 0, 0), m=None, coll=None, bevel=0.0):
    bm = bmesh.new(); bmesh.ops.create_cube(bm, size=1.0)
    for v in bm.verts:
        v.co = Vector((v.co.x * sx + center[0], v.co.y * sy + center[1], v.co.z * sz + center[2]))
    me = bpy.data.meshes.new(name); bm.to_mesh(me); bm.free()
    ob = bpy.data.objects.new(name, me); coll.objects.link(ob)
    if m:
        me.materials.append(m)
    if bevel:
        md = ob.modifiers.new("bevel", "BEVEL"); md.width = bevel; md.segments = 2
    return ob


def cylmesh(name, r, depth, center=(0, 0, 0), m=None, coll=None, seg=32):
    bm = bmesh.new(); bmesh.ops.create_cone(bm, cap_ends=True, segments=seg, radius1=r, radius2=r, depth=depth)
    for v in bm.verts:
        v.co += Vector(center)
    me = bpy.data.meshes.new(name); bm.to_mesh(me); bm.free()
    me.polygons.foreach_set("use_smooth", [True] * len(me.polygons))
    ob = bpy.data.objects.new(name, me); coll.objects.link(ob)
    if m:
        me.materials.append(m)
    return ob


ARM = dict(base=(0.25, 0.11), L1=0.30, L2=0.30, z_link=0.80, ceiling=1.6, exit_z=0.70,  # ceiling-mounted SCARA: column + links above the frame (z>=0.58)
           wrist=0.040, palm_y=-0.034, z_travel=0.52,
           finger_r=W["outer_diameter_m"] / 2 + 0.0022, finger_open=0.004, elbow_sign=-1.0)


def build_arm():
    coll = bpy.data.collections.new("RUL_Arm"); bpy.context.scene.collection.children.link(coll)
    orange = mat("RUL_ArmDarkSatin", (0.035, 0.035, 0.04), 0.42, 0.5); grey = mat("RUL_ArmGrey", (0.06, 0.06, 0.065), 0.45, 0.6)
    steel = mat("RUL_ArmSteel", (0.12, 0.12, 0.125), 0.38, 1.0); rubber = mat("RUL_ArmPad", (0.03, 0.03, 0.03), 0.8)
    bx, by = ARM["base"]; zl, zc = ARM["z_link"], ARM["ceiling"]
    o = {}
    o["base"] = cylmesh("RUL_ArmBase", 0.04, 0.02, (bx, by, zc - 0.01), grey, coll, 48)        # ceiling mount (out of frame)
    o["column"] = cylmesh("RUL_ArmColumn", 0.014, zc - zl, (bx, by, (zc + zl) / 2), orange, coll)
    o["carriage"] = boxmesh("RUL_ArmCarriage", 0.036, 0.036, 0.03, (0, 0, 0), orange, coll, 0.003)
    o["link1"] = boxmesh("RUL_ArmLink1", ARM["L1"] + 0.024, 0.02, 0.014, (ARM["L1"] / 2, 0, 0), orange, coll, 0.003)
    o["link2"] = boxmesh("RUL_ArmLink2", ARM["L2"] + 0.018, 0.016, 0.012, (ARM["L2"] / 2, 0, 0), orange, coll, 0.003)
    o["quill"] = cylmesh("RUL_ArmQuill", 0.003, 1.0, (0, 0, -0.5), steel, coll)                  # unit rod, Z-scaled (telescoping)
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.005, segments=24, ring_count=12)
    wb = bpy.context.object; wb.name = "RUL_ArmWrist"; wb.data.materials.append(grey); bpy.ops.object.shade_smooth()
    for c in wb.users_collection:
        c.objects.unlink(wb)
    coll.objects.link(wb); o["wrist"] = wb
    py = ARM["palm_y"]
    o["gripper"] = boxmesh("RUL_ArmGripper", 0.046, 0.012, 0.005, (0, py, 0.0015), grey, coll, 0.0015)  # bottom 1.25 mm above the weight bottom
    o["gripper_post"] = boxmesh("RUL_ArmGripperPost", 0.008, 0.008, ARM["wrist"] - 0.012, (0, py, (ARM["wrist"] - 0.012) / 2 + 0.005), grey, coll, 0.001)
    o["gripper_post"].parent = o["gripper"]
    for s, nm in ((1, "R"), (-1, "L")):
        f = boxmesh(f"RUL_ArmFinger{nm}", 0.004, -py + 0.010, 0.003, (0, (py + 0.010) / 2 + 0.002, 0), rubber, coll, 0.0006)
        f.parent = o["gripper"]; f["side"] = s; o["finger" + nm] = f
    return o


def scara(target_xy):
    bx, by = ARM["base"]; dx, dy = target_xy[0] - bx, target_xy[1] - by
    r = math.hypot(dx, dy); L1, L2 = ARM["L1"], ARM["L2"]
    c2 = (r * r - L1 * L1 - L2 * L2) / (2 * L1 * L2); c2 = max(-1.0, min(1.0, c2))
    q2 = ARM["elbow_sign"] * math.acos(c2)
    q1 = math.atan2(dy, dx) - math.atan2(L2 * math.sin(q2), L1 + L2 * math.cos(q2))
    return q1, q2, r <= L1 + L2


# ------------------------------------------------------------------ main
def main():
    bpy.ops.wm.open_mainfile(filepath=str(ROOT / "scene/out/ruler_ff.blend"))
    sc = bpy.context.scene
    t, mid, sched, src, extra = load_deformation()
    N = len(t)
    sc.frame_start, sc.frame_end, sc.frame_current = 0, NFR - 1, 0
    sc.render.fps, sc.render.fps_base = FPS, 1.0
    ru = bpy.data.objects["RUL_Ruler"]
    nv = len(ru.data.vertices); rest = np.empty(nv * 3); ru.data.vertices.foreach_get("co", rest); rest = rest.reshape(-1, 3)
    origin = np.array(ru.matrix_world.translation)
    assert np.allclose(np.array(ru.matrix_world)[:3, :3], np.eye(3))
    pos, (jj, ii, side) = solid_positions(mid, rest)
    pc2 = OUT.with_name(OUT.stem + "_strip.pc2"); OUT.parent.mkdir(parents=True, exist_ok=True)
    write_pc2(pc2, pos - origin)
    mc = ru.modifiers.new("RUL_StripCache", "MESH_CACHE"); mc.cache_format = "PC2"; mc.filepath = str(pc2)
    mc.time_mode = "FRAME"; mc.play_mode = "SCENE"; mc.frame_start = 0.0; mc.frame_scale = float(SPF)
    mc.interpolation = "LINEAR"; mc.deform_mode = "OVERWRITE"; mc.forward_axis = "POS_Y"; mc.up_axis = "POS_Z"
    ru["deformation_source"] = src; ru["deformation_npz"] = str(DEF)

    # --- tip body: tipbody_matrix_world (solver) maps UNDEFORMED world coords of platform/peg/seated weights to deformed ones
    if "tipbody_matrix_world" in extra:
        TB = np.asarray(extra["tipbody_matrix_world"], np.float64)
    else:  # analytic test npz: rigid tip frame relative to the straight strip
        TB = tip_frames(mid) @ np.linalg.inv(T(R["center_xy_m"][0], R["center_xy_m"][1], R["top_z_m"]))
    tip = bpy.data.objects.new("RUL_TipFrame", None); sc.collection.objects.link(tip)
    tip.empty_display_type = "ARROWS"; tip.empty_display_size = 0.02
    key_matrices(tip, t, TB)
    for o in [o for o in bpy.data.objects if o.name.startswith("RUL_Platform")]:
        o.parent = tip; o.matrix_parent_inverse = Matrix.Identity(4)          # basis = undeformed world pose

    # --- feeder ring (7 pedestals) + feed weights; the bench stack stays as untouched spares
    arm = build_arm()
    tw, nplace = W["thickness_m"], len(sched)
    O_z = SCHED_META.get("O_z", R["top_z_m"] - PL["tab_size_m"][2])
    offs = SCHED_META.get("offsets", [0.015 + tw * j for j in range(nplace)])
    SEATZ = [O_z + offs[j] + 1e-5 * (j + 1) for j in range(nplace)]     # weight bottoms (10 um contact gaps), undeformed coords
    FR, PHI0, DPHI = 0.055, 0.0, math.radians(40.0)
    POS = [0, 2, 4, 6, 1, 3, 5][:nplace] if nplace == 7 else list(range(nplace))   # consecutive weights >= 80 deg apart
    phi = [PHI0 + DPHI * POS[j] for j in range(nplace)]                        # ring opening faces -Y (sway side) and the camera
    H = [SEATZ[j] + 0.004 + 0.0 * j for j in range(nplace)]            # pedestal top = slide height (just above stack top)
    psi = [p + math.pi for p in phi]                                      # slot (+X local) points at the peg
    fcol = bpy.data.collections.new("RUL_Feeder"); sc.collection.children.link(fcol)
    dark = mat("RUL_FeederDarkSatin", (0.035, 0.035, 0.04), 0.42, 0.5)
    zr = O_z + 0.007                                                      # ring height (below the plate top 0.378)
    bm = bmesh.new(); seg_n, tube_n, rt = 64, 12, 0.0025
    a_s = np.linspace(min(phi) - math.radians(8), max(phi) + math.radians(8), seg_n)
    rings = []
    for a in a_s:
        c = np.array([FR * math.cos(a), FR * math.sin(a), zr]); rdir = np.array([math.cos(a), math.sin(a), 0.0])
        rings.append([bm.verts.new(tuple(c + rt * (math.cos(b) * rdir + math.sin(b) * np.array([0, 0, 1.0])))) for b in np.linspace(0, 2 * math.pi, tube_n, endpoint=False)])
    for i in range(seg_n - 1):
        for k in range(tube_n):
            bm.faces.new((rings[i][k], rings[i][(k + 1) % tube_n], rings[i + 1][(k + 1) % tube_n], rings[i + 1][k]))
    bm.faces.new(rings[0][::-1]); bm.faces.new(rings[-1])
    me = bpy.data.meshes.new("RUL_FeederRing"); bm.to_mesh(me); bm.free(); me.polygons.foreach_set("use_smooth", [True] * len(me.polygons))
    ring = bpy.data.objects.new("RUL_FeederRing", me); fcol.objects.link(ring); me.materials.append(dark)
    yb = 0.12
    boxmesh("RUL_FeederBar", 0.006, yb - FR + 0.004, 0.005, (0, (yb + FR) / 2, zr), dark, fcol, 0.001)
    cylmesh("RUL_FeederPost", 0.005, zr + 0.0025, (0, yb, (zr + 0.0025) / 2), dark, fcol)
    cylmesh("RUL_FeederFoot", 0.03, 0.006, (0, yb, 0.003), dark, fcol, 48)
    feeder = [o for o in fcol.objects]
    src_w = bpy.data.objects["RUL_Weight00"]
    wobs, PED = [], []
    for j in range(nplace):
        x, y = FR * math.cos(phi[j]), FR * math.sin(phi[j])
        d = cylmesh(f"RUL_FeederDisc{j}", 0.006, 0.002, (x, y, H[j] - 1e-5 - 0.001), dark, fcol)
        p_ = cylmesh(f"RUL_FeederPin{j}", 0.0015, H[j] - 0.002 - zr, (x, y, zr + (H[j] - 0.002 - zr) / 2), dark, fcol)
        feeder += [d, p_]
        ob = src_w.copy(); ob.name = f"RUL_FeedWeight{j}"; fcol.objects.link(ob)
        for m_ in [m_ for m_ in ob.modifiers if m_.type == "BEVEL"]:
            ob.modifiers.remove(m_)                                      # exact 4.5 mm extent -> exact seat offsets
        PED.append(T(x, y, H[j]) @ Rz(psi[j])); ob.matrix_world = Matrix(PED[j].tolist()); wobs.append(ob)
    GW = T(0, 0, tw / 2) @ Rz(-math.pi / 2)                              # gripper: palm radially outward (local -Y = weight -X)
    iGW = np.linalg.inv(GW)
    tq = lambda x: int(round(x * FPS * SPF))                            # noqa: E731
    TBt = lambda x: TB[min(max(tq(x), 0), N - 1)]                       # noqa: E731
    seat = [T(0, 0, SEATZ[j]) @ Rz(psi[j]) for j in range(nplace)]      # undeformed-world seat poses (Child Of basis)

    # --- gripper waypoint program: (time, kind, pose, open) ; kind 'w' = static world, 't' = tip-relative (undeformed coords)
    keys = []; EXIT_T = [None]
    def above(j, dz):
        return T(0, 0, dz) @ PED[j] @ GW
    keys.append((0.0, "w", T(0, 0, max(H) + tw + 0.004 - H[0]) @ PED[0] @ GW, 1.0))
    seg = []
    for j, t0 in enumerate(sched):
        tp = t0 - 0.24
        Lped = None                                                      # pedestal pose in undeformed coords at pick time
        keys += [(t0 - 0.295, "w", T(0, 0, max(H[j:]) + tw + 0.004 - H[j]) @ PED[j] @ GW, 1.0), (t0 - 0.265, "w", PED[j] @ GW, 1.0), (tp, "w", PED[j] @ GW, 0.0)]
        Lped = np.linalg.inv(TBt(tp)) @ PED[j]
        ce, se = math.cos(phi[j]), math.sin(phi[j])
        keys += [(t0 - 0.22, "t", T(0, 0, 0.0008) @ Lped @ GW, 0.0),
                 (t0 - 0.15, "t", T(0.022 * ce, 0.022 * se, H[j] + 0.0008) @ Rz(psi[j]) @ GW, 0.0),   # exact radial line before the slot meets the peg
                 (t0 - 0.10, "t", T(0, 0, H[j] + 0.0008) @ Rz(psi[j]) @ GW, 0.0),
                 (t0, "t", seat[j] @ GW, 0.0), (t0 + 0.15, "t", seat[j] @ GW, 0.0), (t0 + 0.165, "t", seat[j] @ GW, 1.0)]
        xo, yo = FR * math.cos(phi[j]), FR * math.sin(phi[j])
        keys.append((t0 + 0.19, "t", T(0, 0, H[j] + 0.0008) @ Rz(psi[j]) @ GW, 1.0))       # reverse of the carry path
        keys.append((t0 + 0.235, "t", T(xo, yo, H[j] + 0.0008) @ Rz(psi[j]) @ GW, 1.0))
        nz = max(H[j + 1:] or [H[j] + 0.02]) + tw + 0.004                   # above every remaining feeder weight
        keys.append((t0 + 0.255 if j + 1 < nplace else t0 + 0.27, "t", T(xo, yo, nz) @ Rz(psi[j]) @ GW, 1.0))
        if j + 1 == nplace:
            EXIT_T[0] = t0 + 0.27 + 0.55                                          # straight up, out of the frame (peak ~0.7 m/s)
            keys.append((EXIT_T[0], "w", T(0, 0, ARM["exit_z"] - (TBt(t0 + 0.27) @ T(xo, yo, nz) @ Rz(psi[j]) @ GW)[2, 3]) @ TBt(t0 + 0.27) @ T(xo, yo, nz) @ Rz(psi[j]) @ GW, 1.0))
        seg.append(dict(grip=t0 - 0.265, close=tp, seat=t0, release=t0 + 0.15))
    keys.sort(key=lambda k: k[0])

    def kpose(k, tau):
        return k[2] if k[1] == "w" else TBt(tau) @ k[2]

    G, fing, wmats = [], [], [[] for _ in range(nplace)]
    for n in range(N):
        tau = t[n]
        i = max([q for q in range(len(keys)) if keys[q][0] <= tau + 1e-9])
        A = keys[i]; B = keys[min(i + 1, len(keys) - 1)]
        u = 0.0 if B is A else ss((tau - A[0]) / max(B[0] - A[0], 1e-9))
        M = blend(kpose(A, tau), kpose(B, tau), u)
        G.append(M); fing.append(ARM["finger_open"] * (A[3] * (1 - u) + B[3] * u))
        for j in range(nplace):
            p = seg[j]
            if tau < p["close"] - 1e-9:
                wmats[j].append(PED[j])
            elif tau < p["seat"] - 1e-9:
                wmats[j].append(M @ iGW)
            else:
                wmats[j].append(seat[j])                                 # basis in undeformed coords (Child Of tip body on)
    key_matrices(arm["gripper"], t, G)
    for nm in ("R", "L"):
        f = arm["finger" + nm]; s = f["side"]
        key_matrices(f, t, [T(s * (ARM["finger_r"] + fo), 0, 0) for fo in fing])
    # arm chain from the gripper pose (wrist -> quill -> SCARA)
    Q, C, L1m, L2m, reach_ok, rmax = [], [], [], [], True, 0.0
    for M in G:
        wr = M[:3, 3] + M[:3, :3] @ np.array([0, ARM["palm_y"], ARM["wrist"]])
        zl = ARM["z_link"]
        q1, q2, ok = scara(wr[:2]); reach_ok &= ok
        rmax = max(rmax, math.hypot(wr[0] - ARM["base"][0], wr[1] - ARM["base"][1]))
        bx, by = ARM["base"]
        Q.append(T(*wr)); C.append(T(bx, by, zl) @ Rz(q1))
        L1m.append(T(bx, by, zl + 0.0) @ Rz(q1))
        e = np.array([bx + ARM["L1"] * math.cos(q1), by + ARM["L1"] * math.sin(q1)])
        L2m.append(T(e[0], e[1], zl - 0.020) @ Rz(q1 + q2))
    key_matrices(arm["wrist"], t, Q)
    zq = ARM["z_link"] - 0.020
    key_matrices(arm["quill"], t, [T(m[0, 3], m[1, 3], zq) for m in Q])
    sc_ = np.array([zq - m[2, 3] for m in Q]); arm["quill"].keyframe_insert("scale", index=2, frame=0.0)
    fc = next(f for f in fcurves_of(arm["quill"]) if f.data_path == "scale" and f.array_index == 2)
    fc.keyframe_points.clear(); fc.keyframe_points.add(len(t)); co = np.empty(2 * len(t)); co[0::2] = np.asarray(t) * FPS; co[1::2] = sc_
    fc.keyframe_points.foreach_set("co", co); [setattr(k_, "interpolation", "LINEAR") for k_ in fc.keyframe_points]; fc.update()
    key_matrices(arm["carriage"], t, C); key_matrices(arm["link1"], t, L1m); key_matrices(arm["link2"], t, L2m)
    for k, ob in enumerate(wobs):
        sidx = tq(seg[k]["seat"])
        key_matrices(ob, t, wmats[k], constant_after=sidx - 1)
        con = ob.constraints.new("CHILD_OF"); con.name = "RUL_RideTip"; con.target = tip
        con.inverse_matrix = Matrix.Identity(4); con.set_inverse_pending = False
        con.influence = 0.0; con.keyframe_insert("influence", frame=0.0)
        con.influence = 1.0; con.keyframe_insert("influence", frame=t[sidx] * FPS)
        for fc in fcurves_of(ob):
            if "influence" in fc.data_path:
                for kp in fc.keyframe_points:
                    kp.interpolation = "CONSTANT"
        ob["seat_time_s"] = float(t[sidx]); ob["placement_index"] = k
    TF = TB
    WM = [np.array([TB[n] @ wmats[j][n] if t[n] >= seg[j]["seat"] - 1e-9 else wmats[j][n] for n in range(N)]) for j in range(nplace)]
    GEO = dict(exit_t=EXIT_T[0], G=np.array(G), WM=WM, iGW=iGW, feeder=feeder, O_z=O_z, offs=offs, SEATZ=SEATZ)
    sc["anim_contract"] = json.dumps({"fps": FPS, "frames": [0, NFR - 1], "samples_per_frame": SPF, "deformation": str(DEF),
                                      "source": src, "weight_times_s": [float(t[tq(s)]) for s in sched], "placed_weights": [o.name for o in wobs]})
    sc.render.use_motion_blur = True; sc.render.motion_blur_shutter = 0.5      # PC2 subframes + subframe keys feed the blur
    bpy.ops.wm.save_as_mainfile(filepath=str(OUT), compress=True)

    # ------------------------------------------------------------------ checks
    res = {"deformation": str(DEF), "source": src, "n_samples": N, "weight_times_s": [float(t[tq(s)]) for s in sched],
           "arm_reach_ok": bool(reach_ok), "arm_reach_max_m": rmax, "arm_reach_limit_m": ARM["L1"] + ARM["L2"], "pc2": str(pc2), "pc2_sha256": hashlib.sha256(pc2.read_bytes()).hexdigest()}
    dump = hashlib.sha256()
    for ob in sorted([o for o in bpy.data.objects if o.animation_data and o.animation_data.action], key=lambda o: o.name):
        for fc in sorted(fcurves_of(ob), key=lambda f: (f.data_path, f.array_index)):
            co = np.empty(2 * len(fc.keyframe_points)); fc.keyframe_points.foreach_get("co", co)
            dump.update(ob.name.encode() + fc.data_path.encode() + np.round(co, 12).tobytes())
    res["anim_keys_sha256"] = dump.hexdigest()
    if not NOCHECK:
        res.update(run_checks(sc, ru, mid, pos, rest, origin, jj, ii, side, tip, TF, wobs, seat, seg, arm, t, GEO))
    res["determinism_note"] = "compare pc2_sha256 + anim_keys_sha256 across two builds"
    CHECKS.write_text(json.dumps(res, indent=1)); print("CHECKS", json.dumps({k: v for k, v in res.items() if not isinstance(v, (list, dict))}))
    if RFR:
        render(sc)


def run_checks(sc, ru, mid, pos, rest, origin, jj, ii, side, tip, TF, wobs, seat, seg, arm, t, GEO):
    import collections
    dg = bpy.context.evaluated_depsgraph_get()
    cam = json.loads((ROOT / "config/camera_ff.json").read_text())
    Wpx, Hpx = cam["intrinsics"]["resolution_px"]
    clamp = rest[:, 2] <= R["clamped_length_m"] + 1e-9
    L_npz = np.linalg.norm(np.diff(mid[:, :, NX // 2], axis=1), axis=-1).sum(1)
    N = len(t); dt = t[1] - t[0]
    out = {"max_attach_err_m": 0.0, "max_len_rel_err_vs_npz": 0.0, "max_len_rel_err_vs_0.3m": float(np.max(np.abs(L_npz / R["length_m"] - 1))),
           "max_thickness_err_m": 0.0, "max_cache_readback_err_m": 0.0, "max_tip_follow_err_m": 0.0, "max_camera_px": 0.0,
           "max_strip_below_jaw_outside_clamp_m": 0.0, "intersections": [], "frames_checked": 0, "bvh_evaluations": 0}
    # ---- kinematic checks on the full 1/120 s grid (evaluated in Blender: matrix_world)
    rim = np.array([[0, 0, 0, 1], [0.0125, 0, 0, 1], [-0.0125, 0, 0, 1], [0, 0.0125, 0.0045, 1], [0, -0.0125, 0.0045, 1]]).T
    WW = {o.name: np.zeros((N, 4, 4)) for o in wobs}; GG = np.zeros((N, 4, 4))
    for n in range(N):
        f = t[n] * FPS; sc.frame_set(int(math.floor(f + 1e-9)), subframe=float(f - math.floor(f + 1e-9)))
        for o in wobs:
            WW[o.name][n] = np.array(o.matrix_world)
        GG[n] = np.array(arm["gripper"].matrix_world)
    spd, step, gd, seat_err = {}, {}, {}, {}
    for j, o in enumerate(wobs):
        P = np.einsum("nij,jk->nki", WW[o.name], rim)[:, :, :3]
        v = np.linalg.norm(np.diff(P, axis=0), axis=-1).max(1)
        spd[o.name] = float(v.max() / dt); step[o.name] = float(v.max())
        car = (t >= seg[j]["close"] - 1e-9) & (t <= seg[j]["seat"] + 0.15 + 1e-9)
        rel = np.einsum("nij,njk->nik", GG[car], np.broadcast_to(GEO["iGW"], (car.sum(), 4, 4)))
        gd[o.name] = float(np.abs(rel[:, :3, 3] - WW[o.name][car][:, :3, 3]).max())
        aft = t >= seg[j]["seat"] - 1e-9
        loc = np.einsum("nij,njk->nik", np.linalg.inv(TF[aft]), WW[o.name][aft])
        seat_err[o.name] = float(np.abs(loc[:, 2, 3] - (GEO["O_z"] + GEO["offs"][j])).max())
        out["max_tip_follow_err_m"] = max(out["max_tip_follow_err_m"], float(np.abs(WW[o.name][aft] - np.einsum("nij,jk->nik", TF[aft], seat[j])).max()))
    out["weight_peak_speed_mps"] = spd; out["max_weight_peak_speed_mps"] = max(spd.values())
    out["weight_max_step_per_subframe_m"] = step; out["gripper_weight_dist_while_carried_m"] = gd
    out["seat_offset_err_vs_meta_m"] = seat_err; out["max_seat_offset_err_m"] = max(seat_err.values())
    gv = np.linalg.norm(np.diff(GG[:, :3, 3], axis=0), axis=-1); out["gripper_peak_speed_mps"] = float(gv.max() / dt)
    out["gripper_peak_speed_at_s"] = float(t[int(gv.argmax())])
    out["visibility_animated"] = [o.name for o in bpy.data.objects if o.animation_data and o.animation_data.action and
                                  any("hide" in fc.data_path for fc in fcurves_of(o))]
    # ---- clearance: feeder (static) and gripper vs strip (mid-surface +- t/2) and tip body (plate/tabs/peg)
    def verts_world(o):
        eo = o.evaluated_get(dg); m = eo.to_mesh(); a = np.empty(len(m.vertices) * 3); m.vertices.foreach_get("co", a)
        a = a.reshape(-1, 3); Mw = np.array(eo.matrix_world); eo.to_mesh_clear(); return a @ Mw[:3, :3].T + Mw[:3, 3]
    sc.frame_set(0)
    FV = np.concatenate([verts_world(o) for o in GEO["feeder"]])
    tb_objs = [o for o in bpy.data.objects if o.name.startswith("RUL_Platform")]
    TBV = np.concatenate([np.array([o.matrix_basis @ v.co for v in o.data.vertices]) for o in tb_objs])  # undeformed coords
    TBV = np.c_[TBV, np.ones(len(TBV))]
    strip = mid[:, 150::2, ::5]                                           # rows z > 0.225 m, every 2 mm, 3 columns
    gpts = np.concatenate([np.array([o.matrix_basis @ v.co for v in o.data.vertices]) for o in (arm["gripper"],)])
    cmin_f_strip = cmin_f_tb = cmin_g_strip = 1e9
    for n in range(N):
        sp = strip[n].reshape(-1, 3)
        d = np.sqrt(((sp[:, None, :] - FV[None, ::3, :]) ** 2).sum(-1)).min() - TH / 2
        cmin_f_strip = min(cmin_f_strip, d)
        tbw = (TF[n] @ TBV.T).T[:, :3]
        cmin_f_tb = min(cmin_f_tb, np.sqrt(((tbw[::2, None, :] - FV[None, ::3, :]) ** 2).sum(-1)).min())
        gw = gpts @ GG[n][:3, :3].T + GG[n][:3, 3]
        cmin_g_strip = min(cmin_g_strip, np.sqrt(((sp[:, None, :] - gw[None]) ** 2).sum(-1)).min() - TH / 2)
    out["clearance_feeder_vs_strip_m"] = float(cmin_f_strip); out["clearance_feeder_vs_tipbody_m"] = float(cmin_f_tb)
    out["clearance_gripper_palm_vs_strip_m"] = float(cmin_g_strip)
    # ---- per-frame geometry + BVH intersections: every frame, plus every subframe while the gripper touches a weight
    movers = [o for o in bpy.data.objects if o.name.startswith("RUL_Arm") and o.type == "MESH"] + wobs
    statics = [bpy.data.objects[n] for n in ("RUL_PlatformPeg", "RUL_PlatformPlate", "RUL_PlatformTab", "RUL_PlatformTabBack") if n in bpy.data.objects]
    statics += [o for o in bpy.data.objects if o.name.startswith("RUL_Vise") or o.name.startswith("RUL_Jaw")] + GEO["feeder"]
    statics += [o for o in bpy.data.objects if o.name.startswith("RUL_Weight")]   # bench spares
    armp = [o for o in movers if o.name.startswith("RUL_Arm")]
    pairs = [(ru, m) for m in movers] + [(a, w) for a in armp for w in wobs] + [(a, b) for i, a in enumerate(wobs) for b in wobs[i + 1:]] \
        + [(m, s) for m in movers for s in statics]
    widx = {o.name: k for k, o in enumerate(wobs)}
    gripping = lambda a, o, tau: a.name.startswith("RUL_ArmFinger") and o.name in widx and seg[widx[o.name]]["grip"] - 1e-9 <= tau <= seg[widx[o.name]]["release"] + 0.03  # noqa: E731
    contact = lambda tau: any(s_["grip"] - 0.01 <= tau <= s_["release"] + 0.03 for s_ in seg)  # noqa: E731
    samples = [n for n in range(N) if n % SPF == 0 or contact(t[n])]
    # arm (all in-frame parts) vs strip + tip body after the last release, every sample
    rel_last = seg[-1]["release"]; armm = [o for o in bpy.data.objects if o.name.startswith("RUL_Arm") and o.type == "MESH" and o.name not in ("RUL_ArmBase", "RUL_ArmColumn", "RUL_ArmCarriage", "RUL_ArmLink1", "RUL_ArmLink2")]
    armloc = [(o, np.array([v.co[:] for v in o.data.vertices])) for o in armm]
    tbw_all = TBV; cl = 1e9
    for n in [n for n in range(N) if t[n] >= rel_last + 0.02]:
        f = t[n] * FPS; fi = int(math.floor(f + 1e-9)); sc.frame_set(fi, subframe=float(f - fi))
        A = np.concatenate([lv @ np.array(o.matrix_world)[:3, :3].T + np.array(o.matrix_world)[:3, 3] for o, lv in armloc])
        sp = mid[n][150::2, ::5].reshape(-1, 3); tbw = (TF[n] @ tbw_all.T).T[:, :3]
        sw = np.concatenate([(TF[n] @ np.c_[np.array([seat[j][:3, 3] + [dx, dy, 0] for dx, dy in ((0.0125, 0), (-0.0125, 0), (0, 0.0125), (0, -0.0125))]), np.ones(4)].T).T[:, :3] for j in range(len(wobs))])
        B = np.concatenate([sp, tbw, sw])
        cl = min(cl, float(np.sqrt(((A[::2, None, :] - B[None, :, :]) ** 2).sum(-1)).min()))
    out["clearance_arm_vs_strip_tipbody_after_last_release_m"] = cl; out["arm_exit_time_s"] = GEO["exit_t"]
    for n in samples:
        f = t[n] * FPS; fi = int(math.floor(f + 1e-9)); sc.frame_set(fi, subframe=float(f - fi)); tau = t[n]
        ev = ru.evaluated_get(dg); me = ev.to_mesh(); co = np.empty(len(me.vertices) * 3); me.vertices.foreach_get("co", co)
        co = co.reshape(-1, 3) + origin; ev.to_mesh_clear()
        out["max_cache_readback_err_m"] = max(out["max_cache_readback_err_m"], float(np.abs(co - pos[n]).max()))
        out["max_attach_err_m"] = max(out["max_attach_err_m"], float(np.abs(co[clamp] - (rest[clamp] + origin)).max()))
        midc = np.zeros((NS + 1, NX + 1, 3)); np.add.at(midc, (jj, ii), 0.5 * co)
        L = np.linalg.norm(np.diff(midc[:, NX // 2], axis=0), axis=-1).sum()
        out["max_len_rel_err_vs_npz"] = max(out["max_len_rel_err_vs_npz"], float(abs(L / L_npz[n] - 1)))
        pair = np.zeros((NS + 1, NX + 1, 3)); np.add.at(pair, (jj, ii), side[:, None] * co)
        out["max_thickness_err_m"] = max(out["max_thickness_err_m"], float(np.abs(np.linalg.norm(pair, axis=-1) - TH).max()))
        out["max_strip_below_jaw_outside_clamp_m"] = max(out["max_strip_below_jaw_outside_clamp_m"], float(max(0.0, JAW_Z - 1e-9 - co[~clamp][:, 2].min())))
        out["frames_checked"] += n % SPF == 0
        trees = {}
        for o in set([x for p in pairs for x in p]):
            eo = o.evaluated_get(dg); m2 = eo.to_mesh(); bm = bmesh.new(); bm.from_mesh(m2); bm.transform(eo.matrix_world)
            trees[o.name] = BVHTree.FromBMesh(bm); bm.free(); eo.to_mesh_clear()
        out["bvh_evaluations"] += 1
        for a, b in pairs:
            if gripping(a, b, tau):
                continue
            if trees[a.name].overlap(trees[b.name]):
                out["intersections"].append([round(float(f), 2), a.name, b.name])
        if GEO["exit_t"] is not None and tau >= GEO["exit_t"] - 1e-9:
            c = sc.camera
            for ob_ in [o for o in bpy.data.objects if o.name.startswith("RUL_Arm") and o.type == "MESH"]:
                for v in ob_.bound_box:
                    pb = world_to_camera_view(sc, c, ob_.matrix_world @ Vector(v))
                    if pb.z > 0 and -0.01 <= pb.x <= 1.01 and -0.01 <= pb.y <= 1.01:
                        out["arm_in_frame_after_exit"] = out.get("arm_in_frame_after_exit", 0) + 1
        if n % (4 * SPF) == 0:
            c = sc.camera
            for nm in ("RUL_ArmBase", "RUL_ArmColumn", "RUL_ArmCarriage", "RUL_ArmLink1", "RUL_ArmLink2"):
                ob_ = bpy.data.objects[nm]
                for v in ob_.bound_box:
                    pb = world_to_camera_view(sc, c, ob_.matrix_world @ Vector(v))
                    if pb.z > 0 and 0 <= pb.x <= 1 and 0 <= pb.y <= 1:
                        out["arm_parts_in_frame"] = out.get("arm_parts_in_frame", 0) + 1
        if n in (0, 600, 1200):
            c = sc.camera
            for nm, v in cam["project_test"].items():
                pb = world_to_camera_view(sc, c, Vector(v["world_m"])); d = math.hypot(pb.x * Wpx - v["px_blender"][0], (1 - pb.y) * Hpx - v["px_blender"][1])
                out["max_camera_px"] = max(out["max_camera_px"], d)
    out["n_intersections"] = len(out["intersections"])
    out["intersection_pairs"] = {f"{a}|{b}": [c, min(f for f, x, y in out["intersections"] if (x, y) == (a, b))] for (a, b), c in collections.Counter((a, b) for _, a, b in out["intersections"]).items()}
    out["intersections"] = out["intersections"][:20]
    out["pass"] = {"attach<=1e-6m": out["max_attach_err_m"] <= 1e-6, "length_rel<=1e-6": out["max_len_rel_err_vs_npz"] <= 1e-6,
                   "thickness<=1e-6m": out["max_thickness_err_m"] <= 1e-6, "no_intersections": out["n_intersections"] == 0,
                   "camera<=0.05px": out["max_camera_px"] <= 0.05, "arm_column_links_out_of_frame": out.get("arm_parts_in_frame", 0) == 0,
                   "tip_follow<=1e-6m": out["max_tip_follow_err_m"] <= 1e-6, "weight_peak_speed<=1.0": out["max_weight_peak_speed_mps"] <= 1.0,
                   "weight_peak_speed<=0.8(target)": out["max_weight_peak_speed_mps"] <= 0.8,
                   "gripper_weight_dist<=1e-6m": max(gd.values()) <= 1e-6, "seat_offset<=0.1mm": out["max_seat_offset_err_m"] <= 1e-4,
                   "feeder_clearance>=5mm": min(out["clearance_feeder_vs_strip_m"], out["clearance_feeder_vs_tipbody_m"]) >= 0.005,
                   "gripper_vs_strip>=5mm": out["clearance_gripper_palm_vs_strip_m"] >= 0.005, "no_visibility_animation": not out["visibility_animated"],
                   "arm_out_of_frame_after_exit": out.get("arm_in_frame_after_exit", 0) == 0,
                   "arm_clearance_after_last_release>=5mm": out["clearance_arm_vs_strip_tipbody_after_last_release_m"] >= 0.005}
    return out


def render(sc):
    RDIR.mkdir(parents=True, exist_ok=True)
    sc.render.resolution_percentage = 50
    sc.cycles.samples = int(arg("--samples", "32")); sc.cycles.use_adaptive_sampling = True
    for f in RFR:
        sc.frame_set(f); sc.render.filepath = str(RDIR / f"{RPREFIX}_f{f:03d}.png"); bpy.ops.render.render(write_still=True)
        print("RENDERED", sc.render.filepath)


main()
