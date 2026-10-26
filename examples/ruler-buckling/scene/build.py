"""Batch-2 concept A (ruler buckling): deterministic first-frame scene builder.

  blender -b --factory-startup -P scene/build.py -- [--render OUT.png] [--preview] [--samples N]

Reads config/scene_params.json (1 BU = 1 m, Z up, bench top z=0, camera on -Y looking +Y),
textures from scene/tex (make_textures.py). Writes scene/out/ruler_ff.blend and config/camera_ff.json.
Object names: RUL_* . The ruler (RUL_Ruler) is a subdivided solid (UV: u across width, v along length,
origin at the bottom-centre of the mid-surface) so later FEA results can be baked as a mesh cache.
"""
import json
import math
import pathlib
import sys

import bmesh
import bpy
import numpy as np
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Matrix, Vector

ROOT = pathlib.Path(__file__).resolve().parents[1]
P = json.loads((ROOT / "config/scene_params.json").read_text())
TEX = ROOT / "scene/tex"
argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
RENDER = argv[argv.index("--render") + 1] if "--render" in argv else None
PREVIEW = "--preview" in argv
SAMPLES = int(argv[argv.index("--samples") + 1]) if "--samples" in argv else None


# ------------------------------------------------------------------ helpers
def reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.unit_settings.system = "METRIC"
    sc.unit_settings.scale_length = 1.0
    return sc


def link(ob, coll=None):
    (coll or bpy.context.scene.collection).objects.link(ob)
    return ob


def mesh_obj(name, verts, faces, uvs=None, coll=None):
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(map(float, v)) for v in verts], [], [tuple(int(i) for i in f) for f in faces])
    if uvs is not None:
        uvl = me.uv_layers.new(name="UVMap")
        for poly in me.polygons:
            for li in poly.loop_indices:
                uvl.data[li].uv = uvs[me.loops[li].vertex_index]
    me.validate(); me.update()
    for p in me.polygons:
        p.use_smooth = False
    return link(bpy.data.objects.new(name, me), coll)


def box(name, x0, x1, y0, y1, z0, z1, mat=None, bevel=0.0, coll=None):
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    for v in bm.verts:
        v.co = Vector(((x0 + x1) / 2 + v.co.x * (x1 - x0), (y0 + y1) / 2 + v.co.y * (y1 - y0), (z0 + z1) / 2 + v.co.z * (z1 - z0)))
    me = bpy.data.meshes.new(name); bm.to_mesh(me); bm.free()
    ob = link(bpy.data.objects.new(name, me), coll)
    if bevel > 0:
        m = ob.modifiers.new("bevel", "BEVEL"); m.width = bevel; m.segments = 3; m.limit_method = "ANGLE"
        m.harden_normals = False
        ob.data.polygons.foreach_set("use_smooth", [True] * len(ob.data.polygons))
        ob.modifiers.new("wn", "WEIGHTED_NORMAL").keep_sharp = True
    if mat:
        ob.data.materials.append(mat)
    return ob


def cyl(name, radius, depth, loc, axis="Z", mat=None, verts=48, coll=None, bevel=0.0):
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False, segments=verts, radius1=radius, radius2=radius, depth=depth)
    me = bpy.data.meshes.new(name); bm.to_mesh(me); bm.free()
    ob = link(bpy.data.objects.new(name, me), coll)
    ob.location = loc
    ob.rotation_euler = {"Z": (0, 0, 0), "X": (0, math.pi / 2, 0), "Y": (math.pi / 2, 0, 0)}[axis]
    me.polygons.foreach_set("use_smooth", [True] * len(me.polygons))
    if bevel > 0:
        m = ob.modifiers.new("bevel", "BEVEL"); m.width = bevel; m.segments = 2; m.limit_method = "ANGLE"
        ob.modifiers.new("wn", "WEIGHTED_NORMAL").keep_sharp = True
    if mat:
        me.materials.append(mat)
    return ob


def ring(name, r_in, r_out, t, z0, loc_xy, rot_z, mat, seg=64, coll=None):
    vs, fs = [], []
    a = np.linspace(0, 2 * np.pi, seg, endpoint=False)
    for r, z in ((r_out, 0), (r_out, t), (r_in, t), (r_in, 0)):
        vs += [(r * np.cos(q), r * np.sin(q), z) for q in a]
    for k in range(4):
        k2 = (k + 1) % 4
        for i in range(seg):
            j = (i + 1) % seg
            fs.append((k * seg + i, k * seg + j, k2 * seg + j, k2 * seg + i))
    ob = mesh_obj(name, vs, fs, coll=coll)
    ob.location = (loc_xy[0], loc_xy[1], z0); ob.rotation_euler = (0, 0, rot_z)
    m = ob.modifiers.new("bevel", "BEVEL"); m.width = 0.00025; m.segments = 2; m.limit_method = "ANGLE"
    ob.data.polygons.foreach_set("use_smooth", [True] * len(ob.data.polygons))
    ob.modifiers.new("wn", "WEIGHTED_NORMAL").keep_sharp = True
    ob.data.materials.append(mat)
    return ob


def slotted(name, W, z0, loc_xy, rot_z, mat, coll, n=72):
    """Slotted mass: annulus (hole r_i) with a radial slot of width s along local +X, extruded to thickness t."""
    ro, ri, s, t = W["outer_diameter_m"] / 2, W["hole_diameter_m"] / 2, W["slot_width_m"] / 2, W["thickness_m"]
    a0, a1 = math.asin(s / ro), math.asin(s / ri)
    ts = np.linspace(0, 1, n)
    prof = [(ro, 0.0, a0), (ro, t, a0), (ri, t, a1), (ri, 0.0, a1)]
    vs = [(r * math.cos(a + u * (2 * math.pi - 2 * a)), r * math.sin(a + u * (2 * math.pi - 2 * a)), z) for (r, z, a) in prof for u in ts]
    fs = []
    for k in range(4):
        k2 = (k + 1) % 4
        fs += [(k * n + i, k * n + i + 1, k2 * n + i + 1, k2 * n + i) for i in range(n - 1)]
    fs.append(tuple(k * n for k in (3, 2, 1, 0))); fs.append(tuple(k * n + n - 1 for k in range(4)))
    ob = mesh_obj(name, vs, fs, coll=coll)
    ob.location = (loc_xy[0], loc_xy[1], z0); ob.rotation_euler = (0, 0, rot_z)
    bpy.context.view_layer.objects.active = ob
    m = ob.modifiers.new("bevel", "BEVEL"); m.width = 0.0003; m.segments = 2; m.limit_method = "ANGLE"
    ob.data.polygons.foreach_set("use_smooth", [True] * len(ob.data.polygons))
    ob.modifiers.new("wn", "WEIGHTED_NORMAL").keep_sharp = True
    ob.data.materials.append(mat)
    return ob


class NT:
    def __init__(self, name):
        self.mat = bpy.data.materials.new(name); self.mat.use_nodes = True
        self.t = self.mat.node_tree; self.t.nodes.clear()
        self.out = self.n("ShaderNodeOutputMaterial")
        self.bsdf = self.n("ShaderNodeBsdfPrincipled")
        self.l(self.bsdf.outputs[0], self.out.inputs["Surface"])

    def n(self, kind, **kw):
        nd = self.t.nodes.new(kind)
        for k, v in kw.items():
            setattr(nd, k, v)
        return nd

    def l(self, a, b):
        self.t.links.new(a, b)

    def set(self, **kw):
        names = {"base_color": "Base Color", "roughness": "Roughness", "metallic": "Metallic", "anisotropic": "Anisotropic",
                 "specular_ior_level": "Specular IOR Level", "anisotropic_rotation": "Anisotropic Rotation"}
        for k, v in kw.items():
            if k not in names:
                continue
            s = self.bsdf.inputs[names[k]]
            s.default_value = (*v, 1.0) if k == "base_color" else v
        return self

    def math(self, op, a, b):
        m = self.n("ShaderNodeMath", operation=op)
        for i, v in enumerate((a, b)):
            if v is None:
                continue
            if hasattr(v, "is_output"):
                self.l(v, m.inputs[i])
            else:
                m.inputs[i].default_value = v
        return m.outputs[0]

    def mixc(self, fac, a, b):
        m = self.n("ShaderNodeMix", data_type="RGBA")
        self.l(fac, m.inputs["Factor"])
        for sock, v in ((m.inputs[6], a), (m.inputs[7], b)):
            if hasattr(v, "is_output"):
                self.l(v, sock)
            else:
                sock.default_value = (*v, 1.0)
        return m.outputs[2]


def plain(name, spec):
    return NT(name).set(**spec).mat


def noise_mod(nt, scale, detail=6.0, amp=0.04, base=0.5, coord="Object", stretch=(1, 1, 1)):
    tc = nt.n("ShaderNodeTexCoord")
    mp = nt.n("ShaderNodeMapping"); mp.inputs["Scale"].default_value = stretch
    nt.l(tc.outputs[coord], mp.inputs["Vector"])
    nz = nt.n("ShaderNodeTexNoise"); nz.inputs["Scale"].default_value = scale; nz.inputs["Detail"].default_value = detail
    nt.l(mp.outputs[0], nz.inputs["Vector"])
    v = nt.math("SUBTRACT", nz.outputs["Fac"], 0.5)
    return nt.math("ADD", nt.math("MULTIPLY", v, amp * 2), base)


# ------------------------------------------------------------------ materials
def oak(S):
    """Oak workbench top: distorted wave rings along X (grain runs along X), pores, tool-mark bump, oil-finish roughness."""
    nt = NT("RUL_BenchOak").set(roughness=S["roughness"])
    tc = nt.n("ShaderNodeTexCoord"); mp = nt.n("ShaderNodeMapping"); mp.inputs["Scale"].default_value = (0.15, 3.0, 3.0)
    nt.l(tc.outputs["Object"], mp.inputs["Vector"])
    wv = nt.n("ShaderNodeTexWave", wave_type="BANDS", bands_direction="Y")
    wv.inputs["Scale"].default_value = S["grain_scale"]; wv.inputs["Distortion"].default_value = 9.0
    wv.inputs["Detail"].default_value = 4.0; wv.inputs["Detail Scale"].default_value = 1.5
    nt.l(mp.outputs[0], wv.inputs["Vector"])
    big = noise_mod(nt, 1.2, detail=4.0, amp=0.5, base=0.5)
    fac = nt.math("ADD", nt.math("MULTIPLY", wv.outputs["Fac"], 0.7), nt.math("MULTIPLY", big, 0.3))
    pores = noise_mod(nt, 1.0, detail=2.0, amp=0.5, base=0.5, stretch=(60.0, 2500.0, 2500.0))
    fac = nt.math("MULTIPLY", fac, nt.math("ADD", nt.math("MULTIPLY", pores, 0.35), 0.75))
    nt.l(nt.mixc(fac, S["dark"], S["light"]), nt.bsdf.inputs["Base Color"])
    marks = noise_mod(nt, 1.0, detail=3.0, amp=0.5, base=0.5, stretch=(3.0, 140.0, 140.0))  # plane / tool marks across the grain
    bp = nt.n("ShaderNodeBump"); bp.inputs["Strength"].default_value = 0.35; bp.inputs["Distance"].default_value = 0.0008
    nt.l(nt.math("ADD", marks, nt.math("MULTIPLY", pores, 0.5)), bp.inputs["Height"]); nt.l(bp.outputs[0], nt.bsdf.inputs["Normal"])
    nt.l(nt.math("ADD", nt.math("MULTIPLY", big, 0.25), S["roughness"] - 0.12), nt.bsdf.inputs["Roughness"])
    return nt.mat


def pegboard(S, BD):
    """Painted pegboard: saturated paint with mottling and a hole grid (holes dark)."""
    nt = NT("RUL_Pegboard").set(roughness=S["roughness"])
    tc = nt.n("ShaderNodeTexCoord"); sep = nt.n("ShaderNodeSeparateXYZ"); nt.l(tc.outputs["Object"], sep.inputs[0])
    pch = BD["hole_pitch_m"]
    fx = nt.math("SUBTRACT", nt.math("FRACT", nt.math("DIVIDE", sep.outputs[0], pch), None), 0.5)
    fz = nt.math("SUBTRACT", nt.math("FRACT", nt.math("DIVIDE", sep.outputs[2], pch), None), 0.5)
    r2 = nt.math("ADD", nt.math("MULTIPLY", fx, fx), nt.math("MULTIPLY", fz, fz))
    hole = nt.math("LESS_THAN", r2, (BD["hole_radius_m"] / pch) ** 2)
    mot = noise_mod(nt, 3.0, detail=8.0, amp=0.5, base=0.5)
    paint = nt.mixc(mot, [c * 0.85 for c in S["base_color"]], [min(1, c * 1.12) for c in S["base_color"]])
    nt.l(nt.mixc(hole, paint, S["hole_color"]), nt.bsdf.inputs["Base Color"])
    return nt.mat


def materials():
    M = P["materials"]; mats = {}
    mats["bench"] = oak(M["bench"])
    mats["backdrop"] = pegboard(M["backdrop"], P["backdrop"])
    # cast iron: near-black, slightly rough powder-coat with orange-peel bump
    nt = NT("RUL_CastIron").set(**M["cast_iron"])
    tc = nt.n("ShaderNodeTexCoord"); nz = nt.n("ShaderNodeTexNoise"); nz.inputs["Scale"].default_value = 900.0
    nt.l(tc.outputs["Object"], nz.inputs["Vector"])
    bp = nt.n("ShaderNodeBump"); bp.inputs["Strength"].default_value = 0.08; bp.inputs["Distance"].default_value = 0.0002
    nt.l(nz.outputs["Fac"], bp.inputs["Height"]); nt.l(bp.outputs[0], nt.bsdf.inputs["Normal"])
    nz2 = nt.n("ShaderNodeTexNoise"); nz2.inputs["Scale"].default_value = 180.0; nz2.inputs["Detail"].default_value = 10.0
    nt.l(tc.outputs["Object"], nz2.inputs["Vector"])
    bp.inputs["Strength"].default_value = 0.22; bp.inputs["Distance"].default_value = 0.0004
    hsum = nt.math("ADD", nz.outputs["Fac"], nt.math("MULTIPLY", nz2.outputs["Fac"], 1.5)); nt.l(hsum, bp.inputs["Height"])
    geo = nt.n("ShaderNodeNewGeometry")
    wr = nt.n("ShaderNodeMapRange"); nt.l(geo.outputs["Pointiness"], wr.inputs["Value"]); wr.inputs["From Min"].default_value = 0.53; wr.inputs["From Max"].default_value = 0.60
    wear = nt.math("GREATER_THAN", nz2.outputs["Fac"], 0.70)  # sparse small paint chips (pointiness unreliable on low-poly boxes)
    bare = nt.n("ShaderNodeBsdfPrincipled"); bare.inputs["Base Color"].default_value = (0.32, 0.31, 0.30, 1); bare.inputs["Metallic"].default_value = 1.0
    bare.inputs["Roughness"].default_value = 0.42; nt.l(bp.outputs[0], bare.inputs["Normal"])
    ms = nt.n("ShaderNodeMixShader"); nt.l(wear, ms.inputs[0]); nt.l(nt.bsdf.outputs[0], ms.inputs[1]); nt.l(bare.outputs[0], ms.inputs[2])
    nt.l(ms.outputs[0], nt.out.inputs["Surface"])
    nt.l(nt.mixc(noise_mod(nt, 25.0, amp=0.5, base=0.5), [c * 0.8 for c in M["cast_iron"]["base_color"]], [c * 1.15 for c in M["cast_iron"]["base_color"]]), nt.bsdf.inputs["Base Color"])
    mats["cast_iron"] = nt.mat
    mats["jaw_steel"] = plain("RUL_JawSteel", M["jaw_steel"])
    nt = NT("RUL_JawSerrated").set(**M["jaw_steel"])
    tc = nt.n("ShaderNodeTexCoord"); wv = nt.n("ShaderNodeTexWave", wave_type="BANDS", bands_direction="X", wave_profile="TRI")
    wv.inputs["Scale"].default_value = 0.1 / 0.0016 * 0 + 60.0  # ~1.6 mm pitch across a 0.1 m jaw (object coords)
    sep = nt.n("ShaderNodeSeparateXYZ"); com = nt.n("ShaderNodeCombineXYZ"); nt.l(tc.outputs["Object"], sep.inputs[0])
    nt.l(sep.outputs[2], com.inputs[0]); nt.l(sep.outputs[1], com.inputs[1]); nt.l(com.outputs[0], wv.inputs["Vector"])
    bp = nt.n("ShaderNodeBump"); bp.inputs["Strength"].default_value = 0.6; bp.inputs["Distance"].default_value = 0.0005
    nt.l(wv.outputs["Fac"], bp.inputs["Height"]); nt.l(bp.outputs[0], nt.bsdf.inputs["Normal"])
    mats["jaw_serrated"] = nt.mat
    mats["aluminium"] = plain("RUL_Aluminium", M["aluminium"])
    mats["screw"] = plain("RUL_Screw", M["screw"])
    nt = NT("RUL_Brass").set(**M["brass"])
    fp = noise_mod(nt, 260.0, detail=3.0, amp=0.5, base=0.5)  # smudge / fingerprint scale
    nt.l(nt.math("ADD", nt.math("MULTIPLY", fp, 0.3), M["brass"]["roughness"] - 0.12), nt.bsdf.inputs["Roughness"])
    tarn = noise_mod(nt, 60.0, detail=6.0, amp=0.5, base=0.5)
    nt.l(nt.mixc(tarn, M["brass"]["base_color"], [0.62, 0.43, 0.22]), nt.bsdf.inputs["Base Color"])
    mats["brass"] = nt.mat
    # ruler: satin brushed stainless + ink mask (UV)
    R = M["ruler"]
    nt = NT("RUL_RulerSteel").set(**R)
    nt.bsdf.inputs["Anisotropic Rotation"].default_value = 0.25  # UV tangent is along u (width) -> rotate 90 deg: brushed along length
    tg = nt.n("ShaderNodeTangent", direction_type="UV_MAP"); tg.uv_map = "UVMap"; nt.l(tg.outputs[0], nt.bsdf.inputs["Tangent"])
    uv = nt.n("ShaderNodeUVMap"); uv.uv_map = "UVMap"
    img = nt.n("ShaderNodeTexImage"); img.image = bpy.data.images.load(str(TEX / "ruler_ink.png"))
    img.image.colorspace_settings.name = "Non-Color"; img.interpolation = "Cubic"; img.extension = "CLIP"
    nt.l(uv.outputs[0], img.inputs["Vector"])
    ink = img.outputs["Color"]
    if not P["ruler"]["markings"].get("enabled", True):  # plain strip: no scale
        zv = nt.n("ShaderNodeValue"); zv.outputs[0].default_value = 0.0; ink = zv.outputs[0]
    streak = noise_mod(nt, 1.0, detail=8.0, amp=0.5, base=0.5, stretch=(4000.0, 4000.0, 6.0))
    steel = nt.mixc(streak, [c * 0.92 for c in R["base_color"]], [min(1, c * 1.06) for c in R["base_color"]])
    nt.l(nt.mixc(ink, steel, R["ink_color"]), nt.bsdf.inputs["Base Color"])
    rough_steel = nt.math("ADD", nt.math("MULTIPLY", streak, 0.08), R["roughness"] - 0.04)
    mr = nt.n("ShaderNodeMix", data_type="FLOAT"); nt.l(ink, mr.inputs["Factor"]); nt.l(rough_steel, mr.inputs[2])
    mr.inputs[3].default_value = R["ink_roughness"]; nt.l(mr.outputs[0], nt.bsdf.inputs["Roughness"])
    nt.l(nt.math("SUBTRACT", 1.0, ink), nt.bsdf.inputs["Metallic"])
    mats["ruler"] = nt.mat
    # card: white paper + grid ink (object coords mapped to card extents)
    C = P["grid_card"]
    nt = NT("RUL_GridCard"); nt.set(roughness=M["card"]["roughness"])
    tc = nt.n("ShaderNodeTexCoord")
    mp = nt.n("ShaderNodeMapping")  # object space (origin at card bottom-centre) -> [0,1]^2 on (x, z)
    mp.inputs["Location"].default_value = (0.5, 0, 0)
    mp.inputs["Scale"].default_value = (1 / C["width_m"], 1, 1)
    sep = nt.n("ShaderNodeSeparateXYZ"); com = nt.n("ShaderNodeCombineXYZ")
    nt.l(tc.outputs["Object"], sep.inputs[0])
    nt.l(nt.math("ADD", nt.math("DIVIDE", sep.outputs[0], C["width_m"]), 0.5), com.inputs[0])
    nt.l(nt.math("DIVIDE", sep.outputs[2], C["height_m"]), com.inputs[1])
    img = nt.n("ShaderNodeTexImage"); img.image = bpy.data.images.load(str(TEX / "card_ink.png"))
    img.image.colorspace_settings.name = "Non-Color"; img.interpolation = "Cubic"; img.extension = "CLIP"
    nt.l(com.outputs[0], img.inputs["Vector"])
    a = C["paper_albedo"]
    nt.l(nt.mixc(img.outputs["Color"], [a, a, a * 0.985], M["card"]["ink_color"]), nt.bsdf.inputs["Base Color"])
    mats["card"] = nt.mat
    mats["card_stand"] = plain("RUL_CardStand", {"base_color": [0.03, 0.03, 0.03], "roughness": 0.6})
    return mats


# ------------------------------------------------------------------ geometry
def build_geometry(mats):
    B = P["bench"]; V = P["vise"]; R = P["ruler"]; C = P["grid_card"]; PL = P["platform"]; W = P["slotted_weights"]; BD = P["backdrop"]
    set_coll = bpy.data.collections.new("RUL_Set"); bpy.context.scene.collection.children.link(set_coll)
    bd_coll = bpy.data.collections.new("RUL_BackdropLit"); bpy.context.scene.collection.children.link(bd_coll)
    box("RUL_Bench", B["x_min_m"], B["x_max_m"], B["y_min_m"], B["y_max_m"], -B["thickness_m"], 0.0, mats["bench"], bevel=0.003, coll=set_coll)
    # backdrop: vertical wall with a cove into the bench (warm grey)
    rc = BD["cove_radius_m"]; yw = BD["wall_y_m"]; hw = BD["width_m"] / 2
    if BD.get("style") in ("wall", "pegboard"):
        rc = 0.0
    prof = [(yw, BD["z_min_m"]), (yw, BD["height_m"])] if BD.get("style") in ("wall", "pegboard") else [(yw - rc - 0.15, 0.0005)] + [(yw - rc + rc * math.sin(t), 0.0005 + rc - rc * math.cos(t)) for t in np.linspace(0, math.pi / 2, 12)] + [(yw, BD["height_m"])] if BD.get("style") not in ("wall", "pegboard") else prof
    vs = [(x, y, z) for (y, z) in prof for x in (-hw, hw)]
    fs = [(2 * i, 2 * i + 1, 2 * i + 3, 2 * i + 2) for i in range(len(prof) - 1)]
    bdo = mesh_obj("RUL_Backdrop", vs, fs, coll=bd_coll); bdo.data.materials.append(mats["backdrop"])
    bdo.data.polygons.foreach_set("use_smooth", [True] * len(bdo.data.polygons))

    # --- vise (cast iron), jaws close along Y on the ruler's thickness
    cx, cy = V["center_xy_m"]; zt = V["jaw_top_z_m"]; jw = V["jaw_width_x_m"] / 2; t2 = R["thickness_m"] / 2
    vise = bpy.data.collections.new("RUL_Vise"); bpy.context.scene.collection.children.link(vise)
    bx, by, bz = V["base"]["size_m"]
    box("RUL_ViseBase", cx - bx / 2, cx + bx / 2, cy - by / 2 + 0.01, cy + by / 2 + 0.01, 0.0, bz, mats["cast_iron"], 0.004, vise)
    box("RUL_ViseFixedJaw", cx - jw, cx + jw, cy + t2 + 0.004, cy + 0.050, bz, zt, mats["cast_iron"], 0.003, vise)
    box("RUL_ViseMovingJaw", cx - jw, cx + jw, cy - 0.046, cy - t2 - 0.004, zt - 0.050, zt, mats["cast_iron"], 0.003, vise)
    box("RUL_ViseSlide", cx - 0.032, cx + 0.032, cy - 0.115, cy + 0.045, bz, zt - 0.050, mats["cast_iron"], 0.003, vise)
    box("RUL_ViseEndPlate", cx - 0.038, cx + 0.038, cy - 0.128, cy - 0.110, bz, zt - 0.030, mats["cast_iron"], 0.003, vise)
    jh = V["jaw_height_m"]
    box("RUL_JawInsertFront", cx - jw, cx + jw, cy - t2 - 0.004, cy - t2, zt - jh, zt, mats["jaw_serrated"], 0.0006, vise)
    box("RUL_JawInsertBack", cx - jw, cx + jw, cy + t2, cy + t2 + 0.004, zt - jh, zt, mats["jaw_serrated"], 0.0006, vise)
    FS = V.get("front_screws")
    if FS:  # jaw-insert screws visible on the front face of the moving jaw
        for k, xo in enumerate(FS["x_offsets_m"]):
            cyl(f"RUL_ViseJawScrew{k}", FS["head_diameter_m"] / 2, FS["head_height_m"],
                (cx + xo, cy - 0.046 - FS["head_height_m"] / 2 + 0.0005, zt - FS["z_below_top_m"]), "Y", mats["jaw_steel"], verts=6, coll=vise, bevel=0.0005)
    zs = zt - 0.040
    cyl("RUL_ViseScrew", V["screw_diameter_m"] / 2, 0.070, (cx, cy - 0.150, zs), "Y", mats["jaw_steel"], coll=vise)
    cyl("RUL_ViseHub", 0.013, 0.022, (cx, cy - 0.180, zs), "Y", mats["cast_iron"], coll=vise, bevel=0.002)
    hl = V["handle_length_m"]; hx = cx + 0.045  # handle slid to the right, like a real T-bar left after tightening
    cyl("RUL_ViseHandle", V["handle_diameter_m"] / 2, hl, (hx, cy - 0.180, zs), "X", mats["jaw_steel"], coll=vise)
    for s, nm in ((-1, "L"), (1, "R")):
        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.0085, location=(hx + s * hl / 2, cy - 0.180, zs), segments=32, ring_count=16)
        b = bpy.context.object; b.name = f"RUL_ViseHandleBall{nm}"; b.data.materials.append(mats["jaw_steel"])
        bpy.ops.object.shade_smooth()
        for c in b.users_collection:
            c.objects.unlink(b)
        vise.objects.link(b)

    # --- ruler: subdivided mid-surface grid -> solidify (applied) ; origin at bottom-centre of mid-surface
    nx, nz = R["mesh_segments_width"], R["mesh_segments_length"]
    xs = np.linspace(-R["width_m"] / 2, R["width_m"] / 2, nx + 1); zz = np.linspace(0.0, R["length_m"], nz + 1)
    verts = [(x, 0.0, z) for z in zz for x in xs]
    uvs = [((x + R["width_m"] / 2) / R["width_m"], z / R["length_m"]) for z in zz for x in xs]
    faces = [(j * (nx + 1) + i, j * (nx + 1) + i + 1, (j + 1) * (nx + 1) + i + 1, (j + 1) * (nx + 1) + i) for j in range(nz) for i in range(nx)]
    ru = mesh_obj("RUL_Ruler", verts, faces, uvs)
    ru.location = (R["center_xy_m"][0], R["center_xy_m"][1], R["bottom_z_m"])
    sm = ru.modifiers.new("solidify", "SOLIDIFY"); sm.thickness = R["thickness_m"]; sm.offset = 0.0; sm.use_even_offset = True
    bpy.context.view_layer.objects.active = ru; ru.select_set(True)
    bpy.ops.object.modifier_apply(modifier="solidify"); ru.select_set(False)
    ru.data.materials.append(mats["ruler"])
    # Face normals: the grid's normal is -Y (verts ordered x then z, CCW seen from -Y) -> front face; fine for UVs.

    # --- aluminium platform on the top end + 2 screws on the front tab
    zt_r = R["top_z_m"]; px, py, pz = PL["plate_size_m"]
    box("RUL_PlatformPlate", -px / 2, px / 2, -py / 2, py / 2, zt_r, zt_r + pz, mats["aluminium"], 0.0004)
    tx, ty, tz = PL["tab_size_m"]
    box("RUL_PlatformTab", -tx / 2, tx / 2, -t2 - ty, -t2, zt_r - tz, zt_r, mats["aluminium"], 0.0003)
    box("RUL_PlatformTabBack", -tx / 2, tx / 2, t2, t2 + ty, zt_r - tz, zt_r, mats["aluminium"], 0.0003)
    S = PL["screws"]
    for k, xo in enumerate(S["x_offsets_m"]):
        cyl(f"RUL_PlatformScrew{k}", S["head_diameter_m"] / 2, S["head_height_m"], (xo, -t2 - ty - S["head_height_m"] / 2, zt_r - S["z_from_top_m"]),
            "Y", mats["screw"], verts=32, bevel=0.0004)

    # --- peg on the platform (slotted weights stack on it)
    PG = PL["peg"]
    cyl("RUL_PlatformPeg", PG["diameter_m"] / 2, PG["height_m"], (0.0, 0.0, zt_r + pz + PG["height_m"] / 2), "Z", mats["screw"], verts=32, bevel=0.0003)
    # --- slotted brass weights, neat stack
    rng = np.random.default_rng(W["seed"])
    wc = bpy.data.collections.new("RUL_Weights"); bpy.context.scene.collection.children.link(wc)
    for i in range(W["count"]):
        j = rng.uniform(-W["jitter_xy_m"], W["jitter_xy_m"], 2)
        rot = math.radians(W["slot_azimuth_deg"] + rng.uniform(-W["jitter_rot_deg"], W["jitter_rot_deg"]))
        slotted(f"RUL_Weight{i:02d}", W, i * W["thickness_m"], (W["stack_center_xy_m"][0] + j[0], W["stack_center_xy_m"][1] + j[1]), rot, mats["brass"], wc)

    gc = None
    if C.get("enabled", True):
        # --- grid card (origin bottom-centre), with a black wedge stand behind it
        gc = mesh_obj("RUL_GridCard", [], [], coll=bd_coll)
        bm = bmesh.new(); bmesh.ops.create_cube(bm, size=1.0)
        for v in bm.verts:
            v.co = Vector((v.co.x * C["width_m"], v.co.y * C["thickness_m"], (v.co.z + 0.5) * C["height_m"]))
        bm.to_mesh(gc.data); bm.free(); gc.data.materials.append(mats["card"])
        gc.location = (C["x_center_m"], C["y_center_m"], C["bottom_z_m"])
        yb = C["y_center_m"] + C["thickness_m"] / 2
        for k, x in enumerate(C.get("_stand_x_m", (-0.12, 0.12))):
            box(f"RUL_CardStand{k}", x - 0.015, x + 0.015, yb, yb + 0.05, 0.0, 0.03, mats["card_stand"], 0.002, set_coll)
    pc = bpy.data.collections.new("RUL_Props"); bpy.context.scene.collection.children.link(pc)
    for pr in P.get("props", []):
        mt = plain("RUL_Prop_" + pr["name"], {"base_color": pr["color"], "roughness": 0.35 if pr.get("metal") else 0.45, "metallic": 1.0 if pr.get("metal") else 0.0})
        if pr["kind"] in ("cyl", "cyl_z"):
            cyl("RUL_Prop_" + pr["name"], pr["r"], pr["len"], tuple(pr["loc"]), "Z", mt, coll=pc, bevel=0.001)
        elif pr["kind"] == "box":
            sx, sy, sz = pr["size"]; x, y, z = pr["loc"]
            box("RUL_Prop_" + pr["name"], x - sx / 2, x + sx / 2, y - sy / 2, y + sy / 2, z - sz / 2, z + sz / 2, mt, 0.004, pc)
        elif pr["kind"] == "torus":
            bpy.ops.mesh.primitive_torus_add(major_radius=pr["R"], minor_radius=pr["r"], location=tuple(pr["loc"]), rotation=(math.pi / 2, 0, 0))
            o = bpy.context.object; o.name = "RUL_Prop_" + pr["name"]; o.data.materials.append(mt); bpy.ops.object.shade_smooth()
    rc_ = bpy.data.collections.new("RUL_RulerLit")  # light-link target only (objects stay in their own collections)
    for o in [ru] + [o for o in bpy.data.objects if o.name.startswith("RUL_Platform")]:
        rc_.objects.link(o)
    return dict(ruler=ru, card=gc, backdrop=bdo, bd_coll=bd_coll, ruler_coll=rc_)


# ------------------------------------------------------------------ lights, camera, render
def aim(ob, loc, target):
    ob.location = loc
    d = Vector(target) - Vector(loc)
    ob.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()


def lights(objs):
    L = P["lights"]
    for name in [k for k in ("key", "fill", "top", "back", "ruler_reflect", "rim") if k in L]:
        s = L[name]
        ld = bpy.data.lights.new(f"RUL_Light_{name}", "AREA"); ld.shape = "RECTANGLE"
        ld.size, ld.size_y = s["size_m"]; ld.energy = s["energy_W"]; ld.color = s["color"]
        ob = link(bpy.data.objects.new(f"RUL_Light_{name}", ld)); aim(ob, s["loc_m"], s["target_m"])
        if name == "back":
            ob.light_linking.receiver_collection = objs["bd_coll"]
        if name == "ruler_reflect":
            ob.light_linking.receiver_collection = objs["ruler_coll"]
    w = bpy.data.worlds.new("RUL_World"); w.use_nodes = True
    w.node_tree.nodes["Background"].inputs["Color"].default_value = (0.5, 0.48, 0.45, 1)
    w.node_tree.nodes["Background"].inputs["Strength"].default_value = L["world_strength"]
    bpy.context.scene.world = w


def camera():
    c = P["camera"]
    cd = bpy.data.cameras.new("RUL_Camera"); cd.lens = c["lens_mm"]; cd.sensor_width = c["sensor_width_mm"]
    cd.sensor_fit = c["sensor_fit"]; cd.shift_x, cd.shift_y = c["shift"]; cd.clip_start = c["clip_start_m"]; cd.clip_end = c["clip_end_m"]
    ob = link(bpy.data.objects.new("RUL_Camera", cd))
    p = math.radians(c["pitch_down_deg"]); t = Vector(c["target_m"]); d = c["distance_m"]
    az = math.radians(c.get("azimuth_deg", 0.0))
    ob.location = t + Vector((d * math.sin(az) * math.cos(p), -d * math.cos(az) * math.cos(p), d * math.sin(p)))
    ob.rotation_euler = (math.pi / 2 - p, 0.0, az)
    ob.lock_location = ob.lock_rotation = ob.lock_scale = (True, True, True)
    if c.get("dof", {}).get("use"):
        cd.dof.use_dof = True; bpy.context.view_layer.update()
        fpt = Vector((0.0, 0.0, 0.225)); fdir = ob.matrix_world.to_3x3() @ Vector((0, 0, -1))
        cd.dof.focus_distance = (fpt - ob.location).dot(fdir); cd.dof.aperture_fstop = c["dof"]["f_stop"]; cd.dof.aperture_blades = c["dof"]["blades"]
    sc = bpy.context.scene; sc.camera = ob
    sc.render.resolution_x, sc.render.resolution_y = c["resolution_px"]; sc.render.resolution_percentage = 100
    return ob


def render_settings():
    r = P["render"]; sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    try:
        pr = bpy.context.preferences.addons["cycles"].preferences; pr.compute_device_type = "METAL"; pr.get_devices()
        for dv in pr.devices:
            dv.use = True
        sc.cycles.device = "GPU"
    except Exception as e:  # noqa
        print("GPU setup failed, CPU:", e)
    sc.cycles.samples = SAMPLES or (48 if PREVIEW else r["samples"])
    sc.cycles.use_denoising = r["denoise"]; sc.cycles.use_adaptive_sampling = True
    sc.cycles.max_bounces = 8
    sc.view_settings.view_transform = r["view_transform"]; sc.view_settings.look = r["look"]; sc.view_settings.exposure = r["exposure"]
    sc.render.film_transparent = False
    sc.render.image_settings.file_format = "PNG"; sc.render.image_settings.color_depth = "8"
    if PREVIEW:
        sc.render.resolution_percentage = 50


def camera_json(cam, objs):
    sc = bpy.context.scene; cd = cam.data
    W, H = P["camera"]["resolution_px"]
    fx = cd.lens / cd.sensor_width * W; fy = fx  # sensor_fit HORIZONTAL, square pixels
    cx = W / 2 - cd.shift_x * max(W, H); cy = H / 2 + cd.shift_y * max(W, H)
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    Mwc = np.array(cam.matrix_world.inverted())  # world -> Blender camera (x right, y up, -z forward)
    flip = np.diag([1, -1, -1, 1]); T = flip @ Mwc   # world -> OpenCV camera (x right, y down, z forward)
    R_, t_ = T[:3, :3], T[:3, 3]

    def proj(p):
        pc = R_ @ np.asarray(p) + t_; uv = K @ (pc / pc[2]); return [float(uv[0]), float(uv[1])], float(pc[2])

    def bl(p):
        v = world_to_camera_view(sc, cam, Vector(p)); return [float(v.x * W), float((1 - v.y) * H)]

    R = P["ruler"]; C = P["grid_card"]; lines = json.loads((TEX / "card_lines.json").read_text())
    yf = -R["thickness_m"] / 2; ycard = C["y_center_m"] - C["thickness_m"] / 2
    pts = {
        "ruler_tip_centre_front": [0.0, yf, R["top_z_m"]],
        "ruler_top_left_front": [-R["width_m"] / 2, yf, R["top_z_m"]],
        "ruler_top_right_front": [R["width_m"] / 2, yf, R["top_z_m"]],
        "ruler_at_jaw_top_centre": [0.0, yf, P["vise"]["jaw_top_z_m"]],
        "platform_top_centre": [0.0, 0.0, R["top_z_m"] + P["platform"]["plate_size_m"][2]],
        "peg_top_centre": [0.0, 0.0, R["top_z_m"] + P["platform"]["plate_size_m"][2] + P["platform"]["peg"]["height_m"]],
    }
    pts["weights_stack_top_centre"] = [P["slotted_weights"]["stack_center_xy_m"][0], P["slotted_weights"]["stack_center_xy_m"][1], P["slotted_weights"]["count"] * P["slotted_weights"]["thickness_m"]]
    zpk = R["top_z_m"] + P["platform"]["plate_size_m"][2] + P["slotted_weights"]["count"] * P["slotted_weights"]["thickness_m"]
    for s, nm in ((-1, "minusY"), (1, "plusY"), (-1, "minusX"), (1, "plusX")):
        off = [0.0, 0.15 * s, zpk - 0.03] if "Y" in nm else [0.15 * s, 0.0, zpk - 0.03]
        pts[f"bend_room_tip_{nm}_150mm_with_12_weights"] = off
    for nm, x, z in () if not C.get("enabled", True) else (("BL", lines["x_lines_m"][0], lines["z_lines_m"][0]), ("BR", lines["x_lines_m"][-1], lines["z_lines_m"][0]),
                     ("TL", lines["x_lines_m"][0], lines["z_lines_m"][-1]), ("TR", lines["x_lines_m"][-1], lines["z_lines_m"][-1]),
                     ("TL_inframe_z400mm", lines["x_lines_m"][0], 0.40), ("TR_inframe_z400mm", lines["x_lines_m"][-1], 0.40),
                     ("origin_x0_z100mm", 0.0, 0.10)):
        pts[f"grid_{nm}"] = [x, ycard, z]
    test = {}
    for k, p in pts.items():
        uv, depth = proj(p); ub = bl(p)
        test[k] = {"world_m": p, "px_pinhole": [round(v, 3) for v in uv], "px_blender": [round(v, 3) for v in ub],
                   "diff_px": round(float(np.hypot(uv[0] - ub[0], uv[1] - ub[1])), 6), "depth_m": round(depth, 5),
                   "in_frame": bool(0 <= uv[0] < W and 0 <= uv[1] < H)}
    tip, jaw = test["ruler_tip_centre_front"]["px_pinhole"], test["ruler_at_jaw_top_centre"]["px_pinhole"]
    out = {
        "_doc": "Pinhole camera of record for firstframe/frame0.png. OpenCV convention: x right, y down, z forward; x_cam = R @ x_world + t; "
                "u = fx*X/Z + cx, v = fy*Y/Z + cy in continuous pixel coords where the image spans [0,W]x[0,H] (pixel i's centre is at i+0.5). "
                "No distortion. Units: metres, pixels. project_test compares this pinhole with Blender's own world_to_camera_view.",
        "intrinsics": {"focal_mm": cd.lens, "sensor_width_mm": cd.sensor_width, "sensor_height_mm_effective": cd.sensor_width * H / W,
                       "sensor_fit": cd.sensor_fit, "resolution_px": [W, H], "shift_xy": [cd.shift_x, cd.shift_y],
                       "fx_px": fx, "fy_px": fy, "cx_px": cx, "cy_px": cy, "K": K.tolist(), "distortion": [0, 0, 0, 0, 0],
                       "hfov_deg": math.degrees(2 * math.atan(cd.sensor_width / 2 / cd.lens)),
                       "vfov_deg": math.degrees(2 * math.atan(cd.sensor_width * H / W / 2 / cd.lens))},
        "extrinsics_world_to_camera_opencv": {"R": R_.tolist(), "t_m": t_.tolist(), "T_4x4": T.tolist()},
        "blender_camera": {"location_m": list(cam.location), "rotation_euler_xyz_rad": list(cam.rotation_euler),
                           "matrix_world": [list(r) for r in cam.matrix_world], "pitch_down_deg": P["camera"]["pitch_down_deg"]},
        "scale_px_per_mm_at_ruler_plane": fx / (-(np.array(cam.matrix_world.inverted()) @ np.array([0, 0, 0.2, 1]))[2]) / 1000.0,
        "ruler_visible_length_px": round(float(np.hypot(tip[0] - jaw[0], tip[1] - jaw[1])), 2),
        "ruler_full_length_frac_of_height": round(float(R["length_m"] * fx / test["ruler_tip_centre_front"]["depth_m"] / H), 4),
        "project_test": test,
        "max_pinhole_vs_blender_diff_px": max(v["diff_px"] for v in test.values()),
    }
    (ROOT / "config/camera_ff.json").write_text(json.dumps(out, indent=1))
    print("camera_ff.json max diff px", out["max_pinhole_vs_blender_diff_px"], "tip", tip)


def main():
    sc = reset()
    mats = materials()
    objs = build_geometry(mats)
    lights(objs)
    cam = camera()
    render_settings()
    sc.frame_start = sc.frame_end = sc.frame_current = 1
    bpy.context.view_layer.update()
    camera_json(cam, objs)
    out = ROOT / "scene/out"; out.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(out / "ruler_ff.blend"), compress=True)
    bpy.ops.file.make_paths_relative()  # textures as //../tex/*.png (relative to the .blend)
    bpy.ops.wm.save_mainfile(compress=True)
    if RENDER:
        sc.render.filepath = str(pathlib.Path(RENDER).resolve())
        bpy.ops.render.render(write_still=True)
        print("RENDERED", sc.render.filepath)


main()
